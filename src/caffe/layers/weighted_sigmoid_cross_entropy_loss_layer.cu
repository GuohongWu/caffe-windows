#include <vector>

#include "caffe/layers/weighted_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void WeightedSigmoidCrossEntropyLossComputeWeights(const int nthreads,
		const Dtype* target, const bool has_ignore_label_, const int ignore_label_,
		int* valid_count, int* pos_count) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			const int target_value = static_cast<int>(target[i]);
			if (has_ignore_label_ && target_value == ignore_label_) {
				continue;
			}

			++(*valid_count);

			if (target_value == 1)
				++(*pos_count);
		}
	}

	template <typename Dtype>
	__global__ void WeightedSigmoidCrossEntropyLossForwardGPU(const int nthreads,
		const Dtype* input_data, const Dtype* target, Dtype* loss,
		const bool has_ignore_label_, const int ignore_label_,
		Dtype pos_weight_, Dtype neg_weight_) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			const int target_value = static_cast<int>(target[i]);
			if (has_ignore_label_ && target_value == ignore_label_) {
				loss[i] = 0;
			}
			else {
				loss[i] = ((target_value == 1) ? pos_weight_ : neg_weight_) * (input_data[i] * (target[i] - (input_data[i] >= 0)) -
					log(1 + exp(input_data[i] - 2 * input_data[i] *
					(input_data[i] >= 0))));
			}
		}
	}

	template <typename Dtype>
	__global__ void WeightedSigmoidCrossEntropyLossIgnoreDiffGPU(const int count,
		const int ignore_label, const Dtype* target, Dtype* diff) {
		CUDA_KERNEL_LOOP(i, count) {
			const int target_value = static_cast<int>(target[i]);
			if (target_value == ignore_label) {
				diff[i] = 0;
			}
		}
	}


	template <typename Dtype>
	void WeightedSigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// The forward pass computes the sigmoid outputs.
		sigmoid_bottom_vec_[0] = bottom[0];
		sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
		// Compute the loss (negative log likelihood)
		const int count = bottom[0]->count();
		// Stable version of loss computation from input data
		const Dtype* input_data = bottom[0]->gpu_data();
		const Dtype* target = bottom[1]->gpu_data();
		// Since this memory is not used for anything until it is overwritten
		// on the backward pass, we use it here to avoid having to allocate new GPU
		// memory to accumulate intermediate results in the kernel.
		Dtype* loss_data = bottom[0]->mutable_gpu_diff();
		//Dtype* count_data = bottom[1]->mutable_gpu_diff();
		int valid_count = 0;
		int pos_count = 0;
		//WeightedSigmoidCrossEntropyLossComputeWeights<Dtype> << <CAFFE_GET_BLOCKS(count),
		//	CAFFE_CUDA_NUM_THREADS >> >(count, target,
		//		has_ignore_label_, ignore_label_, &valid_count, &pos_count);

		const Dtype* target_cpu = bottom[1]->cpu_data();
		for (int i = 0; i < count; ++i) {
			const int target_value = static_cast<int>(target_cpu[i]);
			if (has_ignore_label_ && target_value == ignore_label_) {
				continue;
			}
			++valid_count;

			if (target_value == 1)
				++pos_count;
		}
		if (pos_count == 0)
			pos_count = 1;
		else if (pos_count == valid_count)
			pos_count = valid_count - 1;
		this->pos_weight_ = Dtype(0.5) * Dtype(valid_count) / Dtype(pos_count); // n/(2*k)
		this->neg_weight_ = Dtype(0.5) * Dtype(valid_count) / Dtype(valid_count - pos_count); // n/(2*(n-k))

		// NOLINT_NEXT_LINE(whitespace/operators)
		WeightedSigmoidCrossEntropyLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, input_data, target, loss_data,
				has_ignore_label_, ignore_label_, this->pos_weight_, this->neg_weight_);
		
		Dtype loss;
		caffe_gpu_asum(count, loss_data, &loss);
		normalizer_ = get_normalizer(normalization_, valid_count);
		top[0]->mutable_cpu_data()[0] = loss / normalizer_;
	}

	template <typename Dtype>
	void WeightedSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]) {
			// First, compute the diff
			const int count = bottom[0]->count();
			const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
			const Dtype* target = bottom[1]->gpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

			caffe_gpu_mul(count, sigmoid_output_data, target, bottom_diff);  // Pn * Pn^
			caffe_gpu_axpby(count, this->neg_weight_, sigmoid_output_data, this->pos_weight_ - this->neg_weight_, bottom_diff);  // neg_weight_ * Pn^ + (pos_weight_ - neg_weight_) * bottom_diff
			caffe_gpu_axpy(count, -this->pos_weight_, target, bottom_diff);  // bottom_diff - pos_weight_ * Pn

			//caffe_copy(count, sigmoid_output_data, bottom_diff);
			//caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
			// Zero out gradient of ignored targets.
			if (has_ignore_label_) {
				// NOLINT_NEXT_LINE(whitespace/operators)
				WeightedSigmoidCrossEntropyLossIgnoreDiffGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
					CAFFE_CUDA_NUM_THREADS >> >(count, ignore_label_, target, bottom_diff);
			}
			// Scale down gradient
			Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
			caffe_gpu_scal(count, loss_weight, bottom_diff);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(WeightedSigmoidCrossEntropyLossLayer);

}  // namespace caffe
