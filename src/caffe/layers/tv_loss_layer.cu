#include <vector>

#include "caffe/layers/tv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void mask_kernel(const int count, const int size,
		const int H, const int W, Dtype* mask) {
		CUDA_KERNEL_LOOP(i, count) {
			const int unit_pos = i % size;
			if (unit_pos % W == W - 1 || unit_pos / W == H - 1) {
				mask[i] = (Dtype)0;
			}
			else {
				mask[i] = (Dtype)1;
			}
		}
	}

	template <typename Dtype>
	void TVLossLayer<Dtype>::create_mask_gpu(const int count,
		const int H, const int W, Dtype* mask) {
		// NOLINT_NEXT_LINE(whitespace/operators)
		mask_kernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, H*W, H, W, mask);
	}

	template void TVLossLayer<float>::create_mask_gpu(const int count,
		const int H, const int W, float* mask);
	template void TVLossLayer<double>::create_mask_gpu(const int count,
		const int H, const int W, double* mask);

	template <typename Dtype>
	void TVLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int W = bottom[0]->shape(-1);
		const int count = bottom[0]->count();
		caffe_gpu_sub(count - 1, bottom[0]->gpu_data(), &bottom[0]->gpu_data()[1],
			x_diff_.mutable_gpu_data());
		caffe_gpu_mul(count, x_diff_.gpu_data(), mask_.gpu_data(),
			x_diff_.mutable_gpu_data());
		caffe_gpu_sub(count - W, bottom[0]->gpu_data(), &bottom[0]->gpu_data()[W],
			y_diff_.mutable_gpu_data());
		caffe_gpu_mul(count, y_diff_.gpu_data(), mask_.gpu_data(),
			y_diff_.mutable_gpu_data());
		caffe_gpu_mul(count, x_diff_.gpu_data(), x_diff_.gpu_data(),
			grad_norm_.mutable_gpu_data());  // X_diff^2
		caffe_gpu_mul(count, y_diff_.gpu_data(), y_diff_.gpu_data(),
			tmp_.mutable_gpu_data());  // Y_diff^2
		caffe_gpu_add(count, tmp_.gpu_data(), grad_norm_.gpu_data(),
			grad_norm_.mutable_gpu_data());  // X_diff^2 + Y_diff^2
		caffe_gpu_powx(count, grad_norm_.gpu_data(),
			(Dtype)this->layer_param_.tv_loss_param().beta() / 2,
			tmp_.mutable_gpu_data());  // (X_diff^2 + Y_diff^2)^(beta/2)
		caffe_gpu_asum(count, tmp_.gpu_data(), top[0]->mutable_cpu_data());
	}

	template <typename Dtype>
	void TVLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const int W = bottom[0]->shape(-1);
		const int count = bottom[0]->count();
		caffe_gpu_powx(count, grad_norm_.gpu_data(),
			(Dtype)this->layer_param_.tv_loss_param().beta() / 2 - 1,
			grad_norm_.mutable_gpu_data());
		caffe_gpu_scal(count, (Dtype)this->layer_param_.tv_loss_param().beta() / 2,
			grad_norm_.mutable_gpu_data());
		caffe_gpu_mul(count, x_diff_.gpu_data(), grad_norm_.gpu_data(),
			x_diff_.mutable_gpu_data());
		caffe_gpu_scal(count, (Dtype)2, x_diff_.mutable_gpu_data());  // dX_diff
		caffe_gpu_mul(count, y_diff_.gpu_data(), grad_norm_.gpu_data(),
			y_diff_.mutable_gpu_data());
		caffe_gpu_scal(count, (Dtype)2, y_diff_.mutable_gpu_data());  // dY_diff
		caffe_gpu_axpy(count, (Dtype)1, x_diff_.gpu_data(),
			bottom[0]->mutable_gpu_diff());
		caffe_gpu_axpy(count, (Dtype)1, y_diff_.gpu_data(),
			bottom[0]->mutable_gpu_diff());
		caffe_gpu_axpy(count - 1, (Dtype)-1, x_diff_.gpu_data(),
			&bottom[0]->mutable_gpu_diff()[1]);
		caffe_gpu_axpy(count - W, (Dtype)-1, y_diff_.gpu_data(),
			&bottom[0]->mutable_gpu_diff()[W]);
		caffe_gpu_scal(count, top[0]->cpu_diff()[0], bottom[0]->mutable_gpu_diff());
	}


	INSTANTIATE_LAYER_GPU_FUNCS(TVLossLayer);

}  // namespace caffe