#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>

#include "caffe/layers/MHE_loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void kernel_Forward(const int num, const int cls_num, const int feat_dim,
		const Dtype* bottom_data, const Dtype* label_data, Dtype* dot_prod_data, Dtype* loss_eltwise) {
		CUDA_KERNEL_LOOP(k, num) {
			int b_i = k / cls_num;
			int c_k = k % cls_num;
			loss_eltwise[k] = Dtype(0);

			int gt_label = static_cast<int>(label_data[b_i]);
			if (c_k == gt_label)
				continue;

			const Dtype* bottom_data_label = bottom_data + gt_label * feat_dim;
			const Dtype* bottom_data_k = bottom_data + c_k * feat_dim;
			Dtype dot_prod_result(0);
			for (int i = 0; i < feat_dim; ++i) {
				dot_prod_result += bottom_data_label[i] * bottom_data_k[i];
			}
			dot_prod_data[k] = Dtype(1) - dot_prod_result;
			loss_eltwise[k] = Dtype(1) - log(max(dot_prod_data[k], Dtype(FLT_MIN)));
		}
	}

	// -= or += may produce wrong results
	template <typename Dtype>
	__global__ void kernel_Backward(const int num, const int cls_num, const int feat_dim,
		const Dtype* bottom_data, const Dtype* label_data, const Dtype* dot_prod_data, const Dtype alpha, Dtype* bout) {
		CUDA_KERNEL_LOOP(k, num) {
			int b_i = k / cls_num;
			int c_k = k % cls_num;

			int gt_label = static_cast<int>(label_data[b_i]);
			if (c_k == gt_label)
				continue;

			Dtype scal_ = alpha / max(dot_prod_data[k], Dtype(FLT_MIN));

			const Dtype* bottom_data_label = bottom_data + gt_label * feat_dim;
			const Dtype* bottom_data_k = bottom_data + c_k * feat_dim;
			Dtype* bout_label = bout + gt_label * feat_dim;
			Dtype* bout_k = bout + c_k * feat_dim;
			for (int i = 0; i < feat_dim; ++i) {
				bout_label[i] += scal_ * bottom_data_k[i];
				bout_k[i] += scal_ * bottom_data_label[i];
			}
		}
	}



	template <typename Dtype>
	void MHELossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		if (this->use_lambda_curve) {
			this->lambda_m = this->lambda_max * (Dtype(1) - pow(Dtype(1) + this->gamma * this->iter, -this->power));
			this->iter += Dtype(1);
		}

		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* label_data = bottom[1]->gpu_data();
		Dtype* dot_prod_data = this->dot_prod_.mutable_gpu_data();
		Dtype* loss_temp_data = this->loss_temp_.mutable_gpu_data();

		int cls_num = bottom[0]->num();
		int batch_num = bottom[1]->num();
		int feat_dim = bottom[0]->count() / cls_num;

		kernel_Forward<Dtype> << <CAFFE_GET_BLOCKS(batch_num * cls_num),
			CAFFE_CUDA_NUM_THREADS >> >(batch_num * cls_num, cls_num, feat_dim, bottom_data, label_data, dot_prod_data, loss_temp_data);

		Dtype loss;
		caffe_gpu_asum(this->loss_temp_.count(), loss_temp_data, &loss);
		top[0]->mutable_cpu_data()[0] = (loss / Dtype(batch_num * (cls_num - 1)) - Dtype(1)) * this->lambda_m;

		if (top.size() == 2)
			top[1]->mutable_cpu_data()[0] = this->lambda_m;
	}

	template <typename Dtype>
	void MHELossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* dot_prod_data = this->dot_prod_.mutable_cpu_data();

		int cls_num = bottom[0]->num();
		int batch_num = bottom[1]->num();
		int feat_dim = bottom[0]->count() / cls_num;

		const Dtype alpha = top[0]->cpu_diff()[0] / Dtype(batch_num * (cls_num - 1)) * this->lambda_m;
		if (propagate_down[0]) {
			Dtype* bout = bottom[0]->mutable_gpu_diff();
			caffe_gpu_set(bottom[0]->count(), Dtype(0), bout);

			for (int i = 0; i < batch_num; ++i) {
				int gt_label = static_cast<int>(label_data[i]);
				CHECK_GE(gt_label, 0);
				CHECK_LT(gt_label, cls_num);
				for (int k = 0; k < cls_num; ++k) {
					if (k == gt_label)
						continue;

					Dtype dot_prod_data_sub = std::max(dot_prod_data[i * cls_num + k], Dtype(FLT_MIN));
					caffe_gpu_axpy(feat_dim, alpha / dot_prod_data_sub, bottom_data + k * feat_dim, bout + gt_label * feat_dim);
					caffe_gpu_axpy(feat_dim, alpha / dot_prod_data_sub, bottom_data + gt_label * feat_dim, bout + k * feat_dim);
				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MHELossLayer);
}