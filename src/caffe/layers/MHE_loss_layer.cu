#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/MHE_loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void MHELossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* vec_res_data = this->vec_res_temp_.mutable_gpu_data();

		Dtype loss(0.0);
		int cls_num = bottom[0]->num();
		int batch_num = bottom[1]->num();
		int feat_dim = bottom[0]->count() / cls_num;
		for (int i = 0; i < batch_num; ++i) {
			int gt_label = static_cast<int>(label_data[i]);
			CHECK_GE(gt_label, 0);
			CHECK_LT(gt_label, cls_num);
			for (int k = 0; k < cls_num; ++k) {
				if (k == gt_label)
					continue;
				caffe_gpu_sub(feat_dim, bottom_data + gt_label * feat_dim, bottom_data + k * feat_dim, vec_res_data);
				Dtype sqr_len_;
				caffe_gpu_dot(feat_dim, vec_res_data, vec_res_data, &sqr_len_);
				loss -= Dtype(0.5) * log(std::max(sqr_len_, Dtype(FLT_MIN)));
			}
		}

		top[0]->mutable_cpu_data()[0] = loss / Dtype(batch_num * (cls_num - 1)) * this->lambda_m;
	}

	template <typename Dtype>
	void MHELossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* vec_res_data = this->vec_res_temp_.mutable_gpu_data();

		int cls_num = bottom[0]->num();
		int batch_num = bottom[1]->num();
		int feat_dim = bottom[0]->count() / cls_num;

		const Dtype alpha = -top[0]->cpu_diff()[0] / Dtype(batch_num * (cls_num - 1)) * this->lambda_m;
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
					caffe_gpu_sub(feat_dim, bottom_data + gt_label * feat_dim, bottom_data + k * feat_dim, vec_res_data);
					Dtype sqr_len_;
					caffe_gpu_dot(feat_dim, vec_res_data, vec_res_data, &sqr_len_);
					caffe_gpu_axpy(feat_dim, alpha / sqr_len_, vec_res_data, bout + gt_label * feat_dim);
					caffe_gpu_axpy(feat_dim, -alpha / sqr_len_, vec_res_data, bout + k * feat_dim);
				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MHELossLayer);
}