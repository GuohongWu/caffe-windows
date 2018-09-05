#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/MHE_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

	template <typename Dtype>
	void MHELossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		CHECK_EQ(bottom[1]->height(), 1);
		CHECK_EQ(bottom[1]->width(), 1);
		CHECK_EQ(bottom[1]->channels(), 1);

		this->lambda_m = this->layer_param_.mhe_loss_param().lambda_m();
		this->vec_res_temp_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
	}

	template <typename Dtype>
	void MHELossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void MHELossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* vec_res_data = this->vec_res_temp_.mutable_cpu_data();

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
				caffe_sub(feat_dim, bottom_data + gt_label * feat_dim, bottom_data + k * feat_dim, vec_res_data);
				Dtype sqr_len_ = caffe_cpu_dot(feat_dim, vec_res_data, vec_res_data);
				loss -= Dtype(0.5) * log(std::max(sqr_len_, Dtype(FLT_MIN)));
			}
		}

		top[0]->mutable_cpu_data()[0] = loss / Dtype(batch_num * (cls_num - 1)) * this->lambda_m;
	}

	template <typename Dtype>
	void MHELossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* vec_res_data = this->vec_res_temp_.mutable_cpu_data();

		int cls_num = bottom[0]->num();
		int batch_num = bottom[1]->num();
		int feat_dim = bottom[0]->count() / cls_num;

		const Dtype alpha = - top[0]->cpu_diff()[0] / Dtype(batch_num * (cls_num - 1)) * this->lambda_m;
		if (propagate_down[0]) {
			Dtype* bout = bottom[0]->mutable_cpu_diff();
			caffe_set(bottom[0]->count(), Dtype(0), bout);

			for (int i = 0; i < batch_num; ++i) {
				int gt_label = static_cast<int>(label_data[i]);
				CHECK_GE(gt_label, 0);
				CHECK_LT(gt_label, cls_num);
				for (int k = 0; k < cls_num; ++k) {
					if (k == gt_label)
						continue;
					caffe_sub(feat_dim, bottom_data + gt_label * feat_dim, bottom_data + k * feat_dim, vec_res_data);
					Dtype sqr_len_ = caffe_cpu_dot(feat_dim, vec_res_data, vec_res_data);
					caffe_axpy(feat_dim, alpha / sqr_len_, vec_res_data, bout + gt_label * feat_dim);
					caffe_axpy(feat_dim, -alpha / sqr_len_, vec_res_data, bout + k * feat_dim);
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MHELossLayer);
#endif

	INSTANTIATE_CLASS(MHELossLayer);
	REGISTER_LAYER_CLASS(MHELoss);

}