#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>

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

		this->dot_prod_.Reshape(bottom[1]->num(), bottom[0]->num(), 1, 1);
		this->loss_temp_.ReshapeLike(this->dot_prod_);

		this->use_lambda_curve = !this->layer_param_.mhe_loss_param().has_lambda_m();
		this->lambda_m = this->layer_param_.mhe_loss_param().lambda_m();
		this->lambda_max = this->layer_param_.mhe_loss_param().lambda_max();
		this->gamma = this->layer_param_.mhe_loss_param().gamma();
		this->power = this->layer_param_.mhe_loss_param().power();
		this->iter = this->layer_param_.mhe_loss_param().iter();
	}

	template <typename Dtype>
	void MHELossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);

		if (top.size() == 2)
			top[1]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void MHELossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		if (this->use_lambda_curve) {
			this->lambda_m = this->lambda_max * (Dtype(1) - pow(Dtype(1) + this->gamma * this->iter, -this->power));
			this->iter += Dtype(1);
		}

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* dot_prod_data = this->dot_prod_.mutable_cpu_data();

		Dtype loss(0.0);
		int cls_num = bottom[0]->num();
		int batch_num = bottom[1]->num();
		int feat_dim = bottom[0]->count() / cls_num;

//#pragma omp parallel for (With omp ON, the result is NOT correct!)
		for (int i = 0; i < batch_num; ++i) {
			int gt_label = static_cast<int>(label_data[i]);
			CHECK_GE(gt_label, 0);
			CHECK_LT(gt_label, cls_num);
			for (int k = 0; k < cls_num; ++k) {
				if (k == gt_label)
					continue;
				//caffe_sub(feat_dim, bottom_data + gt_label * feat_dim, bottom_data + k * feat_dim, vec_res_data);
				dot_prod_data[i * cls_num + k] = Dtype(1) - caffe_cpu_dot(feat_dim, bottom_data + gt_label * feat_dim, bottom_data + k * feat_dim);
				loss -= log(std::max(dot_prod_data[i * cls_num + k], Dtype(FLT_MIN)));
			}
		}

		/*int i, k;
#pragma omp parallel for
		for (int idx = 0; idx < batch_num * cls_num; ++idx) {
			i = idx / cls_num;
			k = idx % cls_num;

			int gt_label = static_cast<int>(label_data[i]);
			CHECK_GE(gt_label, 0);
			CHECK_LT(gt_label, cls_num);
			if (k == gt_label)
				continue;

			dot_prod_data[idx] = Dtype(1) - caffe_cpu_dot(feat_dim, bottom_data + gt_label * feat_dim, bottom_data + k * feat_dim);
			loss -= log(std::max(dot_prod_data[idx], Dtype(FLT_MIN)));
		}*/

		top[0]->mutable_cpu_data()[0] = loss / Dtype(batch_num * (cls_num - 1)) * this->lambda_m;

		if (top.size() == 2)
			top[1]->mutable_cpu_data()[0] = this->lambda_m;
	}

	template <typename Dtype>
	void MHELossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* dot_prod_data = this->dot_prod_.mutable_cpu_data();

		int cls_num = bottom[0]->num();
		int batch_num = bottom[1]->num();
		int feat_dim = bottom[0]->count() / cls_num;

		const Dtype alpha = top[0]->cpu_diff()[0] / Dtype(batch_num * (cls_num - 1)) * this->lambda_m;
		if (propagate_down[0]) {
			Dtype* bout = bottom[0]->mutable_cpu_diff();
			caffe_set(bottom[0]->count(), Dtype(0), bout);

//#pragma omp parallel for (With omp ON, the result is NOT correct!)
			for (int i = 0; i < batch_num; ++i) {
				int gt_label = static_cast<int>(label_data[i]);
				CHECK_GE(gt_label, 0);
				CHECK_LT(gt_label, cls_num);
				for (int k = 0; k < cls_num; ++k) {
					if (k == gt_label)
						continue;

					Dtype dot_prod_data_sub = std::max(dot_prod_data[i * cls_num + k], Dtype(FLT_MIN));
					caffe_axpy(feat_dim, alpha / dot_prod_data_sub, bottom_data + k * feat_dim, bout + gt_label * feat_dim);
					caffe_axpy(feat_dim, alpha / dot_prod_data_sub, bottom_data + gt_label * feat_dim, bout + k * feat_dim);
				}
			}

			/*int i, k;
#pragma omp parallel for
			for (int idx = 0; idx < batch_num * cls_num; ++idx) {
				i = idx / cls_num;
				k = idx % cls_num;

				int gt_label = static_cast<int>(label_data[i]);
				CHECK_GE(gt_label, 0);
				CHECK_LT(gt_label, cls_num);
				if (k == gt_label)
					continue;

				Dtype dot_prod_data_sub = std::max(dot_prod_data[idx], Dtype(FLT_MIN));
				caffe_axpy(feat_dim, alpha / dot_prod_data_sub, bottom_data + k * feat_dim, bout + gt_label * feat_dim);
				caffe_axpy(feat_dim, alpha / dot_prod_data_sub, bottom_data + gt_label * feat_dim, bout + k * feat_dim);
			}*/
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MHELossLayer);
#endif

	INSTANTIATE_CLASS(MHELossLayer);
	REGISTER_LAYER_CLASS(MHELoss);

}