#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/cosine_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void CosineContrastiveLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		if (this->add_weighted) {
			int total_nums = bottom[2]->num();
			int pos_count = 0;
			for (int i = 0; i < total_nums; ++i) {
				if (static_cast<int>(bottom[2]->cpu_data()[i]))
					++pos_count;
			}
			if (pos_count == 0 || pos_count == total_nums) {
				this->pos_weight_ = this->neg_weight_ = Dtype(1);
			}
			else {
				this->pos_weight_ = Dtype(0.5) * Dtype(total_nums) / Dtype(pos_count); // n/(2*k)
				this->neg_weight_ = Dtype(0.5) * Dtype(total_nums) / Dtype(total_nums - pos_count); // n/(2*(n-k))
			}
		}

		Dtype margin = this->layer_param_.contrastive_loss_param().margin();
		Dtype loss(0.0);
		const int channels = bottom[0]->channels();
		for (int i = 0; i < bottom[0]->num(); ++i) {
			Dtype dot_prod_out, x_len_out, y_len_out;
			caffe_gpu_dot(channels, bottom[0]->gpu_data() + (i*channels), bottom[1]->gpu_data() + (i*channels), &dot_prod_out);
			caffe_gpu_dot(channels, bottom[0]->gpu_data() + (i*channels), bottom[0]->gpu_data() + (i*channels), &x_len_out);
			caffe_gpu_dot(channels, bottom[1]->gpu_data() + (i*channels), bottom[1]->gpu_data() + (i*channels), &y_len_out);

			//this->dot_prod_.mutable_cpu_data()[i] = dot_out;
			this->x_len_.mutable_cpu_data()[i] = sqrt(x_len_out);
			this->y_len_.mutable_cpu_data()[i] = sqrt(y_len_out);

			Dtype cur_cosine_val = dot_prod_out / (this->x_len_.cpu_data()[i] + Dtype(1e-6)) / (this->y_len_.cpu_data()[i] + Dtype(1e-6));
			this->cos_theta_.mutable_cpu_data()[i] = cur_cosine_val;
			this->sin_theta_.mutable_cpu_data()[i] = sqrt(Dtype(1) - cur_cosine_val * cur_cosine_val);

			Dtype cos_theta_plus_m = cur_cosine_val * this->cos_m - this->sin_m * this->sin_theta_.cpu_data()[i];
			if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
				if (cur_cosine_val + this->cos_m >= Dtype(0))
					loss += this->pos_weight_ * (Dtype(1) - cos_theta_plus_m);
				else
					loss += this->pos_weight_ * (Dtype(3) + cos_theta_plus_m);
			}
			else {  // dissimilar pairs
				if (cur_cosine_val + this->cos_m >= Dtype(0))
					loss += this->neg_weight_ * std::max(margin, cos_theta_plus_m);
				else
					loss += this->neg_weight_ * std::max(margin, Dtype(-2) - cos_theta_plus_m);
			}
		}
		loss = loss / static_cast<Dtype>(bottom[0]->num());
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void CosineContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		Dtype margin = this->layer_param_.contrastive_loss_param().margin();
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(num);

		if (propagate_down[0]) {
			Dtype* bout = bottom[0]->mutable_gpu_diff();
			for (int j = 0; j < num; ++j) {
				Dtype cur_cos_theta = this->cos_theta_.cpu_data()[j];
				Dtype cur_sin_theta = this->sin_theta_.cpu_data()[j];
				Dtype diff_to_cosTheta = Dtype(0);

				if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
					diff_to_cosTheta = -this->pos_weight_ * (this->cos_m + this->sin_m * cur_cos_theta / (cur_sin_theta + Dtype(1e-6)));
					if (cur_cos_theta + this->cos_m < Dtype(0))
						diff_to_cosTheta = -diff_to_cosTheta;

					caffe_copy(channels, bottom[1]->gpu_data() + (j * channels), bout + (j * channels));
					caffe_gpu_axpby(channels,
						-alpha * diff_to_cosTheta * cur_cos_theta / (this->x_len_.cpu_data()[j] * this->x_len_.cpu_data()[j] + Dtype(1e-6)),
						bottom[0]->gpu_data() + (j * channels),
						alpha * diff_to_cosTheta / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bout + (j * channels));
				}
				else {
					Dtype mdist = this->cos_m * cur_cos_theta - this->sin_m * cur_sin_theta;
					if (cur_cos_theta + this->cos_m < Dtype(0))
						mdist = Dtype(-2) - mdist;

					if (mdist > margin) {
						diff_to_cosTheta = this->neg_weight_ * (this->cos_m + this->sin_m * cur_cos_theta / (cur_sin_theta + Dtype(1e-6)));
						if (cur_cos_theta + this->cos_m < Dtype(0))
							diff_to_cosTheta = -diff_to_cosTheta;

						caffe_copy(channels, bottom[1]->gpu_data() + (j * channels), bout + (j * channels));
						caffe_gpu_axpby(channels,
							-alpha * diff_to_cosTheta * cur_cos_theta / (this->x_len_.cpu_data()[j] * this->x_len_.cpu_data()[j] + Dtype(1e-6)),
							bottom[0]->gpu_data() + (j * channels),
							alpha * diff_to_cosTheta / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
							bout + (j * channels));
					}
					else {
						caffe_gpu_set(channels, Dtype(0), bout + (j*channels));
					}
				}
			}
		}

		if (propagate_down[1]) {
			Dtype* bout = bottom[1]->mutable_gpu_diff();
			for (int j = 0; j < num; ++j) {
				Dtype cur_cos_theta = this->cos_theta_.cpu_data()[j];
				Dtype cur_sin_theta = this->sin_theta_.cpu_data()[j];
				Dtype diff_to_cosTheta = Dtype(0);

				if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
					diff_to_cosTheta = -this->pos_weight_ * (this->cos_m + this->sin_m * cur_cos_theta / (cur_sin_theta + Dtype(1e-6)));
					if (cur_cos_theta + this->cos_m < Dtype(0))
						diff_to_cosTheta = -diff_to_cosTheta;

					caffe_copy(channels, bottom[1]->gpu_data() + (j * channels), bout + (j * channels));
					caffe_gpu_axpby(channels,
						alpha * diff_to_cosTheta / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bottom[0]->gpu_data() + (j * channels),
						-alpha * diff_to_cosTheta * cur_cos_theta / (this->y_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bout + (j * channels));
				}
				else {
					Dtype mdist = this->cos_m * cur_cos_theta - this->sin_m * cur_sin_theta;
					if (cur_cos_theta + this->cos_m < Dtype(0))
						mdist = Dtype(-2) - mdist;

					if (mdist > margin) {
						diff_to_cosTheta = this->neg_weight_ * (this->cos_m + this->sin_m * cur_cos_theta / (cur_sin_theta + Dtype(1e-6)));
						if (cur_cos_theta + this->cos_m < Dtype(0))
							diff_to_cosTheta = -diff_to_cosTheta;

						caffe_copy(channels, bottom[1]->gpu_data() + (j * channels), bout + (j * channels));
						caffe_gpu_axpby(channels,
							alpha * diff_to_cosTheta / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
							bottom[0]->gpu_data() + (j * channels),
							-alpha * diff_to_cosTheta * cur_cos_theta / (this->y_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
							bout + (j * channels));
					}
					else {
						caffe_gpu_set(channels, Dtype(0), bout + (j*channels));
					}
				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(CosineContrastiveLossLayer);

}  // namespace caffe
