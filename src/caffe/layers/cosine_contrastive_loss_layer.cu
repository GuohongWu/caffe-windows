#include <algorithm>
#include <vector>

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
			Dtype x_len_out, y_len_out;
			caffe_gpu_dot(channels, bottom[0]->gpu_data() + (i*channels), bottom[1]->gpu_data() + (i*channels), this->dot_prod_.mutable_cpu_data() + i);
			caffe_gpu_dot(channels, bottom[0]->gpu_data() + (i*channels), bottom[0]->gpu_data() + (i*channels), &x_len_out);
			caffe_gpu_dot(channels, bottom[1]->gpu_data() + (i*channels), bottom[1]->gpu_data() + (i*channels), &y_len_out);

			//this->dot_prod_.mutable_cpu_data()[i] = dot_out;
			this->x_len_.mutable_cpu_data()[i] = sqrt(x_len_out);
			this->y_len_.mutable_cpu_data()[i] = sqrt(y_len_out);

			Dtype cur_cosine_val = this->dot_prod_.cpu_data()[i] / (this->x_len_.cpu_data()[i] + Dtype(1e-6)) / (this->y_len_.cpu_data()[i] + Dtype(1e-6));
			if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
				loss += this->pos_weight_ * (Dtype(1) - cur_cosine_val);
			}
			else {  // dissimilar pairs
				loss += this->neg_weight_ * std::max(margin, cur_cosine_val);
			}
		}
		loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void CosineContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		Dtype margin = this->layer_param_.contrastive_loss_param().margin();
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		const Dtype alpha = Dtype(0.5) * top[0]->cpu_diff()[0] /
			static_cast<Dtype>(num);

		if (propagate_down[0]) {
			Dtype* bout = bottom[0]->mutable_gpu_diff();
			for (int j = 0; j < num; ++j) {
				Dtype cur_cosine_val = this->dot_prod_.cpu_data()[j] / (this->x_len_.cpu_data()[j] + Dtype(1e-6)) / (this->y_len_.cpu_data()[j] + Dtype(1e-6));
				if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
					caffe_copy(channels, bottom[1]->gpu_data() + (j * channels), bout + (j * channels));
					caffe_gpu_axpby(channels,
						alpha * this->pos_weight_ * cur_cosine_val / (this->x_len_.cpu_data()[j] * this->x_len_.cpu_data()[j] + Dtype(1e-6)),
						bottom[0]->gpu_data() + (j * channels),
						-alpha * this->pos_weight_ / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bout + (j * channels));
				}
				else {
					if (cur_cosine_val > margin) {
						caffe_copy(channels, bottom[1]->gpu_data() + (j * channels), bout + (j * channels));
						caffe_gpu_axpby(channels,
							-alpha * this->neg_weight_ * cur_cosine_val / (this->x_len_.cpu_data()[j] * this->x_len_.cpu_data()[j] + Dtype(1e-6)),
							bottom[0]->gpu_data() + (j * channels),
							alpha * this->neg_weight_ / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
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
				Dtype cur_cosine_val = this->dot_prod_.cpu_data()[j] / (this->x_len_.cpu_data()[j] + Dtype(1e-6)) / (this->y_len_.cpu_data()[j] + Dtype(1e-6));
				if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
					caffe_copy(channels, bottom[1]->gpu_data() + (j * channels), bout + (j * channels));
					caffe_gpu_axpby(channels,
						-alpha * this->pos_weight_ / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bottom[0]->gpu_data() + (j * channels),
						alpha * this->pos_weight_ * cur_cosine_val / (this->y_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bout + (j * channels));
				}
				else {
					if (cur_cosine_val > margin) {
						caffe_copy(channels, bottom[1]->gpu_data() + (j * channels), bout + (j * channels));
						caffe_gpu_axpby(channels,
							alpha * this->neg_weight_ / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
							bottom[0]->gpu_data() + (j * channels),
							-alpha * this->neg_weight_ * cur_cosine_val / (this->y_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
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
