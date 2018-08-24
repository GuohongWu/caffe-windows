#include <algorithm>
#include <vector>

#include "caffe/layers/cosine_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void CosineContrastiveLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
		CHECK_EQ(bottom[0]->height(), 1);
		CHECK_EQ(bottom[0]->width(), 1);
		CHECK_EQ(bottom[1]->height(), 1);
		CHECK_EQ(bottom[1]->width(), 1);
		CHECK_EQ(bottom[2]->channels(), 1);
		CHECK_EQ(bottom[2]->height(), 1);
		CHECK_EQ(bottom[2]->width(), 1);
		this->dot_prod_.Reshape(bottom[0]->num(), 1, 1, 1);
		this->x_len_.Reshape(bottom[0]->num(), 1, 1, 1);
		this->y_len_.Reshape(bottom[0]->num(), 1, 1, 1);

		this->add_weighted = this->layer_param_.contrastive_loss_param().add_weighted();
		this->pos_weight_ = this->neg_weight_ = Dtype(1);
	}

	template <typename Dtype>
	void CosineContrastiveLossLayer<Dtype>::Forward_cpu(
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
			this->dot_prod_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
				bottom[0]->cpu_data() + (i*channels), bottom[1]->cpu_data() + (i*channels));
			this->x_len_.mutable_cpu_data()[i] = sqrt(caffe_cpu_dot(channels,
				bottom[0]->cpu_data() + (i*channels), bottom[0]->cpu_data() + (i*channels)));
			this->y_len_.mutable_cpu_data()[i] = sqrt(caffe_cpu_dot(channels,
				bottom[1]->cpu_data() + (i*channels), bottom[1]->cpu_data() + (i*channels)));

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
	void CosineContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		Dtype margin = this->layer_param_.contrastive_loss_param().margin();
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();

		const Dtype alpha = Dtype(0.5) * top[0]->cpu_diff()[0] /
			static_cast<Dtype>(num);
		if (propagate_down[0]) {
			Dtype* bout = bottom[0]->mutable_cpu_diff();
			for (int j = 0; j < num; ++j) {
				Dtype cur_cosine_val = this->dot_prod_.cpu_data()[j] / (this->x_len_.cpu_data()[j] + Dtype(1e-6)) / (this->y_len_.cpu_data()[j] + Dtype(1e-6));
				if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
					caffe_copy(channels, bottom[1]->cpu_data() + (j * channels), bout + (j * channels));
					caffe_cpu_axpby(channels, 
						alpha * this->pos_weight_ * cur_cosine_val / (this->x_len_.cpu_data()[j] * this->x_len_.cpu_data()[j] + Dtype(1e-6)),
						bottom[0]->cpu_data() + (j * channels),
						-alpha * this->pos_weight_ / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bout + (j * channels));
				}
				else {
					if (cur_cosine_val > margin) {
						caffe_copy(channels, bottom[1]->cpu_data() + (j * channels), bout + (j * channels));
						caffe_cpu_axpby(channels,
							-alpha * this->neg_weight_ * cur_cosine_val / (this->x_len_.cpu_data()[j] * this->x_len_.cpu_data()[j] + Dtype(1e-6)),
							bottom[0]->cpu_data() + (j * channels),
							alpha * this->neg_weight_ / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
							bout + (j * channels));
					}
					else {
						caffe_set(channels, Dtype(0), bout + (j*channels));
					}
				}
			}
		}

		if (propagate_down[1]) {
			Dtype* bout = bottom[1]->mutable_cpu_diff();
			for (int j = 0; j < num; ++j) {
				Dtype cur_cosine_val = this->dot_prod_.cpu_data()[j] / (this->x_len_.cpu_data()[j] + Dtype(1e-6)) / (this->y_len_.cpu_data()[j] + Dtype(1e-6));
				if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
					caffe_copy(channels, bottom[1]->cpu_data() + (j * channels), bout + (j * channels));
					caffe_cpu_axpby(channels,
						-alpha * this->pos_weight_ / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bottom[0]->cpu_data() + (j * channels),
						alpha * this->pos_weight_ * cur_cosine_val / (this->y_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
						bout + (j * channels));
				}
				else {
					if (cur_cosine_val > margin) {
						caffe_copy(channels, bottom[1]->cpu_data() + (j * channels), bout + (j * channels));
						caffe_cpu_axpby(channels,
							alpha * this->neg_weight_ / (this->x_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
							bottom[0]->cpu_data() + (j * channels),
							-alpha * this->neg_weight_ * cur_cosine_val / (this->y_len_.cpu_data()[j] * this->y_len_.cpu_data()[j] + Dtype(1e-6)),
							bout + (j * channels));
					}
					else {
						caffe_set(channels, Dtype(0), bout + (j*channels));
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(CosineContrastiveLossLayer);
#endif

	INSTANTIATE_CLASS(CosineContrastiveLossLayer);
	REGISTER_LAYER_CLASS(CosineContrastiveLoss);

}  // namespace caffe
