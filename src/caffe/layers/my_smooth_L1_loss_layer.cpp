#include "caffe/util/math_functions.hpp"
#include "caffe/layers/my_smooth_L1_loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void MySmoothL1LossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			MySmoothL1LossParameter loss_param = this->layer_param_.my_smooth_l1_loss_param();
			sigma2_ = loss_param.sigma() * loss_param.sigma();
			CHECK_EQ(bottom.size(),3) << "you must add the label info to exclude some elements here";
	}

	template <typename Dtype>
	void MySmoothL1LossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			LossLayer<Dtype>::Reshape(bottom, top);
			CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
			CHECK_EQ(bottom[0]->height(), bottom[1]->height());
			CHECK_EQ(bottom[0]->width(), bottom[1]->width());
			CHECK_EQ(bottom[2]->channels(),1)<<"the channel of the label must be, that is a label map";
			CHECK_EQ(bottom[2]->height(),bottom[0]->height());
			CHECK_EQ(bottom[2]->width(),bottom[0]->width());
			// inner diff
			diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());
			// outer diff
			errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width());

			// vector of count indices for counting the valid example, which is greater than 0
			ones_.Reshape(bottom[2]->num(), bottom[2]->channels(),
				bottom[2]->height(), bottom[2]->width());

			// add the following aux variables
			non_zero_ = Dtype(0);

			inlier_num_ = bottom[0]->count(1);

			// check the channels
			channels_ = bottom[0]->shape(1);

			// width and height number
			width_height_ = bottom[0]->count(2);

	}

	template <typename Dtype>
	void MySmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			NOT_IMPLEMENTED;
	}

	template <typename Dtype>
	void MySmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			NOT_IMPLEMENTED;
	}

#ifdef CPU_ONLY
	STUB_GPU(MySmoothL1LossLayer);
#endif

	INSTANTIATE_CLASS(MySmoothL1LossLayer);
	REGISTER_LAYER_CLASS(MySmoothL1Loss);

}