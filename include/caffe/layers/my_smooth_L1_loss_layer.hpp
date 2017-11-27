#ifndef CAFFE_MY_SMOOTH_L1_LOSS_LAYER_HPP_
#define CAFFE_MY_SMOOTH_L1_LOSS_LAYER_HPP_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe {

	template <typename Dtype>
	class MySmoothL1LossLayer : public LossLayer<Dtype> {
	public:
		explicit MySmoothL1LossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), diff_() {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "MySmoothL1Loss"; }

		// 0 and 1 can be BP
		// 2 can not be BP, because it is label
		// the total loss function is \sum_{i}{yi(xi - zi)^2}
		// the label is N x C, indicating that c is 
		virtual inline int ExactNumBottomBlobs() const {return 3;}

		// 
		virtual inline int ExactNumTopBlobs() const {return 1;}

		// 2 is the layer that can not be backpropogated
		// because it is the label
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return bottom_index != 2;
		}

	protected:
		/// @copydoc EuclideanLossLayer
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		// store the difference part
		Blob<Dtype> diff_;
		Blob<Dtype> ones_;
		Blob<Dtype> errors_;
		Dtype non_zero_;
		// store margin paramters here
		float sigma2_;
		// the total number of samples
		int outer_num_;
		// the inlier number
		int inlier_num_;
		// the channel number
		int channels_;
		// the width * height
		int width_height_;

	};

}



#endif