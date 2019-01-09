#ifndef CAFFE_TV_LOSS_LAYER_HPP_
#define CAFFE_TV_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	/**
	* @brief Computes Total Variation loss, as described in [1].
	*
	* [1] "Understanding Deep Image Representations by Inverting Them",
	*     A. Mahendran and A. Vedaldi, CVPR 2015
	*/
	template <typename Dtype>
	class TVLossLayer : public LossLayer<Dtype> {
	public:
		explicit TVLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param), grad_norm_(), mask_(),
			x_diff_(), y_diff_(), tmp_() {}

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "TVLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> grad_norm_;
		Blob<Dtype> mask_;
		Blob<Dtype> x_diff_;
		Blob<Dtype> y_diff_;
		Blob<Dtype> tmp_;

	private:
		void create_mask_cpu(const vector<int>& shape, const int count,
			const int H, const int W);
#ifndef CPU_ONLY
		void create_mask_gpu(const int count, const int H, const int W, Dtype* mask);
#endif
	};

}  // namespace caffe

#endif  // CAFFE_TV_LOSS_LAYER_HPP_