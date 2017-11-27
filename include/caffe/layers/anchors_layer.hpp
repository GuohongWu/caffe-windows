#ifndef CAFFE_ANCHORS_LAYER_HPP_
#define CAFFE_ANCHORS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {
	template <typename Dtype>
	class AnchorsLayer : public Layer<Dtype> {
	public:
		explicit AnchorsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Anchors"; }
		virtual inline int ExactNumBottomBlobs() const { return 3; }
		virtual inline int MaxTopBlobs() const { return 4; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		//	const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		// virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		//	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int base_size_;
		int anchor_stride_;
		Dtype nms_upper_thresh_;
		Dtype nms_lower_thresh_;
		Blob<Dtype> anchors_0_0_;
	};

}  // namespace caffe

#endif  // CAFFE_ANCHORS_LAYER_HPP_
