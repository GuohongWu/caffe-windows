#ifndef CAFFE_PROPOSAL_RCNN_TRAIN_LAYER_HPP_
#define CAFFE_PROPOSAL_RCNN_TRAIN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	// the proposal layer for faster rcnn during the TRAINING phase, which contains 3 tops for the roi pooling, bounding box regression,object class
	// this will generate multiple tops to continue the training
	// rois format: batch_ind,x1,y1,x2,y2
	template <typename Dtype>
	class ProposalRCNNTrainLayer : public Layer<Dtype> {
	public:
		explicit ProposalRCNNTrainLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
				//LOG(FATAL) << "Reshaping happens during the call to forward.";
		}

		virtual inline const char* type() const { return "ProposalRCNNTrain"; }

	protected:
		// 
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {NOT_IMPLEMENTED;}

		int num_class_;
		int base_size_;
		int feat_stride_;
		int pre_nms_topn_;
		int post_nms_topn_;
		Dtype nms_thresh_;
		int min_size_;
		Blob<Dtype> anchors_;
		Blob<Dtype> proposals_;
		Blob<int> roi_indices_;
		Blob<int> nms_mask_;
	};

}  // namespace caffe





#endif