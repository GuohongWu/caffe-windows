#ifndef CAFFE_YOLO_LOSS_LAYER_HPP_
#define CAFFE_YOLO_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	// implement YOLO(v1) loss presented in the paper:
	// You Only Look Once: Unified, Real-Time Object Detection, 2016.

	template <typename Dtype>
	class YOLOLossLayer : public LossLayer<Dtype> {
	public:
		explicit YOLOLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "YOLOLoss"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		//	const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		//virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		//	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		
		int grid_w;
		int grid_h;
		int cls_nums;
		int bbox_nums;
		Dtype lambda_obj;
		Dtype lambda_noobj;
		Dtype lambda_coord;
		Blob<Dtype> bottom0_diff_;
	};

}

#endif

