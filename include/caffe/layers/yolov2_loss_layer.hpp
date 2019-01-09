#ifndef CAFFE_YOLOV2_LOSS_LAYER_HPP_
#define CAFFE_YOLOV2_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	// implement YOLO(v2) loss presented in the paper:
	// YOLO9000: Better, Faster, Stronger, 2016.

	template <typename Dtype>
	class YOLOV2LossLayer : public LossLayer<Dtype> {
	public:
		explicit YOLOV2LossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "YOLOV2Loss"; }

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
		vector<Dtype> bbox_priors;  // w1,h1,w2,h2,...,wn,hn
		Blob<Dtype> bottom0_diff_;
	};

}

#endif
