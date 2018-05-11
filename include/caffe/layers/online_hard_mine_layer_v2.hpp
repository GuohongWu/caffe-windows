#ifndef CAFFE_ONLINE_HARD_MINE_LAYER_V2_HPP_
#define CAFFE_ONLINE_HARD_MINE_LAYER_V2_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{

	template <typename Dtype>
	class OnlineHardMineV2Layer : public Layer<Dtype> {
	public:

		explicit OnlineHardMineV2Layer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "OnlineHardMineV2"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

		// do not allow any backward
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return false;
		}

	protected:
		// no backward process is implemented
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		/// @brief Not implemented (non-differentiable function)
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			NOT_IMPLEMENTED;
		}
		// grasp the top N samples
		Dtype neg_vs_pos_ratio_;  // e.g. 3 = 3:1
		Dtype top_ignore_ratio_;
	};

}

#endif