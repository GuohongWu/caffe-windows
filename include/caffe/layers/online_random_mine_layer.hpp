#ifndef CAFFE_ONLINE_RANDOM_MINE_LAYER_HPP_
#define CAFFE_ONLINE_RANDOM_MINE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
	/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class OnlineRandomMineLayer : public Layer<Dtype> {
 public:

  explicit OnlineRandomMineLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OnlineRandomMine"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
	
  virtual inline int ExactNumTopBlobs() const {return 1;}

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
  // select the random batch from the current label image
  int batch_size_;
  int half_batch_size_;
  // the total number of samples
  //int outer_num_;
  // the inlier number
  int inlier_num_;
  // the channel number
  int channels_;
  // the width * height
  int width_height_;
};

}

#endif