#ifndef CAFFE_ONLINE_HARD_MINE_LAYER_HPP_
#define CAFFE_ONLINE_HARD_MINE_LAYER_HPP_

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
class OnlineHardMineLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with AccuracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit OnlineHardMineLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OnlineHardMine"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
	
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
  // grasp the top N samples
  Dtype top_ratio_;
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