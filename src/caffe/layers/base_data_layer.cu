#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
  }
  if (this->output_weights_) {
    // Reshape to loaded weights.
    top[2]->ReshapeLike(prefetch_current_->weight_);
    // Copy the weights.
    top[2]->set_gpu_data(prefetch_current_->weight_.mutable_gpu_data());
  }
  if (this->output_segMask_) {
	  top[3]->ReshapeLike(prefetch_current_->segMask_);
	  top[3]->set_gpu_data(prefetch_current_->segMask_.mutable_gpu_data());
  }
  if (this->output_poseKeypoints_) {
	  if (this->output_segMask_) {
		  top[4]->ReshapeLike(prefetch_current_->poseKeypoints_);
		  top[4]->set_gpu_data(prefetch_current_->poseKeypoints_.mutable_gpu_data());
	  }
	  else {
		  top[3]->ReshapeLike(prefetch_current_->poseKeypoints_);
		  top[3]->set_gpu_data(prefetch_current_->poseKeypoints_.mutable_gpu_data());
	  }
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
