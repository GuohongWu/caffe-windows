#include <stdint.h>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/image_mix_hdf5_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void ImageMixHDF5DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 
    if (this->prefetch_current_) {
		this->prefetch_free_.push(this->prefetch_current_);
    }
	this->prefetch_current_ = this->prefetch_full_.pop("Waiting for data");
	// Reshape to loaded data.
	top[0]->ReshapeLike(this->prefetch_current_->data_);
	top[0]->set_gpu_data(this->prefetch_current_->data_.mutable_gpu_data());
	// Reshape to loaded labels.
	top[1]->ReshapeLike(this->prefetch_current_->label_);
	top[1]->set_gpu_data(this->prefetch_current_->label_.mutable_gpu_data());

	Blob<Dtype> bbox_startloc;
	bbox_startloc.ReshapeLike(this->prefetch_current_->weight_);
	bbox_startloc.set_gpu_data(this->prefetch_current_->weight_.mutable_gpu_data());

	// top[2] reshape
	const int batch_size = this->layer_param_.image_mix_hdf5_data_param().batch_size();
	vector<int> top2_shape;
	top2_shape.resize(this->hdf5_blobs_->num_axes());
	top2_shape[0] = batch_size;
	for (int i = 1; i < top2_shape.size(); ++i) {
		top2_shape[i] = this->hdf5_blobs_->shape(i);
	}
	top[2]->Reshape(top2_shape);

	const Dtype* bbox_startloc_data = bbox_startloc.cpu_data();
	Dtype* top1_data = top[1]->mutable_gpu_data();
	
	Dtype n_i = 0;
	Dtype copy_nums;
	int data_dim = top[2]->count(1);
	for (int i = 0; i < batch_size; ++i) {
		caffe_copy(1, top1_data + i, &copy_nums);
		if (int(n_i + copy_nums) > top[2]->shape(0)) {
			Blob<Dtype> data_tmp;
			data_tmp.CopyFrom(*top[2], false, true);
			vector<int> new_top_shape(top[2]->num_axes());
			new_top_shape[0] = n_i + copy_nums;
			for (int k = 1; k < new_top_shape.size(); ++k)
				new_top_shape[k] = top[2]->shape(k);
			top[2]->Reshape(new_top_shape);
			caffe_copy(data_tmp.count(), data_tmp.gpu_data(), top[2]->mutable_gpu_data());
		}
		caffe_copy(int(copy_nums) * data_dim, &this->hdf5_blobs_->gpu_data()[int(bbox_startloc_data[i]) * data_dim], &top[2]->mutable_gpu_data()[int(n_i) * data_dim]);
		caffe_copy(1, &n_i, top1_data + i);

		n_i += copy_nums;
	}

}

INSTANTIATE_LAYER_GPU_FORWARD(ImageMixHDF5DataLayer);

}  // namespace caffe
