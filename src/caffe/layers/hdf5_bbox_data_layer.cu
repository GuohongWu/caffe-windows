/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_bbox_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void HDF5BboxDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_bbox_data_param().batch_size();
  const int top_size = this->layer_param_.top_size();
  vector<int> top_back_shape(hdf_blobs_[top_size - 1]->num_axes());
  top_back_shape[0] = batch_size;
  for (int j = 1; j < top_back_shape.size(); ++j) {
	  top_back_shape[j] = hdf_blobs_[top_size - 1]->shape(j);
  }
  top.back()->Reshape(top_back_shape);

  Dtype n_i = 0;
  for (int i = 0; i < batch_size; ++i) {
    while (Skip()) {
      Next();
    }
	int j;
			for (j = 0; j + 2 < top_size; ++j) {
				int data_dim = top[j]->count() / top[j]->shape(0);
				caffe_copy(data_dim,
					&hdf_blobs_[j]->gpu_data()[current_row_
					* data_dim], &top[j]->mutable_gpu_data()[i * data_dim]);
			}
			// assgin top[top_size - 2]
			caffe_copy(1, &n_i, &top[j++]->mutable_gpu_data()[i]);

			// copy bbox_lists into top[top_size - 1] 
			// top[top_size - 1] need to be reshaped here.
			int data_dim = top[j]->count() / top[j]->shape(0);
			Dtype copy_nums = hdf_blobs_[j]->shape(0);
			if(current_row_ + 1 != hdf_blobs_[j - 1]->shape(0)) {
				caffe_copy(1, &hdf_blobs_[j - 1]->gpu_data()[current_row_ + 1], &copy_nums);
			}

			Dtype minuend_tmp;
			caffe_copy(1, &hdf_blobs_[j - 1]->gpu_data()[current_row_], &minuend_tmp);
			copy_nums -= minuend_tmp;

			if (n_i + copy_nums > top[j]->shape(0)) {
				Blob<Dtype> data_tmp;
				data_tmp.CopyFrom(*top[j], false, true);
				vector<int> new_top_shape(top[j]->num_axes());
				new_top_shape[0] = n_i + copy_nums;
				for (int k = 1; k < new_top_shape.size(); ++k)
					new_top_shape[k] = top[j]->shape(k);
				top[j]->Reshape(new_top_shape);
				caffe_copy(data_tmp.count(), data_tmp.gpu_data(), top[j]->mutable_gpu_data());
			}

			caffe_copy(data_dim * int(copy_nums),
				&hdf_blobs_[j]->gpu_data()[int(minuend_tmp) * data_dim], &top[j]->mutable_gpu_data()[int(n_i) * data_dim]);
			n_i += copy_nums;

    Next();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HDF5BboxDataLayer);

}  // namespace caffe
