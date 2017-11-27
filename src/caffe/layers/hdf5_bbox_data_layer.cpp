/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
:: use util functions caffe_copy, and Blob->offset()
:: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_bbox_data_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

	template <typename Dtype>
	HDF5BboxDataLayer<Dtype>::~HDF5BboxDataLayer<Dtype>() { }

	// Load data and label from HDF5 filename into the class property blobs.
	template <typename Dtype>
	void HDF5BboxDataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
		DLOG(INFO) << "Loading HDF5 file: " << filename;
		hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file_id < 0) {
			LOG(FATAL) << "Failed opening HDF5 file: " << filename;
		}

		int top_size = this->layer_param_.top_size();
		hdf_blobs_.resize(top_size);

		const int MIN_DATA_DIM = 1;
		const int MAX_DATA_DIM = INT_MAX;

		for (int i = 0; i < top_size; ++i) {
			hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
			// Allow reshape here, as we are loading data not params
			hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
				MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get(), true);
		}

		herr_t status = H5Fclose(file_id);
		CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

		// MinTopBlobs==3 guarantees at least one top blob
		CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
		DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0) << " rows";
	}

	template <typename Dtype>
	void HDF5BboxDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Refuse transformation parameters since HDF5 is totally generic.
		CHECK(!this->layer_param_.has_transform_param()) <<
			this->type() << " does not transform data.";
		// Read the source to parse the filenames.
		const string& source = this->layer_param_.hdf5_bbox_data_param().source();
		LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
		hdf_filenames_.clear();
		std::ifstream source_file(source.c_str());
		if (source_file.is_open()) {
			std::string line;
			while (source_file >> line) {
				hdf_filenames_.push_back(line);
			}
		}
		else {
			LOG(FATAL) << "Failed to open source file: " << source;
		}
		source_file.close();
		num_files_ = hdf_filenames_.size();
		current_file_ = 0;
		LOG(INFO) << "Number of HDF5 files: " << num_files_;
		CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
			<< source;

		// Load the first HDF5 file and initialize the line counter.
		LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
		current_row_ = 0;

		// Reshape blobs.
		const int batch_size = this->layer_param_.hdf5_bbox_data_param().batch_size();
		const int top_size = this->layer_param_.top_size();
		vector<int> top_shape;

		for (int i = 0; i < top_size; ++i) {
			top_shape.resize(hdf_blobs_[i]->num_axes());
			top_shape[0] = batch_size;
			for (int j = 1; j < top_shape.size(); ++j) {
				top_shape[j] = hdf_blobs_[i]->shape(j);
			}
			top[i]->Reshape(top_shape);
		}
	}

	template <typename Dtype>
	bool HDF5BboxDataLayer<Dtype>::Skip() {
		int size = Caffe::solver_count();
		int rank = Caffe::solver_rank();
		bool keep = (offset_ % size) == rank ||
			// In test mode, only rank 0 runs, so avoid skipping
			this->layer_param_.phase() == TEST;
		return !keep;
	}

	template<typename Dtype>
	void HDF5BboxDataLayer<Dtype>::Next() {
		if (++current_row_ == hdf_blobs_[0]->shape(0)) {
			if (num_files_ > 1) {
				++current_file_;
				if (current_file_ == num_files_) {
					current_file_ = 0;
					DLOG(INFO) << "Looping around to first file.";
				}
				LoadHDF5FileData(hdf_filenames_[current_file_].c_str());
			}
			current_row_ = 0;
		}
		offset_++;
	}

	template <typename Dtype>
	void HDF5BboxDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int batch_size = this->layer_param_.hdf5_bbox_data_param().batch_size();
		//hsize_t end_row_ = std::min(current_row_ + batch_size, hdf_blobs_[0]->shape(0));
		const int top_size = this->layer_param_.top_size();
		top.back()->Reshape(vector<int>{batch_size, 4});
		vector<int> top_back_shape(hdf_blobs_[top_size - 1]->num_axes());
		top_back_shape[0] = batch_size;
		for (int j = 1; j < top_back_shape.size(); ++j) {
			top_back_shape[j] = hdf_blobs_[top_size - 1]->shape(j);
		}
		top.back()->Reshape(top_back_shape);

		int n_i = 0;
		for (int i = 0; i < batch_size; ++i) {
			while (Skip()) {
				Next();
			}
			int j;
			for (j = 0; j + 2 < top_size; ++j) {
				int data_dim = top[j]->count() / top[j]->shape(0);
				caffe_copy(data_dim,
					&hdf_blobs_[j]->cpu_data()[current_row_
					* data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
			}
			// assgin top[top_size - 2]
			top[j++]->mutable_cpu_data()[i] = Dtype(n_i);

			// copy bbox_lists into top[top_size - 1] 
			// top[top_size - 1] need to be reshaped here.
			int data_dim = top[j]->count() / top[j]->shape(0);
			int copy_nums = (current_row_ + 1 == hdf_blobs_[j - 1]->shape(0)) ? hdf_blobs_[j]->shape(0) : hdf_blobs_[j - 1]->cpu_data()[current_row_ + 1];
			copy_nums -= hdf_blobs_[j - 1]->cpu_data()[current_row_];

			if (n_i + copy_nums > top[j]->shape(0)) {
				Blob<Dtype> data_tmp;
				data_tmp.CopyFrom(*top[j], false, true);
				vector<int> new_top_shape(top[j]->num_axes());
				new_top_shape[0] = n_i + copy_nums;
				for (int k = 1; k < new_top_shape.size(); ++k)
					new_top_shape[k] = top[j]->shape(k);
				top[j]->Reshape(new_top_shape);
				top[j]->CopyFrom(data_tmp);
			}

			int idx = hdf_blobs_[j - 1]->gpu_data()[current_row_] * data_dim;
			caffe_copy(data_dim * copy_nums,
				&hdf_blobs_[j]->cpu_data()[idx], &top[j]->mutable_cpu_data()[n_i * data_dim]);
			n_i += copy_nums;

			Next();
		}
	}

#ifdef CPU_ONLY
	STUB_GPU_FORWARD(HDF5BboxDataLayer, Forward);
#endif

	INSTANTIATE_CLASS(HDF5BboxDataLayer);
	REGISTER_LAYER_CLASS(HDF5BboxData);

}  // namespace caffe
