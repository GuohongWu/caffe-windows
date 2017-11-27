#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_mix_hdf5_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"
#include "caffe/util/hdf5.hpp"

namespace caffe {

	template <typename Dtype>
	ImageMixHDF5DataLayer<Dtype>::~ImageMixHDF5DataLayer<Dtype>() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void ImageMixHDF5DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Read the img_source with filenames and labels
		const string& img_source = this->layer_param_.image_mix_hdf5_data_param().img_source();
		LOG(INFO) << "Opening file " << img_source;
		std::ifstream infile(img_source.c_str());
		string line;
		size_t pos;
		int label_;
		int weight_;
		lines_.clear();
		while (std::getline(infile, line)) {
			pos = line.find_last_of(' ');
			weight_ = atoi(line.substr(pos + 1).c_str());
			while (line[pos] == ' ')
				--pos;

			line = line.substr(0, pos + 1);
			pos = line.find_last_of(' ');
			label_ = atoi(line.substr(pos + 1).c_str());
			while (line[pos] == ' ')
				--pos;

			lines_.push_back(std::make_tuple(line.substr(0, pos + 1), label_, weight_));
		}
		CHECK(!lines_.empty()) << "File is empty";

		// Read the bbox_source to parse the filenames.
		const string& bbox_source = this->layer_param_.image_mix_hdf5_data_param().bbox_source();
		LoadBBoxHDF5FileData(bbox_source.c_str());

		if (this->layer_param_.image_mix_hdf5_data_param().shuffle()) {
			// randomly shuffle data
			LOG(INFO) << "Shuffling data";
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
			ShuffleImages();
		}
		else {
			if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
				this->layer_param_.image_mix_hdf5_data_param().rand_skip() == 0) {
				LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
			}
		}
		LOG(INFO) << "A total of " << lines_.size() << " images.";

		lines_id_ = 0;
		// Check if we would need to randomly skip a few data points
		if (this->layer_param_.image_mix_hdf5_data_param().rand_skip()) {
			unsigned int skip = caffe_rng_rand() %
				this->layer_param_.image_mix_hdf5_data_param().rand_skip();
			LOG(INFO) << "Skipping first " << skip << " data points.";
			CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
			lines_id_ = skip;
		}

		this->PreFetchReshape(bottom, top);
	}

	template <typename Dtype>
	void ImageMixHDF5DataLayer<Dtype>::PreFetchReshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int new_height = this->layer_param_.image_mix_hdf5_data_param().new_height();
		int new_width = this->layer_param_.image_mix_hdf5_data_param().new_width();
		const bool is_color = this->layer_param_.image_mix_hdf5_data_param().is_color();
		const int batch_size = this->layer_param_.image_mix_hdf5_data_param().batch_size();

		CHECK_GT(batch_size, 0) << "Positive batch size required";
		CHECK((new_height == 0 && new_width == 0) ||
			(new_height > 0 && new_width > 0)) << "Current implementation requires "
			"new_height and new_width to be set at the same time.";

		// Read an image, and use it to initialize the top blob.
		if (new_height == 0) {
			string root_folder = this->layer_param_.image_mix_hdf5_data_param().root_folder();
			cv::Mat cv_img = ReadImageToCVMat(root_folder + std::get<0>(lines_[lines_id_]), is_color);
			CHECK(cv_img.data) << "Could not load " << std::get<0>(lines_[lines_id_]);

			new_height = cv_img.rows;
			new_width = cv_img.cols;
		}
		
		vector<int> top_shape{ 1, (is_color?3:1), new_height, new_width };
		this->transformed_data_.Reshape(top_shape);
		// Reshape prefetch_data		
		top_shape[0] = batch_size;
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->data_.Reshape(top_shape); // top[0]
		}
		top[0]->Reshape(top_shape);

		LOG(INFO) << "output data size: " << batch_size << ","
			<< this->prefetch_[0]->data_.channels() << "," << new_height << ","
			<< new_width;

		// label_
		vector<int> label_shape(1, batch_size);
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->label_.Reshape(label_shape); // top[1]
		}
		top[1]->Reshape(label_shape);

		// weight_
		bbox_init_shape.resize(this->hdf5_blobs_->num_axes());
		bbox_init_shape[0] = batch_size;
		for (int i = 1; i < bbox_init_shape.size(); ++i) {
			bbox_init_shape[i] = this->hdf5_blobs_->shape(i);
		}
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->weight_.Reshape(bbox_init_shape); // top[2]
		}
		top[2]->Reshape(bbox_init_shape);

		// optional: segMask_
		if (this->output_segMask_) {
			top_shape[1] = 1;
			for (int i = 0; i < this->prefetch_.size(); ++i) {
				this->prefetch_[i]->segMask_.Reshape(top_shape); // top[3]
			}
			top[3]->Reshape(top_shape);

			top_shape[0] = 1;
			this->transformed_segMask_.Reshape(top_shape);
		}

		// optional: poseKeypoints_
		if (this->output_poseKeypoints_) {
			poseKeypoints_init_shape.resize(this->hdf5_poseKeypoints_blobs_->num_axes());
			poseKeypoints_init_shape[0] = batch_size;
			for (int i = 1; i < poseKeypoints_init_shape.size(); ++i) {
				poseKeypoints_init_shape[i] = this->hdf5_poseKeypoints_blobs_->shape(i);
			}
			for (int i = 0; i < this->prefetch_.size(); ++i) {
				this->prefetch_[i]->poseKeypoints_.Reshape(poseKeypoints_init_shape); // top[3] or top[4]
			}
			if (this->output_segMask_)
				top[4]->Reshape(poseKeypoints_init_shape);
			else
				top[3]->Reshape(poseKeypoints_init_shape);
		}
	}

	template <typename Dtype>
	void ImageMixHDF5DataLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void ImageMixHDF5DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());
		ImageMixHDF5DataParameter image_mix_hdf5_data_param = this->layer_param_.image_mix_hdf5_data_param();
		const int batch_size = image_mix_hdf5_data_param.batch_size();
		const bool is_color = image_mix_hdf5_data_param.is_color();
		string root_folder = image_mix_hdf5_data_param.root_folder();

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();
		Dtype* prefetch_label = batch->label_.mutable_cpu_data();
		batch->weight_.Reshape(bbox_init_shape);
		//Dtype* prefetch_weight = batch->weight_.mutable_cpu_data();
		Dtype* prefetch_segMask = nullptr;
		if (this->output_segMask_)
			prefetch_segMask = batch->segMask_.mutable_cpu_data();
		if (this->output_poseKeypoints_)
			batch->poseKeypoints_.Reshape(poseKeypoints_init_shape);

		// datum scales
		const int lines_size = lines_.size();
		int cur_start_loc = 0;
		int weight_data_dim = batch->weight_.count() / batch->weight_.num();
		int poseKeypoints_data_dim = 0;
		if (this->output_poseKeypoints_)
			poseKeypoints_data_dim = batch->poseKeypoints_.count() / batch->poseKeypoints_.num();
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			timer.Start();
			bool valid_sample = false;
			while (!valid_sample) {
				CHECK_GT(lines_size, lines_id_);
				std::tuple<std::string, int, int> this_line = lines_[lines_id_];

				cv::Mat cv_img = ReadImageToCVMat(root_folder + std::get<0>(this_line), is_color);
				if (!cv_img.data) {
					LOG(INFO) << "Could not load " << root_folder + std::get<0>(this_line);
					valid_sample = false;
				}
				else {
					valid_sample = true;
				}
				cv::Mat cv_mask_img;
				if (this->output_segMask_) {
					const string mask_file_dir = root_folder + "../segMaskData/";
					string mask_img_name = std::get<0>(this_line);
					int pos = mask_img_name.find_last_of('.');
					mask_img_name = mask_img_name.substr(0, pos) + ".png";
					cv_mask_img = ReadImageToCVMat(mask_file_dir + mask_img_name, false);
					if (!cv_mask_img.data) {
						LOG(INFO) << "Could not load " << mask_file_dir + mask_img_name;
						valid_sample = false;
					}
					else {
						valid_sample = true;
					}
					CHECK_EQ(cv_img.size(), cv_mask_img.size()) << "segMask image must have the same size as the input image.";
				}
				read_time += timer.MicroSeconds();

				// go to the next iter
				lines_id_++;
				if (lines_id_ >= lines_size) {
					// We have reached the end. Restart from the first.
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
					if (this->layer_param_.image_mix_hdf5_data_param().shuffle()) {
						ShuffleImages();
					}
				}

				if (valid_sample) {
					timer.Start();
					// Apply transformations (mirror, crop...) to the image
					prefetch_label[item_id] = cur_start_loc;
					int cur_bbox_nums = std::get<1>(this_line);
					int hdf5_read_loc = std::get<2>(this_line);
					if (cur_start_loc + cur_bbox_nums > batch->weight_.num()) {
						Blob<Dtype> data_tmp;
						data_tmp.CopyFrom(batch->weight_, false, true);
						vector<int> new_weight_shape(bbox_init_shape);
						new_weight_shape[0] = cur_start_loc + cur_bbox_nums;
						batch->weight_.Reshape(new_weight_shape);
						caffe_copy(data_tmp.count(), data_tmp.cpu_data(), batch->weight_.mutable_cpu_data());

						if (this->output_poseKeypoints_) {
							data_tmp.CopyFrom(batch->poseKeypoints_, false, true);
							new_weight_shape = poseKeypoints_init_shape;
							new_weight_shape[0] = cur_start_loc + cur_bbox_nums;
							batch->poseKeypoints_.Reshape(new_weight_shape);
							caffe_copy(data_tmp.count(), data_tmp.cpu_data(), batch->poseKeypoints_.mutable_cpu_data());
						}
					}
					Dtype* prefetch_weight_write_loc = batch->weight_.mutable_cpu_data() + cur_start_loc * weight_data_dim;
					caffe_copy(cur_bbox_nums * weight_data_dim, &this->hdf5_blobs_->cpu_data()[hdf5_read_loc * weight_data_dim], prefetch_weight_write_loc);

					Dtype* prefetch_poseKeypoints_write_loc = nullptr;
					if (this->output_poseKeypoints_) {
						prefetch_poseKeypoints_write_loc = batch->poseKeypoints_.mutable_cpu_data() + cur_start_loc * poseKeypoints_data_dim;
						caffe_copy(cur_bbox_nums * poseKeypoints_data_dim, &this->hdf5_poseKeypoints_blobs_->cpu_data()[hdf5_read_loc * poseKeypoints_data_dim], prefetch_poseKeypoints_write_loc);
					}
					cur_start_loc += cur_bbox_nums;

					int offset = batch->data_.offset(item_id);
					this->transformed_data_.set_cpu_data(prefetch_data + offset);
					if (this->output_segMask_) {
						int seg_offset = batch->segMask_.offset(item_id);
						this->transformed_segMask_.set_cpu_data(prefetch_segMask + seg_offset);
					}
					this->data_transformer_->Transform(cv_img, &(this->transformed_data_), cv_mask_img, &(this->transformed_segMask_), prefetch_weight_write_loc, prefetch_poseKeypoints_write_loc, cur_bbox_nums);
					trans_time += timer.MicroSeconds();
				}				
			}
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	// Load bbox-data from given HDF5 filename into this->hdf5_blob_.
	template <typename Dtype>
	void ImageMixHDF5DataLayer<Dtype>::LoadBBoxHDF5FileData(const char* filename) {
		DLOG(INFO) << "Loading HDF5 file: " << filename;
		hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file_id < 0) {
			LOG(FATAL) << "Failed opening HDF5 file: " << filename;
		}

		const int MIN_DATA_DIM = 1;
		const int MAX_DATA_DIM = INT_MAX;

		this->hdf5_blobs_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
		// Allow reshape here, as we are loading data not params
		hdf5_load_nd_dataset(file_id, this->layer_param_.top(2).c_str(),
				MIN_DATA_DIM, MAX_DATA_DIM, this->hdf5_blobs_.get(), true);

		if (this->output_poseKeypoints_) {
			this->hdf5_poseKeypoints_blobs_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
			if(this->output_segMask_)
				hdf5_load_nd_dataset(file_id, this->layer_param_.top(4).c_str(),
					MIN_DATA_DIM, MAX_DATA_DIM, this->hdf5_poseKeypoints_blobs_.get(), true);
			else
				hdf5_load_nd_dataset(file_id, this->layer_param_.top(3).c_str(),
					MIN_DATA_DIM, MAX_DATA_DIM, this->hdf5_poseKeypoints_blobs_.get(), true);
			DLOG(INFO) << "Successfully loaded " << this->hdf5_poseKeypoints_blobs_->shape(0) << " rows";
		}

		herr_t status = H5Fclose(file_id);
		CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

		CHECK_GE(this->hdf5_blobs_->num_axes(), 1) << "Input must have at least 1 axis.";
		DLOG(INFO) << "Successfully loaded " << this->hdf5_blobs_->shape(0) << " rows";
	}

	INSTANTIATE_CLASS(ImageMixHDF5DataLayer);
	REGISTER_LAYER_CLASS(ImageMixHDF5Data);

}  // namespace caffe
#endif  // USE_OPENCV
