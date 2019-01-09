#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/layers/single_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template <typename Dtype>
	SingleImageDataLayer<Dtype>::~SingleImageDataLayer<Dtype>() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void SingleImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Read the img_source with filenames and labels
		const string& source_ = this->layer_param_.image_data_param().source();
		LOG(INFO) << "Opening file " << source_;
		std::ifstream infile(source_.c_str());
		string line;
		this->lines_.clear();
		while (std::getline(infile, line)) {
			lines_.emplace_back(line);
		}
		CHECK(!lines_.empty()) << "File is empty";

		if (this->layer_param_.image_data_param().shuffle()) {
			// randomly shuffle data
			LOG(INFO) << "Shuffling data";
			const unsigned int prefetch_rng_seed = caffe_rng_rand();
			prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
			ShuffleImages();
		}
		else {
			if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
				this->layer_param_.image_data_param().rand_skip() == 0) {
				LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
			}
		}
		LOG(INFO) << "A total of " << lines_.size() << " images.";

		lines_id_ = 0;
		// Check if we would need to randomly skip a few data points
		if (this->layer_param_.image_data_param().rand_skip()) {
			unsigned int skip = caffe_rng_rand() %
				this->layer_param_.image_data_param().rand_skip();
			LOG(INFO) << "Skipping first " << skip << " data points.";
			CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
			lines_id_ = skip;
		}

		this->PreFetchReshape(bottom, top);
	}

	template <typename Dtype>
	void SingleImageDataLayer<Dtype>::PreFetchReshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int new_height = this->layer_param_.image_data_param().new_height();
		int new_width = this->layer_param_.image_data_param().new_width();
		const bool is_color = this->layer_param_.image_data_param().is_color();
		const int batch_size = this->layer_param_.image_data_param().batch_size();

		CHECK_GT(batch_size, 0) << "Positive batch size required";
		CHECK((new_height == 0 && new_width == 0) ||
			(new_height > 0 && new_width > 0)) << "Current implementation requires "
			"new_height and new_width to be set at the same time.";

		// Read an image, and use it to initialize the top blob.
		if (new_height == 0) {
			string root_folder = this->layer_param_.image_data_param().root_folder();
			cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_], is_color);
			CHECK(cv_img.data) << "Could not load " << lines_[lines_id_];

			new_height = cv_img.rows;
			new_width = cv_img.cols;
		}

		vector<int> top0_shape{ 1, (is_color ? 3 : 1), new_height, new_width };
		vector<int> top1_shape(top0_shape);
		this->transformed_data_.Reshape(top0_shape);
		// Reshape prefetch_data		
		top0_shape[0] = top1_shape[0] = batch_size;
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->data_.Reshape(top0_shape); // top[0]
			this->prefetch_[i]->label_.Reshape(top1_shape); // top[1]
		}
		top[0]->Reshape(top0_shape);
		top[1]->Reshape(top1_shape);

		LOG(INFO) << "output data size: " << batch_size << ","
			<< top[0]->channels() << "," << new_height << ","
			<< new_width;
	}

	template <typename Dtype>
	void SingleImageDataLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void SingleImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());
		CHECK(batch->label_.count());

		const bool is_color = this->layer_param_.image_data_param().is_color();
		const int batch_size = this->layer_param_.image_data_param().batch_size();
		string root_folder = this->layer_param_.image_data_param().root_folder();

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();
		Dtype* prefetch_label = batch->label_.mutable_cpu_data();

		// datum scales
		const int lines_size = lines_.size();
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			timer.Start();
			bool valid_sample = false;
			while (!valid_sample) {
				CHECK_GT(lines_size, lines_id_);
				const std::string this_line = lines_[lines_id_];

				cv::Mat cv_img = ReadImageToCVMat(root_folder + this_line, is_color);
				if (!cv_img.data) {
					LOG(INFO) << "Could not load " << root_folder + this_line;
					valid_sample = false;
				}
				else {
					valid_sample = true;
				}
				read_time += timer.MicroSeconds();

				// go to the next iter
				lines_id_++;
				if (lines_id_ >= lines_size) {
					// We have reached the end. Restart from the first.
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
					if (this->layer_param_.image_data_param().shuffle()) {
						ShuffleImages();
					}
				}

				if (valid_sample) {
					timer.Start();
					// Apply transformations (mirror, crop...) to the image
					int offset1 = batch->label_.offset(item_id);
					Dtype* copy_label_data = prefetch_label + offset1;
					cv::Mat cv_label_img = cv_img;
					std::vector<cv::Mat> split_;
					cv::split(cv_label_img, split_);
					for (int c_ = 0; c_ < split_.size(); ++c_) {					
						for (auto cv_label_img_iter = split_[c_].begin<uchar>(); cv_label_img_iter != split_[c_].end<uchar>(); ++cv_label_img_iter)
							*copy_label_data++ = Dtype(*cv_label_img_iter) / Dtype(255); // image scale		
					}								

					int offset0 = batch->data_.offset(item_id);
					this->transformed_data_.set_cpu_data(prefetch_data + offset0);
					this->data_transformer_->Transform(cv_img, &(this->transformed_data_), false);
					trans_time += timer.MicroSeconds();
				}
			}
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}


	INSTANTIATE_CLASS(SingleImageDataLayer);
	REGISTER_LAYER_CLASS(SingleImageData);

}  // namespace caffe
#endif  // USE_OPENCV
