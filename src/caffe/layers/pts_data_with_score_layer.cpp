#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/layers/pts_data_with_score_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template <typename Dtype>
	PtsDataWithScoreLayer<Dtype>::~PtsDataWithScoreLayer<Dtype>() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void PtsDataWithScoreLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Read the img_source with filenames and labels
		const string& source_ = this->layer_param_.image_data_param().source();
		LOG(INFO) << "Opening file " << source_;
		std::ifstream in_file(source_.c_str());
		int rows_, cols;
		in_file >> rows_ >> cols;
		this->lines_.resize(rows_);
		for (int i = 0; i < rows_; ++i) {
			this->lines_[i].resize(cols);
			for (int j = 0; j < cols; ++j)
				in_file >> this->lines_[i][j];
		}

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
	void PtsDataWithScoreLayer<Dtype>::PreFetchReshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const int batch_size = this->layer_param_.image_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";

		vector<int> top0_shape{ 1, (int)this->lines_[0].size() - 1, 1, 1 };
		this->transformed_data_.Reshape(top0_shape);
		// Reshape prefetch_data		
		top0_shape[0] = batch_size;
		vector<int> top1_shape(top0_shape);
		top1_shape[1] = 1;
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->data_.Reshape(top0_shape); // top[0]
			this->prefetch_[i]->label_.Reshape(top1_shape); // top[1]
		}
		top[0]->Reshape(top0_shape);
		top[1]->Reshape(top1_shape);

		LOG(INFO) << "output data size: " << batch_size << ","
			<< top[0]->channels() << "," << 1 << ","
			<< 1;
	}

	template <typename Dtype>
	void PtsDataWithScoreLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void PtsDataWithScoreLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(batch->label_.count());
		CHECK(this->transformed_data_.count());

		const int batch_size = this->layer_param_.image_data_param().batch_size();
		Dtype* prefetch_data = batch->data_.mutable_cpu_data();
		Dtype* prefetch_label = batch->label_.mutable_cpu_data();

		// datum scales
		const int lines_size = lines_.size();
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			timer.Start();
			CHECK_GT(lines_size, lines_id_);
			vector<Dtype> this_line_pts = lines_[lines_id_];
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

			timer.Start();
			int offset1 = batch->label_.offset(item_id);
			*(prefetch_label + offset1) = this_line_pts.back();

			int offset0 = batch->data_.offset(item_id);
			this->transformed_data_.set_cpu_data(prefetch_data + offset0);
			Dtype* prefetch_blob = this->transformed_data_.mutable_cpu_data();

			// copy values
			for (int k = 0; k + 1 < this_line_pts.size(); ++k)
				prefetch_blob[k] = this_line_pts[k];
			trans_time += timer.MicroSeconds();
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}


	INSTANTIATE_CLASS(PtsDataWithScoreLayer);
	REGISTER_LAYER_CLASS(PtsDataWithScore);

}  // namespace caffe
#endif  // USE_OPENCV
