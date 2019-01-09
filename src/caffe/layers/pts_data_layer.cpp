#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/layers/pts_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template <typename Dtype>
	PtsDataLayer<Dtype>::~PtsDataLayer<Dtype>() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void PtsDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
	void PtsDataLayer<Dtype>::PreFetchReshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const int batch_size = this->layer_param_.image_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";

		vector<int> top0_shape{ 1, (int)this->lines_[0].size(), 1, 1 };
		this->transformed_data_.Reshape(top0_shape);
		// Reshape prefetch_data		
		top0_shape[0] = batch_size;
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->data_.Reshape(top0_shape); // top[0]
			this->prefetch_[i]->label_.Reshape(top0_shape); // top[1]
		}
		top[0]->Reshape(top0_shape);
		top[1]->Reshape(top0_shape);

		LOG(INFO) << "output data size: " << batch_size << ","
			<< top[0]->channels() << "," << 1 << ","
			<< 1;
	}

	template <typename Dtype>
	void PtsDataLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void PtsDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
			auto this_line_pts = lines_[lines_id_];
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
			Dtype* gt_data = prefetch_label + offset1;
			// copy values
			for (int k = 0; k < this_line_pts.size(); ++k)
				gt_data[k] = this_line_pts[k];

			int offset0 = batch->data_.offset(item_id);
			this->transformed_data_.set_cpu_data(prefetch_data + offset0);
			Dtype* prefetch_blob = this->transformed_data_.mutable_cpu_data();

			// this->PtsDistort(this_line_pts);

			// copy values
			for (int k = 0; k < this_line_pts.size(); ++k)
				prefetch_blob[k] = this_line_pts[k];
			trans_time += timer.MicroSeconds();			
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	template <typename Dtype>
	void PtsDataLayer<Dtype>::PtsDistort(vector<Dtype> &in_pts) {
		int prob_ints = 5;
		cv::RNG uni_rng(time(0));
		//1. mouth
		if (uni_rng.uniform(0, prob_ints) == 0)
		{
			Dtype x_offset = 0.03;
			if (uni_rng.uniform(0, 2)) {
				for(int i = 46; i < 49; ++i)
					in_pts[2 * i] += x_offset;
				for (int i = 56; i < 59; ++i)
					in_pts[2 * i] += x_offset;
				in_pts[2 * 63] += x_offset;
			}
			else {
				for (int i = 50; i < 55; ++i)
					in_pts[2 * i] -= x_offset;
				for (int i = 60; i < 62; ++i)
					in_pts[2 * i] -= x_offset;
			}			
		}
		if (uni_rng.uniform(0, prob_ints) == 0)
		{
			cv::Point_<Dtype> mouth_center(0, 0);
			for (int i = 46; i < 64; ++i) {
				mouth_center.x += in_pts[2 * i];
				mouth_center.y += in_pts[2 * i + 1];
			}
			mouth_center.x /= Dtype(18);
			mouth_center.y /= Dtype(18);
			cv::Mat rot_mat;
			if(uni_rng.uniform(0, 2))
				rot_mat = cv::getRotationMatrix2D(mouth_center, CV_PI/22, 1.0);
			else
				rot_mat = cv::getRotationMatrix2D(mouth_center, -CV_PI / 22, 1.0);

			cv::Point_<Dtype> pt_tmp;
			for (int i = 46; i < 64; ++i) {
				pt_tmp.x = in_pts[2 * i] * rot_mat.at<double>(0) + in_pts[2 * i + 1] * rot_mat.at<double>(1) + rot_mat.at<double>(2);
				pt_tmp.y = in_pts[2 * i] * rot_mat.at<double>(3) + in_pts[2 * i + 1] * rot_mat.at<double>(4) + rot_mat.at<double>(5);
				in_pts[2 * i] = pt_tmp.x;
				in_pts[2 * i + 1] = pt_tmp.y;
			}
		}
		if (uni_rng.uniform(0, prob_ints) == 0)
		{
			Dtype x_offset = 0.05;
			if (uni_rng.uniform(0, 2))
				x_offset = -x_offset;
			for (int i = 46; i < 64; ++i)
				in_pts[2 * i] += x_offset;
		}

		// nose
		if (uni_rng.uniform(0, prob_ints) == 0)
		{
			cv::Point_<Dtype> nose_tip;
			nose_tip.x = in_pts[2 * 64];
			nose_tip.y = in_pts[2 * 64 + 1];
			cv::Mat rot_mat = cv::getRotationMatrix2D(nose_tip, 0.0, 1.35);

			cv::Point_<Dtype> pt_tmp;
			for (int i = 35; i < 46; ++i) {
				pt_tmp.x = in_pts[2 * i] * rot_mat.at<double>(0) + in_pts[2 * i + 1] * rot_mat.at<double>(1) + rot_mat.at<double>(2);
				pt_tmp.y = in_pts[2 * i] * rot_mat.at<double>(3) + in_pts[2 * i + 1] * rot_mat.at<double>(4) + rot_mat.at<double>(5);
				in_pts[2 * i] = pt_tmp.x;
				in_pts[2 * i + 1] = pt_tmp.y;
			}
		}

		// eyes
		if (uni_rng.uniform(0, prob_ints) == 0)
		{
			// left
			if (uni_rng.uniform(0, 2))
			{
				cv::Point_<Dtype> pt_res;
				pt_res.x = 0.25 * (in_pts[2 * 66] - in_pts[2 * 65]);
				pt_res.y = 0.25 * (in_pts[2 * 66 + 1] - in_pts[2 * 65 + 1]);
				in_pts[2 * 65] += pt_res.x;
				in_pts[2 * 66] -= pt_res.x;
				in_pts[2 * 65 + 1] += pt_res.y;
				in_pts[2 * 66 + 1] -= pt_res.y;

				pt_res.x = 0.25 * (in_pts[2 * 67] - in_pts[2 * 68]);
				pt_res.y = 0.25 * (in_pts[2 * 67 + 1] - in_pts[2 * 68 + 1]);
				in_pts[2 * 68] += pt_res.x;
				in_pts[2 * 67] -= pt_res.x;
				in_pts[2 * 68 + 1] += pt_res.y;
				in_pts[2 * 67 + 1] -= pt_res.y;

				pt_res.x = 0.25 * (in_pts[2 * 30] - in_pts[2 * 28]);
				pt_res.y = 0.25 * (in_pts[2 * 30 + 1] - in_pts[2 * 28 + 1]);
				in_pts[2 * 28] += pt_res.x;
				in_pts[2 * 30] -= pt_res.x;
				in_pts[2 * 28 + 1] += pt_res.y;
				in_pts[2 * 30 + 1] -= pt_res.y;
			}
			else // right
			{
				cv::Point_<Dtype> pt_res;
				pt_res.x = 0.25 * (in_pts[2 * 70] - in_pts[2 * 69]);
				pt_res.y = 0.25 * (in_pts[2 * 70 + 1] - in_pts[2 * 69 + 1]);
				in_pts[2 * 69] += pt_res.x;
				in_pts[2 * 70] -= pt_res.x;
				in_pts[2 * 69 + 1] += pt_res.y;
				in_pts[2 * 70 + 1] -= pt_res.y;

				pt_res.x = 0.25 * (in_pts[2 * 71] - in_pts[2 * 72]);
				pt_res.y = 0.25 * (in_pts[2 * 71 + 1] - in_pts[2 * 72 + 1]);
				in_pts[2 * 72] += pt_res.x;
				in_pts[2 * 71] -= pt_res.x;
				in_pts[2 * 72 + 1] += pt_res.y;
				in_pts[2 * 71 + 1] -= pt_res.y;

				pt_res.x = 0.25 * (in_pts[2 * 34] - in_pts[2 * 32]);
				pt_res.y = 0.25 * (in_pts[2 * 34 + 1] - in_pts[2 * 32 + 1]);
				in_pts[2 * 32] += pt_res.x;
				in_pts[2 * 34] -= pt_res.x;
				in_pts[2 * 32 + 1] += pt_res.y;
				in_pts[2 * 34 + 1] -= pt_res.y;
			}
		}
		
		// face contours
		if (uni_rng.uniform(0, prob_ints) == 0)
		{
			cv::Point_<Dtype> pt_center;
			pt_center.x = 0.5 * (in_pts[2 * 0] + in_pts[2 * 14]);
			pt_center.y = 0.5 * (in_pts[2 * 0 + 1] + in_pts[2 * 14 + 1]);
			cv::Mat rot_mat;
			// zoom in
			if (uni_rng.uniform(0, 2))
				rot_mat = cv::getRotationMatrix2D(pt_center, 0.0, 1.35);
			else // zoom out
				rot_mat = cv::getRotationMatrix2D(pt_center, 0.0, 0.75);

			cv::Point_<Dtype> pt_tmp;
			for (int i = 0; i < 15; ++i) {
				pt_tmp.x = in_pts[2 * i] * rot_mat.at<double>(0) + in_pts[2 * i + 1] * rot_mat.at<double>(1) + rot_mat.at<double>(2);
				pt_tmp.y = in_pts[2 * i] * rot_mat.at<double>(3) + in_pts[2 * i + 1] * rot_mat.at<double>(4) + rot_mat.at<double>(5);
				in_pts[2 * i] = pt_tmp.x;
				in_pts[2 * i + 1] = pt_tmp.y;
			}
		}
		if (uni_rng.uniform(0, prob_ints) == 0)
		{
			Dtype y_offset = 0.1;
			if (uni_rng.uniform(0, 2))
				y_offset = -y_offset;
			for (int i = 0; i < 15; ++i) {
				in_pts[2 * i + 1] += y_offset;
			}
		}
	}


	INSTANTIATE_CLASS(PtsDataLayer);
	REGISTER_LAYER_CLASS(PtsData);

}  // namespace caffe
#endif  // USE_OPENCV
