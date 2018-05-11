#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/cpm_data_layer.hpp"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

	template <typename TypeParam>
	class CPMDataLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		CPMDataLayerTest()
			: seed_(1701),
			blob_top_data_(new Blob<Dtype>()),
			blob_top_label_(new Blob<Dtype>()) { }
		virtual void SetUp() {
			blob_top_vec_.push_back(blob_top_data_);
			blob_top_vec_.push_back(blob_top_label_);
			Caffe::set_random_seed(seed_);
		}

		virtual ~CPMDataLayerTest() {
			delete blob_top_data_;
			delete blob_top_label_;
		}

		int seed_;
		string filename_;
		Blob<Dtype>* const blob_top_data_;
		Blob<Dtype>* const blob_top_label_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(CPMDataLayerTest, TestDtypesAndDevices);

	TYPED_TEST(CPMDataLayerTest, TestTops) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter param;
		param.add_top("data");
		param.add_top("label");

		CPMTransformationParameter* cpm_transform_param = param.mutable_cpm_transform_param();
		cpm_transform_param->set_stride(8);
		cpm_transform_param->set_max_rotate_degree(20);
		cpm_transform_param->set_visualize(true);
		cpm_transform_param->set_crop_size(368);
		cpm_transform_param->set_target_dist(0.6);
		cpm_transform_param->set_center_perterb_max(40);
		cpm_transform_param->set_num_parts(56);
		cpm_transform_param->set_np_in_lmdb(17);
		cpm_transform_param->set_do_clahe(false);

		DataParameter* data_param = param.mutable_data_param();
		data_param->set_batch_size(1);
		data_param->set_source("D:/haomaiyi/Realtime_Multi-Person_Pose_Estimation-master/training/lmdb_trainVal");
		data_param->set_backend(caffe::DataParameter_DB::DataParameter_DB_LMDB);

		CPMDataLayer<Dtype> layer(param);
		layer.DataLayerSetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		// Forward
		for (int iter = 0; iter < 100; ++iter) {
			layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

			int data_ch = this->blob_top_data_->channels();
			int data_h = this->blob_top_data_->height();
			int data_w = this->blob_top_data_->width();

			int label_ch = this->blob_top_label_->channels();
			int label_h = this->blob_top_label_->height();
			int label_w = this->blob_top_label_->width();

			for (int bz = 0; bz < this->blob_top_data_->num(); ++bz) {
				cv::Mat ch1(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + bz * 3 * 640 * 640);
				cv::Mat ch2(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + 640 * 640 + bz * 3 * 640 * 640);
				cv::Mat ch3(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + 2 * 640 * 640 + bz * 3 * 640 * 640);
				vector<cv::Mat> chs{ ch1, ch2, ch3 };
				cv::Mat cv_img;
				cv::merge(chs, cv_img);
				cv_img.convertTo(cv_img, CV_8UC3);



			}
		}
	}
}  // namespace caffe
#endif  // USE_OPENCV
