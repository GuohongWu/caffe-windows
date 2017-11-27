#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/fcn_image_data_layer.hpp"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

	template <typename TypeParam>
	class FCNImageDataLayerTest : public GPUDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		FCNImageDataLayerTest()
			: seed_(1701),
			blob_top_data_(new Blob<Dtype>()),
			blob_top_segMask_(new Blob<Dtype>()) { }
		virtual void SetUp() {
			blob_top_vec_.push_back(blob_top_data_);
			blob_top_vec_.push_back(blob_top_segMask_);
			Caffe::set_random_seed(seed_);
		}

		virtual ~FCNImageDataLayerTest() {
			delete blob_top_data_;
			delete blob_top_segMask_;
		}

		int seed_;
		string filename_;
		Blob<Dtype>* const blob_top_data_;
		Blob<Dtype>* const blob_top_segMask_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(FCNImageDataLayerTest, TestDtypesAndDevices);

	TYPED_TEST(FCNImageDataLayerTest, TestTops) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter param;

		FCNImageDataParameter* fcn_image_data_param = param.mutable_fcn_image_data_param();
		fcn_image_data_param->set_classification_nums(1);

		ImageDataParameter* image_data_param = param.mutable_image_data_param();
		image_data_param->set_batch_size(2);
		image_data_param->set_root_folder("B:/Object_Detection_And_Segmentation/Faster_RCNN/TrainData/imgData/");
		image_data_param->set_source("B:/Object_Detection_And_Segmentation/Faster_RCNN/TrainData/imgData/fcn_img_info.txt");
		image_data_param->set_shuffle(false);
		image_data_param->set_new_height(640);
		image_data_param->set_new_width(640);

		TransformationParameter* trans_param = param.mutable_transform_param();
		trans_param->add_mean_value(0.0f);
		trans_param->add_mean_value(0.0f);
		trans_param->add_mean_value(0.0f);
		trans_param->set_scale(1.0f);
		trans_param->set_force_color(true);
		trans_param->set_color_distort(false);
		trans_param->set_scaled_resize(true);
		trans_param->set_mirror(true);

		FCNImageDataLayer<Dtype> layer(param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		// Forward
		for (int iter = 0; iter < 100; ++iter) {
			layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

			EXPECT_EQ(this->blob_top_data_->num(), 2);
			EXPECT_EQ(this->blob_top_data_->channels(), 3);
			EXPECT_EQ(this->blob_top_data_->height(), 640);
			EXPECT_EQ(this->blob_top_data_->width(), 640);
			EXPECT_EQ(this->blob_top_segMask_->num(), 2);
			EXPECT_EQ(this->blob_top_segMask_->channels(), 1);
			EXPECT_EQ(this->blob_top_segMask_->height(), 640);
			EXPECT_EQ(this->blob_top_segMask_->width(), 640);

			for (int bz = 0; bz < this->blob_top_data_->num(); ++bz) {
				cv::Mat ch1(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + bz * 3 * 640 * 640);
				cv::Mat ch2(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + 640 * 640 + bz * 3 * 640 * 640);
				cv::Mat ch3(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + 2 * 640 * 640 + bz * 3 * 640 * 640);
				vector<cv::Mat> chs{ ch1, ch2, ch3 };
				cv::Mat cv_img;
				cv::merge(chs, cv_img);
				cv_img.convertTo(cv_img, CV_8UC3);

				// test segMask
					cv::Mat cv_mask_img(640, 640, CV_32FC1, this->blob_top_segMask_->mutable_cpu_data() + bz * 1 * 640 * 640);
					cv::Mat cv_mask_img_clone = cv_mask_img.clone();
					chs.push_back(cv_mask_img_clone * 100);
					cv::Mat cv_img_seg;
					cv::merge(chs, cv_img_seg);
					cv_img_seg.convertTo(cv_img_seg, CV_8UC4);
					cv_mask_img.convertTo(cv_mask_img, CV_8UC1);
			}
		}
	}
}  // namespace caffe
#endif  // USE_OPENCV
