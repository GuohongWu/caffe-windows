#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/image_mix_hdf5_data_layer.hpp"

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
	class ImageMixHDF5DataLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		ImageMixHDF5DataLayerTest()
			: seed_(1701),
			blob_top_data_(new Blob<Dtype>()),
			blob_top_bboxStartloc_(new Blob<Dtype>()),
		    blob_top_bboxLists_(new Blob<Dtype>()),
			blob_top_segMask_(new Blob<Dtype>()),
			blob_top_poseKeypoints_(new Blob<Dtype>()) { }
		virtual void SetUp() {
			blob_top_vec_.push_back(blob_top_data_);
			blob_top_vec_.push_back(blob_top_bboxStartloc_);
			blob_top_vec_.push_back(blob_top_bboxLists_);
			blob_top_vec_.push_back(blob_top_segMask_);
			blob_top_vec_.push_back(blob_top_poseKeypoints_);
			Caffe::set_random_seed(seed_);
		}

		virtual ~ImageMixHDF5DataLayerTest() {
			delete blob_top_data_;
			delete blob_top_bboxStartloc_;
			delete blob_top_bboxLists_;
			delete blob_top_segMask_;
			delete blob_top_poseKeypoints_;
		}

		int seed_;
		string filename_;
		Blob<Dtype>* const blob_top_data_;
		Blob<Dtype>* const blob_top_bboxStartloc_;
		Blob<Dtype>* const blob_top_bboxLists_;
		Blob<Dtype>* const blob_top_segMask_;
		Blob<Dtype>* const blob_top_poseKeypoints_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(ImageMixHDF5DataLayerTest, TestDtypesAndDevices);

	TYPED_TEST(ImageMixHDF5DataLayerTest, TestTops) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter param;
		param.add_top("data");
		param.add_top("bbox_startloc");
		param.add_top("bbox_lists");
		param.add_top("segMask_data");
		param.add_top("poseKeypoints_lists");

		ImageMixHDF5DataParameter* image_data_param = param.mutable_image_mix_hdf5_data_param();
		image_data_param->set_batch_size(2);
		image_data_param->set_root_folder("B:/Object_Detection_And_Segmentation/Faster_RCNN/TrainData/imgData/");
		image_data_param->set_img_source("B:/Object_Detection_And_Segmentation/Faster_RCNN/TrainData/imgData/img_name_info.txt");
		image_data_param->set_bbox_source("B:/Object_Detection_And_Segmentation/Faster_RCNN/TrainData/imgData/CoCo_bbox_lists.h5");
		image_data_param->set_shuffle(false);
		image_data_param->set_new_height(640);
		image_data_param->set_new_width(640);

		TransformationParameter* trans_param = param.mutable_transform_param();
		trans_param->add_mean_value(0.0f);
		trans_param->add_mean_value(0.0f);
		trans_param->add_mean_value(0.0f);
		trans_param->set_scale(1.0f);
		trans_param->set_force_color(true);
		trans_param->set_scaled_resize(true);
		trans_param->set_mirror(true);

		ImageMixHDF5DataLayer<Dtype> layer(param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		
		// Forward
		for (int iter = 0; iter < 100; ++iter) {
			layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

			EXPECT_EQ(this->blob_top_data_->num(), 2);
			EXPECT_EQ(this->blob_top_data_->channels(), 3);
			EXPECT_EQ(this->blob_top_data_->height(), 640);
			EXPECT_EQ(this->blob_top_data_->width(), 640);
			EXPECT_EQ(this->blob_top_bboxStartloc_->num(), 2);
			EXPECT_EQ(this->blob_top_bboxStartloc_->channels(), 1);
			EXPECT_EQ(this->blob_top_bboxStartloc_->height(), 1);
			EXPECT_EQ(this->blob_top_bboxStartloc_->width(), 1);
			EXPECT_EQ(this->blob_top_bboxLists_->channels(), 5);
			EXPECT_EQ(this->blob_top_bboxLists_->height(), 1);
			EXPECT_EQ(this->blob_top_bboxLists_->width(), 1);
			EXPECT_EQ(this->blob_top_segMask_->num(), 2);
			EXPECT_EQ(this->blob_top_segMask_->channels(), 1);
			EXPECT_EQ(this->blob_top_segMask_->height(), 640);
			EXPECT_EQ(this->blob_top_segMask_->width(), 640);
			EXPECT_EQ(this->blob_top_poseKeypoints_->num(), this->blob_top_bboxLists_->num());
			EXPECT_EQ(this->blob_top_poseKeypoints_->channels(), 1);
			EXPECT_EQ(this->blob_top_poseKeypoints_->height(), 17);
			EXPECT_EQ(this->blob_top_poseKeypoints_->width(), 3);

			for (int bz = 0; bz < this->blob_top_data_->num(); ++bz) {
				cv::Mat ch1(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + bz * 3 * 640 * 640);
				cv::Mat ch2(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + 640 * 640 + bz * 3 * 640 * 640);
				cv::Mat ch3(640, 640, CV_32FC1, this->blob_top_data_->mutable_cpu_data() + 2 * 640 * 640 + bz * 3 * 640 * 640);
				vector<cv::Mat> chs{ ch1, ch2, ch3 };
				cv::Mat cv_img;
				cv::merge(chs, cv_img);
				cv_img.convertTo(cv_img, CV_8UC3);

				// test segMask
				cv::Mat cv_mask_img;
				if (this->blob_top_vec_.size() >= 4) {
					cv::Mat cv_mask_img(640, 640, CV_32FC1, this->blob_top_segMask_->mutable_cpu_data() + bz * 1 * 640 * 640);
					cv::Mat cv_mask_img_clone = cv_mask_img.clone();
					chs.push_back(cv_mask_img_clone * 100);
					cv::Mat cv_img_seg;
					cv::merge(chs, cv_img_seg);
					cv_img_seg.convertTo(cv_img_seg, CV_8UC4);

					cv_mask_img.convertTo(cv_mask_img, CV_8UC1);
				}

				std::cout << "\n\nblob_top_bboxStartloc_ Values: " << this->blob_top_bboxStartloc_->cpu_data()[bz] << std::endl;
				std::cout << "\n\nblob_top_bboxLists_ Values: \n";
				int data_dim = 5;
				const Dtype* bbox_data = this->blob_top_bboxLists_->cpu_data() + int(this->blob_top_bboxStartloc_->cpu_data()[bz]) * data_dim;
				for (int i = this->blob_top_bboxStartloc_->cpu_data()[bz]; i < ((bz + 1 == 2) ? this->blob_top_bboxLists_->num() : this->blob_top_bboxStartloc_->cpu_data()[bz + 1]); ++i) {
					cv::rectangle(cv_img, cv::Point(bbox_data[0], bbox_data[1]), cv::Point(bbox_data[2], bbox_data[3]), cv::Scalar(0, 255, 0));
					for (int j = 0; j < data_dim; ++j)
						std::cout << bbox_data[j] << ' ';
					std::cout << std::endl;
					bbox_data += 5;
				}

				// test poseKeypoints
				int points_data_dim = 17 * 3;
				const Dtype* points_data = this->blob_top_poseKeypoints_->cpu_data() + int(this->blob_top_bboxStartloc_->cpu_data()[bz]) * points_data_dim;
				for (int i = this->blob_top_bboxStartloc_->cpu_data()[bz]; i < ((bz + 1 == 2) ? this->blob_top_bboxLists_->num() : this->blob_top_bboxStartloc_->cpu_data()[bz + 1]); ++i) {
					for (int j = 0; j < 17; ++j) {
						if(points_data[2])
							cv::circle(cv_img, cv::Point(points_data[0], points_data[1]), 2.5, cv::Scalar(0, 255, 0), -1);
						else
							cv::circle(cv_img, cv::Point(points_data[0], points_data[1]), 2.5, cv::Scalar(0, 0, 255), -1);

						points_data += 3;
					}
				}
			}
		}
	}
}  // namespace caffe
#endif  // USE_OPENCV
