#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/gram_mat_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {

#ifndef CPU_ONLY
	extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

	template <typename TypeParam>
	class GramMatLayerTest : public CPUDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		GramMatLayerTest()
			: blob_bottom_(new Blob<Dtype>(4, 2, 3, 1)),
			blob_top_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);

			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~GramMatLayerTest() {
			delete blob_bottom_;
			delete blob_top_;
		}
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	//TYPED_TEST_CASE(GramMatLayerTest, TestDtypesAndDevices);
	TYPED_TEST_CASE(GramMatLayerTest, ::testing::Types<CPUDeviceTest<float>>);

	TYPED_TEST(GramMatLayerTest, TestForward) {
		typedef typename TypeParam::Dtype Dtype;
			LayerParameter layer_param;

			GramMatLayer<Dtype> layer(layer_param);
			layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

			cv::Mat X_(this->blob_bottom_->channels(), this->blob_bottom_->height(), CV_32FC1, this->blob_bottom_->mutable_cpu_data());
			cv::Mat X_t = X_.t();
			cv::Mat Y = X_ * X_t;
			std::cout << Y << std::endl;

			cv::Mat top_T(this->blob_top_->channels(), this->blob_top_->height(), CV_32FC1, this->blob_top_->mutable_cpu_data());
			std::cout << top_T << std::endl;
	}

	TYPED_TEST(GramMatLayerTest, TestGradient) {
		typedef typename TypeParam::Dtype Dtype;
		
		bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
		IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
		if (Caffe::mode() == Caffe::CPU ||
			sizeof(Dtype) == 4 || IS_VALID_CUDA) {
			LayerParameter layer_param;
			
			GramMatLayer<Dtype> layer(layer_param);
			layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				this->blob_top_vec_, 0);
		}
		else {
			LOG(ERROR) << "Skipping test due to old architecture.";
		}
	}

}  // namespace caffe
