#include <cmath>
#include <vector>
#include <conio.h>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/roi_alignment_layer.hpp"
#include "caffe/custom_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;



namespace caffe {

	template <typename TypeParam>
	class ROIPoolingLayerTest : public GPUDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		ROIPoolingLayerTest()
			: blob_bottom_data_1(new Blob<Dtype>(1, 3, 38, 38)),
			blob_bottom_data_2(new Blob<Dtype>(1,5,1,1)),
			blob_top_loss_(new Blob<Dtype>()) {
				// fill the values
				FillerParameter filler_param;
				//filler_param.set_std(10);
				GaussianFiller<Dtype> filler(filler_param);
				filler.Fill(this->blob_bottom_data_1);
				//filler.Fill(this->blob_bottom_data_2);

				blob_bottom_vec_.push_back(blob_bottom_data_1);
				for(int i = 0;i < blob_bottom_data_2->num();++i)
				{
					blob_bottom_data_2->mutable_cpu_data()[i*5] = 0;
					blob_bottom_data_2->mutable_cpu_data()[i*5+1] = 10;
					blob_bottom_data_2->mutable_cpu_data()[i*5+2] = 10;
					blob_bottom_data_2->mutable_cpu_data()[i*5+3] = 80;
					blob_bottom_data_2->mutable_cpu_data()[i*5+4] = 80;
				}

				blob_bottom_vec_.push_back(blob_bottom_data_2);

				blob_top_vec_.push_back(blob_top_loss_);
		}
		virtual ~ROIPoolingLayerTest() {
			delete blob_bottom_data_1;
			delete blob_bottom_data_2;
			delete blob_top_loss_;
		}

		Blob<Dtype>* const blob_bottom_data_1;
		Blob<Dtype>* const blob_bottom_data_2;

		Blob<Dtype>* const blob_top_loss_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	template <typename TypeParam>
	class ROIAlignmentLayerTest : public GPUDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;
	protected:
		ROIAlignmentLayerTest()
			: blob_bottom_data_1(new Blob<Dtype>(1, 3, 10, 10)),
			blob_bottom_data_2(new Blob<Dtype>(1,5,1,1)),
			blob_top_loss_(new Blob<Dtype>()) {
				// fill the values
				FillerParameter filler_param;
				//filler_param.set_std(10);
				GaussianFiller<Dtype> filler(filler_param);
				filler.Fill(this->blob_bottom_data_1);
				//filler.Fill(this->blob_bottom_data_2);

				blob_bottom_vec_.push_back(blob_bottom_data_1);
				for(int i = 0;i < blob_bottom_data_2->num();++i)
				{
					blob_bottom_data_2->mutable_cpu_data()[i*5] = (Dtype)0;
					blob_bottom_data_2->mutable_cpu_data()[i*5+1] = (Dtype)1;
					blob_bottom_data_2->mutable_cpu_data()[i*5+2] = (Dtype)1;
					blob_bottom_data_2->mutable_cpu_data()[i*5+3] = (Dtype)8;
					blob_bottom_data_2->mutable_cpu_data()[i*5+4] = (Dtype)8;
				}

				blob_bottom_vec_.push_back(blob_bottom_data_2);

				blob_top_vec_.push_back(blob_top_loss_);
		}
		virtual ~ROIAlignmentLayerTest() {
			delete blob_bottom_data_1;
			delete blob_bottom_data_2;
			delete blob_top_loss_;
		}

		Blob<Dtype>* const blob_bottom_data_1;
		Blob<Dtype>* const blob_bottom_data_2;

		Blob<Dtype>* const blob_top_loss_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(ROIAlignmentLayerTest, ::testing::Types<GPUDeviceTest<float>>);

	TYPED_TEST(ROIAlignmentLayerTest, TestGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		layer_param.add_loss_weight(1);

		layer_param.mutable_roi_alignment_param()->set_pooled_h(7);
		layer_param.mutable_roi_alignment_param()->set_pooled_w(7);
		layer_param.mutable_roi_alignment_param()->set_spatial_scale(1);

		ROIAlignmentLayer<Dtype> layer(layer_param);
		
		layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
		// check the gradient here
		GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_, 0);
	}


	//TYPED_TEST_CASE(ROIPoolingLayerTest, ::testing::Types<GPUDevice<float> >);

	//TYPED_TEST(ROIPoolingLayerTest, TestGradient) {
	//	typedef typename TypeParam::Dtype Dtype;
	//	LayerParameter layer_param;
	//	layer_param.add_loss_weight(1);

	//	layer_param.mutable_roi_pooling_param()->set_pooled_h(7);
	//	layer_param.mutable_roi_pooling_param()->set_pooled_w(7);
	//	layer_param.mutable_roi_pooling_param()->set_spatial_scale(1);

	//	ROIPoolingLayer<Dtype> layer(layer_param);
	//	
	//	layer.SetUp(this->blob_bottom_vec_,this->blob_top_vec_);
	//	// check the gradient here
	//	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	//	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	//		this->blob_top_vec_, 0);
	//}

	//TYPED_TEST_CASE(ROIAlignmentLayerTest, ::testing::Types<GPUDevice<float> >);
	//TYPED_TEST_CASE(ROIAlignmentLayerTest,TestDtypesAndDevices);
	//TYPED_TEST(ROIAlignmentLayerTest, TestCPU) {

	//	typedef typename TypeParam::Dtype Dtype;
	//	LayerParameter layer_param;	
	//	layer_param.add_loss_weight(1);

	//	layer_param.mutable_roi_alignment_param()->set_pooled_h(7);
	//	layer_param.mutable_roi_alignment_param()->set_pooled_w(7);
	//	layer_param.mutable_roi_alignment_param()->set_spatial_scale(1);
	//	ROIAlignmentLayer<Dtype> layer(layer_param);

	//	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	//	layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	//	// Now, check values
	//	//const TypeParam* bottom_data = this->blob_bottom_0->cpu_data();
	//	//const TypeParam* top_data = this->blob_top_->cpu_data();
	//	//for(int i = 0;i < 4;++i)
	//	//	cout<<this->blob_top_->shape(i)<<endl;
	//	//cout<<"ratio "<<top_data[0]<<endl;

	//}


}