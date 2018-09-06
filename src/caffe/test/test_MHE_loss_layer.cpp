#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/MHE_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename TypeParam>
	class MHELossLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		MHELossLayerTest()
			: blob_bottom_data_i_(new Blob<Dtype>(100, 32, 1, 1)),
			blob_bottom_y_(new Blob<Dtype>(16, 1, 1, 1)),
			blob_top_loss_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			filler_param.set_min(-1.0);
			filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
			UnitSphereFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_data_i_);
			blob_bottom_vec_.push_back(blob_bottom_data_i_);
			for (int i = 0; i < blob_bottom_y_->count(); ++i) {
				blob_bottom_y_->mutable_cpu_data()[i] = caffe_rng_rand() % 100;  // 0 ~ 99
			}

			blob_bottom_vec_.push_back(blob_bottom_y_);
			blob_top_vec_.push_back(blob_top_loss_);
		}
		virtual ~MHELossLayerTest() {
			delete blob_bottom_data_i_;
			delete blob_bottom_y_;
			delete blob_top_loss_;
		}

		Blob<Dtype>* const blob_bottom_data_i_;
		Blob<Dtype>* const blob_bottom_y_;
		Blob<Dtype>* const blob_top_loss_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(MHELossLayerTest, TestDtypesAndDevices);



	TYPED_TEST(MHELossLayerTest, TestGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		layer_param.mutable_mhe_loss_param()->set_lambda_m(10.0f);

		MHELossLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		GradientChecker<Dtype> checker(1e-3, 1e-3, caffe_rng_rand());
		// check the gradient for the first two bottom layers
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_, 0);
	}

}  // namespace caffe
