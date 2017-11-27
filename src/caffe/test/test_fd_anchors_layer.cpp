#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/fd_anchors_layer.hpp"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

	template <typename TypeParam>
	class FDAnchorsLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		FDAnchorsLayerTest()
			: seed_(1701) { 
			blob_bottom_vec_.resize(3);
			blob_bottom_vec_[0] = new Blob<Dtype>(1, 3, 640, 640);
			blob_bottom_vec_[1] = new Blob<Dtype>(1, 1, 1, 1);
			blob_bottom_vec_[2] = new Blob<Dtype>(1, 5, 1, 1);

			blob_top_vec_.resize(12);
			for (int i = 0; i < 12; ++i) {
				blob_top_vec_[i] = new Blob<Dtype>();
			}
		}
		virtual void SetUp() {
			Caffe::set_random_seed(seed_);
			blob_bottom_vec_[1]->mutable_cpu_data()[0] = Dtype(0);

			blob_bottom_vec_[2]->mutable_cpu_data()[0] = Dtype(10);
			blob_bottom_vec_[2]->mutable_cpu_data()[1] = Dtype(10);
			blob_bottom_vec_[2]->mutable_cpu_data()[2] = Dtype(30);
			blob_bottom_vec_[2]->mutable_cpu_data()[3] = Dtype(30);
			blob_bottom_vec_[2]->mutable_cpu_data()[4] = Dtype(1);
		}

		virtual ~FDAnchorsLayerTest() {
			for(auto &v : blob_bottom_vec_)
				delete v;

			for(auto &v : blob_top_vec_)
				delete v;
		}

		int seed_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(FDAnchorsLayerTest, TestDtypesAndDevices);

	TYPED_TEST(FDAnchorsLayerTest, TestTops) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter param;
		//param.add_top("label_map");
		//param.add_top("reg_map");

		FDAnchorsParameter* fd_anchors_param = param.mutable_fd_anchors_param();
		fd_anchors_param->set_anchor_size(16);
		fd_anchors_param->set_anchor_stride(4);
		fd_anchors_param->set_nms_upper_thresh(0.35);
		fd_anchors_param->set_nms_lower_thresh(0.1);
		fd_anchors_param->add_scale(1);
		fd_anchors_param->add_scale(2);
		fd_anchors_param->add_scale(4);
		fd_anchors_param->add_scale(8);
		fd_anchors_param->add_scale(16);
		fd_anchors_param->add_scale(32);

		FDAnchorsLayer<Dtype> layer(param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		// Forward
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	}
}  // namespace caffe
#endif  // USE_OPENCV
