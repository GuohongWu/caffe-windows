#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

int main(int argc, char** argv) {

	std::ifstream in_data(argv[1], std::ios::in);
	int nums_, channels_, height_, width_;
	in_data >> nums_ >> channels_ >> height_ >> width_;

	caffe::BlobProto sum_blob;
	sum_blob.set_num(nums_);
	sum_blob.set_channels(channels_);
	sum_blob.set_height(height_);
	sum_blob.set_width(width_);
	const int data_size = nums_ * channels_ * height_ * width_;
	for (int i = 0; i < data_size; ++i) {
		sum_blob.add_data(0.);
	}

	float temp_;
	for (int i = 0; i < data_size; ++i) {
		in_data >> temp_;
		sum_blob.set_data(i, temp_);
	}

	WriteProtoToBinaryFile(sum_blob, argv[2]);

	return 0;
}