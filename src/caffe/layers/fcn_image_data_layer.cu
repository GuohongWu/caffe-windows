#include "caffe/layers/fcn_image_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void kernel_pixelwise_clip(const int total_nums, const int class_nums, Dtype* mask_data) {
		CUDA_KERNEL_LOOP(index, total_nums) {
			if (mask_data[index] != Dtype(255) && mask_data[index] > class_nums)
				mask_data[index] = class_nums;
		}
	}

	template <typename Dtype>
	__global__ void kernel_channel_eval(const int total_nums, const int channel_nums, const int img_size, const Dtype* mask_data, Dtype* top1_data) {
		CUDA_KERNEL_LOOP(index, total_nums) {
			int n = index / (img_size * channel_nums);
			int idn = index % img_size;
			int c = index % (img_size * channel_nums) / img_size;
			int mask_idx = n * img_size + idn;
			top1_data[index] = (mask_data[mask_idx] == Dtype(c+1) || mask_data[mask_idx] == Dtype(255)) ? mask_data[mask_idx] : Dtype(0);
		}
	}

	template <typename Dtype>
	void FCNImageDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (this->prefetch_current_) {
			this->prefetch_free_.push(this->prefetch_current_);
		}
		this->prefetch_current_ = this->prefetch_full_.pop("Waiting for data");
		top[0]->set_gpu_data(this->prefetch_current_->data_.mutable_gpu_data());

		const int class_nums = this->layer_param_.fcn_image_data_param().classification_nums();
		const int total_nums = this->prefetch_current_->label_.count();
		Dtype* mask_data = this->prefetch_current_->label_.mutable_gpu_data();
		kernel_pixelwise_clip<Dtype> << <CAFFE_GET_BLOCKS(total_nums),
			CAFFE_CUDA_NUM_THREADS >> >(total_nums, class_nums, mask_data);
		CUDA_POST_KERNEL_CHECK;

		CHECK_EQ(class_nums, top[1]->channels());
		CHECK_EQ(1, this->prefetch_current_->label_.channels());
		int nums = top[1]->count();
		int img_size = top[1]->height() * top[1]->width();
		Dtype* top1_data = top[1]->mutable_gpu_data();
		kernel_channel_eval<Dtype> << <CAFFE_GET_BLOCKS(nums),
			CAFFE_CUDA_NUM_THREADS >> >(nums, class_nums, img_size, mask_data, top1_data);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FORWARD(FCNImageDataLayer);

}  // namespace caffe
