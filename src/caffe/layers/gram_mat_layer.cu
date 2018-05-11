#include <vector>

#include "caffe/layers/gram_mat_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

	template <typename Dtype>
	void GramMatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data_t = bottom[0]->gpu_data();

		Dtype scal_term_ = Dtype(this->C_ * this->H_);
		for (int index = 0; index < this->N_; ++index) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				C_, C_, H_, (Dtype)1.0 / scal_term_,
				bottom_data + index * C_ * H_, bottom_data_t + index * C_ * H_,
				(Dtype)0., top_data + index * C_ * C_);
		}
	}

	template <typename Dtype>
	void GramMatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const Dtype* bottom_data = bottom[0]->gpu_data();

			Dtype scal_term_ = Dtype(this->C_ * this->H_);
			for (int index = 0; index < this->N_; ++index) {
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
					C_, H_, C_, (Dtype)1.0 / scal_term_,
					top_diff + index * C_ * C_, bottom_data + index * C_ * H_,
					(Dtype)0., bottom_diff + index * C_ * H_);
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
					C_, H_, C_, (Dtype)1.0 / scal_term_,
					top_diff + index * C_ * C_, bottom_data + index * C_ * H_,
					(Dtype)1., bottom_diff + index * C_ * H_);
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(GramMatLayer);

}  // namespace caffe
