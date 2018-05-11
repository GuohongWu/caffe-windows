#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/gram_mat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void GramMatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		this->N_ = bottom[0]->num();
		this->C_ = bottom[0]->channels();
		this->H_ = bottom[0]->height();
		this->W_ = bottom[0]->width();
		CHECK_EQ(this->W_, 1);
	}

	template <typename Dtype>
	void GramMatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Figure out the dimensions
		std::vector<int> output_shape{this->N_, this->C_, this->C_, 1};
		top[0]->Reshape(output_shape);
	}

	template <typename Dtype>
	void GramMatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* bottom_data_t = bottom[0]->cpu_data();

		Dtype scal_term_ = Dtype(this->C_ * this->H_);
		for (int n = 0; n < this->N_; ++n) {
			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				this->C_, this->C_, this->H_, (Dtype)1.0 / scal_term_,
				bottom_data, bottom_data_t,
				(Dtype)0., top_data);

			bottom_data += this->C_ * this->H_;
			bottom_data_t += this->C_ * this->H_;
			top_data += this->C_ * this->C_;
		}		
	}

	template <typename Dtype>
	void GramMatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* bottom_data = bottom[0]->cpu_data();

			Dtype scal_term_ = Dtype(this->C_ * this->H_);
			for (int n = 0; n < this->N_; ++n) {
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
					this->C_, this->H_, this->C_, (Dtype)1.0 / scal_term_,
					top_diff, bottom_data,
					(Dtype)0., bottom_diff);
				caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
					this->C_, this->H_, this->C_, (Dtype)1.0 / scal_term_,
					top_diff, bottom_data,
					(Dtype)1., bottom_diff);

				bottom_data += this->C_ * this->H_;
				bottom_diff += this->C_ * this->H_;
				top_diff += this->C_ * this->C_;
			}		
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(GramMatLayer);
#endif

	INSTANTIATE_CLASS(GramMatLayer);
	REGISTER_LAYER_CLASS(GramMat);

}  // namespace caffe
