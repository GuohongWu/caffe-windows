#include <vector>

#include "caffe/layers/tv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void TVLossLayer<Dtype>::create_mask_cpu(const vector<int>& shape,
		const int count, const int H, const int W) {
		mask_.Reshape(shape);
		Dtype* data = mask_.mutable_cpu_data();
		const int size = H*W;
		for (int i = 0; i < count; ++i) {
			const int unit_pos = i % size;
			if (unit_pos % W == W - 1 || unit_pos / W == H - 1) {
				data[i] = (Dtype)0;
			}
			else {
				data[i] = (Dtype)1;
			}
		}
	}

	template <typename Dtype>
	void TVLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		switch (Caffe::mode()) {
		case Caffe::CPU: {
			create_mask_cpu(bottom[0]->shape(), bottom[0]->count(),
				bottom[0]->shape(-2), bottom[0]->shape(-1));
			break;
		}
		case Caffe::GPU: {
#ifndef CPU_ONLY
			mask_.ReshapeLike(*bottom[0]);
			create_mask_gpu(bottom[0]->count(),
				bottom[0]->shape(-2), bottom[0]->shape(-1), mask_.mutable_gpu_data());
#endif
			break;
		}
		default:
			LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}
		grad_norm_.ReshapeLike(*bottom[0]);
		x_diff_.ReshapeLike(*bottom[0]);
		y_diff_.ReshapeLike(*bottom[0]);
		tmp_.ReshapeLike(*bottom[0]);
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void TVLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int W = bottom[0]->shape(-1);
		const int count = bottom[0]->count();
		caffe_sub(count - 1, bottom[0]->cpu_data(), &bottom[0]->cpu_data()[1],
			x_diff_.mutable_cpu_data());
		caffe_mul(count, x_diff_.cpu_data(), mask_.cpu_data(),
			x_diff_.mutable_cpu_data());
		caffe_sub(count - W, bottom[0]->cpu_data(), &bottom[0]->cpu_data()[W],
			y_diff_.mutable_cpu_data());
		caffe_mul(count, y_diff_.cpu_data(), mask_.cpu_data(),
			y_diff_.mutable_cpu_data());
		caffe_mul(count, x_diff_.cpu_data(), x_diff_.cpu_data(),
			grad_norm_.mutable_cpu_data());  // X_diff^2
		caffe_mul(count, y_diff_.cpu_data(), y_diff_.cpu_data(),
			tmp_.mutable_cpu_data());  // Y_diff^2
		caffe_add(count, tmp_.cpu_data(), grad_norm_.cpu_data(),
			grad_norm_.mutable_cpu_data());  // X_diff^2 + Y_diff^2
		caffe_powx(count, grad_norm_.cpu_data(),
			(Dtype)this->layer_param_.tv_loss_param().beta() / 2,
			tmp_.mutable_cpu_data());  // (X_diff^2 + Y_diff^2)^(beta/2)
		top[0]->mutable_cpu_data()[0] = caffe_cpu_asum(count, tmp_.cpu_data());
	}

	template <typename Dtype>
	void TVLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const int W = bottom[0]->shape(-1);
		const int count = bottom[0]->count();
		caffe_powx(count, grad_norm_.cpu_data(),
			(Dtype)this->layer_param_.tv_loss_param().beta() / 2 - 1,
			grad_norm_.mutable_cpu_data());
		caffe_scal(count, (Dtype)this->layer_param_.tv_loss_param().beta() / 2,
			grad_norm_.mutable_cpu_data());
		caffe_mul(count, x_diff_.cpu_data(), grad_norm_.cpu_data(),
			x_diff_.mutable_cpu_data());
		caffe_scal(count, (Dtype)2, x_diff_.mutable_cpu_data());  // dX_diff
		caffe_mul(count, y_diff_.cpu_data(), grad_norm_.cpu_data(),
			y_diff_.mutable_cpu_data());
		caffe_scal(count, (Dtype)2, y_diff_.mutable_cpu_data());  // dY_diff
		caffe_axpy(count, (Dtype)1, x_diff_.cpu_data(),
			bottom[0]->mutable_cpu_diff());
		caffe_axpy(count, (Dtype)1, y_diff_.cpu_data(),
			bottom[0]->mutable_cpu_diff());
		caffe_axpy(count - 1, (Dtype)-1, x_diff_.cpu_data(),
			&bottom[0]->mutable_cpu_diff()[1]);
		caffe_axpy(count - W, (Dtype)-1, y_diff_.cpu_data(),
			&bottom[0]->mutable_cpu_diff()[W]);
		caffe_scal(count, top[0]->cpu_diff()[0], bottom[0]->mutable_cpu_diff());
	}

#ifdef CPU_ONLY
	STUB_GPU(TVLossLayer);
#endif

	INSTANTIATE_CLASS(TVLossLayer);
	REGISTER_LAYER_CLASS(TVLoss);

}  // namespace caffe