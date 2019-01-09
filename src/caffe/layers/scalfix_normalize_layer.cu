#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/scalfix_normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim, Dtype epsilon,
                                const Dtype* data, Dtype* norm_data) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    norm_data[index] = sum + epsilon;
  }
}

template <typename Dtype>
__global__ void kernel_channel_scale(const int num, const int channels, const int spatial_dim,
                                     Dtype alpha, const Dtype* data, const Dtype* norm_data,
                                     Dtype beta, Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    output_data[index] = alpha * data[index] * norm_data[n * spatial_dim + s] + beta * output_data[index];
  }
}

template <typename Dtype>
__global__ void kernel_channel_self_scale(const int num, const int channels, const int spatial_dim,
                                          const Dtype* norm_data, Dtype* input_output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    input_output_data[index] *= norm_data[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int num, const int channels, const int spatial_dim,
                                   Dtype alpha, const Dtype* data, const Dtype* norm_data,
                                   Dtype beta, Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    output_data[index] = alpha * data[index] / norm_data[n * spatial_dim + s] + beta * output_data[index];
  }
}

template <typename Dtype>
__global__ void kernel_channel_self_div(const int num, const int channels, const int spatial_dim,
                                     const Dtype* norm_data, Dtype* input_output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    input_output_data[index] /= norm_data[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
                                   const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
                                   Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
              * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
__global__ void kernel_sign(const int count, const Dtype* input, Dtype* sign_out) {
  CUDA_KERNEL_LOOP(index, count) {
    sign_out[index] = (Dtype(0) < input[index]) - (input[index] < Dtype(0));
  }
}

template <typename Dtype>
void ScalFixNormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* square_data = squared_.mutable_gpu_data();
  Dtype* norm_data = (top.size() == 2) ? top[1]->mutable_gpu_data() : norm_.mutable_gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  if (normalize_type_ == "L2") {
    caffe_gpu_powx(num*channels*spatial_dim, bottom_data, Dtype(2), square_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-12, square_data, norm_data);
    caffe_gpu_powx(num * spatial_dim, norm_data, Dtype(0.5), norm_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_div<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, Dtype(1), bottom_data, norm_data, Dtype(0), top_data);
  }
  else if (normalize_type_ == "L1") {
    caffe_gpu_abs(num*channels*spatial_dim, bottom_data, square_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-6, square_data, norm_data);
    //caffe_gpu_powx(num * spatial_dim, norm_data, Dtype(-1), norm_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_div<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, Dtype(1), bottom_data, norm_data, Dtype(0), top_data);
  }
  else {
    NOT_IMPLEMENTED;
  }
}



template <typename Dtype>
void ScalFixNormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* square_data = squared_.mutable_gpu_data();
  const Dtype* norm_data = (top.size() == 2) ? top[1]->gpu_data() : norm_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* temp_diff = norm_.mutable_gpu_diff();

  int num = top[0]->num();
  int channels = top[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  
  if (propagate_down[0]) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_dot<Dtype> << <CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, top_data, top_diff, temp_diff);

    if (normalize_type_ == "L2") {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
        CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, Dtype(1), top_data, temp_diff, Dtype(0), bottom_diff);
    }
    else if (normalize_type_ == "L1") {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_sign<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
        CAFFE_CUDA_NUM_THREADS >> >(num*channels*spatial_dim, bottom_data, square_data);
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
        CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, Dtype(1), square_data, temp_diff, Dtype(0), bottom_diff);
    }
    else {
      NOT_IMPLEMENTED;
    }

    caffe_gpu_sub(num * channels * spatial_dim, top_diff, bottom_diff, bottom_diff);
    if (fix_gradient_) {
      //// NOLINT_NEXT_LINE(whitespace/operators)
      //kernel_channel_self_scale<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
      //  CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, norm_data, bottom_diff);
    }
    else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_channel_self_div<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
        CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, norm_data, bottom_diff);
    }
    
  }
  
  if (bp_norm_) {
    const Dtype* norm_diff = top[1]->gpu_diff();
    if (normalize_type_ == "L2") {
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
        CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, Dtype(1), top_data, norm_diff, Dtype(1), bottom_diff);
    }
    else if (normalize_type_ == "L1") {
      if (!propagate_down[0]) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        kernel_sign<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
          CAFFE_CUDA_NUM_THREADS >> >(num*channels*spatial_dim, bottom_data, square_data);
      }
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
        CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, Dtype(1), square_data, norm_diff, Dtype(1), bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScalFixNormalizeLayer);


}  // namespace caffe