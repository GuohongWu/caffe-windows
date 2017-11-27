#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/my_smooth_L1_loss_layer.hpp"


// add the label colum here for training the smooth L1 loss with label
namespace caffe {

template <typename Dtype>
__global__ void SmoothL1Forward(const int n,const int inlier_num, const int channel_num,const int width_height_,
								const Dtype* in, const Dtype* in_indices,Dtype* out,Dtype* out_indices,
    Dtype sigma2) {
  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
  //        |x| - 0.5 / sigma / sigma    otherwise
  CUDA_KERNEL_LOOP(index, n) {

	// compute the channel indices and grab the corresponding channel  value
	// batch no
	int a = index / inlier_num;
	// channel indices
	int b = (index / width_height_) % channel_num;
	// width and height indices here
	int c = index % width_height_;
	// compute the input indices n x 1 x h x w
	int in_ind = a * width_height_ + c;
	
	int valid = in_indices[in_ind] > Dtype(0.1) ? 1 : 0;
	// assign the value in the first channel indices
	if(b == 0)
		out_indices[in_ind] = Dtype(valid);

	if(valid)
	{
		Dtype val = in[index];
		Dtype abs_val = abs(val);
		if (abs_val < 1.0 / sigma2) {
		  out[index] = 0.5 * val * val * sigma2;
		} else {
		  out[index] = abs_val - 0.5 / sigma2;
		}
	}
	else
	{
		out[index] = 0;
	}

  }
}

template <typename Dtype>
void MySmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1

  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, inlier_num_,channels_,width_height_,diff_.gpu_data(), bottom[2]->gpu_data(),errors_.mutable_gpu_data(),ones_.mutable_gpu_data(),sigma2_);
  CUDA_POST_KERNEL_CHECK;


  caffe_gpu_asum(ones_.count(),ones_.gpu_data(),&non_zero_);
  Dtype loss;
  if(non_zero_ == Dtype(0))
	  loss = Dtype(0);
  else
	{
		  caffe_gpu_asum(count, errors_.gpu_data(), &loss);
		//  const Dtype* ones_data = ones_.cpu_data();
		 // const Dtype* b1_data = bottom[0]->cpu_data();
		 // const Dtype* b2_data = bottom[1]->cpu_data();
		  //for(int i = 0;i < count;++i)
		  //{
			//if(ones_data[i])
			//{
			//	LOG(INFO)<<"pred is "<<b1_data[i];
			//	LOG(INFO)<<"target is "<<b2_data[i];
			//}			
		  //}
		  //LOG(INFO)<<"loss is "<<loss;
		  //LOG(INFO)<<"non zero is "<<non_zero_;
		  loss /= non_zero_;
		  
	}
  //top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const int inlier_num,const int width_height_,
								 const Dtype* in, const Dtype* in_indices,Dtype* out,
    Dtype sigma2) {
  // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
  //       = sign(x)                   otherwise
  CUDA_KERNEL_LOOP(index, n) {

	int a = index / inlier_num;
	// width and height indices here
	int c = index % width_height_;
	// compute the input indices n x 1 x h x w
	int in_ind = a * width_height_ + c;
	
	int valid = in_indices[in_ind] > Dtype(0.1) ? 1 : 0;
	if(valid)
	{
		Dtype val = in[index];
		Dtype abs_val = abs(val);
		if (abs_val < 1.0 / sigma2) {
		  out[index] = sigma2 * val;
		} else {
		  out[index] = (Dtype(0) < val) - (val < Dtype(0));
		}
	}
	else
		out[index] = 0;

  }
}

template <typename Dtype>
void MySmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // after forwards, diff_ holds w_in * (b0 - b1)
  int count = diff_.count();
  SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, inlier_num_,width_height_,diff_.gpu_data(), bottom[2]->gpu_data(),diff_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
	  Dtype alpha = Dtype(0);
	  if(non_zero_)
		 //alpha = sign * top[0]->cpu_diff()[0] / (non_zero_ * bottom[i]->num());
		 alpha = sign * top[0]->cpu_diff()[0] / (non_zero_);

      caffe_gpu_axpby(
          count,                           // count
          alpha,                           // alpha
          diff_.gpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // y
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MySmoothL1LossLayer);

}