// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_alignment_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, Dtype* acc_feature_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w + 1, (Dtype)1);
    Dtype roi_height = max(roi_end_h - roi_start_h + 1, (Dtype)1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    Dtype hstart = (static_cast<Dtype>(ph)
                                        * bin_size_h);
    Dtype wstart = (static_cast<Dtype>(pw)
                                        * bin_size_w);
    Dtype hend = (static_cast<Dtype>(ph + 1)
                                     * bin_size_h);
    Dtype wend = (static_cast<Dtype>(pw + 1)
                                     * bin_size_w);

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, (Dtype)0), (Dtype)height);
    hend = min(max(hend + roi_start_h, (Dtype)0), (Dtype)height);
    wstart = min(max(wstart + roi_start_w, (Dtype)0), (Dtype)width);
    wend = min(max(wend + roi_start_w, (Dtype)0), (Dtype)width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);
	// is the roi correct?
	if(is_empty)
	{
		top_data[index] = 0;
		return;
	}

    // here we must compute the 
    Dtype mean_val = 0;
	int cnt = 0;
	// 
    bottom_data += (roi_batch_ind * channels + c) * height * width;
	// set to certain roi
	acc_feature_data += (n*pooled_height*pooled_width*width*height + (ph*pooled_width + pw)*width*height);

	// NOTE compute the cnt at first!!!
	for (Dtype h = hstart; h < hend; h += 1) {
      for (Dtype w = wstart; w < wend; w += 1) {
		  cnt++;
	  }
	}
	Dtype cnt1 = Dtype(1)/Dtype(cnt);
    for (Dtype h = hstart; h < hend; h += 1) {
      for (Dtype w = wstart; w < wend; w += 1) {
		// perform bilinear sampling and store the weights into the bottom matrix, this is used for further computation
		// though this will cost large memory, it can speed up the backward process
		  int x = static_cast<int>(w);
		  int y = static_cast<int>(h);

		  int x1 = min(x+1,width-1);
		  int y1 = min(y+1,height-1);
		  Dtype wx = w - (Dtype)x;
		  Dtype wy = h - (Dtype)y;
		  Dtype wx1 = 1-wx;
		  Dtype wy1 = 1-wy;
		  //compute the sampling value by 
		  mean_val += (wx1*wy1*bottom_data[y*width+x] + wx1*wy*bottom_data[y1*width+x] + 
			  wx*wy1*bottom_data[y*width+x1]+wx*wy*bottom_data[y1*width+x1]);

		  //mean_val += bottom_data[y*width+x];
		  // the sampling value is set at once, for all the channel use the same bilinear sampling weights
		  if(c == 0)
		  {
			  //acc_feature_data[y*width+x] += 1;
			  acc_feature_data[ y*width+x] += (wx1*wy1*cnt1);
			  acc_feature_data[ y1*width+x] += (wx1*wy*cnt1);
			  acc_feature_data[ y*width+x1] += (wx*wy1*cnt1);
			  acc_feature_data[ y1*width+x1] += (wx*wy*cnt1);
		  }
      }
    }
	// store back to the top data
    //top_data[index] = mean_val/Dtype(cnt);
	top_data[index] = mean_val * cnt1;
  }
}

template <typename Dtype>
void ROIAlignmentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // make sure the number of rois must be greater than 0
  CHECK(bottom[1]->num() > 0)<<"the number of roi must be greater than 0";
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(acc_feature.count(),Dtype(0),acc_feature.mutable_gpu_data());
  Dtype* acc_feature_data = acc_feature.mutable_gpu_data();

  // compute the 
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, acc_feature_data);
  CUDA_POST_KERNEL_CHECK;
}

// ROI Alignment layer shown here
template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
      Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
      Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
      Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;

	  // for the roi alignment layer, the pixel need not to be included in the results 
      // use bilinear sampling, the w and h can be one-minus the roi_start_w (floor(roi_start)) and ceil(roi_end)
      //const bool in_roi = (w >= floor(roi_start_w) && w <= ceil(roi_end_w) &&
      //                     h >= floor(roi_start_h) && h <= ceil(roi_end_h));
      //if (!in_roi) {
      //  continue;
      //}

	  // locate the roi offset by the top diff
      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
	  int width_height_ = width * height;
	  int offset1 = roi_n*pooled_height*pooled_width*width_height_;
      const Dtype* offset_top_diff = top_diff + offset;
	  // note that the arg data is of size nroi x (poolw*poolh) x height x width
      const Dtype* offset_argmax_data = argmax_data + offset1;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w + 1, (Dtype)1);
      Dtype roi_height = max(roi_end_h - roi_start_h + 1, (Dtype)1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

	  // consider larger region for correct gradient computation
	  // find the pooling region that contains the current output pixel
      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);
	  // use larger field to ensure the correctness of gradient computation
      phstart = min(max(phstart-1, 0), pooled_height);
      phend = min(max(phend+1, 0), pooled_height);
      pwstart = min(max(pwstart-1, 0), pooled_width);
      pwend = min(max(pwend+1, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
	  //for (int ph = 0; ph < pooled_height; ++ph) {
   //     for (int pw = 0; pw < pooled_width; ++pw) {
			// accmuluate the gradient by searching over all possible pool region that cover the current output pixel
			gradient += (offset_argmax_data[(ph*pooled_width + pw)*width_height_ + (h*width+w)] * offset_top_diff[ph * pooled_width + pw]);
        }
      }
    }

    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIAlignmentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const Dtype* argmax_data = acc_feature.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignmentLayer);

}  // namespace caffe
