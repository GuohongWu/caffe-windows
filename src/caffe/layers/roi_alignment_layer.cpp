// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Congyi Wang
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_alignment_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

	template <typename Dtype>
	void ROIAlignmentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			RoiAlignmentParameter roi_pool_param = this->layer_param_.roi_alignment_param();
			CHECK_GT(roi_pool_param.pooled_h(), 0)
				<< "pooled_h must be > 0";
			CHECK_GT(roi_pool_param.pooled_w(), 0)
				<< "pooled_w must be > 0";
			pooled_height_ = roi_pool_param.pooled_h();
			pooled_width_ = roi_pool_param.pooled_w();
			spatial_scale_ = roi_pool_param.spatial_scale();
			LOG(INFO) << "Spatial scale: " << spatial_scale_;
	}

	template <typename Dtype>
	void ROIAlignmentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			channels_ = bottom[0]->channels();
			height_ = bottom[0]->height();
			width_ = bottom[0]->width();
			top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
				pooled_width_);
		    // store the bilinear sample weights in the original feature map size
			acc_feature.Reshape(bottom[1]->num(), pooled_height_*pooled_width_, height_,
				width_);
			// width*height
			width_height_ = width_ * height_;
	}

	template <typename Dtype>
	void ROIAlignmentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			const Dtype* bottom_rois = bottom[1]->cpu_data();
			// Number of ROIs
			int num_rois = bottom[1]->num();
			int batch_size = bottom[0]->num();
			int top_count = top[0]->count();
			Dtype* top_data = top[0]->mutable_cpu_data();
			//caffe_set(top_count, Dtype(-FLT_MAX), top_data);

			// first set all acc feature as zero
			Dtype* acc_feature_data = acc_feature.mutable_cpu_data();
			caffe_set(acc_feature.count(), Dtype(0), acc_feature_data);


			// For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
			for (int n = 0; n < num_rois; ++n) {
				int roi_batch_ind = bottom_rois[0];

				Dtype roi_start_w = bottom_rois[1] * spatial_scale_;
				Dtype roi_start_h = bottom_rois[2] * spatial_scale_;
				Dtype roi_end_w = bottom_rois[3] * spatial_scale_;
				Dtype roi_end_h = bottom_rois[4] * spatial_scale_;

				CHECK_GE(roi_batch_ind, 0);
				CHECK_LT(roi_batch_ind, batch_size);

				Dtype roi_height = max(roi_end_h - roi_start_h + 1, (Dtype)1);
				Dtype roi_width = max(roi_end_w - roi_start_w + 1, (Dtype)1);
				const Dtype bin_size_h = (roi_height)
					/ static_cast<Dtype>(pooled_height_);
				const Dtype bin_size_w = (roi_width)
					/ static_cast<Dtype>(pooled_width_);

				const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

				for (int c = 0; c < channels_; ++c) {
					for (int ph = 0; ph < pooled_height_; ++ph) {
						for (int pw = 0; pw < pooled_width_; ++pw) {
							// Compute pooling region for this output unit:
							//  start (included) = floor(ph * roi_height / pooled_height_)
							//  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
							Dtype hstart = static_cast<Dtype>(ph)	* bin_size_h;
							Dtype wstart = static_cast<Dtype>(pw)	* bin_size_w;
							Dtype hend = static_cast<Dtype>(ph + 1)* bin_size_h;
							Dtype wend = static_cast<Dtype>(pw + 1)* bin_size_w;

							hstart = min(max(hstart + roi_start_h, (Dtype)0), (Dtype)height_);
							hend = min(max(hend + roi_start_h, (Dtype)0), (Dtype)height_);
							wstart = min(max(wstart + roi_start_w, (Dtype)0), (Dtype)width_);
							wend = min(max(wend + roi_start_w, (Dtype)0), (Dtype)width_);

							bool is_empty = (hend <= hstart) || (wend <= wstart);

							const int pool_index = ph * pooled_width_ + pw;
							if (is_empty) {
								top_data[pool_index] = 0;
								continue;
							}
							// get the bilinear sampling weights
							int cnt = 0;
							Dtype total_val = Dtype(0);
							for (Dtype h = hstart; h < hend; h += 1) {
								for (Dtype w = wstart; w < wend; w += 1) {
									// count the bilinear sampling weights using x,y,x+1,y+1,
									int x = static_cast<int>(w);
									int y = static_cast<int>(h);
									int x1 = min(x+1,width_-1);
									int y1 = min(y+1,height_-1);
									Dtype wx = w - (Dtype)x;
									Dtype wy = h - (Dtype)y;
									Dtype wx1 = 1-wx;
									Dtype wy1 = 1-wy;
									// compute the sampling value by 
									total_val += (wx1*wy1*batch_data[y*width_+x] + wx1*wy*batch_data[y1*width_+x] + 
										wx*wy1*batch_data[y*width_+x1]+wx*wy*batch_data[y1*width_+x1]);

									// the sampling value is set at once, for all the channel use the same bilinear sampling weights
									if(c == 0)
									{
										acc_feature_data[pool_index*width_height_ + y*width_+x] += wx1*wy1;
										acc_feature_data[pool_index*width_height_ + y1*width_+x] += wx1*wy;
										acc_feature_data[pool_index*width_height_ + y*width_+x1] += wx*wy1;
										acc_feature_data[pool_index*width_height_ + y1*width_+x1] += wx*wy;
									}
									// count the sample weights
									cnt++;
								}
							}
							// compute the final average value and fill in the sampling weights 
							top_data[pool_index] = total_val/Dtype(cnt);
							if(c == 0)
							{
								// divide by cnt is needed at once
								Dtype cnt_1 = Dtype(1)/Dtype(cnt);
								caffe_scal(width_height_,cnt_1,acc_feature_data+pool_index*width_height_);
							}
							
						}
					}
					// Increment all data pointers by one channel
					batch_data += bottom[0]->offset(0, 1);
					top_data += top[0]->offset(0, 1);
				}
				// Increment ROI data pointer
				bottom_rois += bottom[1]->offset(1);
				acc_feature_data += acc_feature.offset(1);
			}
	}

	// very slow speed, we must try to compute the small region contained in the target feature map size
	template <typename Dtype>
	void ROIAlignmentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			//NOT_IMPLEMENTED;
			if (!propagate_down[0]) {
				return;
			}
			 //given the top diff, compute the bottom diff using the top diff
			const Dtype* bottom_rois = bottom[1]->cpu_data();
			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* acc_feature_data = acc_feature.cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const int count = bottom[0]->count();
			caffe_gpu_set(count, Dtype(0.), bottom_diff);

			int pool_wh = pooled_height_ * pooled_width_;
			for(int n = 0;n < bottom[1]->num();++n)
			{
				// find the batchid and batch diff
				int roi_batch_ind = bottom_rois[n*5];

				Dtype* batch_data = bottom_diff + bottom[0]->offset(roi_batch_ind);
				// multiply the 
				for(int c = 0;c < channels_;++c)
				{
					Dtype* bdata = batch_data + c*width_height_;

					for(int h = 0;h < pooled_height_;++h)
						for(int w = 0;w < pooled_width_;++w)
						{
							int pool_ind = h * pooled_width_ + w;
							// add the diff into the bottom diff
							for(int i = 0;i < height_;++i)
								for(int j = 0;j < width_;++j)
								{
									bdata[i*width_ + j] += (top_diff[c*pool_wh + pool_ind]*acc_feature_data[pool_ind*pool_wh + i*width_ + j]);				
								}
						}
				}
				// foward one
				top_diff += top[0]->offset(1);
				acc_feature_data += acc_feature.offset(1);
			}
	}


#ifdef CPU_ONLY
	STUB_GPU(ROIAlignmentLayer);
#endif

	INSTANTIATE_CLASS(ROIAlignmentLayer);
	REGISTER_LAYER_CLASS(ROIAlignment);

}  // namespace caffe
