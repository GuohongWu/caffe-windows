#include "caffe/layers/attention_map_layer.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {

	template <typename Dtype>
	void BboxAttentionMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const BboxAttentionMapParameter &param_ = this->layer_param_.bbox_attention_map_param();
		this->img_width_ = param_.img_width();
		this->img_height_ = param_.img_height();
		this->map_width_ = param_.map_width();
		this->map_height_ = param_.map_height();
	}

	template <typename Dtype>
	void BboxAttentionMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_GT(this->map_width_, 0);
		CHECK_GT(this->map_height_, 0);
		CHECK_GT(this->img_width_, 0);
		CHECK_GT(this->img_height_, 0);

		CHECK_EQ(bottom[0]->channels(), 4);
		CHECK_EQ(bottom[0]->width(), 1);
		CHECK_EQ(bottom[0]->height(), 1);

		top[0]->Reshape(bottom[0]->num(), 1, this->map_height_, this->map_width_);
	}

	template <typename Dtype>
	void BboxAttentionMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bbox_data = bottom[0]->cpu_data();
		Dtype* map_data = top[0]->mutable_cpu_data();
		int map_area = this->map_width_ * this->map_height_;

		cv::Rect org_img_rect(0, 0, this->img_width_, this->img_height_);
		cv::Mat org_att_map(this->img_height_, this->img_width_, CV_32FC1);
		cv::Mat out_att_map(this->map_height_, this->map_width_, CV_32FC1);
		for (int k = 0; k < bottom[0]->num(); ++k) {
			cv::Rect cur_bbox(bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]);
			cur_bbox &= org_img_rect;

			org_att_map = 0;
			if (cur_bbox.area() > 0) {
				org_att_map(cur_bbox) = 1;
			}
			cv::resize(org_att_map, out_att_map, out_att_map.size(), 0, 0, CV_INTER_AREA);
			for (int i = 0; i < map_area; ++i)
				*map_data++ = Dtype(out_att_map.at<float>(i));

			bbox_data += 4;
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(BboxAttentionMapLayer);
#endif

	INSTANTIATE_CLASS(BboxAttentionMapLayer);
	REGISTER_LAYER_CLASS(BboxAttentionMap);

}