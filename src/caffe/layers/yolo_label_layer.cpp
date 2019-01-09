#include "caffe/layers/yolo_label_layer.hpp"


namespace caffe {

	template <typename Dtype>
	void YOLOLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		YOLOLabelParameter yolo_label_param = this->layer_param_.yolo_label_param();
		this->grid_h = yolo_label_param.grid_h();
		this->grid_w = yolo_label_param.grid_w();
		CHECK_GT(this->grid_h, 0);
		CHECK_LT(this->grid_h, bottom[0]->height());
		CHECK_GT(this->grid_w, 0);
		CHECK_LT(this->grid_w, bottom[0]->width());
	}

	template <typename Dtype>
	void YOLOLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[1]->count(1), 1);
		CHECK_EQ(bottom[2]->count(1), 5);
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());

		int batch_size = bottom[0]->num();
		top[0]->Reshape(batch_size, 5, this->grid_h, this->grid_w);
	}

	template <typename Dtype>
	void YOLOLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype* bbox_start_loc = bottom[1]->cpu_data();
		const Dtype* bbox_data = bottom[2]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		caffe_set(top[0]->count(), Dtype(0), top_data);

		int top_data_dim_ = top[0]->count(1);

		int batch_size = bottom[0]->num();
		Dtype img_height = bottom[0]->height();
		Dtype img_width = bottom[0]->width();

		Dtype scal_h = Dtype(this->grid_h) / img_height;
		Dtype scal_w = Dtype(this->grid_w) / img_width;
		for (int n = 0; n < batch_size; ++n) {
			int cur_bbox_start_loc_ = bbox_start_loc[n];
			int cur_bbox_end_loc_ = ((n + 1 == batch_size) ? bottom[2]->num() : bbox_start_loc[n + 1]);
			for (int k = cur_bbox_start_loc_; k < cur_bbox_end_loc_; ++k) {
				const Dtype* cur_bbox_data = bbox_data + k * 5;
				Dtype cen_x = Dtype(0.5) * (cur_bbox_data[0] + cur_bbox_data[2]) * scal_w;
				Dtype cen_y = Dtype(0.5) * (cur_bbox_data[1] + cur_bbox_data[3]) * scal_h;
				Dtype w_ = (cur_bbox_data[2] - cur_bbox_data[0]) * scal_w;
				Dtype h_ = (cur_bbox_data[3] - cur_bbox_data[1]) * scal_h;

				int grid_w_idx = std::max(0, std::min(this->grid_w - 1, int(cen_x)));
				int grid_h_idx = std::max(0, std::min(this->grid_h - 1, int(cen_y)));
				int cls_idx = top[0]->offset(n, 0, grid_h_idx, grid_w_idx);
				top_data[cls_idx] = cur_bbox_data[4];
				int cen_x_idx = top[0]->offset(n, 1, grid_h_idx, grid_w_idx);
				top_data[cen_x_idx] = cen_x;
				int cen_y_idx = top[0]->offset(n, 2, grid_h_idx, grid_w_idx);
				top_data[cen_y_idx] = cen_y;
				int w_idx = top[0]->offset(n, 3, grid_h_idx, grid_w_idx);
				top_data[w_idx] = w_;
				int h_idx = top[0]->offset(n, 4, grid_h_idx, grid_w_idx);
				top_data[h_idx] = h_;
			}
		}
	}

	template <typename Dtype>
	void YOLOLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}

#ifdef CPU_ONLY
	STUB_GPU(YOLOLabelLayer);
#endif

	INSTANTIATE_CLASS(YOLOLabelLayer);
	REGISTER_LAYER_CLASS(YOLOLabel);
}