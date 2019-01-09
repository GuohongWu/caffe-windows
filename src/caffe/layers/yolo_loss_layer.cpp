#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

	template <typename Dtype>
	Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
		Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
		Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
		return right - left;
	}

	// format: cen_x, cen_y, width, height
	template <typename Dtype>
	Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
		Dtype w = Overlap(box[0], box[2], truth[0], truth[2]);
		Dtype h = Overlap(box[1], box[3], truth[1], truth[3]);
		if (w < 0 || h < 0) return 0;
		Dtype inter_area = w * h;
		Dtype union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
		return inter_area / union_area;
	}

	template <typename Dtype>
	void YOLOLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		YOLOLossParameter yolo_loss_param = this->layer_param_.yolo_loss_param();
		this->cls_nums = yolo_loss_param.cls_nums();
		this->bbox_nums = yolo_loss_param.bbox_nums();
		this->lambda_obj = yolo_loss_param.lambda_obj();
		this->lambda_noobj = yolo_loss_param.lambda_noobj();
		this->lambda_coord = yolo_loss_param.lambda_coord();

		this->grid_h = bottom[1]->height();
		this->grid_w = bottom[1]->width();

		CHECK_GT(this->cls_nums, 0);
		CHECK_GT(this->bbox_nums, 0);
		CHECK_EQ(this->grid_h, bottom[0]->height());
		CHECK_EQ(this->grid_w, bottom[0]->width());
		CHECK_EQ(5 * this->bbox_nums + this->cls_nums, bottom[0]->channels());
	}

	template <typename Dtype>
	void YOLOLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);

		this->bottom0_diff_.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void YOLOLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* reg_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		Dtype* diff_data = this->bottom0_diff_.mutable_cpu_data();
		caffe_set(this->bottom0_diff_.count(), Dtype(0), diff_data);

		Dtype loss(0), coord_loss(0), obj_loss(0), noobj_loss(0), cls_loss(0);
		for (int n = 0; n < bottom[0]->num(); ++n) {
			for (int h = 0; h < this->grid_h; ++h) {
				for (int w = 0; w < this->grid_w; ++w) {
					int gt_cls_idx = bottom[1]->offset(n, 0, h, w);
					if (label_data[gt_cls_idx] < 1 || label_data[gt_cls_idx] > this->cls_nums) // this grid cell holds no object.
					{
						for (int c = 0; c < this->bbox_nums; ++c) {
							int bbox_confidence_idx = bottom[0]->offset(n, 4 + 5 * c, h, w);
							diff_data[bbox_confidence_idx] = this->lambda_noobj * reg_data[bbox_confidence_idx];
							noobj_loss += diff_data[bbox_confidence_idx] * reg_data[bbox_confidence_idx];
						}
						continue;
					}
					else
					{
						// compute cls_loss
						for (int cls_idx = 0; cls_idx < this->cls_nums; ++cls_idx) {
							int cls_offset = bottom[0]->offset(n, 5 * this->bbox_nums + cls_idx, h, w);
							diff_data[cls_offset] = reg_data[cls_offset] - (label_data[gt_cls_idx] == (cls_idx + 1));
							cls_loss += diff_data[cls_offset] * diff_data[cls_offset];
						}

						// compute obj_loss and coord_loss
						Dtype tar_iou(0);
						int tar_idx = 0;
						vector<Dtype> gt_bbox(4), pred_bbox(4);
						int cen_x_idx = bottom[1]->offset(n, 1, h, w);
						gt_bbox[0] = label_data[cen_x_idx];
						int cen_y_idx = bottom[1]->offset(n, 2, h, w);
						gt_bbox[1] = label_data[cen_y_idx];
						int w_idx = bottom[1]->offset(n, 3, h, w);
						gt_bbox[2] = label_data[w_idx];
						int h_idx = bottom[1]->offset(n, 4, h, w);
						gt_bbox[3] = label_data[h_idx];

						for (int c = 0; c < this->bbox_nums; ++c) {
							cen_x_idx = bottom[0]->offset(n, 0 + 5 * c, h, w);
							pred_bbox[0] = reg_data[cen_x_idx] + Dtype(w);
							cen_y_idx = bottom[0]->offset(n, 1 + 5 * c, h, w);
							pred_bbox[1] = reg_data[cen_y_idx] + Dtype(h);
							w_idx = bottom[0]->offset(n, 2 + 5 * c, h, w);
							pred_bbox[2] = reg_data[w_idx] * reg_data[w_idx] * Dtype(this->grid_w);
							h_idx = bottom[0]->offset(n, 3 + 5 * c, h, w);
							pred_bbox[3] = reg_data[h_idx] * reg_data[h_idx] * Dtype(this->grid_h);

							Dtype cur_iou = Calc_iou(pred_bbox, gt_bbox);
							if (cur_iou > tar_iou) {
								tar_iou = cur_iou;
								tar_idx = c;
							}
						}

						// obj_loss
						int bbox_confidence_idx = bottom[0]->offset(n, 4 + 5 * tar_idx, h, w);
						diff_data[bbox_confidence_idx] = this->lambda_obj * (reg_data[bbox_confidence_idx] - tar_iou);
						obj_loss += diff_data[bbox_confidence_idx] * (reg_data[bbox_confidence_idx] - tar_iou);

						// coord_loss
						gt_bbox[0] -= Dtype(w);
						gt_bbox[1] -= Dtype(h);
						gt_bbox[2] = sqrt(gt_bbox[2] / Dtype(this->grid_w));
						gt_bbox[3] = sqrt(gt_bbox[3] / Dtype(this->grid_h));
						cen_x_idx = bottom[0]->offset(n, 0 + 5 * tar_idx, h, w);
						cen_y_idx = bottom[0]->offset(n, 1 + 5 * tar_idx, h, w);
						w_idx = bottom[0]->offset(n, 2 + 5 * tar_idx, h, w);
						h_idx = bottom[0]->offset(n, 3 + 5 * tar_idx, h, w);
						coord_loss += this->lambda_coord * (pow(reg_data[cen_x_idx] - gt_bbox[0], 2) + pow(reg_data[cen_y_idx] - gt_bbox[1], 2) + 
																		pow(reg_data[w_idx] - gt_bbox[2], 2) + pow(reg_data[h_idx] - gt_bbox[3], 2));
						diff_data[cen_x_idx] = this->lambda_coord * (reg_data[cen_x_idx] - gt_bbox[0]);
						diff_data[cen_y_idx] = this->lambda_coord * (reg_data[cen_y_idx] - gt_bbox[1]);
						diff_data[w_idx] = this->lambda_coord * (reg_data[w_idx] - gt_bbox[2]);
						diff_data[h_idx] = this->lambda_coord * (reg_data[h_idx] - gt_bbox[3]);
					}
				}
			}
		}

		loss = coord_loss + obj_loss + noobj_loss + cls_loss;
		loss /= Dtype(2 * bottom[0]->num());
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void YOLOLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]) {
			const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
			caffe_cpu_axpby(
				bottom[0]->count(),
				alpha,
				this->bottom0_diff_.cpu_data(),
				Dtype(0),
				bottom[0]->mutable_cpu_diff());
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(YOLOLossLayer);
#endif

	INSTANTIATE_CLASS(YOLOLossLayer);
	REGISTER_LAYER_CLASS(YOLOLoss);

}