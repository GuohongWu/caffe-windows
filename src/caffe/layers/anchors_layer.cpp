#include "caffe/layers/anchors_layer.hpp"
#include "caffe/util/nms.hpp"
#include <algorithm>
#include <tuple>

#define ROUND(x) ((int)((x) + (Dtype)0.5))

using std::max;
using std::min;

namespace caffe {

	template <typename Dtype>
	static void generate_anchors(int base_size,
		const Dtype ratios[],
		const Dtype scales[],
		const int num_ratios,
		const int num_scales,
		Dtype anchors[])
	{
		// base box's width & height & center location
		const Dtype base_area = (Dtype)(base_size * base_size);
		const Dtype center = (Dtype)0.5 * (base_size - (Dtype)1);

		// enumerate all transformed boxes
		Dtype *p_anchors = anchors;
		for (int i = 0; i < num_ratios; ++i)
		{
			// transformed width & height for given ratio factors
			const Dtype ratio_w = (Dtype)(sqrt(base_area / ratios[i]));
			const Dtype ratio_h = (Dtype)(ratio_w * ratios[i]);

			for (int j = 0; j < num_scales; ++j)
			{
				// transformed width & height for given scale factors
				const Dtype scale_w = (Dtype)0.5 * ROUND(ratio_w * scales[j] * 1.0 - (Dtype)1);
				const Dtype scale_h = (Dtype)0.5 * ROUND(ratio_h * scales[j] * 1.0 - (Dtype)1);

				// (x1, y1, x2, y2) for transformed box
				p_anchors[0] = center - scale_w;
				p_anchors[1] = center - scale_h;
				p_anchors[2] = center + scale_w;
				p_anchors[3] = center + scale_h;
				p_anchors += 4;
			} // endfor j
		}
	}

	template <typename Dtype>
	void AnchorsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top)
	{
		AnchorsParameter param = this->layer_param_.anchors_param();
		anchor_stride_ = param.anchor_stride();
		base_size_ = param.base_size();
		nms_upper_thresh_ = param.nms_upper_thresh();
		nms_lower_thresh_ = param.nms_lower_thresh();

		vector<Dtype> ratios(param.ratio_size());
		for (int i = 0; i < param.ratio_size(); ++i)
		{
			ratios[i] = param.ratio(i);
		}
		vector<Dtype> scales(param.scale_size());
		for (int i = 0; i < param.scale_size(); ++i)
		{
			scales[i] = param.scale(i);
		}

		vector<int> anchors_shape(2);
		anchors_shape[0] = ratios.size() * scales.size();
		anchors_shape[1] = 4;
		anchors_0_0_.Reshape(anchors_shape);
		caffe::generate_anchors(base_size_, &ratios[0], &scales[0],
			ratios.size(), scales.size(),
			anchors_0_0_.mutable_cpu_data());
	}

	template <typename Dtype>
	void AnchorsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batch_size = bottom[0]->num();
		int anchor_map_H = ceilf(float(bottom[0]->height()) / this->anchor_stride_);
		int anchor_map_W = ceilf(float(bottom[0]->width()) / this->anchor_stride_);
		int anchor_nums = anchors_0_0_.num();

		top[0]->Reshape(batch_size, anchor_nums, anchor_map_H, anchor_map_W); // label_map
		top[1]->Reshape(batch_size, 4 * anchor_nums, anchor_map_H, anchor_map_W); // reg_map
		if(this->layer_param_.top_size() >= 3) {
			top[2]->Reshape(1, 4, 1, 1); // img_info
			Dtype* top2_data = top[2]->mutable_cpu_data();
			top2_data[0] = bottom[0]->height();
			top2_data[1] = bottom[0]->width();
			top2_data[2] = 1;
			top2_data[3] = 1;
		}

		// top[3] output max-IoU corre to each anchors. (For debug)
		if (this->layer_param_.top_size() >= 4) {
			top[3]->Reshape(batch_size, 5 * anchor_nums, anchor_map_H, anchor_map_W);
		}
	}

	template <typename Dtype>
	void AnchorsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {
		int nums = top[0]->num();
		int anchor_nums = top[0]->channels();
		int height = top[0]->height();
		int width = top[0]->width();
		int feature_map_area = height * width;
		int anchor_counts = anchor_nums * feature_map_area;
		Dtype* top0_data = top[0]->mutable_cpu_data();
		Dtype* top1_data = top[1]->mutable_cpu_data();
		caffe_set(top[0]->count(), Dtype(-1), top0_data);

		const Dtype* bbox_start_loc = bottom[1]->cpu_data();
		const Dtype* bbox_data = bottom[2]->cpu_data();
		int bbox_data_dim_ = bottom[2]->count(1);
		const Dtype* anchors_data_ = this->anchors_0_0_.cpu_data();
		Dtype curAnc[4];

		for (int n = 0; n < nums; ++n) {
			int bbox_end_loc_ = ((n + 1 == nums) ? bottom[2]->num() : bbox_start_loc[n + 1]);
			vector<bool> bbox_bind_to_noAnc(bbox_end_loc_ - bbox_start_loc[n], false);

			// 1.assign each anchor to the unique face with highest IoU. Meanwhile, this IoU must be larger than a threshold (e.g. nms_upper_thresh_)
			int offset_0_first = -1 + n * anchor_counts;
			for (int c = 0; c < anchor_nums; ++c) {
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						++offset_0_first;
						
						std::copy(anchors_data_ + 4 * c, anchors_data_ + 4 * (c + 1), curAnc);
						curAnc[0] += w * this->anchor_stride_;
						curAnc[1] += h * this->anchor_stride_;
						curAnc[2] += w * this->anchor_stride_;
						curAnc[3] += h * this->anchor_stride_;
						if (curAnc[0] < 0 || curAnc[2] > bottom[0]->width() || curAnc[1] < 0 || curAnc[3] > bottom[0]->height()) {  // cross-boundary anchors are ignored. 
							*(top0_data + offset_0_first) = -2;   // Temporarily assgin to -2, to distinguish.
							continue;
						}						

						Dtype max_iou = 0;
						int max_bbox_idx = -1;
						for (int k = bbox_start_loc[n]; k < bbox_end_loc_; ++k) {
							Dtype cur_iou = IoU_cpu(bbox_data + k * bbox_data_dim_, curAnc);
							if (cur_iou > max_iou) {
								max_iou = cur_iou;
								max_bbox_idx = k;
							}
							else if (cur_iou == max_iou && max_iou > 0) {
								if(caffe::caffe_rng_rand() % 2)
									max_bbox_idx = k;
							}
						}
						if (max_iou > this->nms_upper_thresh_) {
							*(top0_data + offset_0_first) = max_bbox_idx + 1;
							bbox_bind_to_noAnc[max_bbox_idx - bbox_start_loc[n]] = true;
							for (int pos = 0; pos < 4; ++pos) {
								int offset_1 = top[1]->offset(n, pos * anchor_nums + c, h, w);
								*(top1_data + offset_1) = curAnc[pos];
							}
						}
						else if (max_iou < this->nms_lower_thresh_) {
							*(top0_data + offset_0_first) = 0; // sure to be background!
						}
					}
				}
			}

			// 2. For other Bboxes and other Anchors, Assign each bboxes to the anchor(s) with highest Intersection - overUnion(IoU) overlap 
			vector<std::tuple<Dtype, int, vector<int> > > bbox_corres_anchors;
			for (int k = bbox_start_loc[n]; k < bbox_end_loc_; ++k) {
				if (bbox_bind_to_noAnc[k - bbox_start_loc[n]])
					continue;

				bbox_corres_anchors.emplace_back(0, k + 1, vector<int>());
				vector<int>& max_offset_idn = std::get<2>(bbox_corres_anchors.back());
				Dtype& max_iou = std::get<0>(bbox_corres_anchors.back());
				//vector<int> max_offset_idn;
				int offset_0_second = -1 + n * anchor_counts;
				for (int c = 0; c < anchor_nums; ++c) {
					for (int h = 0; h < height; ++h) {
						for (int w = 0; w < width; ++w) {
							int curBbox_idx = *(top0_data + (++offset_0_second));
							if (curBbox_idx > 0 || (curBbox_idx == 0 && max_iou >= this->nms_lower_thresh_) || curBbox_idx == -2)
								continue;

							std::copy(anchors_data_ + 4 * c, anchors_data_ + 4 * (c + 1), curAnc);
							curAnc[0] += w * this->anchor_stride_;
							curAnc[1] += h * this->anchor_stride_;
							curAnc[2] += w * this->anchor_stride_;
							curAnc[3] += h * this->anchor_stride_;

							Dtype cur_iou = IoU_cpu(bbox_data + k * bbox_data_dim_, curAnc);
							if (cur_iou > max_iou) {
								max_iou = cur_iou;
								max_offset_idn.clear();
								max_offset_idn.push_back(offset_0_second);
							}
							else if (cur_iou == max_iou && max_iou > 0) {
								max_offset_idn.push_back(offset_0_second);
							}
						}
					}
				}
			}
			std::sort(bbox_corres_anchors.begin(), bbox_corres_anchors.end(), [](const std::tuple<Dtype, int, vector<int> >& A, const std::tuple<Dtype, int, vector<int> >& B) -> bool { return std::get<0>(A) > std::get<0>(B); });
			for (auto &curBbox_corres : bbox_corres_anchors) {
				for (auto &cur_offset0 : std::get<2>(curBbox_corres)) {
					if (*(top0_data + cur_offset0) > 0)
						continue;

					*(top0_data + cur_offset0) = std::get<1>(curBbox_corres);

					int chan_ = cur_offset0 % anchor_counts / feature_map_area;
					int w_ = cur_offset0 % feature_map_area % width;
					int h_ = cur_offset0 % feature_map_area / width;
					std::copy(anchors_data_ + 4 * chan_, anchors_data_ + 4 * (chan_ + 1), curAnc);
					curAnc[0] += w_ * this->anchor_stride_;
					curAnc[1] += h_ * this->anchor_stride_;
					curAnc[2] += w_ * this->anchor_stride_;
					curAnc[3] += h_ * this->anchor_stride_;
					for (int pos = 0; pos < 4; ++pos) {
						int offset_1 = top[1]->offset(n, pos * anchor_nums + chan_, h_, w_);
						*(top1_data + offset_1) = curAnc[pos];
					}
				}
			}

			// 3. compute transform params and classes
			int offset_1[4];
			for (int c = 0; c < anchor_nums; ++c) {
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						int offset_0 = top[0]->offset(n, c, h, w);
						int curBbox_idx = *(top0_data + offset_0);
						if (--curBbox_idx >= 0) {
							const Dtype* bbox_data_gt = bbox_data + curBbox_idx * bbox_data_dim_;
							*(top0_data + offset_0) = *(bbox_data_gt + 4);

							for (int pos = 0; pos < 4; ++pos)
								offset_1[pos] = top[1]->offset(n, pos * anchor_nums + c, h, w);
							// get the transform parameters from the box to gbox
							Dtype src_w = std::max(*(top1_data + offset_1[2]) - *(top1_data + offset_1[0]) + (Dtype)1, (Dtype)1);
							Dtype src_h = std::max(*(top1_data + offset_1[3]) - *(top1_data + offset_1[1]) + (Dtype)1, (Dtype)1);
							Dtype src_cenx = *(top1_data + offset_1[0]) + (Dtype)0.5 * src_w;
							Dtype src_ceny = *(top1_data + offset_1[1]) + (Dtype)0.5 * src_h;
							Dtype tar_w = std::max(bbox_data_gt[2] - bbox_data_gt[0] + (Dtype)1, (Dtype)1);
							Dtype tar_h = std::max(bbox_data_gt[3] - bbox_data_gt[1] + (Dtype)1, (Dtype)1);
							Dtype tar_cenx = bbox_data_gt[0] + (Dtype)0.5 * tar_w;
							Dtype tar_ceny = bbox_data_gt[1] + (Dtype)0.5 * tar_h;

							*(top1_data + offset_1[0]) = (tar_cenx - src_cenx) / src_w;
							*(top1_data + offset_1[1]) = (tar_ceny - src_ceny) / src_h;
							*(top1_data + offset_1[2]) = std::log(tar_w / src_w);
							*(top1_data + offset_1[3]) = std::log(tar_h / src_h);
							//caffe::get_transform_params(top1_data + offset_1, bbox_data + curBbox_idx * bbox_data_dim_, top1_data + offset_1);
						}
						else if (curBbox_idx == -3)
							*(top0_data + offset_0) = -1;
					}
				}
			}
		}

		// For debug
		if (this->layer_param_.top_size() == 4) {
			Dtype* top3_data = top[3]->mutable_cpu_data();
			for (int n = 0; n < nums; ++n) {
				for (int c = 0; c < anchor_nums; ++c) {
					for (int h = 0; h < height; ++h) {
						for (int w = 0; w < width; ++w) {
							std::copy(anchors_data_ + 4 * c, anchors_data_ + 4 * (c + 1), curAnc);
							curAnc[0] += w * this->anchor_stride_;
							curAnc[1] += h * this->anchor_stride_;
							curAnc[2] += w * this->anchor_stride_;
							curAnc[3] += h * this->anchor_stride_;

							for (int pos = 1; pos < 5; ++pos) {
								int offset_3 = top[3]->offset(n, pos * anchor_nums + c, h, w);
								*(top3_data + offset_3) = curAnc[pos - 1];
							}

							Dtype max_iou = 0;
							for (int k = bbox_start_loc[n]; k < ((n + 1 == nums) ? bottom[2]->num() : bbox_start_loc[n + 1]); ++k) {
								Dtype cur_iou = IoU_cpu(bbox_data + k * bbox_data_dim_, curAnc);
								if (cur_iou > max_iou) {
									max_iou = cur_iou;
								}
							}
							int offset_3 = top[3]->offset(n, c, h, w);
							*(top3_data + offset_3) = Dtype(max_iou);
						}
					}
				}
			}
		}

		
		//int bigger_than_zero_counts = 0;
		//for(int i = 0; i < top[0]->count(); ++i)
		//	if(top[0]->cpu_data()[i] > 0)
		//		++bigger_than_zero_counts;
		//if(bigger_than_zero_counts == 0) {
		//	for(int i = 0; i < bottom[1]->count(); ++i)
		//		LOG(INFO) << bottom[1]->cpu_data()[i];
		//	
		//	LOG(INFO) << '\n';
		//	for(int i = 0; i < bottom[2]->count(); ++i)
		//		LOG(INFO) << bottom[2]->cpu_data()[i];
		//}
		//CHECK_GT(bigger_than_zero_counts, 0) << "There must exist at least one anchor(s) matching any bounding boxes.";
	}

	template <typename Dtype>
	void AnchorsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}

#ifdef CPU_ONLY
	STUB_GPU(AnchorsLayer);
#endif

	INSTANTIATE_CLASS(AnchorsLayer);
	REGISTER_LAYER_CLASS(Anchors);

}