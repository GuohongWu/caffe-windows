#include "caffe/layers/fd_anchors_layer.hpp"
#include "caffe/util/nms.hpp"
#include <queue>


namespace caffe {

	template <typename Dtype>
	void FDAnchorsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		FDAnchorsParameter fd_param = this->layer_param_.fd_anchors_param();
		this->anchor_size_ = fd_param.anchor_size();
		this->anchor_stride_ = fd_param.anchor_stride();
		nms_upper_thresh_ = fd_param.nms_upper_thresh();
		nms_lower_thresh_ = fd_param.nms_lower_thresh();
		vector<Dtype> ratios_(fd_param.ratio_size());
		for (int i = 0; i < fd_param.ratio_size(); ++i) {
			ratios_[i] = fd_param.ratio(i);
		}

		CHECK_EQ(fd_param.scale_size() * 2, top.size());
		this->scales.resize(fd_param.scale_size());
		for (int i = 0; i < fd_param.scale_size(); ++i) {
			scales[i] = fd_param.scale(i);
		}

		vector<int> anchor_shape{ int(scales.size()), int(ratios_.size()), 1, 4 };
		this->anchors_0_0_.Reshape(anchor_shape);
		Dtype* anchor_data = this->anchors_0_0_.mutable_cpu_data();
		for (int s = 0; s < scales.size(); ++s) {
			for (int c = 0; c < ratios_.size(); ++c) {
				Dtype cur_W = this->anchor_size_ / sqrt(ratios_[c]) * this->scales[s];
				Dtype cur_H = cur_W * ratios_[c];

				anchor_data[0] = Dtype(0.5 * (this->anchor_stride_ * this->scales[s] - cur_W));
				anchor_data[1] = Dtype(0.5 * (this->anchor_stride_ * this->scales[s] - cur_H));
				anchor_data[2] = anchor_data[0] + cur_W - Dtype(1);
				anchor_data[3] = anchor_data[1] + cur_H - Dtype(1);
				anchor_data += 4;
			}
		}
		/*
		for (int i = 0; i < scales.size(); ++i) {
			anchor_data[0] = anchor_data[1] = Dtype(0.5 * scales[i] * (anchor_stride_ - anchor_size_));
			anchor_data[2] = anchor_data[3] = Dtype(anchor_data[0] + scales[i] * anchor_size_ - 1);
			anchor_data += 4;
		}*/		
	}

	template <typename Dtype>
	void FDAnchorsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batch_size = bottom[0]->num();
		int img_height = bottom[0]->height();
		int img_width = bottom[0]->width();

		int ratio_nums = this->layer_param_.fd_anchors_param().ratio_size();
		int scale_nums = this->scales.size();
		float base_anchor_map_H = float(img_height) / this->anchor_stride_;
		float base_anchor_map_W = float(img_width) / this->anchor_stride_;
		for (int s = 0; s < scale_nums; ++s) {
			int cur_anchor_H = ceilf(base_anchor_map_H / this->scales[s]);
			int cur_anchor_W = ceilf(base_anchor_map_W / this->scales[s]);
			top[s]->Reshape(batch_size, ratio_nums, cur_anchor_H, cur_anchor_W); // label_map
			top[s + scale_nums]->Reshape(batch_size, 4 * ratio_nums, cur_anchor_H, cur_anchor_W); // reg_map
		}
	}

	template <typename Dtype>
	void FDAnchorsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {
		int nums = top[0]->num();
		int scal_nums = this->scales.size();
		int anchor_nums = top[0]->channels();

		vector<int> height_(scal_nums, -1);
		vector<int> width_(scal_nums, -1);
		vector<Dtype*> top0_data(scal_nums, nullptr);
		vector<Dtype*> top1_data(scal_nums, nullptr);
		for (int i = 0; i < scal_nums; ++i) {
			height_[i] = top[i]->height();
			width_[i] = top[i]->width();
			top0_data[i] = top[i]->mutable_cpu_data();
			top1_data[i] = top[i + scal_nums]->mutable_cpu_data();
			caffe_set(top[i]->count(), Dtype(-1), top0_data[i]);
		}

		const Dtype* bbox_start_loc = bottom[1]->cpu_data();
		const Dtype* bbox_data = bottom[2]->cpu_data();
		int bbox_data_dim_ = bottom[2]->count(1);
		const Dtype* anchors_data_ = this->anchors_0_0_.cpu_data();
		Dtype curAnc[4];

		for (int n = 0; n < nums; ++n) {
			int bbox_end_loc_ = ((n + 1 == nums) ? bottom[2]->num() : bbox_start_loc[n + 1]);
			vector<int> bbox_bind_to_AncNums(bbox_end_loc_ - bbox_start_loc[n], 0);

			// 1.assign each anchor to the unique face with highest IoU. Meanwhile, this IoU must be larger than a threshold (e.g. nms_upper_thresh_)
			for (int s = 0; s < scal_nums; ++s) {
				int offset_0_first = -1 + n * anchor_nums * height_[s] * width_[s];
				for (int c = 0; c < anchor_nums; ++c) {
					for (int h = 0; h < height_[s]; ++h) {
						for (int w = 0; w < width_[s]; ++w) {
							++offset_0_first;

							std::copy(anchors_data_ + 4 * anchor_nums * s + 4 * c, anchors_data_ + 4 * anchor_nums * s + 4 * (c + 1), curAnc);
							curAnc[0] += w * this->anchor_stride_ * this->scales[s];
							curAnc[1] += h * this->anchor_stride_ * this->scales[s];
							curAnc[2] += w * this->anchor_stride_ * this->scales[s];
							curAnc[3] += h * this->anchor_stride_ * this->scales[s];

							Dtype max_iou = 0;
							int max_bbox_idx = -1;
							for (int k = bbox_start_loc[n]; k < bbox_end_loc_; ++k) {
								Dtype cur_iou = IoU_cpu(bbox_data + k * bbox_data_dim_, curAnc);
								if (cur_iou > max_iou) {
									max_iou = cur_iou;
									max_bbox_idx = k;
								}
								else if (cur_iou == max_iou && max_iou > 0) {
									if (caffe::caffe_rng_rand() % 2)
										max_bbox_idx = k;
								}
							}
							if (max_iou > this->nms_upper_thresh_) {
								*(top0_data[s] + offset_0_first) = max_bbox_idx + 1;
								++bbox_bind_to_AncNums[max_bbox_idx - bbox_start_loc[n]];
								for (int pos = 0; pos < 4; ++pos) {
									int offset_1 = top[s + scal_nums]->offset(n, pos * anchor_nums + c, h, w);
									*(top1_data[s] + offset_1) = curAnc[pos];
								}
							}
							else if (max_iou < this->nms_lower_thresh_) {
								*(top0_data[s] + offset_0_first) = 0; // sure to be background!
							}
						}
					}
				}
			}

			// 2. For other Bboxes and other Anchors, Assign each bboxes to the anchor(s) with highest Intersection - overUnion(IoU) overlap 
			struct AncWithHighestIOU {
				Dtype max_iou;
				int max_bbox_idx;
				std::vector<std::tuple<int, int, int, int> > max_offset_idn;

				AncWithHighestIOU(Dtype iou_, int idx_) : max_iou(iou_), max_bbox_idx(idx_) { this->max_offset_idn.clear(); }
			};
			vector<AncWithHighestIOU> bbox_corres_anchors;
			for (int k = bbox_start_loc[n]; k < bbox_end_loc_; ++k) {
				if (bbox_bind_to_AncNums[k - bbox_start_loc[n]])
					continue;

				bbox_corres_anchors.emplace_back(Dtype(0), k + 1);
				vector<std::tuple<int, int, int, int> >& max_offset_idn = bbox_corres_anchors.back().max_offset_idn;
				Dtype& max_iou = bbox_corres_anchors.back().max_iou;
				for (int s = 0; s < scal_nums; ++s) {
					for (int c = 0; c < anchor_nums; ++c) {
						for (int h = 0; h < height_[s]; ++h) {
							for (int w = 0; w < width_[s]; ++w) {
								int offset_0_second = top[s]->offset(n, c, h, w);
								int curBbox_idx = *(top0_data[s] + offset_0_second);
								if (curBbox_idx >= 0)
									continue;

								std::copy(anchors_data_ + 4 * anchor_nums * s + 4 * c, anchors_data_ + 4 * anchor_nums * s + 4 * (c + 1), curAnc);
								curAnc[0] += w * this->anchor_stride_ * this->scales[s];
								curAnc[1] += h * this->anchor_stride_ * this->scales[s];
								curAnc[2] += w * this->anchor_stride_ * this->scales[s];
								curAnc[3] += h * this->anchor_stride_ * this->scales[s];

								Dtype cur_iou = IoU_cpu(bbox_data + k * bbox_data_dim_, curAnc);
								if (cur_iou > max_iou) {
									max_iou = cur_iou;
									max_offset_idn.clear();
									max_offset_idn.emplace_back(s, c, h, w);
								}
								else if (cur_iou == max_iou && max_iou > 0) {
									max_offset_idn.emplace_back(s, c, h, w);
								}
							}
						}
					}
				}
			}
			std::sort(bbox_corres_anchors.begin(), bbox_corres_anchors.end(), [](const AncWithHighestIOU& A, const AncWithHighestIOU& B) -> bool { return A.max_iou > B.max_iou; });
			for (auto &curBbox_corres : bbox_corres_anchors) {
				for (auto &cur_offset0 : curBbox_corres.max_offset_idn) {
					int s_ = std::get<0>(cur_offset0);
					int c_ = std::get<1>(cur_offset0);
					int h_ = std::get<2>(cur_offset0);
					int w_ = std::get<3>(cur_offset0);
					int offset_0_second = top[s_]->offset(n, c_, h_, w_);
					if (*(top0_data[s_] + offset_0_second) > 0)
						continue;

					*(top0_data[s_] + offset_0_second) = curBbox_corres.max_bbox_idx;
					++bbox_bind_to_AncNums[curBbox_corres.max_bbox_idx - 1 - bbox_start_loc[n]];

					std::copy(anchors_data_ + 4 * anchor_nums * s_ + 4 * c_, anchors_data_ + 4 * anchor_nums * s_ + 4 * (c_ + 1), curAnc);
					curAnc[0] += w_ * this->anchor_stride_ * this->scales[s_];
					curAnc[1] += h_ * this->anchor_stride_ * this->scales[s_];
					curAnc[2] += w_ * this->anchor_stride_ * this->scales[s_];
					curAnc[3] += h_ * this->anchor_stride_ * this->scales[s_];
					for (int pos = 0; pos < 4; ++pos) {
						int offset_1 = top[s_ + scal_nums]->offset(n, pos * anchor_nums + c_, h_, w_);
						*(top1_data[s_] + offset_1) = curAnc[pos];
					}
				}
			}

			// 3. some tiny and outer faces may still do not match enough anchors
			/*
			picking out anchors whose jaccard overlap with this face are higher than 0:1, 
			then sorting them to select top-N as matched anchors of this face. 
			We set N as the average number from previous stages.
			*/
			int N = 0;
			for (auto &n_i : bbox_bind_to_AncNums)
				N += n_i;
			N /= bbox_bind_to_AncNums.size();
			DLOG(INFO) << "N = " << N;
			struct AncWithUniqueIOU {
				Dtype cur_iou;
				int max_s;
				int max_c;
				int max_h;
				int max_w;

				bool operator < (const AncWithUniqueIOU& B) const {
					return (this->cur_iou < B.cur_iou);
				}

				bool operator > (const AncWithUniqueIOU& B) const {
					return (this->cur_iou > B.cur_iou);
				}
			};
			for (int k = bbox_start_loc[n]; k < bbox_end_loc_; ++k) {
				int need_to_add_nums = N - 2 * bbox_bind_to_AncNums[k - bbox_start_loc[n]];  // if less than N/2 anchor nums, then need to add more.
				if (need_to_add_nums <= 0)
					continue;

				std::priority_queue <AncWithUniqueIOU, std::vector<AncWithUniqueIOU>, std::greater<AncWithUniqueIOU>> pri_que;  // min-heap
				AncWithUniqueIOU curAnc_struct;
				for (int s = 0; s < scal_nums; ++s) {
					for(int c = 0; c < anchor_nums; ++c) {
						for (int h = 0; h < height_[s]; ++h) {
							for (int w = 0; w < width_[s]; ++w) {
								int offset_0_second = top[s]->offset(n, c, h, w);
								int curBbox_idx = *(top0_data[s] + offset_0_second);
								if (curBbox_idx >= 0)
									continue;

								std::copy(anchors_data_ + 4 * anchor_nums * s + 4 * c, anchors_data_ + 4 * anchor_nums * s + 4 * (c + 1), curAnc);
								curAnc[0] += w * this->anchor_stride_ * this->scales[s];
								curAnc[1] += h * this->anchor_stride_ * this->scales[s];
								curAnc[2] += w * this->anchor_stride_ * this->scales[s];
								curAnc[3] += h * this->anchor_stride_ * this->scales[s];

								curAnc_struct.cur_iou = IoU_cpu(bbox_data + k * bbox_data_dim_, curAnc);
								if (curAnc_struct.cur_iou > this->nms_lower_thresh_) {
									curAnc_struct.max_s = s;
									curAnc_struct.max_c = c;
									curAnc_struct.max_h = h;
									curAnc_struct.max_w = w;
									if (pri_que.size() < need_to_add_nums) {
										pri_que.push(curAnc_struct);
									}
									else if (pri_que.top().cur_iou < curAnc_struct.cur_iou) {
										pri_que.pop();
										pri_que.push(curAnc_struct);
									}
								}
							}
						}
					}
				}
				while (!pri_que.empty()) {
					curAnc_struct = pri_que.top();
					int s_ = curAnc_struct.max_s;
					int c_ = curAnc_struct.max_c;
					int h_ = curAnc_struct.max_h;
					int w_ = curAnc_struct.max_w;
					int offset_0_second = top[s_]->offset(n, c_, h_, w_);
					*(top0_data[s_] + offset_0_second) = k + 1;

					std::copy(anchors_data_ + 4 * anchor_nums * s_ + 4 * c_, anchors_data_ + 4 * anchor_nums * s_ + 4 * (c_ + 1), curAnc);
					curAnc[0] += w_ * this->anchor_stride_ * this->scales[s_];
					curAnc[1] += h_ * this->anchor_stride_ * this->scales[s_];
					curAnc[2] += w_ * this->anchor_stride_ * this->scales[s_];
					curAnc[3] += h_ * this->anchor_stride_ * this->scales[s_];
					for (int pos = 0; pos < 4; ++pos) {
						int offset_1 = top[s_ + scal_nums]->offset(n, pos * anchor_nums + c_, h_, w_);
						*(top1_data[s_] + offset_1) = curAnc[pos];
					}

					pri_que.pop();
				}
			}

			// 4. compute transform params and classes
			int offset_1[4];
			for (int s = 0; s < scal_nums; ++s) {
				for (int c = 0; c < anchor_nums; ++c) {
					for (int h = 0; h < height_[s]; ++h) {
						for (int w = 0; w < width_[s]; ++w) {
							int offset_0 = top[s]->offset(n, c, h, w);
							int curBbox_idx = *(top0_data[s] + offset_0);
							if (--curBbox_idx >= 0) {
								const Dtype* bbox_data_gt = bbox_data + curBbox_idx * bbox_data_dim_;
								*(top0_data[s] + offset_0) = *(bbox_data_gt + 4);

								for (int pos = 0; pos < 4; ++pos)
									offset_1[pos] = top[s + scal_nums]->offset(n, pos * anchor_nums + c, h, w);
								// get the transform parameters from the box to gbox
								Dtype src_w = std::max(*(top1_data[s] + offset_1[2]) - *(top1_data[s] + offset_1[0]) + (Dtype)1, (Dtype)1);
								Dtype src_h = std::max(*(top1_data[s] + offset_1[3]) - *(top1_data[s] + offset_1[1]) + (Dtype)1, (Dtype)1);
								Dtype src_cenx = *(top1_data[s] + offset_1[0]) + (Dtype)0.5 * src_w;
								Dtype src_ceny = *(top1_data[s] + offset_1[1]) + (Dtype)0.5 * src_h;
								Dtype tar_w = std::max(bbox_data_gt[2] - bbox_data_gt[0] + (Dtype)1, (Dtype)1);
								Dtype tar_h = std::max(bbox_data_gt[3] - bbox_data_gt[1] + (Dtype)1, (Dtype)1);
								Dtype tar_cenx = bbox_data_gt[0] + (Dtype)0.5 * tar_w;
								Dtype tar_ceny = bbox_data_gt[1] + (Dtype)0.5 * tar_h;

								*(top1_data[s] + offset_1[0]) = (tar_cenx - src_cenx) / src_w;
								*(top1_data[s] + offset_1[1]) = (tar_ceny - src_ceny) / src_h;
								*(top1_data[s] + offset_1[2]) = std::log(tar_w / src_w);
								*(top1_data[s] + offset_1[3]) = std::log(tar_h / src_h);
							}
							else
								*(top0_data[s] + offset_0) = 0;
						}
					}
				}
			}
		}


	}

	template <typename Dtype>
	void FDAnchorsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}

#ifdef CPU_ONLY
	STUB_GPU(FDAnchorsLayer);
#endif

	INSTANTIATE_CLASS(FDAnchorsLayer);
	REGISTER_LAYER_CLASS(FDAnchors);
}