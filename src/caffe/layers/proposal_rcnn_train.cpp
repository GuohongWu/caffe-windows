#include "caffe/layers/proposal_rcnn_train.hpp"
#include "caffe/util/nms.hpp"

#ifndef ROUND
#define ROUND(x) ((int)((x) + (Dtype)0.5))
#endif

using std::max;
using std::min;

namespace caffe
{
	// get the transformation between two bounding boxes, used for training phase
	template<typename Dtype>
	static void get_transform(Dtype box[],Dtype gbox[],Dtype& dx,Dtype& dy,
		Dtype& d_log_w, Dtype& d_log_h)
	{
		// get the transform parameters from the box to gbox
		Dtype src_w = std::max(box[2] - box[0] + (Dtype)1,(Dtype)1);
		Dtype src_h = std::max(box[3] - box[1] + (Dtype)1,(Dtype)1);
		Dtype src_cenx = box[0] + (Dtype)0.5 * src_w;
		Dtype src_ceny = box[1] + (Dtype)0.5 * src_h;
		Dtype tar_w = std::max(gbox[2] - gbox[0] + (Dtype)1,(Dtype)1);
		Dtype tar_h = std::max(gbox[3] - gbox[1] + (Dtype)1,(Dtype)1);
		Dtype tar_cenx = gbox[0] + (Dtype)0.5 * tar_w;
		Dtype tar_ceny = gbox[1] + (Dtype)0.5 * tar_h;

		dx = (tar_cenx - src_cenx)/src_w;
		dy = (tar_ceny - src_ceny)/src_h;

		d_log_w = std::log(tar_w/src_w);
		d_log_h = std::log(tar_h/src_h);

	}
	// transform the bounding box from anchor to true bounding box
	template <typename Dtype>
	static int transform_box(Dtype box[],
		const Dtype dx, const Dtype dy,
		const Dtype d_log_w, const Dtype d_log_h,
		const Dtype img_W, const Dtype img_H,
		const Dtype min_box_W, const Dtype min_box_H)
	{
		// width & height of box
		const Dtype w = box[2] - box[0] + (Dtype)1;
		const Dtype h = box[3] - box[1] + (Dtype)1;
		// center location of box
		const Dtype ctr_x = box[0] + (Dtype)0.5 * w;
		const Dtype ctr_y = box[1] + (Dtype)0.5 * h;

		// new center location according to gradient (dx, dy)
		const Dtype pred_ctr_x = dx * w + ctr_x;
		const Dtype pred_ctr_y = dy * h + ctr_y;
		// new width & height according to gradient d(log w), d(log h)
		const Dtype pred_w = exp(d_log_w) * w;
		const Dtype pred_h = exp(d_log_h) * h;

		// update upper-left corner location
		box[0] = pred_ctr_x - (Dtype)0.5 * pred_w;
		box[1] = pred_ctr_y - (Dtype)0.5 * pred_h;
		// update lower-right corner location
		box[2] = pred_ctr_x + (Dtype)0.5 * pred_w - (Dtype)1;
		box[3] = pred_ctr_y + (Dtype)0.5 * pred_h - (Dtype)1;

		// the bounding box that are outside the image width and height will not be considered here 
		// exclude the bounding box outside the rectangle
		//bool in_box = (box[0] >= 0 && box[0]<img_W) && (box[1] >= 0 && box[1]<img_H) && (box[2] >= 0 && box[2]<img_W) && (box[3] >= 0 && box[3]<img_H);
		//if(!in_box)
		//	return 0;

		// adjust new corner locations to be within the image region,
		box[0] = std::max((Dtype)0, std::min(box[0], img_W - (Dtype)1));
		box[1] = std::max((Dtype)0, std::min(box[1], img_H - (Dtype)1));
		box[2] = std::max((Dtype)0, std::min(box[2], img_W - (Dtype)1));
		box[3] = std::max((Dtype)0, std::min(box[3], img_H - (Dtype)1));

		// recompute new width & height
		const Dtype box_w = box[2] - box[0] + (Dtype)1;
		const Dtype box_h = box[3] - box[1] + (Dtype)1;

		// check if new box's size >= threshold
		return (box_w >= min_box_W) * (box_h >= min_box_H);
	}

	// sample the background and the 

	// sort the box using the detected scores
	template <typename Dtype>
	static void sort_box(Dtype list_cpu[], const int n,const int start, const int end,
		const int num_top)
	{
		const Dtype pivot_score = list_cpu[start * n + 4];
		int left = start + 1, right = end;
		Dtype temp[n];
		while (left <= right)
		{
			while (left <= end && list_cpu[left * n + 4] >= pivot_score)
				++left;
			while (right > start && list_cpu[right * n + 4] <= pivot_score)
				--right;
			if (left <= right)
			{
				for (int i = 0; i < n; ++i)
				{
					temp[i] = list_cpu[left * n + i];
				}
				for (int i = 0; i < n; ++i)
				{
					list_cpu[left * n + i] = list_cpu[right * n + i];
				}
				for (int i = 0; i < n; ++i)
				{
					list_cpu[right * n + i] = temp[i];
				}
				++left;
				--right;
			}
		}

		if (right > start)
		{
			for (int i = 0; i < n; ++i)
			{
				temp[i] = list_cpu[start * n + i];
			}
			for (int i = 0; i < n; ++i)
			{
				list_cpu[start * n + i] = list_cpu[right * n + i];
			}
			for (int i = 0; i < n; ++i)
			{
				list_cpu[right * n + i] = temp[i];
			}
		}

		if (start < right - 1)
		{
			sort_box(list_cpu,n, start, right - 1, num_top);
		}
		if (right + 1 < num_top && right + 1 < end)
		{
			sort_box(list_cpu,n,right + 1, end, num_top);
		}
	}

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
				//const Dtype scale_w = (Dtype)0.5 * ROUND(ratio_w * scales[j] * 1.0 / base_size - (Dtype)1);
				//const Dtype scale_h = (Dtype)0.5 * ROUND(ratio_h * scales[j] * 1.0 / base_size - (Dtype)1);
				const Dtype scale_w = (Dtype)0.5 * ROUND(ratio_w * scales[j] - (Dtype)1);
				const Dtype scale_h = (Dtype)0.5 * ROUND(ratio_h * scales[j] - (Dtype)1);
				// (x1, y1, x2, y2) for transformed box
				p_anchors[0] = center - scale_w;
				p_anchors[1] = center - scale_h;
				p_anchors[2] = center + scale_w;
				p_anchors[3] = center + scale_h;
				p_anchors += 4;
			} // endfor j
		}
	}

	// utility function: sample randomly with replacement from the indcies here
	void RandKN1(vector<int>& vec,int s)
	{
		// random seledt s from the vec  by in-place permutation
		int i;
		int tmp;
		int n = vec.size();
		for(i = 0;i < s;++i)
		{

			tmp = i + caffe_rng_rand() % (n-i);
			// swap vec(i) with vec(tmp)
			int b;
			b = vec[tmp];
			vec[tmp] = vec[i];
			vec[i] = b;
		}
	}
	// get the ground truth proposal and bounding box using the input data
	// we must get all the valid negative and positive indices for random sampling
	template <typename Dtype>
	static void enumerate_proposals_cpu(const Dtype bottom4d[],
		const Dtype d_anchor4d[],
		const Dtype g_bottom4d[],const Dtype g_anchor4d[],
		const Dtype anchors[],
		Dtype proposals[],const int n,
		const int num_anchors,
		const int bottom_H, const int bottom_W,
		const Dtype img_H, const Dtype img_W,
		const Dtype min_box_H, const Dtype min_box_W,
		const int feat_stride,vector<int>& pos_indice,vector<int>& neg_indices)
	{
		// note that we must exclude the ground truth label that are assigned -1, this will be of no use to the training
		// also, the background label have no ground truth transform parameters
		// 
		pos_indice.clear();
		neg_indices.clear();
		Dtype *p_proposal = proposals;
		const int bottom_area = bottom_H * bottom_W;
		const int props_area = num_anchors * bottom_H * bottom_W;

		Dtype gtmp[4];
		Dtype gtmp_anchor[4];

		int cur_cnt = 0;
		for (int h = 0; h < bottom_H; ++h)
		{
			for (int w = 0; w < bottom_W; ++w)
			{
				const Dtype x = w * feat_stride;
				const Dtype y = h * feat_stride;
				const Dtype *p_box = d_anchor4d + h * bottom_W + w;
				const Dtype *p_score = bottom4d + h * bottom_W + w;
				// the ground truth box transformation and score used to generate ground truth roi information
				const Dtype *g_box = g_anchor4d + h * bottom_W + w;
				const Dtype *g_score = g_bottom4d + h*bottom_W + w;

				for (int k = 0; k < num_anchors; ++k)
				{

					////////////////////// stored as n x k x 4 x h x w
					//const Dtype dx = p_box[(k * 4 + 0) * bottom_area];
					//const Dtype dy = p_box[(k * 4 + 1) * bottom_area];
					//const Dtype d_log_w = p_box[(k * 4 + 2) * bottom_area];
					//const Dtype d_log_h = p_box[(k * 4 + 3) * bottom_area];


					//// the ground truth bounding box
					//const Dtype gx = g_box[(k * 4 + 0) * bottom_area];
					//const Dtype gy = g_box[(k * 4 + 1) * bottom_area];
					//const Dtype g_log_w = g_box[(k * 4 + 2) * bottom_area];
					//const Dtype g_log_h = g_box[(k * 4 + 3) * bottom_area];
					///////////////////// stored as n x 4 x k x h x w
					const Dtype dx = p_box[k*bottom_area];
					const Dtype dy = p_box[k*bottom_area + props_area];
					const Dtype d_log_w = p_box[k*bottom_area + 2*props_area];
					const Dtype d_log_h = p_box[k*bottom_area + 3*props_area];


					// the ground truth bounding box
					const Dtype gx = g_box[k*bottom_area];
					const Dtype gy = g_box[k*bottom_area + props_area];
					const Dtype g_log_w = g_box[k*bottom_area + 2*props_area];
					const Dtype g_log_h = g_box[k*bottom_area + 3*props_area];


					const Dtype lab = g_score[k*bottom_area];

					// the anchor proposal
					p_proposal[0] = x + anchors[k * 4 + 0];
					p_proposal[1] = y + anchors[k * 4 + 1];
					p_proposal[2] = x + anchors[k * 4 + 2];
					p_proposal[3] = y + anchors[k * 4 + 3];
					// get the ground truth bounding box for computing the ground truth

					gtmp[0] = p_proposal[0];
					gtmp[1] = p_proposal[1];
					gtmp[2] = p_proposal[2];
					gtmp[3] = p_proposal[3];


					// if the anchor is background or ignored, then we do not add the proposal, but set the proposal label as the same lab

					// the prediction proposals
					p_proposal[4] = transform_box(p_proposal,
						dx, dy, d_log_w, d_log_h,
						img_W, img_H, min_box_W, min_box_H) *
						p_score[k * bottom_area];
					// if the label is -1,then we do not select it in our final roi poolings, set to zero score to be excluded!
					if(lab == -1)
						p_proposal[4] = 0;

					Dtype tx,ty,tlogw,tlogh;
					// if the target is not a background, then we can compute the ground truth bounding box here
					if(lab > 0)
					{
						// set the bounding box transformation here 
						transform_box(gtmp,gx,gy,g_log_w,g_log_h,img_W,img_H,min_box_W,min_box_H);
						// compute the transformation parameters between pred to ground truth bounding box gtmp
						get_transform(p_proposal,gtmp,tx,ty,tlogw,tlogh);
					}
					else
					{
						// store as all zero
						tx = ty = tlogw = tlogh = 0;
					}

					// set the information directly from the input data
					p_proposal[5] = lab;
					p_proposal[6] = tx;
					p_proposal[7] = ty;
					p_proposal[8] = tlogw;
					p_proposal[9] = tlogh;
					// count the training examples here, do not consider the rectangle with score 0, and do not consider the label with -1
					if(p_proposal[4] > 0)
					{
						if(lab > 0)
							pos_indice.push_back(cur_cnt);
						else
							neg_indices.push_back(cur_cnt);
					}
					// set the current count and continue the emumerate process
					cur_cnt++;
					p_proposal += n;
				} // endfor k
			}   // endfor w
		}     // endfor h
	}

	template <typename Dtype>
	static void retrieve_rois_cpu(const int num_rois,
		const int item_index,
		const Dtype proposals[],const int n,
		const int roi_indices[],
		Dtype rois[],
		Dtype roi_label[],Dtype roi_trans[],Dtype roi_loss_weight[],int num_class4,int npos = 1)
	{
		
		const Dtype pos_factor = npos != 0 ? Dtype(1)/Dtype(npos) : 0;
		for (int i = 0; i < num_rois; ++i)
		{
			const Dtype *const proposals_index = proposals + roi_indices[i] * n;
			rois[i * 5 + 0] = item_index;
			// the rois are set here
			rois[i * 5 + 1] = proposals_index[0];
			rois[i * 5 + 2] = proposals_index[1];
			rois[i * 5 + 3] = proposals_index[2];
			rois[i * 5 + 4] = proposals_index[3];

			// set the ground truth roi label and the ground truth transformation target variables
			roi_label[i] = proposals_index[5];
			// first set the training target as 0
			caffe_set(num_class4,Dtype(0),roi_trans + i*num_class4);
			caffe_set(num_class4,Dtype(0),roi_loss_weight + i*num_class4);
			// for non-background proposal images, we must assign certain value to some classes position
			int lab = roi_label[i];
			if(lab > 0)
			{
				int lab4 = lab*4;
				roi_trans[i * num_class4 + lab4 + 0] = proposals_index[6];
				roi_trans[i * num_class4 +lab4 +  1] = proposals_index[7];
				roi_trans[i * num_class4 + lab4 + 2] = proposals_index[8];
				roi_trans[i * num_class4 + lab4 + 3] = proposals_index[9];
				roi_loss_weight[i*num_class4 + lab4 + 0] = (Dtype)1 * pos_factor;
				roi_loss_weight[i*num_class4 + lab4 + 1] = (Dtype)1 * pos_factor;
				roi_loss_weight[i*num_class4 + lab4 + 2] = (Dtype)1 * pos_factor;
				roi_loss_weight[i*num_class4 + lab4 + 3] = (Dtype)1 * pos_factor;
			}
		}
	}

	template <typename Dtype>
	void ProposalRCNNTrainLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top)
	{

		ProposalRCNNTrainParameter param = this->layer_param_.proposal_rcnn_train_param();

		base_size_ = param.base_size();
		feat_stride_ = param.feat_stride();
		pre_nms_topn_ = param.pre_nms_topn();
		post_nms_topn_ = param.post_nms_topn();
		nms_thresh_ = param.nms_thresh();
		min_size_ = param.min_size();
		num_class_ = param.num_classes();

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
		anchors_.Reshape(anchors_shape);
		generate_anchors(base_size_, &ratios[0], &scales[0],
			ratios.size(), scales.size(),
			anchors_.mutable_cpu_data());

		vector<int> roi_indices_shape(1);
		roi_indices_shape[0] = post_nms_topn_;
		roi_indices_.Reshape(roi_indices_shape);

		// rois blob : holds R regions of interest, each is a 5 - tuple
		// (n, x1, y1, x2, y2) specifying an image batch index n and a
		// rectangle(x1, y1, x2, y2)
		vector<int> top_shape(2);
		top_shape[0] = bottom[0]->shape(0) * post_nms_topn_;
		top_shape[1] = 5;
		top[0]->Reshape(top_shape);
		// set the roi label and roi target here, the two blob is filled by the class label and the rectangle
		// the roi target must spread into the (nclass+1)*4 format for regression of each class inlcuding the background images
		top_shape[1] = 1;
		top[1]->Reshape(top_shape);
		// the regression target variables 
		top_shape[1] = 4*num_class_;
		top[2]->Reshape(top_shape);
		// the regression target outer loss weight variables
		top_shape[1] = 4*num_class_;
		top[3]->Reshape(top_shape);
		//the regression outer weight variables
		//top[4]->Reshape(top_shape);

	}

	template <typename Dtype>
	void ProposalRCNNTrainLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top)
	{

		const Dtype *p_bottom_item = bottom[0]->cpu_data();
		const Dtype *p_d_anchor_item = bottom[1]->cpu_data();
		const Dtype *p_img_info_cpu = bottom[2]->cpu_data();
		// ground truth label map
		const Dtype *g_bottom_item = bottom[3]->cpu_data();
		const Dtype *g_d_anchor_item = bottom[4]->cpu_data();

		Dtype *p_roi_item = top[0]->mutable_cpu_data();
		Dtype *p_roi_label = top[1]->mutable_cpu_data();
		Dtype *p_roi_target = top[2]->mutable_cpu_data();
		Dtype *p_roi_loss_weight = top[3]->mutable_cpu_data();

		vector<int> proposals_shape(2);
		vector<int> top_shape(2);
		proposals_shape[0] = 0;
		// for xywh and score and gt_label gt_target
		proposals_shape[1] = 10;

		top_shape[0] = 0;
		top_shape[1] = 5;

		// set the positive indices and negative indices
		vector<int> pos_indices;
		vector<int> neg_indices;

		for (int n = 0; n < bottom[0]->shape(0); ++n)
		{
			// bottom shape: (2 x num_anchors) x H x W
			const int bottom_H = bottom[0]->height();
			const int bottom_W = bottom[0]->width();
			// input image height & width
			const Dtype img_H = p_img_info_cpu[0];
			const Dtype img_W = p_img_info_cpu[1];
			// scale factor for height & width
			const Dtype scale_H = p_img_info_cpu[2];
			const Dtype scale_W = p_img_info_cpu[3];
			// minimum box width & height
			const Dtype min_box_H = min_size_ * scale_H;
			const Dtype min_box_W = min_size_ * scale_W;
			// number of all proposals = num_anchors * H * W
			const int num_proposals = anchors_.shape(0) * bottom_H * bottom_W;
			// number of top-n proposals before NMS
			const int pre_nms_topn = std::min(num_proposals, pre_nms_topn_);
			// number of final RoIs
			int num_rois = 0;

			// enumerate all proposals
			//   num_proposals = num_anchors * H * W
			//   (x1, y1, x2, y2, score) for each proposal
			// NOTE: for bottom, only foreground scores are passed
			proposals_shape[0] = num_proposals;
			proposals_.Reshape(proposals_shape);
			enumerate_proposals_cpu(
				p_bottom_item + num_proposals, p_d_anchor_item,g_bottom_item,g_d_anchor_item,
				anchors_.cpu_data(), proposals_.mutable_cpu_data(),proposals_.shape(1), anchors_.shape(0),
				bottom_H, bottom_W, img_H, img_W, min_box_H, min_box_W,
				feat_stride_,pos_indices,neg_indices);

			// random sampling the effective training examples from the dataset
			// 75% negative samples and 25% positive samples
			// as described in the corresponding paper here
			int npos = std::min(static_cast<int>(std::floor(post_nms_topn_*0.25)),static_cast<int>(pos_indices.size()));
			int nneg = std::min(post_nms_topn_ - npos,(int)neg_indices.size());
			RandKN1(pos_indices,npos);
			RandKN1(neg_indices,nneg);
			num_rois = npos + nneg;
			int* p_roi_indices = roi_indices_.mutable_cpu_data();
			for(int u = 0;u < npos;++u)
				p_roi_indices[u] = pos_indices[u];
			for(int u = npos;u < num_rois;++u)
				p_roi_indices[u] = neg_indices[u-npos];

			// retrieve the rois from the proposals_
			retrieve_rois_cpu(
				num_rois, n, proposals_.cpu_data(), proposals_.shape(1),roi_indices_.cpu_data(),
				p_roi_item, p_roi_label,p_roi_target,p_roi_loss_weight,4*num_class_);

			top_shape[0] += num_rois;

			p_bottom_item += bottom[0]->offset(1);
			p_d_anchor_item += bottom[1]->offset(1);
			g_bottom_item += bottom[3]->offset(1);
			g_d_anchor_item += bottom[4]->offset(1);
			// ouptut blobs
			p_roi_item += num_rois * 5;
			p_roi_label += num_rois;
			p_roi_target += (num_rois * 4 * num_class_);
			p_roi_loss_weight += (num_rois * 4 * num_class_);

		}

		// first roi proposal output
		top[0]->Reshape(top_shape);
		top_shape[1] = 1;
		top[1]->Reshape(top_shape);
		// the regression target variables 
		top_shape[1] = 4*num_class_;
		top[2]->Reshape(top_shape);
		// the regression target loss weight variables
		top_shape[1] = 4*num_class_;
		top[3]->Reshape(top_shape);
		// copy the outer weights with the inner target weight for usage in the smoothL1 Loss
		//top[4]->CopyFrom(*top[3]);
		//for(int i = 0; i < top[0]->count(); ++i)
		//	LOG(INFO) << top[0]->cpu_data()[i] << std::endl;
	}

#ifdef CPU_ONLY
	STUB_GPU(ProposalRCNNTrainLayer);
#endif
	INSTANTIATE_CLASS(ProposalRCNNTrainLayer);
	REGISTER_LAYER_CLASS(ProposalRCNNTrain);

}