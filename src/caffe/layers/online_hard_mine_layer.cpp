#include <functional>
#include <utility>
#include <vector>
#include <queue>
#include <omp.h>

#include "caffe/layers/online_hard_mine_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe
{
	template <typename Dtype>
	void OnlineHardMineLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			bool b = this->layer_param_.online_hard_mine_param().has_resample_ratio();
			if(!b)
				top_ratio_ = Dtype(0.01);
			else
				top_ratio_ = this->layer_param_.online_hard_mine_param().resample_ratio();
	}

	template <typename Dtype>
	void OnlineHardMineLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

			CHECK_GE(top_ratio_,0)<<"top ratio must greater than 0";

			CHECK_EQ(bottom[0]->height(), bottom[1]->height());
			CHECK_EQ(bottom[0]->width(), bottom[1]->width());

			// bottom 0 is the softmax layer output
			// bottom 1 is the label map that is waited to be modified
			int label_axis = 1;
			//outer_num_ = bottom[0]->count(0);
			inlier_num_ = bottom[0]->count(label_axis);
			channels_ = bottom[0]->channels();
			width_height_ = bottom[0]->count(label_axis+1);
			// reshape the top blob
			top[0]->ReshapeLike(*bottom[1]);
	}

	template <typename Dtype>
	struct cmp{
		bool operator()(const std::pair<Dtype, int>& a, const std::pair<Dtype, int>& b){
			return a.first > b.first;
		}
	};

	template <typename Dtype>
	void OnlineHardMineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

			int batchnum = bottom[0]->shape(0);
			//int count = bottom[0]->count();
			// first step, find the top k minimum threshold in the cpu data
			const Dtype* bottom0_cpu = bottom[0]->cpu_data();
			const Dtype* bottom1_cpu = bottom[1]->cpu_data();
			// first copy the data from bottom 1 blob
			top[0]->CopyFrom(*bottom[1]);
			Dtype* top0_cpu = top[0]->mutable_cpu_data();
			// use priority heap to get the first k minimum background training examples
			// use openmp to accerelate the process per-batch
			#pragma omp parallel for
			for(int i = 0;i < batchnum;++i)
			{
				std::priority_queue<std::pair<Dtype, int> > q;
				int k = std::floor(top_ratio_ * width_height_);
				if(k == 0)
					continue;
				for(int j = 0;j < width_height_;++j)
				{
					int pos1 = i*width_height_ + j;
					//if(bottom1_cpu[pos1] < Dtype(0.1))
					if(bottom1_cpu[pos1] == 0)
					{
						if(q.size() < k)
							q.push(std::pair<Dtype,int>(bottom0_cpu[i*inlier_num_ + j],j));
						else if(q.top().first > bottom0_cpu[i*inlier_num_+j])
						{
							q.pop();
							q.push(std::pair<Dtype,int>(bottom0_cpu[i*inlier_num_ + j],j));
						}
						top0_cpu[pos1] = Dtype(-1);
					}
				}
				for(int j = 0;j < k;++j)
				{
					int ki = q.top().second;
					top0_cpu[i*width_height_ + ki] = Dtype(0);
					q.pop();
				}
			}
	}
#ifdef CPU_ONLY
	STUB_GPU(OnlineHardMineLayer);
#endif
	INSTANTIATE_CLASS(OnlineHardMineLayer);
	REGISTER_LAYER_CLASS(OnlineHardMine);
}