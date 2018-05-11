#include <queue>

#include "caffe/layers/online_hard_mine_layer_v2.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe
{
	template <typename Dtype>
	void OnlineHardMineV2Layer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		this->neg_vs_pos_ratio_ = this->layer_param_.online_hard_mine_v2_param().neg_vs_pos_ratio();
		this->top_ignore_ratio_ = this->layer_param_.online_hard_mine_v2_param().top_ignore_ratio();

		CHECK_GE(this->neg_vs_pos_ratio_, 0) << "the neg_vs_pos_ratio_ must be greater than 0!";
		CHECK_LE(this->top_ignore_ratio_, 1) << "the top_ignore_ratio_ can not be greater than 1!";
	}

	template <typename Dtype>
	void OnlineHardMineV2Layer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// bottom 0 is the softmax layer output
		// bottom 1 is the gt label map that is waited to be modified
		CHECK_EQ(bottom[0]->height(), bottom[1]->height());
		CHECK_EQ(bottom[0]->width(), bottom[1]->width());
		CHECK_EQ(bottom[0]->channels(), 2);
		CHECK_EQ(bottom[1]->channels(), 1);

		// reshape the top blob
		top[0]->ReshapeLike(*bottom[1]);		
	}

	template <typename Dtype>
	bool operator < (const std::pair<Dtype, int>& a, const std::pair<Dtype, int>& b) {
		return a.first < b.first;
	}

	template <typename Dtype>
	void OnlineHardMineV2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batchnum = bottom[0]->num();
		int inlier_num_ = bottom[0]->count(1);
		int width_height_ = bottom[0]->count(2);

		// first step, find the top k minimum threshold in the cpu data
		const Dtype* bottom0_cpu = bottom[0]->cpu_data();
		const Dtype* bottom1_cpu = bottom[1]->cpu_data();
		top[0]->CopyFrom(*bottom[1]);
		Dtype* top0_cpu = top[0]->mutable_cpu_data();

		// use priority heap to get the first k minimum background training examples
		// time complexity: O(nlog(k))
		for (int n = 0; n < batchnum; ++ n) {
			int positive_nums = 0;
			for (int i = 0; i < width_height_; ++i) {
				if (bottom1_cpu[n * width_height_ + i] > Dtype(0))
					++positive_nums;
			}

			int neg_top_k = std::max(std::round(this->neg_vs_pos_ratio_ * positive_nums), Dtype(10));
			std::priority_queue<std::pair<Dtype, int> > pri_q;
			for (int i = 0; i < width_height_; ++i) {
				int idx = n * width_height_ + i;
				if (bottom1_cpu[idx] < Dtype(0.1)) { // if is negative sample
					top0_cpu[idx] = Dtype(-1);
					if (pri_q.size() < neg_top_k)
						pri_q.emplace(bottom0_cpu[n * inlier_num_ + i], i);
					else if (pri_q.top().first > bottom0_cpu[n * inlier_num_ + i]) {
						pri_q.pop();
						pri_q.emplace(bottom0_cpu[n * inlier_num_ + i], i);
					}
				}
			}

			// set ignore post number
			int neg_ignore_num = std::round(this->top_ignore_ratio_ * pri_q.size());
			while (pri_q.size() > neg_ignore_num) {
				int idx = n * width_height_ + pri_q.top().second;
				top0_cpu[idx] = Dtype(0);
				pri_q.pop();
			}
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(OnlineHardMineV2Layer);
#endif
	INSTANTIATE_CLASS(OnlineHardMineV2Layer);
	REGISTER_LAYER_CLASS(OnlineHardMineV2);
}