#include <functional>
#include <utility>
#include <vector>
#include <queue>

#include "caffe/layers/online_random_mine_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe
{
	template <typename Dtype>
	void OnlineRandomMineLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			batch_size_ = this->layer_param_.online_random_mine_param().sample_size();
			// random sample neg/positive from the current label map
			half_batch_size_ = batch_size_ / 2;
	}

	template <typename Dtype>
	void OnlineRandomMineLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

			CHECK_GE(batch_size_,0)<<"the batch size must greater than 0";
			// bottom 0 is the input label map which contains all the effective positive and negative samples
			// ignoring the label that are assigned -1
			// our goal is to sample batch_size_ samples from the effective samples
			int label_axis = 1;
			//outer_num_ = bottom[0]->count(0);
			inlier_num_ = bottom[0]->count(label_axis);
			channels_ = bottom[0]->channels();
			width_height_ = bottom[0]->count(label_axis+1);
			// reshape the top blob
			top[0]->ReshapeLike(*bottom[0]);
			// 
	}

	template <typename Dtype>
	struct cmp{
		bool operator()(const std::pair<Dtype, int>& a, const std::pair<Dtype, int>& b){
			return a.first > b.first;
		}
	};

	void RandKN(vector<int>& vec,int s)
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

	template <typename Dtype>
	void OnlineRandomMineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

			int batchnum = bottom[0]->shape(0);
			//int count = bottom[0]->count();
			// first step, find the top k minimum threshold in the cpu data
			const Dtype* bottom0_cpu = bottom[0]->cpu_data();
			// note that the input image dew
			caffe_set(top[0]->count(),Dtype(-1),top[0]->mutable_cpu_data());
			Dtype* top0_cpu = top[0]->mutable_cpu_data();

			//// use priority heap to get the first k minimum background training examples
			//// use openmp to accerelate the process per-batch
			// ignore
			////#pragma omp parallel for
			for(int i = 0;i < batchnum;++i)
			{
				vector<int> neg_indices;
				vector<int> pos_indices;
				const Dtype* bcpu = bottom0_cpu + i*inlier_num_;
				Dtype* top_cpu = top0_cpu + i*inlier_num_;

				for(int j = 0;j < inlier_num_;++j)
				{
					// ignore those with label -1 during the training process
					Dtype lab = bcpu[j];
					if(lab > 0)
						pos_indices.push_back(j);
					else if (lab == 0)
						neg_indices.push_back(j);
				}
				int npos = std::min((int)pos_indices.size(),half_batch_size_);
				int nneg = std::min((int)neg_indices.size(),half_batch_size_);
				RandKN(pos_indices,npos);
				RandKN(neg_indices,nneg);
				// sample with replacement, this will generate equal positive and negative samples
				for(int j = 0;j < npos;++j)
				{
					//top_cpu[pos_indices[j]] = bcpu[pos_indices[j]];
					// for proposals, use 1 and 0 for the proposal network RPN
					top_cpu[pos_indices[j]] = 1;
				}
				for(int j = 0;j < nneg;++j)
				{
					//top_cpu[neg_indices[j]] = bcpu[neg_indices[j]];
					top_cpu[neg_indices[j]] = 0;
				}
			}
	}
#ifdef CPU_ONLY
	STUB_GPU(OnlineRandomMineLayer);
#endif
	INSTANTIATE_CLASS(OnlineRandomMineLayer);
	REGISTER_LAYER_CLASS(OnlineRandomMine);
}