#include "caffe/layers/anchors_layer.hpp"
// #include "caffe/util/nms.hpp"

namespace caffe {

template <typename Dtype>
void AnchorsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
   NOT_IMPLEMENTED;
}


template <typename Dtype>
void AnchorsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(AnchorsLayer);

}