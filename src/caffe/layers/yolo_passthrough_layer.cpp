#include "caffe/layers/yolo_passthrough_layer.hpp"


namespace caffe {

	template <typename Dtype>
	void YOLOPassThroughLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) { }

	template <typename Dtype>
	void YOLOPassThroughLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		CHECK_EQ((height & 1), 0);
		CHECK_EQ((width & 1), 0);

		int batch_size = bottom[0]->num();
		top[0]->Reshape(batch_size, 4 * bottom[0]->channels(), height / 2, width / 2);
	}

	template <typename Dtype>
	void YOLOPassThroughLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		int b_channels = bottom[0]->channels();
		int b_c, th_c, b_h, b_w, bottom_offset;
		for (int n = 0; n < top[0]->num(); ++n) {
			for (int c = 0; c < top[0]->channels(); ++c) {
				for (int h = 0; h < top[0]->height(); ++h) {
					for (int w = 0; w < top[0]->width(); ++w) {
						b_c = c % b_channels;
						th_c = c / b_channels;
						b_h = 2 * h;
						b_w = 2 * w;
						if (th_c == 1)
							++ b_w;
						else if (th_c == 2)
							++ b_h;
						else if (th_c == 3) {
							++ b_w;
							++ b_h;
						}

						bottom_offset = bottom[0]->offset(n, b_c, b_h, b_w);
						*top_data++ = bottom_data[bottom_offset];
					}
				}
			}
		}
	}

	template <typename Dtype>
	void YOLOPassThroughLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

			int b_channels = bottom[0]->channels();
			int b_c, th_c, b_h, b_w, bottom_offset;
			for (int n = 0; n < top[0]->num(); ++n) {
				for (int c = 0; c < top[0]->channels(); ++c) {
					for (int h = 0; h < top[0]->height(); ++h) {
						for (int w = 0; w < top[0]->width(); ++w) {
							b_c = c % b_channels;
							th_c = c / b_channels;
							b_h = 2 * h;
							b_w = 2 * w;
							if (th_c == 1)
								++b_w;
							else if (th_c == 2)
								++b_h;
							else if (th_c == 3) {
								++b_w;
								++b_h;
							}

							bottom_offset = bottom[0]->offset(n, b_c, b_h, b_w);
							bottom_diff[bottom_offset] = *top_diff++;
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(YOLOPassThroughLayer);
#endif

	INSTANTIATE_CLASS(YOLOPassThroughLayer);
	REGISTER_LAYER_CLASS(YOLOPassThrough);
}