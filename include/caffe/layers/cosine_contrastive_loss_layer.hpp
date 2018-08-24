#ifndef CAFFE_COSINE_CONTRASTIVE_LOSS_LAYER_HPP_
#define CAFFE_COSINE_CONTRASTIVE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	class CosineContrastiveLossLayer : public LossLayer<Dtype> {
	public:
		explicit CosineContrastiveLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 3; }
		virtual inline const char* type() const { return "CosineContrastiveLoss"; }
		/**
		* Unlike most loss layers, in the ContrastiveLossLayer we can backpropagate
		* to the first two inputs.
		*/
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return bottom_index != 2;
		}

	protected:
		/// @copydoc ContrastiveLossLayer
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		/**
		* @brief Computes the Contrastive error gradient w.r.t. the inputs.
		*
		* Computes the gradients with respect to the two input vectors (bottom[0] and
		* bottom[1]), but not the similarity label (bottom[2]).
		*
		* @param top output Blob vector (length 1), providing the error gradient with
		*      respect to the outputs
		*   -# @f$ (1 \times 1 \times 1 \times 1) @f$
		*      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
		*      as @f$ \lambda @f$ is the coefficient of this layer's output
		*      @f$\ell_i@f$ in the overall Net loss
		*      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
		*      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
		*      (*Assuming that this top Blob is not used as a bottom (input) by any
		*      other layer of the Net.)
		* @param propagate_down see Layer::Backward.
		* @param bottom input Blob vector (length 2)
		*   -# @f$ (N \times C \times 1 \times 1) @f$
		*      the features @f$a@f$; Backward fills their diff with
		*      gradients if propagate_down[0]
		*   -# @f$ (N \times C \times 1 \times 1) @f$
		*      the features @f$b@f$; Backward fills their diff with gradients if
		*      propagate_down[1]
		*/
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> dot_prod_;  // cached for x.dot(y)
		Blob<Dtype> x_len_;  // cached for |x|
		Blob<Dtype> y_len_;  // cached for |y|
		bool add_weighted;
		Dtype pos_weight_;
		Dtype neg_weight_;
	};

}  // namespace caffe

#endif  // CAFFE_COSINE_CONTRASTIVE_LOSS_LAYER_HPP_
