#ifndef CAFFE_IMAGE_MIX_HDF5_DATA_LAYER_HPP_
#define CAFFE_IMAGE_MIX_HDF5_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <tuple>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Provides data to the Net from image files.
	* @brief Provides bbox & label data to the Net from HDF5 files
	*
	* TODO(dox): thorough documentation for Forward and proto params.
	*/
	template <typename Dtype>
	class ImageMixHDF5DataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit ImageMixHDF5DataLayer(const LayerParameter& param)
			: BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~ImageMixHDF5DataLayer();

		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void PreFetchReshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ImageMixHDF5Data"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int MinTopBlobs() const { return 3; }
		virtual inline int MaxTopBlobs() const { return 5; }

		//virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		//	const vector<Blob<Dtype>*>& top);
		//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		//	const vector<Blob<Dtype>*>& top);

	protected:
		shared_ptr<Caffe::RNG> prefetch_rng_;
		virtual void ShuffleImages();
		virtual void load_batch(Batch<Dtype>* batch);
		virtual void LoadBBoxHDF5FileData(const char* filename);

		vector<std::tuple<std::string, int, int> > lines_;
		int lines_id_;
		shared_ptr<Blob<Dtype> > hdf5_blobs_;   // bbox_lists
		shared_ptr<Blob<Dtype> > hdf5_poseKeypoints_blobs_;  // poseKeypoints_lists
		vector<int> bbox_init_shape;  // N * 5 * 1 * 1
		vector<int> poseKeypoints_init_shape;  // N * 1 * 17 * 3
	};


}  // namespace caffe

#endif  // CAFFE_IMAGE_MIX_HDF5_DATA_LAYER_HPP_

