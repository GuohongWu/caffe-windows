#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>
#include <math.h>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_h || crop_w) {
    height = crop_h;
    width = crop_w;
    // We only do random crop when we do training.
    if (phase_ == TRAIN && !param_.center_crop()) {
      h_off = Rand(datum_height - crop_h + 1);
      w_off = Rand(datum_width - crop_w + 1);
    } else {
      h_off = (datum_height - crop_h) / 2;
      w_off = (datum_width - crop_w) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decode and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_h || crop_w) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
  Blob<Dtype>* transformed_blob,
  bool transpose) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob, transpose);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
	Blob<Dtype>* transformed_blob,
	bool transpose) {
	int crop_h = 0;
	int crop_w = 0;
	if (param_.has_crop_size()) {
		crop_h = param_.crop_size();
		crop_w = param_.crop_size();
	}
	if (param_.has_crop_h()) {
		crop_h = param_.crop_h();
	}
	if (param_.has_crop_w()) {
		crop_w = param_.crop_w();
	}
	const int img_channels = cv_img.channels();
	const int img_height = cv_img.rows;
	const int img_width = cv_img.cols;

	// Check dimensions.
	const int channels = transformed_blob->channels();
	const int height = transformed_blob->height();
	const int width = transformed_blob->width();
	const int num = transformed_blob->num();

	CHECK_EQ(channels, img_channels);
	if (transpose) {
		CHECK_LE(height, img_width);
		CHECK_LE(width, img_height);
	}
	else {
		CHECK_LE(height, img_height);
		CHECK_LE(width, img_width);
	}
	CHECK_GE(num, 1);
	//if (transpose) {
	//  std::swap(height, width);
	//}

	//CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

	const Dtype scale = param_.scale();
	const bool do_mirror = param_.mirror() && Rand(2);
	const bool has_mean_file = param_.has_mean_file();
	const bool has_mean_values = mean_values_.size() > 0;

	CHECK_GT(img_channels, 0);
	CHECK_GE(img_height, crop_h);
	CHECK_GE(img_width, crop_w);

	Dtype* mean = NULL;
	if (has_mean_file) {
		CHECK_EQ(img_channels, data_mean_.channels());
		CHECK_EQ(img_height, data_mean_.height());
		CHECK_EQ(img_width, data_mean_.width());
		mean = data_mean_.mutable_cpu_data();
	}
	if (has_mean_values) {
		CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
			"Specify either 1 mean_value or as many as channels: " << img_channels;
		if (img_channels > 1 && mean_values_.size() == 1) {
			// Replicate the mean_value for simplicity
			for (int c = 1; c < img_channels; ++c) {
				mean_values_.push_back(mean_values_[0]);
			}
		}
	}

	// add a random-color mask square in image
	int mask_size_min = param_.mask_size_min();
	int mask_size_max = param_.mask_size_max();
	CHECK_GE(mask_size_max, mask_size_min);
	CHECK_GE(std::min(img_height, img_width), mask_size_max);
	const bool do_mask = (mask_size_min || mask_size_max) && !Rand(4);
	cv::Mat cv_masked_img;
	if (do_mask) {
		cv_masked_img = cv_img.clone();
		cv::RNG uni_rng(time(0));
		int crop_size_ = uni_rng.uniform(mask_size_min, mask_size_max + 1);
		int tl_pos_x_ = uni_rng.uniform(0, img_width - crop_size_ + 1);
		int tl_pos_y_ = uni_rng.uniform(0, img_height - crop_size_ + 1);
		cv::Mat masked_reg(cv_masked_img, cv::Rect(tl_pos_x_, tl_pos_y_, crop_size_, crop_size_));
		vector<uchar> random_pix_color(img_channels);
		for(int i = 0;i < img_channels;++i)
			random_pix_color[i] = uni_rng.uniform(0,256);
		for (int row_ = 0; row_ < masked_reg.rows; ++row_) {
			uchar* ptr = masked_reg.ptr<uchar>(row_);
			int img_index = 0;
			for (int col_ = 0; col_ < masked_reg.cols; ++col_) {
				for (int c_ = 0; c_ < img_channels; ++c_)
					//ptr[img_index++] = uni_rng.uniform(0, 256);
					ptr[img_index++] = random_pix_color[c_];
			}
		}
	}
	else
		cv_masked_img = cv_img;

	int h_off = 0;
	int w_off = 0;
	cv::Mat cv_cropped_img = cv_masked_img;
	if (crop_h || crop_w) {
		CHECK_EQ(crop_h, height);
		CHECK_EQ(crop_w, width);
		// We only do random crop when we do training.
		if (phase_ == TRAIN && !param_.center_crop()) {
			h_off = Rand(img_height - crop_h + 1);
			w_off = Rand(img_width - crop_w + 1);
		}
		else {
			h_off = (img_height - crop_h) / 2;
			w_off = (img_width - crop_w) / 2;
		}
		cv::Rect roi(w_off, h_off, crop_w, crop_h);
		cv_cropped_img = cv_cropped_img(roi);
	}
	else {
		if (transpose) {
			CHECK_EQ(img_width, height);
			CHECK_EQ(img_height, width);
		}
		else {
			CHECK_EQ(img_height, height);
			CHECK_EQ(img_width, width);
		}
	}

	CHECK(cv_cropped_img.data);
	// image color distort.
	if (param_.color_distort() == true) {
		const int distort_type = Rand(7);
		switch (distort_type)
		{
		case 0:
		case 1:
		case 2: // do nothing. Remain to be org image.
			; break;
		case 3: // GaussianBlur
			cv::GaussianBlur(cv_cropped_img, cv_cropped_img, cv::Size(3, 3), 1.5, 1.5); break;
		case 4: // GaussianNoise
		{
			cv::Mat gauss_noise(cv_cropped_img.rows, cv_cropped_img.cols, CV_32FC(img_channels));
			cv::randn(gauss_noise, 0, 0.0316 * 255);
			cv::Mat cv_cropped_img_float;
			cv_cropped_img.convertTo(cv_cropped_img_float, CV_32FC(img_channels));
			cv_cropped_img_float += gauss_noise;
			cv_cropped_img_float.convertTo(cv_cropped_img, CV_8UC(img_channels));
		}   break;
		case 5:
		case 6: // Brightness and Contrast
		{
			cv::RNG uni_rng(time(0));
			float brightness_ = uni_rng.uniform(-0.055f, 0.055f);
			float contrast_ = uni_rng.uniform(-0.085f, 0.085f);
			float slope_k = tan((45 + 44 * contrast_) / 180 * 3.1415926);
			uchar look_up_tab[256];
			for (int idx = 0; idx < 256; ++idx)
				look_up_tab[idx] = cv::saturate_cast<uchar>((idx - 127.5 * (1 - brightness_)) * slope_k + 127.5 * (1 + brightness_));
			cv::Mat lut_mat(1, 256, CV_8UC1, look_up_tab);

			const int rand_channel_idx = Rand(4);
			if(rand_channel_idx == 0) // BGR
				cv::LUT(cv_cropped_img, lut_mat, cv_cropped_img);
			else {
				std::vector<cv::Mat> img_split;
				cv::split(cv_cropped_img, img_split);

				if(rand_channel_idx == 1) // B
					cv::LUT(img_split[0], lut_mat, img_split[0]);
				else if(rand_channel_idx == 2) // G
					cv::LUT(img_split[1], lut_mat, img_split[1]);
				else // R
					cv::LUT(img_split[2], lut_mat, img_split[2]);

				cv::merge(img_split, cv_cropped_img);
			}
		}  break;
		default:   LOG(ERROR) << "Unknown Color_distort type!"; break;
		}
	}

	Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	bool is_float_data = cv_cropped_img.depth() == CV_32F;
	int top_index;
	if (transpose) {
		for (int w = 0; w < width; ++w) {
			const uchar* ptr = cv_cropped_img.ptr<uchar>(w);
			const float* float_ptr = cv_cropped_img.ptr<float>(w);
			int img_index = 0;
			for (int h = 0; h < height; ++h) {
				for (int c = img_channels - 1; c >= 0; --c) {
					if (do_mirror) {
						top_index = (c * height + h) * width + (width - 1 - w);
					}
					else {
						top_index = (c * height + h) * width + w;
					}
					// int top_index = (c * height + h) * width + w;
					Dtype pixel = static_cast<Dtype>(is_float_data ? float_ptr[img_index] : ptr[img_index]);
					img_index++;
					if (has_mean_file) {
						int mean_index;
						mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
						transformed_data[top_index] =
							(pixel - mean[mean_index]) * scale;
					}
					else {
						if (has_mean_values) {
							transformed_data[top_index] =
								(pixel - mean_values_[c]) * scale;
						}
						else {
							transformed_data[top_index] = pixel * scale;
						}
					}
				}
			}
		}
	}
	else {
		for (int h = 0; h < height; ++h) {
			const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
			const float* float_ptr = cv_cropped_img.ptr<float>(h);
			int img_index = 0;
			for (int w = 0; w < width; ++w) {
				for (int c = 0; c < img_channels; ++c) {
					if (do_mirror) {
						top_index = (c * height + h) * width + (width - 1 - w);
					}
					else {
						top_index = (c * height + h) * width + w;
					}
					// int top_index = (c * height + h) * width + w;
					Dtype pixel = static_cast<Dtype>(is_float_data ? float_ptr[img_index] : ptr[img_index]);
					img_index++;
					if (has_mean_file) {
						int mean_index;
						mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
						transformed_data[top_index] =
							(pixel - mean[mean_index]) * scale;
					}
					else {
						if (has_mean_values) {
							transformed_data[top_index] =
								(pixel - mean_values_[c]) * scale;
						}
						else {
							transformed_data[top_index] = pixel * scale;
						}
					}
				}
			}
		}
	}
}

  template<typename Dtype>
  void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob, 
	                                     const cv::Mat& cv_mask_img, Blob<Dtype>* transformed_segMask_blob, 
	                                        Dtype* bbox_blob, Dtype* poseKeypoints_blob, const int bbox_nums) {
	  int img_height = cv_img.rows;
	  int img_width = cv_img.cols;
	  int new_height = transformed_blob->height();
	  int new_width = transformed_blob->width();
	  int channel_ = transformed_blob->channels();
	  CHECK_GT(img_height, 0);
	  CHECK_GT(img_width, 0);
	  CHECK_GT(new_height, 0);
	  CHECK_GT(new_width, 0);
	  CHECK_EQ(channel_, cv_img.channels());
	  CHECK_EQ(transformed_blob->num(), 1);

	  const Dtype scale = param_.scale();
	  const bool has_mean_file = param_.has_mean_file();
	  const bool has_mean_values = mean_values_.size() > 0;

	  Dtype* img_mean = NULL;
	  if (has_mean_file) {
		  CHECK_EQ(channel_, data_mean_.channels());
		  CHECK_EQ(new_height, data_mean_.height());
		  CHECK_EQ(new_width, data_mean_.width());
		  img_mean = data_mean_.mutable_cpu_data();
	  }
	  if (has_mean_values) {
		  CHECK(mean_values_.size() == 1 || mean_values_.size() == channel_) <<
			  "Specify either 1 mean_value or as many as channels: " << channel_;
		  if (channel_ > 1 && mean_values_.size() == 1) {
			  // Replicate the mean_value for simplicity
			  for (int c = 1; c < channel_; ++c) {
				  mean_values_.push_back(mean_values_[0]);
			  }
		  }
	  }

	  //1. do mirror
	  const bool do_mirror = param_.mirror() && Rand(2);
	  cv::Mat cv_img_mirror = cv_img;
	  cv::Mat cv_mask_img_mirror = cv_mask_img;
	  if (do_mirror) {
		  cv::flip(cv_img_mirror, cv_img_mirror, 1);
		  if (cv_mask_img.data)
			  cv::flip(cv_mask_img_mirror, cv_mask_img_mirror, 1);
		  Dtype* bbox_blob_pos = bbox_blob;
		  for (int k = 0; k < bbox_nums; ++k) {
			  Dtype tmp_ = bbox_blob_pos[0];
			  bbox_blob_pos[0] = img_width - bbox_blob_pos[2] - 1;
			  bbox_blob_pos[2] = img_width - tmp_ - 1;
			  bbox_blob_pos += 5;
		  }

		  if (poseKeypoints_blob) {
			  // swap left and right points
			  /* In COCO : (1 - 'nose'	2 - 'left_eye' 3 - 'right_eye' 4 - 'left_ear' 5 - 'right_ear'
				   6 - 'left_shoulder' 7 - 'right_shoulder'	8 - 'left_elbow' 9 - 'right_elbow' 10 - 'left_wrist'
				   11 - 'right_wrist'	12 - 'left_hip' 13 - 'right_hip' 14 - 'left_knee'	15 - 'right_knee'
				   16 - 'left_ankle' 17 - 'right_ankle')
			  */
			  Dtype* poseKeypoints_blob_pos = poseKeypoints_blob;
			  for (int k = 0; k < bbox_nums; ++k) {
				  poseKeypoints_blob_pos[0] = img_width - poseKeypoints_blob_pos[0] - 1;
				  poseKeypoints_blob_pos += 3;  // the first keypoint is 'nose', do not need to be mirrored.
				  for (int i = 0; i < 8; ++i) {
					  poseKeypoints_blob_pos[0] = img_width - poseKeypoints_blob_pos[0] - 1;
					  poseKeypoints_blob_pos[3] = img_width - poseKeypoints_blob_pos[3] - 1;

					  std::swap(poseKeypoints_blob_pos[0], poseKeypoints_blob_pos[3]);
					  std::swap(poseKeypoints_blob_pos[1], poseKeypoints_blob_pos[4]);
					  std::swap(poseKeypoints_blob_pos[2], poseKeypoints_blob_pos[5]);
					  poseKeypoints_blob_pos += 6;
				  }				  
			  }
		  }
	  }

	  //2. random crop.
	  int crop_h = 0;
	  int crop_w = 0;
	  if (param_.has_crop_size()) {
		  crop_h = param_.crop_size();
		  crop_w = param_.crop_size();
	  }
	  if (param_.has_crop_h()) {
		  crop_h = param_.crop_h();
	  }
	  if (param_.has_crop_w()) {
		  crop_w = param_.crop_w();
	  }
	  cv::Mat cv_img_crop = cv_img_mirror;
	  cv::Mat cv_mask_img_crop = cv_mask_img_mirror;
	  if (crop_h > 0 && crop_w > 0 && Rand(2)) {
		  // we adaptively adjust crop_h and crop_w to fit the size of input img.
		  crop_h = std::min(crop_h, img_height);
		  crop_w = std::min(crop_w, img_width);
		  int h_off = 0;
		  int w_off = 0;
		  if (phase_ == TRAIN && !param_.center_crop()) {
			  h_off = Rand(img_height - crop_h + 1);
			  w_off = Rand(img_width - crop_w + 1);
		  }
		  else {
			  h_off = (img_height - crop_h) / 2;
			  w_off = (img_width - crop_w) / 2;
		  }
		  cv::Rect roi(w_off, h_off, crop_w, crop_h);
		  cv_img_crop = cv_img_crop(roi);
		  img_height = crop_h;
		  img_width = crop_w;

		  if(cv_mask_img.data)
			  cv_mask_img_crop = cv_mask_img_crop(roi);

		  Dtype* bbox_blob_pos = bbox_blob;
		  for (int k = 0; k < bbox_nums; ++k) {
			  bbox_blob_pos[0] -= Dtype(w_off);
			  bbox_blob_pos[1] -= Dtype(h_off);
			  bbox_blob_pos[2] -= Dtype(w_off);
			  bbox_blob_pos[3] -= Dtype(h_off);
			  bbox_blob_pos += 5;
		  }

		  if (poseKeypoints_blob) {
			  // all keypoints will be translated.
			  Dtype* poseKeypoints_blob_pos = poseKeypoints_blob;
			  for (int k = 0; k < bbox_nums; ++k) {
				  for (int i = 0; i < 17; ++i) {
					  poseKeypoints_blob_pos[0] -= Dtype(w_off);
					  poseKeypoints_blob_pos[1] -= Dtype(h_off);
					  poseKeypoints_blob_pos += 3;
				  }
			  }
		  }
	  }

	  //3. image scaled resize.
	  const bool do_scaled_resize = param_.scaled_resize();
	  cv::Mat cv_img_scaled = cv_img_crop;
	  cv::Mat cv_mask_img_scaled = cv_mask_img_crop;
	  float scale_rate = 1.0f;
	  if(do_scaled_resize)
		  scale_rate = std::min(1.0f, float(Rand(61)) / 100.0f + 0.7f);
	  float real_scale_rate = std::min(new_height * scale_rate / img_height, new_width * scale_rate / img_width);
	  img_width = std::min(new_width, int(real_scale_rate * img_width + 0.5f));
	  img_height = std::min(new_height, int(real_scale_rate * img_height + 0.5f));
	  cv::resize(cv_img_scaled, cv_img_scaled, cv::Size(img_width, img_height), 0.0, 0.0, Rand(3));
	  if (cv_mask_img.data)
		  cv::resize(cv_mask_img_scaled, cv_mask_img_scaled, cv::Size(img_width, img_height), 0.0, 0.0, cv::INTER_NEAREST);

	  Dtype* bbox_blob_pos = bbox_blob;
	  for (int k = 0; k < bbox_nums; ++k) {
		  for (int pos = 0; pos < 4; ++pos)
			  bbox_blob_pos[pos] *= real_scale_rate;
		  bbox_blob_pos += 5;
	  }

	  if (poseKeypoints_blob) {
		  // all keypoints will be scaled.
		  Dtype* poseKeypoints_blob_pos = poseKeypoints_blob;
		  for (int k = 0; k < bbox_nums; ++k) {
			  for (int i = 0; i < 17; ++i) {
				  poseKeypoints_blob_pos[0] *= real_scale_rate;
				  poseKeypoints_blob_pos[1] *= real_scale_rate;
				  poseKeypoints_blob_pos += 3;
			  }
		  }
	  }

	  //4. add a random-color mask square in image
	  int mask_size_min = param_.mask_size_min();
	  int mask_size_max = param_.mask_size_max();
	  CHECK_GE(mask_size_max, mask_size_min);
	  CHECK_GE(std::min(img_height, img_width), mask_size_max);
	  const bool do_mask = (mask_size_min || mask_size_max) && !Rand(4);
	  cv::Mat cv_masked_img = cv_img_scaled;
	  if (do_mask) {
		  cv::RNG uni_rng(time(0));
		  int crop_size_ = uni_rng.uniform(mask_size_min, mask_size_max + 1);
		  int tl_pos_x_ = uni_rng.uniform(0, img_width - crop_size_ + 1);
		  int tl_pos_y_ = uni_rng.uniform(0, img_height - crop_size_ + 1);
		  cv::Mat masked_reg(cv_masked_img, cv::Rect(tl_pos_x_, tl_pos_y_, crop_size_, crop_size_));

		  cv::Mat mask_masked_reg;
		  if (cv_mask_img.data)
			  mask_masked_reg = cv::Mat(cv_mask_img_scaled, cv::Rect(tl_pos_x_, tl_pos_y_, crop_size_, crop_size_));

		  vector<uchar> random_pix_color(channel_);
		  for (int c_ = 0; c_ < channel_; ++c_)
			  random_pix_color[c_] = uni_rng.uniform(0, 256);
		  for (int row_ = 0; row_ < masked_reg.rows; ++row_) {
			  uchar* ptr = masked_reg.ptr<uchar>(row_);
			  uchar* ptr_mask;
			  if(cv_mask_img.data)
				ptr_mask = mask_masked_reg.ptr<uchar>(row_);
			  int img_index = 0;
			  for (int col_ = 0; col_ < masked_reg.cols; ++col_) {
				  for (int c_ = 0; c_ < channel_; ++c_, ++img_index) {
					  ptr[img_index] = random_pix_color[c_];
					  if (cv_mask_img.data && c_ == 0)
						  ptr_mask[img_index / channel_] = 0;
				  }
			  }
		  }
	  }

	  //5. image color distort.
	  const bool do_color_distort_ = param_.color_distort();
	  cv::Mat cv_img_colored = cv_masked_img;
	  if (do_color_distort_ == true) {
		  const int distort_type = Rand(7);
		  switch (distort_type)
		  {
		  case 0:
		  case 1:
		  case 2: // do nothing. Remain to be org image.
			  ; break;
		  case 3: // GaussianBlur
			  cv::GaussianBlur(cv_img_colored, cv_img_colored, cv::Size(3, 3), 1.5, 1.5); break;
		  case 4: // GaussianNoise
		  {
			  cv::Mat gauss_noise(img_height, img_width, CV_32FC(channel_));
			  cv::randn(gauss_noise, 0, 0.0316 * 255);
			  cv::Mat cv_cropped_img_float;
			  cv_img_colored.convertTo(cv_cropped_img_float, CV_32FC(channel_));
			  cv_cropped_img_float += gauss_noise;
			  cv_cropped_img_float.convertTo(cv_img_colored, CV_8UC(channel_));
		  }   break;
		  case 5:
		  case 6: // Brightness and Contrast
		  {
			  cv::RNG uni_rng(time(0));
			  float brightness_ = uni_rng.uniform(-0.055f, 0.055f);
			  float contrast_ = uni_rng.uniform(-0.085f, 0.085f);
			  float slope_k = tan((45 + 44 * contrast_) / 180 * 3.1415926);
			  uchar look_up_tab[256];
			  for (int idx = 0; idx < 256; ++idx)
				  look_up_tab[idx] = cv::saturate_cast<uchar>((idx - 127.5 * (1 - brightness_)) * slope_k + 127.5 * (1 + brightness_));
			  cv::Mat lut_mat(1, 256, CV_8UC1, look_up_tab);

			  const int rand_channel_idx = Rand(4);
			  if (rand_channel_idx == 0) // BGR
				  cv::LUT(cv_img_colored, lut_mat, cv_img_colored);
			  else {
				  std::vector<cv::Mat> img_split;
				  cv::split(cv_img_colored, img_split);

				  if (rand_channel_idx == 1) // B
					  cv::LUT(img_split[0], lut_mat, img_split[0]);
				  else if (rand_channel_idx == 2) // G
					  cv::LUT(img_split[1], lut_mat, img_split[1]);
				  else // R
					  cv::LUT(img_split[2], lut_mat, img_split[2]);

				  cv::merge(img_split, cv_img_colored);
			  }
		  }  break;
		  default:   LOG(ERROR) << "Unknown Color_distort type!"; break;
		  }
	  }

	  //6. fill into size[new_width, new_height] mat.
	  cv::Mat full_img_uchar(new_height, new_width, CV_8UC(channel_));
	  cv_img_colored.copyTo(full_img_uchar(cv::Rect(0, 0, img_width, img_height)));

	  //7. finally, (imgData - mean_) * scale_
	  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	  int top_index;
	  for (int h = 0; h < new_height; ++h) {
		  const uchar* float_ptr = full_img_uchar.ptr<uchar>(h);
		  int img_index = 0;
		  for (int w = 0; w < new_width; ++w) {
			  for (int c = 0; c < channel_; ++c) {
				  top_index = (c * new_height + h) * new_width + w;
				  Dtype pixel = static_cast<Dtype>(float_ptr[img_index++]);
				  CHECK_EQ(isnan(pixel), false);
				  if (has_mean_file) {
					  int mean_index;
					  if (do_mirror) {
						  mean_index = (c * new_height + h) * new_width + (new_width - 1 - w);
					  }
					  else {
						  mean_index = (c * new_height + h) * new_width + w;
					  }					  
					  transformed_data[top_index] =
						  (pixel - img_mean[mean_index]) * scale;
				  }
				  else if (has_mean_values) {
						  transformed_data[top_index] =
							  (pixel - mean_values_[c]) * scale;
				  }
				  else {
						transformed_data[top_index] = pixel * scale;
				  }
			  }
		  }
	  }

	  // optional: if we have segMask_blob
	  if (cv_mask_img.data) {
		  CHECK_EQ(cv_mask_img_scaled.channels(), 1);
		  cv::Mat filled_mask_img_float(new_height, new_width, CV_32FC1);
		  cv_mask_img_scaled.convertTo(cv_mask_img_scaled, CV_32F);
		  cv_mask_img_scaled.copyTo(filled_mask_img_float(cv::Rect(0, 0, img_width, img_height)));

		  // copy into transformed_segMask_blob
		  CHECK_EQ(transformed_segMask_blob->count(), filled_mask_img_float.total());
		  Dtype* transformed_segMask_data = transformed_segMask_blob->mutable_cpu_data();
		  auto cv_mask_img_iter = filled_mask_img_float.begin<float>();
		  for (int i = 0; i < transformed_segMask_blob->count(); ++i)
			  *transformed_segMask_data++ = Dtype(*cv_mask_img_iter++);
	  }
  }

#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_h && crop_w) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_h, crop_w);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_h && crop_w) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN && !param_.center_crop()) {
      h_off = Rand(input_height - crop_h + 1);
      w_off = Rand(input_width - crop_w + 1);
    } else {
      h_off = (input_height - crop_h) / 2;
      w_off = (input_width - crop_w) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_h)? crop_h : datum_height;
  shape[3] = (crop_w)? crop_w : datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h)? crop_h : img_height;
  shape[3] = (crop_w)? crop_w : img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && (param_.crop_size() || param_.crop_h() || param_.crop_w()) && !param_.center_crop())
	  || param_.color_distort() || param_.scaled_resize() || (param_.mask_size_max() || param_.mask_size_min());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
