/* Originally modified from COLMAP and adapted to OpenSfM by Piero Toffanin
The COLMAP library is licensed under the new BSD license.
Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
https://github.com/colmap/colmap */

#include <features/dspsift.h>

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Core>

extern "C" {
#include <time.h>
#include <vl/covdet.h>
#include <vl/sift.h>
}

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptorsFloat;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptors;
typedef std::vector<float> FeatureKeypoints;

template <typename T1, typename T2>
T2 TruncateCast(const T1 value) {
  return static_cast<T2>(std::min(
      static_cast<T1>(std::numeric_limits<T2>::max()),
      std::max(static_cast<T1>(std::numeric_limits<T2>::min()), value)));
}

FeatureDescriptors FeatureDescriptorsToUnsignedByte(
    const Eigen::Ref<const FeatureDescriptorsFloat>& descriptors) {
  FeatureDescriptors descriptors_unsigned_byte(descriptors.rows(),
                                               descriptors.cols());
  for (Eigen::MatrixXf::Index r = 0; r < descriptors.rows(); ++r) {
    for (Eigen::MatrixXf::Index c = 0; c < descriptors.cols(); ++c) {
      const float scaled_value = std::round(512.0f * descriptors(r, c));
      descriptors_unsigned_byte(r, c) =
          TruncateCast<float, uint8_t>(scaled_value);
    }
  }
  return descriptors_unsigned_byte;
}

namespace features {

py::tuple dspsift(foundation::pyarray_f image, float peak_threshold,
                float edge_threshold, int target_num_features, 
                bool feature_root, bool domain_size_pooling, bool estimate_affine_shape) {
  if (!image.size()) {
    return py::none();
  }

  FeatureDescriptors descriptors;
  FeatureKeypoints keypoints;
  int keypoints_count = 0;
  const int kpDimension = 4;

  {
    py::gil_scoped_release release;

    double dsp_min_scale = 1.0 / 6.0;
    double dsp_max_scale = 3.0;
    int dsp_num_scales = 10;

    // Setup covariant SIFT detector.
    std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
        vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
    if (!covdet) {
      return py::none();
    }

    const int kMaxOctaveResolution = 1000;

    vl_covdet_set_first_octave(covdet.get(), 0);
    vl_covdet_set_octave_resolution(covdet.get(), 3);
    vl_covdet_set_peak_threshold(covdet.get(), peak_threshold);
    vl_covdet_set_edge_threshold(covdet.get(), edge_threshold);

    vl_covdet_put_image(covdet.get(), image.data(), image.shape(1), image.shape(0));
    // vl_covdet_set_non_extrema_suppression_threshold(covdet.get(), 0);

    // vl_covdet_detect(covdet.get(), target_num_features);
    // int num_features = vl_covdet_get_num_features(covdet.get());

    int num_features = 0;
    while(true){
      int prev_num_features = num_features;
      vl_covdet_detect(covdet.get(), target_num_features);
      num_features = vl_covdet_get_num_features(covdet.get());

      if (num_features < target_num_features && peak_threshold > 0.0001 && prev_num_features < num_features){
        peak_threshold = (peak_threshold * 2.0f) / 3.0f;
        vl_covdet_set_peak_threshold(covdet.get(), peak_threshold);
      }else break;
    }

    if (estimate_affine_shape){
      vl_covdet_extract_affine_shape(covdet.get());
    } else {
      vl_covdet_extract_orientations(covdet.get());
    }

    VlCovDetFeature* features = vl_covdet_get_features(covdet.get());
    

    // Sort features according to detected octave and scale.
    std::sort(
        features,
        features + num_features,
        [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
          if (feature1.o == feature2.o) {
            return feature1.s > feature2.s;
          } else {
            return feature1.o > feature2.o;
          }
        });

    // Copy detected keypoints and clamp when maximum number of features
    // reached.
    int prev_octave_scale_idx = std::numeric_limits<int>::max();
    keypoints_count = std::min(target_num_features, num_features);

    keypoints.resize(kpDimension * keypoints_count);
    int i = 0;
    for (; i < keypoints_count; ++i) {
      keypoints[kpDimension * i + 0] = features[i].frame.x;
      keypoints[kpDimension * i + 1] = features[i].frame.y;

      float det = features[i].frame.a11 * features[i].frame.a22 - features[i].frame.a12 * features[i].frame.a21;
      float size = sqrt(fabs(det));
      float angle = atan2(features[i].frame.a21, features[i].frame.a11) * 180.0f / M_PI;
      keypoints[kpDimension * i + 2] = size;
      keypoints[kpDimension * i + 3] = angle;

      const int octave_scale_idx =
          features[i].o * kMaxOctaveResolution + features[i].s;

      if (octave_scale_idx != prev_octave_scale_idx &&
          (i + 1) >= target_num_features) {
        i++;
        break;
      }

      prev_octave_scale_idx = octave_scale_idx;
    }

    keypoints_count = i;

    // Compute the descriptors for the detected keypoints.
    descriptors.resize(keypoints_count, 128);

    const size_t kPatchResolution = 15;
    const size_t kPatchSide = 2 * kPatchResolution + 1;
    const double kPatchRelativeExtent = 7.5;
    const double kPatchRelativeSmoothing = 1;
    const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
    const double kSigma =
        kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

    std::vector<float> patch(kPatchSide * kPatchSide);
    std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

    float d_min_scale = 1;
    float d_scale_step = 0;
    int d_num_scales = 1;
    if (domain_size_pooling) {
      d_min_scale = dsp_min_scale;
      d_scale_step = (dsp_max_scale - dsp_min_scale) /
                        dsp_num_scales;
      d_num_scales = dsp_num_scales;
    }

    FeatureDescriptorsFloat descriptor(1, 128);
    FeatureDescriptorsFloat scaled_descriptors(d_num_scales, 128);

    std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
        vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
    if (!sift) {
      return py::none();
    }

    vl_sift_set_magnif(sift.get(), 3.0);

    for (int i = 0; i < keypoints_count; ++i) {
      for (int s = 0; s < d_num_scales; ++s) {
        const double dsp_scale = d_min_scale + s * d_scale_step;

        VlFrameOrientedEllipse scaled_frame = features[i].frame;
        scaled_frame.a11 *= dsp_scale;
        scaled_frame.a12 *= dsp_scale;
        scaled_frame.a21 *= dsp_scale;
        scaled_frame.a22 *= dsp_scale;

        vl_covdet_extract_patch_for_frame(covdet.get(),
                                          patch.data(),
                                          kPatchResolution,
                                          kPatchRelativeExtent,
                                          kPatchRelativeSmoothing,
                                          scaled_frame);

        vl_imgradient_polar_f(patchXY.data(),
                              patchXY.data() + 1,
                              2,
                              2 * kPatchSide,
                              patch.data(),
                              kPatchSide,
                              kPatchSide,
                              kPatchSide);

        vl_sift_calc_raw_descriptor(sift.get(),
                                    patchXY.data(),
                                    scaled_descriptors.row(s).data(),
                                    kPatchSide,
                                    kPatchSide,
                                    kPatchResolution,
                                    kPatchResolution,
                                    kSigma,
                                    0);
      }

      if (domain_size_pooling) {
        descriptor = scaled_descriptors.colwise().mean();
      } else {
        descriptor = scaled_descriptors;
      }

      if (!feature_root) {
        descriptor.rowwise().normalize();
      } else {
        for (Eigen::MatrixXf::Index r = 0; r < descriptor.rows(); ++r) {
          descriptor.row(r) *= 1 / descriptor.row(r).lpNorm<1>();
          descriptor.row(r) = descriptor.row(r).array().sqrt();
        }
      }

      descriptors.row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
    }

    // *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
  }

  return py::make_tuple(
    foundation::py_array_from_data(keypoints.data(), keypoints_count, kpDimension),
    foundation::py_array_from_data(descriptors.data(), keypoints_count, 128));
}

}  // namespace features
