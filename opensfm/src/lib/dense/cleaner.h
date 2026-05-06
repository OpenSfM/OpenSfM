#pragma once

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/fuser.h>
#include <dense/opencl_utils.h>

#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace dense {

/// Standalone GPU depthmap cleaner.
class GPUDepthmapCleaner {
 public:
  GPUDepthmapCleaner();
  ~GPUDepthmapCleaner();

  void SetSameDepthThreshold(float t);
  void SetMinConsistentViews(int n);
  void SetCarvingThreshold(float t);
  void SetMaxCarvedViews(int n);
  int AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
              const ImageF& depth);
  cv::Mat Clean(int ref_idx, const std::vector<int>& neighbor_ids);
  /// Remove all views and release GPU buffers.
  void Clear();
  void SetDevice(int device_idx);

 private:
  struct ViewEntry {
    cv::Mat depth;  // CV_32F — kept as cv::Mat for OpenCL upload.
    Mat3d K;
    Mat3d R;
    Vec3d t;
    int width = 0;
    int height = 0;
  };
  std::vector<ViewEntry> views_;
  float same_depth_threshold_ = 0.01f;
  int min_consistent_views_ = 3;
  float carving_threshold_ = 0.2f;
  int max_carved_views_ = 1;
  int device_idx_ = 0;
  bool kernel_built_ = false;
  cl::Program program_;
  cl::Kernel k_clean_;

  // Persistent GPU resources — reused across Clean() calls.
  cl::Buffer cl_cameras_;
  cl::Buffer cl_clean_depth_;
  size_t cl_cameras_bytes_ = 0;
  size_t cl_clean_depth_bytes_ = 0;

  // Cached image2d_t pool — reallocated only when dimensions change.
  static constexpr int kMaxCleanSources = 16;
  cl::Image2D cl_ref_depth_img_;
  std::vector<cl::Image2D> cl_src_depth_imgs_;
  int cl_img_w_ = 0;
  int cl_img_h_ = 0;
};

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
