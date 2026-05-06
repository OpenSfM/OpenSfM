#pragma once

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/fuser.h>
#include <dense/opencl_utils.h>

#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace dense {

/// Parameters for the PatchMatch PatchMatch algorithm.
struct DepthmapParams {
  int max_iterations = 3;
  int patch_size = 11;
  int num_images = 17;
  int max_image_size = 3200;
  int radius_increment = 2;
  float sigma_spatial = 5.0f;
  float sigma_color = 3.0f / 255.0f;
  int top_k = 4;
  float census_weight = 0.3f;
  float depth_min = 0.0f;
  float depth_max = 1.0f;
  int hierarchy_levels = 1;
  float smooth_weight = 0.0f;
};

/// Result of an PatchMatch depth estimation.
struct DepthmapResult {
  ImageF depth;        // (H, W)
  PixelData3f normal;  // (3, H*W)
  ImageF cost;         // (H, W)
  ImageF confidence;   // (H, W) — Bayesian surface confidence [0, 1]
};

class DepthmapEstimator {
 public:
  DepthmapEstimator();
  ~DepthmapEstimator();

  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               const ImageU8& image);
  void SetSfMPoints(
      const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points);
  void SetDepthRange(float min_depth, float max_depth);
  void SetParams(const DepthmapParams& params);
  void SetGeomConsistencyWeight(float weight);
  void SetDevice(int device_idx);
  void SetPreviousDepths(const std::vector<std::pair<int, ImageF>>& depths);
  void ClearPreviousDepths();
  int Prepare();
  void RunLevel(int level, int total_levels, DepthmapResult* result);
  void Run(DepthmapResult* result);

  void ReleaseGpuBuffers();
  void ReleaseBuffers();

 private:
  void InitOpenCL();
  void BuildKernels();
  void UploadData();
  void RandomInit(int width, int height);
  void PriorReinit(int width, int height);
  void RunIteration(int iter, int width, int height);
  void RunCheckerboardFilter(int width, int height);
  void BuildImagePyramid(float scale, std::vector<cv::Mat>& scaled_images,
                         std::vector<Mat3d>& scaled_Ks) const;
  void UpsampleDepthNormal(const DepthmapResult& coarse, int src_w, int src_h,
                           int dst_w, int dst_h);
  void ComputeSfMPlanarPrior(int width, int height);
  void UploadPreviousDepths(int width, int height);
  void ReadBackResults(DepthmapResult* result, int width, int height,
                       bool apply_median = true);

  // Images stay as cv::Mat for OpenCL upload and cv::resize.
  std::vector<cv::Mat> images_;
  std::vector<Mat3d> Ks_;
  std::vector<Mat3d> Rs_;
  std::vector<Vec3d> ts_;

  std::vector<Vec3d> sfm_points_;

  std::vector<std::pair<int, ImageF>> prev_depth_entries_;

  float geom_weight_ = 0.0f;

  std::vector<cv::Mat> orig_images_;
  std::vector<Mat3d> orig_Ks_;
  DepthmapResult prev_level_result_;
  int prev_level_w_ = 0;
  int prev_level_h_ = 0;
  bool prepared_ = false;

  DepthmapParams params_;
  int device_idx_ = 0;

  bool cl_initialised_ = false;
  cl::Program program_;

  cl::Kernel k_random_init_;
  cl::Kernel k_patchmatch_red_;
  cl::Kernel k_patchmatch_black_;
  cl::Kernel k_upsample_;
  cl::Kernel k_prior_reinit_;
  cl::Kernel k_checkerboard_filter_red_;
  cl::Kernel k_checkerboard_filter_black_;

  std::vector<cl::Image2D> cl_images_;
  cl::Buffer cl_cameras_;
  cl::Buffer cl_plane_hypotheses_;
  cl::Buffer cl_costs_;
  cl::Buffer cl_rand_states_;
  cl::Buffer cl_selected_views_;
  cl::Buffer cl_prior_planes_;
  cl::Buffer cl_plane_masks_;
  cl::Buffer cl_prev_depths_;
  cl_uint cl_prev_depth_mask_ = 0u;

  // Host-side prior data for confidence computation.
  ImageF prior_depths_;    // (H, W) — prior depth per pixel
  ImageF triangle_areas_;  // (H, W) — Delaunay triangle area per pixel
  ImageU8 prior_mask_;     // (H, W) — 1 where prior exists

  std::mt19937 rng_{std::random_device{}()};
};

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
