#pragma once

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/cleaner.h>
#include <dense/estimator.h>
#include <dense/opencl_utils.h>

#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace dense {

/// Fused point cloud result from a cluster.
struct ClusterFusedResult {
  std::vector<Vec3f> points;
  std::vector<Vec3f> normals;
  std::vector<Vec3<uint8_t>> colors;
};

/// Parameters for depthmap cleaning.
struct CleanParams {
  float same_depth_threshold = 0.01f;
  int min_consistent_views = 3;
};

/// Parameters for depthmap fusion.
struct FuseParams {
  int min_num_consistent = 3;
  float max_reproj_error = 2.0f;
  float max_depth_error = 0.01f;
  float max_normal_error = 10.0f;
  int border_margin = 5;
  int num_threads = 4;
  int sor_knn = 0;
  float sor_stddev_factor = 2.5f;
};

class DepthmapClusterEstimator {
 public:
  DepthmapClusterEstimator();
  ~DepthmapClusterEstimator();

  void SetParams(const DepthmapParams& params);
  void SetGeomConsistencyWeight(float weight);
  void SetDevice(int device_idx);
  int BeginRefView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                   const ImageU8& image, float depth_min, float depth_max);
  void AddSourceView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                     const ImageU8& image);
  void SetSfMPoints(
      const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points);
  void AddGeomLink(int ref_idx, int source_view_idx, int from_ref_idx);
  void Run(std::vector<DepthmapResult>* results);
  void Clear();
  int NumRefViews() const;
  static size_t GpuMemoryBytes();

 private:
  struct RefEntry {
    DepthmapEstimator estimator;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    std::vector<std::pair<int, int>> geom_links;
    Mat3d K;
    Mat3d R;
    Vec3d t;
  };

  std::vector<RefEntry> refs_;
  DepthmapParams params_;
  float geom_weight_ = 0.0f;
  int device_idx_ = 0;
};

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
