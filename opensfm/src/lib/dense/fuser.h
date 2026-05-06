#pragma once

#include <foundation/types.h>

#include <Eigen/Core>
#include <cstdint>
#include <vector>

namespace dense {

// ---- Image-like type aliases for the dense module ----
// Row-major storage matches cv::Mat / numpy (C-order) memory layout.
using ImageF =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageI =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ImageU8 =
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Per-pixel 3-component data stored column-wise: each column is one pixel's
// Vec3{f,u8}.  Shape: (3, H*W).  ColMajor storage gives interleaved layout
// [x0,y0,z0, x1,y1,z1, ...] matching cv::Mat CV_32FC3 / CV_8UC3.
using PixelData3f = Eigen::Matrix<float, 3, Eigen::Dynamic>;
using PixelData3u8 = Eigen::Matrix<uint8_t, 3, Eigen::Dynamic>;

// ---- Depthmap fusion (CPU, multi-threaded) ----

class DepthmapFuser {
 public:
  DepthmapFuser();

  // ------- Parameters -------
  void SetMinNumConsistent(int n);
  void SetMaxReprojError(float px);
  void SetMaxDepthError(float ratio);
  void SetMaxNormalError(float degrees);
  void SetBorderMargin(int px);
  void SetNumThreads(int n);
  void SetSORParams(int knn, float stddev_factor);
  void SetBehindDepthFactor(float f);

  // ------- Data -------
  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               const ImageF& depth, const PixelData3f& normal,
               const PixelData3u8& color, const ImageU8& mask,
               const std::vector<int>& neighbor_ids, bool primary = true);

  // ------- Run -------
  void Fuse(std::vector<Vec3f>* fused_points, std::vector<Vec3f>* fused_normals,
            std::vector<Vec3<uint8_t>>* fused_colors);

 private:
  struct View {
    ImageF depth;        // (H, W)
    PixelData3f normal;  // (3, H*W)
    PixelData3u8 color;  // (3, H*W)
    ImageU8 mask;        // (H, W) — 0 = invalid (out-of-bounds), >0 = valid
    ImageU8 fused;       // (H, W) — 0 = unfused, 1 = fused
    Mat3d K, R;
    Vec3d t;
    std::vector<int> neighbors;
    bool primary = true;
  };
  std::vector<View> views_;

  int min_num_consistent_;
  float max_reproj_error_sq_;
  float max_depth_error_;
  float min_cos_normal_error_;
  int border_margin_;
  int num_threads_;
  int sor_knn_;
  float sor_stddev_factor_;
  float behind_depth_factor_;
};

}  // namespace dense
