#pragma once

#include <dense/fuser.h>  // reuse ImageF, PixelData3f, etc.

#include <memory>
#include <string>
#include <vector>

namespace dense {

// SVOFuser: TSDF-based depthmap fusion on GPU via OpenCL.
//
// Usage: AddView() → CountVoxels() → Fuse().
// CountVoxels() must be called before Fuse() so the caller can check
// the voxel budget and split if necessary.

class SVOFuser {
 public:
  SVOFuser();

  // ------- Parameters -------
  void SetVoxelSize(float size);
  void SetTruncFactor(float factor);
  void SetMinWeight(float w);
  void SetDevice(int device_idx);
  void SetBBox(const Eigen::Vector3f& min_world,
               const Eigen::Vector3f& max_world);
  static bool IsGPUAvailable();

  // ------- Data -------
  void AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
               const ImageF& depth, const PixelData3f& normal,
               const PixelData3u8& color, const ImageU8& mask,
               const ImageF& weight = ImageF(), const std::string& name = "");

  // ------- Run -------
  // Count unique voxels via a GPU counting pass.
  // Must be called before Fuse().
  uint32_t CountVoxels();

  // Integrate all views and extract surface points.
  // Throws if CountVoxels() was not called first.
  void Fuse(std::vector<Vec3f>* fused_points, std::vector<Vec3f>* fused_normals,
            std::vector<Vec3<uint8_t>>* fused_colors);

 private:
  // Stored views (we keep the Eigen data alive in these buffers).
  struct StoredView {
    Mat3d K, R;
    Vec3d t;
    ImageF depth;
    PixelData3f normal;
    PixelData3u8 color;
    ImageU8 mask;
    ImageF weight;
    std::string name;
  };
  std::vector<StoredView> views_;

  float voxel_size_;
  float trunc_factor_;
  float min_weight_;
  int device_idx_ = 0;             // OpenCL device index
  uint32_t last_voxel_count_ = 0;  // cached from CountVoxels()
  bool has_bbox_ = false;
  Eigen::Vector3f bbox_min_world_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f bbox_max_world_ = Eigen::Vector3f::Zero();
};

}  // namespace dense
