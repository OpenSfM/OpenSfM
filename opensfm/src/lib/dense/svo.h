#pragma once

#include <absl/container/flat_hash_map.h>
#include <foundation/types.h>

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace dense {

// Sparse Voxel Octree for TSDF-based depthmap fusion.
//
// Design: a flat hash-map of voxels keyed by integer coordinate.  Each voxel
// stores a truncated signed distance (TSDF), weight, accumulated color and
// normal — all packed into 14 bytes.
// Only voxels touched by at least one depthmap ray are allocated.
//
// Memory budget (per occupied voxel):
//   SVOVoxel payload:  14 bytes  (was 32 with all-float fields)
//   absl::flat_hash_map overhead: ~1 byte control + inline key/value
//   Total effective:   ~31 bytes (vs ~108 with std::unordered_map + floats)

// ---- snorm16 pack/unpack for values in [-1, 1] ----
static constexpr float kSnorm16Scale = 32767.0f;
static constexpr float kSnorm16InvScale = 1.0f / 32767.0f;

inline int16_t PackSnorm16(float v) {
  return static_cast<int16_t>(
      std::clamp(v * kSnorm16Scale, -kSnorm16Scale, kSnorm16Scale));
}
inline float UnpackSnorm16(int16_t v) {
  return static_cast<float>(v) * kSnorm16InvScale;
}

// Per-voxel payload — tightly packed (14 bytes, alignof=2).
struct SVOVoxel {
  int16_t tsdf = 0;                // signed distance, snorm16 [-1,1]
  uint16_t weight = 0;             // integration count (max 65535)
  int16_t nx = 0, ny = 0, nz = 0;  // weighted-avg normal, snorm16
  uint8_t r = 0, g = 0, b = 0;     // weighted-avg color
  uint8_t _pad = 0;                // align to 14 bytes (sizeof int16_t × 7)
};
static_assert(sizeof(SVOVoxel) == 14, "SVOVoxel must be 14 bytes");

// 3-component integer voxel coordinate.
struct VoxelCoord {
  int32_t x, y, z;
  bool operator==(const VoxelCoord& o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct VoxelCoordHash {
  size_t operator()(const VoxelCoord& c) const {
    // FNV-1a-style combine.
    size_t h = 14695981039346656037ULL;
    h ^= static_cast<size_t>(c.x);
    h *= 1099511628211ULL;
    h ^= static_cast<size_t>(c.y);
    h *= 1099511628211ULL;
    h ^= static_cast<size_t>(c.z);
    h *= 1099511628211ULL;
    return h;
  }
};

using VoxelMap = absl::flat_hash_map<VoxelCoord, SVOVoxel, VoxelCoordHash>;

// Weight scale: per-pixel confidence [0,1] is quantized to [0, kWeightScale]
// so fractional weights accumulate correctly in uint16 SVOVoxel::weight.
static constexpr float kWeightScale = 128.0f;

class SparseVoxelOctree {
 public:
  // voxel_size: side length of each cubic voxel in world units.
  // trunc_factor: truncation distance = trunc_factor * voxel_size.
  explicit SparseVoxelOctree(float voxel_size, float trunc_factor = 4.0f);

  float voxel_size() const { return voxel_size_; }
  float truncation_dist() const { return trunc_dist_; }
  size_t num_voxels() const { return voxels_.size(); }

  // Integrate a single depthmap into the volume.
  // K: 3×3 intrinsic, R: 3×3 rotation, t: 3×1 translation (world→cam).
  // depth: (H,W) float depthmap, normal: (3, H*W), color: (3, H*W).
  // mask: (H,W) uint8 validity mask (0=invalid). Can be empty (all valid).
  // weight: (H,W) float confidence [0,1]. nullptr = all 1.0.
  // bbox_min/bbox_max: optional voxel-coordinate clipping bounds.
  void Integrate(const Mat3f& K, const Mat3f& R, const Vec3f& t,
                 const float* depth, int rows, int cols, const float* normal,
                 const uint8_t* color, const uint8_t* mask,
                 const float* weight = nullptr,
                 const Eigen::Vector3i* bbox_min = nullptr,
                 const Eigen::Vector3i* bbox_max = nullptr);

  // Extract zero-crossing surface points from the TSDF.
  // min_weight: ignore voxels with weight below this.
  void ExtractPoints(float min_weight, std::vector<Vec3f>* points,
                     std::vector<Vec3f>* normals,
                     std::vector<Vec3<uint8_t>>* colors) const;

  // Clear all voxels.
  void Clear() { voxels_.clear(); }

  // Import pre-computed voxels (e.g. from GPU integration).
  void ImportVoxels(VoxelMap&& voxels) { voxels_ = std::move(voxels); }

 private:
  VoxelCoord WorldToVoxel(const Vec3f& p) const;
  Vec3f VoxelCenter(const VoxelCoord& vc) const;

  float voxel_size_;
  float inv_voxel_size_;
  float trunc_dist_;
  VoxelMap voxels_;
};

}  // namespace dense
