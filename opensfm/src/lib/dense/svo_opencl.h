#pragma once

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/opencl_utils.h>
#include <dense/svo.h>
#include <foundation/types.h>

#include <cstdint>

namespace dense {

// Must match kernel-side VoxelSlot exactly (48 bytes).
struct GPUVoxelSlot {
  uint32_t key_ab;   // packed (x+32768)<<16 | (y+32768), EMPTY = 0xFFFFFFFF
  int32_t key_c;     // z coordinate
  int32_t ready;     // initialisation flag
  int32_t count;     // observation count
  int32_t sum_tsdf;  // fixed-point TSDF accumulator
  int32_t sum_nx;    // fixed-point normal.x accumulator
  int32_t sum_ny;
  int32_t sum_nz;
  int32_t sum_r;  // color accumulator
  int32_t sum_g;
  int32_t sum_b;
  int32_t sum_weight;  // accumulated confidence weight (WEIGHT_SCALE units)
};
static_assert(sizeof(GPUVoxelSlot) == 48, "GPUVoxelSlot must be 48 bytes");

// Must match kernel-side CountSlot exactly (12 bytes).
struct GPUCountSlot {
  uint32_t key_ab;
  int32_t key_c;
  int32_t ready;
};
static_assert(sizeof(GPUCountSlot) == 12, "GPUCountSlot must be 12 bytes");

// Must match kernel-side SVOCamera exactly (144 bytes).
// All matrices stored in row-major order.
struct SVOCameraGPU {
  float Kinv[9];
  float Rinv[9];
  float R[9];
  float t[3];
  float cam_pos[3];
  float _pad[3];
};
static_assert(sizeof(SVOCameraGPU) == 144, "SVOCameraGPU must be 144 bytes");

static constexpr int kFPScale = 32768;
static constexpr uint32_t kEmptyKey = 0xFFFFFFFFu;

// GPU-accelerated TSDF integration into an open-addressing hash table.
//
// Usage:
//   SVOIntegratorCL integrator(device_idx);
//   integrator.Initialize(capacity);
//   for (each view) integrator.Integrate(...);
//   VoxelMap voxels = integrator.Download();
class SVOIntegratorCL {
 public:
  explicit SVOIntegratorCL(int device_idx = 0);

  // Allocate the GPU hash table.  |capacity| is rounded up to next power of 2.
  void Initialize(uint32_t capacity);

  // Integrate a single depthmap into the hash table on device.
  // Optional bbox_min/bbox_max clip integration to a sub-volume
  // (voxel integer coordinates).
  void Integrate(const Mat3f& K, const Mat3f& R, const Vec3f& t,
                 const float* depth, int rows, int cols, const float* normal,
                 const uint8_t* color, const uint8_t* mask, const float* weight,
                 float voxel_size, float trunc_dist,
                 const Eigen::Vector3i* bbox_min = nullptr,
                 const Eigen::Vector3i* bbox_max = nullptr);

  // Download the GPU hash table and convert to a CPU-side VoxelMap.
  VoxelMap Download() const;

  // Extract surface points directly on GPU — finds TSDF zero-crossings
  // in the hash table and interpolates position/normal/color.
  // Much faster than Download() + SVO::ExtractPoints() because only
  // the surface points (~5-10% of voxels) are transferred back.
  void ExtractPoints(float min_weight, float voxel_size,
                     std::vector<Vec3f>* points, std::vector<Vec3f>* normals,
                     std::vector<Vec3<uint8_t>>* colors);

  // --- Counting pass (lightweight dry-run) ---
  // Allocate a compact counting table (12 bytes/slot) to count unique
  // voxels across all views without accumulating TSDF / color data.
  void InitializeCounting(uint32_t capacity);

  // Run a count-only kernel for one depthmap (same ray-march, key-only
  // insertion into the counting table).
  // Optional bbox_min/bbox_max clip counting to a sub-volume.
  void Count(const Mat3f& K, const Mat3f& R, const Vec3f& t, const float* depth,
             int rows, int cols, const uint8_t* mask, float voxel_size,
             float trunc_dist, const Eigen::Vector3i* bbox_min = nullptr,
             const Eigen::Vector3i* bbox_max = nullptr);

  // Read back the unique voxel count from the GPU counter.
  uint32_t GetUniqueCount() const;

  uint32_t capacity() const { return capacity_; }

 private:
  void BuildKernels();
  void EnsureFrameBuffers(int rows, int cols, bool has_normal, bool has_color,
                          bool has_mask, bool has_weight);

  int device_idx_;
  uint32_t capacity_ = 0;
  uint32_t capacity_mask_ = 0;

  cl::Program program_;
  cl::Kernel k_clear_;
  cl::Kernel k_integrate_;
  bool kernels_built_ = false;

  cl::Buffer cl_table_;
  cl::Buffer cl_camera_;
  cl::Buffer cl_depth_;
  cl::Buffer cl_normal_;
  cl::Buffer cl_color_;
  cl::Buffer cl_mask_;
  cl::Buffer cl_weight_;
  cl::Buffer cl_dummy_;  // 1-byte placeholder for null pointers

  // Counting pass buffers.
  cl::Kernel k_count_clear_;
  cl::Kernel k_count_;
  cl::Buffer cl_count_table_;
  cl::Buffer cl_counter_;
  uint32_t count_capacity_ = 0;
  uint32_t count_mask_ = 0;

  // Extraction kernel.
  cl::Kernel k_extract_;

  size_t depth_bytes_ = 0;
  size_t normal_bytes_ = 0;
  size_t color_bytes_ = 0;
  size_t mask_bytes_ = 0;
  size_t weight_bytes_ = 0;
};

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
