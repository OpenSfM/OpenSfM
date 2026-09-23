#include <algorithm>
#include <stdexcept>

#include <dense/svo_fuser.h>
#include <foundation/logging.h>

namespace dense {

// Minimal placeholder definition for the OpenCL-disabled build.
// This allows std::unique_ptr<SVOIntegratorCL> to be used in the class
// layout without requiring the OpenCL implementation.
class SVOIntegratorCL {};

SVOFuser::SVOFuser()
    : voxel_size_(0.02f),
      trunc_factor_(4.0f),
      min_weight_(3.0f),
      decimate_flat_(1),
      edge_threshold_(0.15f),
      min_count_(2),
      relative_min_weight_(0.0f),
      dsm_wall_cull_nz_(0.3f),
      num_levels_(1),
      device_idx_(0),
      last_voxel_count_(0),
      has_bbox_(false),
      bbox_min_world_(Eigen::Vector3f::Zero()),
      bbox_max_world_(Eigen::Vector3f::Zero()) {}

SVOFuser::~SVOFuser() = default;

void SVOFuser::SetVoxelSize(float size) { voxel_size_ = std::max(1e-6f, size); }

void SVOFuser::SetTruncFactor(float factor) {
  trunc_factor_ = std::max(1.0f, factor);
}

void SVOFuser::SetMinWeight(float w) { min_weight_ = std::max(0.0f, w); }

void SVOFuser::SetDevice(int device_idx) { device_idx_ = device_idx; }

uint32_t SVOFuser::Capacity() const { return 0u; }

void SVOFuser::ReleaseRefineBuffers() {}

void SVOFuser::SetNumLevels(int n) { num_levels_ = std::max(1, n); }

void SVOFuser::SetDecimateFat(uint32_t n) { decimate_flat_ = std::max(1u, n); }

void SVOFuser::SetEdgeThreshold(float t) {
  edge_threshold_ = std::max(0.0f, std::min(1.0f, t));
}

void SVOFuser::SetMinCount(int n) { min_count_ = std::max(1, n); }

void SVOFuser::SetRelativeMinWeight(float w) {
  relative_min_weight_ = std::max(0.0f, w);
}

void SVOFuser::SetDSMWallCullNz(float nz) {
  dsm_wall_cull_nz_ = std::min(std::max(0.0f, nz), 1.0f);
}

void SVOFuser::SetBBox(const Eigen::Vector3f& min_world,
                       const Eigen::Vector3f& max_world) {
  has_bbox_ = true;
  bbox_min_world_ = min_world;
  bbox_max_world_ = max_world;
}

bool SVOFuser::IsGPUAvailable() { return false; }

void SVOFuser::AddView(
    const Mat3d& K, const Mat3d& R, const Vec3d& t,
    Eigen::Map<const ImageF> depth,
    Eigen::Map<const PixelData3f> normal,
    Eigen::Map<const PixelData3u8> color,
    Eigen::Map<const ImageU8> mask, Eigen::Map<const ImageF> weight,
    const std::string& name) {
  views_.emplace_back(K, R, t, depth, normal, color, mask, weight, name);
}

uint32_t SVOFuser::CountVoxels() {
  if (views_.empty()) {
    return 0u;
  }
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

void SVOFuser::Fuse() {
  if (views_.empty()) {
    return;
  }
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

void SVOFuser::RefineGeometry(
    int iters, float lambda_reg,
    const std::map<std::string, std::vector<std::string>>& neighbors,
    float lambda_anchor, float early_stop_rel) {
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

void SVOFuser::BakeColors(std::vector<Vec3f>& points,
                          std::vector<Vec3f>& normals,
                          std::vector<Vec3<uint8_t>>* colors, int n_final,
                          int irls_iters,
                          const std::vector<uint8_t>* relax_occ,
                          const std::vector<float>* dsm_occ, int dsm_w,
                          int dsm_h, float dsm_origin_x, float dsm_origin_y,
                          float dsm_gsd, float dsm_max_z,
                          std::vector<uint8_t>* out_sharp) {
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

void SVOFuser::PruneByVisibility(int iterations, float carve_margin,
                                 int carve_threshold, int support_min) {
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

void SVOFuser::ExtractPoints(std::vector<Vec3f>* fused_points,
                             std::vector<Vec3f>* fused_normals,
                             std::vector<Vec3<uint8_t>>* fused_colors) {
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

void SVOFuser::RenderDSMOrtho(float origin_x, float origin_y, float gsd,
                              int width, int height, float z_min,
                              float z_max, std::vector<float>* dsm_out,
                              std::vector<uint8_t>* ortho_out,
                              std::vector<float>* normals_out) {
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

void SVOFuser::ExtractMesh(std::vector<Vec3f>* verts,
                           std::vector<Vec3f>* normals,
                           std::vector<int>* tris) {
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

void SVOFuser::Fuse(std::vector<Vec3f>* fused_points,
                    std::vector<Vec3f>* fused_normals,
                    std::vector<Vec3<uint8_t>>* fused_colors) {
  throw std::runtime_error("SVOFuser: OpenCL is not available");
}

}  // namespace dense
