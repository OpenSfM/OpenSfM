#include <dense/svo_fuser.h>
#include <dense/svo_opencl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace dense {

SVOFuser::SVOFuser()
    : voxel_size_(0.02f), trunc_factor_(4.0f), min_weight_(3.0f) {}

void SVOFuser::SetVoxelSize(float size) { voxel_size_ = std::max(1e-6f, size); }

void SVOFuser::SetTruncFactor(float factor) {
  trunc_factor_ = std::max(1.0f, factor);
}

void SVOFuser::SetMinWeight(float w) { min_weight_ = std::max(0.0f, w); }

void SVOFuser::SetDevice(int device_idx) { device_idx_ = device_idx; }

void SVOFuser::SetBBox(const Eigen::Vector3f& min_world,
                       const Eigen::Vector3f& max_world) {
  has_bbox_ = true;
  bbox_min_world_ = min_world;
  bbox_max_world_ = max_world;
}

bool SVOFuser::IsGPUAvailable() {
  return opencl::CLContext::Instance().IsAvailable();
}

void SVOFuser::AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                       const ImageF& depth, const PixelData3f& normal,
                       const PixelData3u8& color, const ImageU8& mask,
                       const ImageF& weight, const std::string& name) {
  StoredView sv;
  sv.K = K;
  sv.R = R;
  sv.t = t;
  sv.depth = depth;
  sv.normal = normal;
  sv.color = color;
  sv.mask = mask;
  sv.weight = weight;
  sv.name = name;
  views_.push_back(std::move(sv));
}

uint32_t SVOFuser::CountVoxels() {
  if (views_.empty()) {
    return 0;
  }

  if (!opencl::CLContext::Instance().IsAvailable()) {
    throw std::runtime_error(
        "SVOFuser: GPU (OpenCL) is required but not available");
  }

  const float trunc_dist = trunc_factor_ * voxel_size_;
  const float inv_vs = 1.0f / voxel_size_;
  Eigen::Vector3i bbox_min_v, bbox_max_v;
  if (has_bbox_) {
    bbox_min_v =
        (bbox_min_world_ * inv_vs).array().floor().cast<int>().matrix();
    bbox_max_v = (bbox_max_world_ * inv_vs).array().ceil().cast<int>().matrix();
  }
  const Eigen::Vector3i* bbox_min_ptr = has_bbox_ ? &bbox_min_v : nullptr;
  const Eigen::Vector3i* bbox_max_ptr = has_bbox_ ? &bbox_max_v : nullptr;

  SVOIntegratorCL integrator(device_idx_);

  uint64_t max_pixels = 0;
  for (const auto& sv : views_) {
    uint64_t px = static_cast<uint64_t>(sv.depth.rows()) * sv.depth.cols();
    if (px > max_pixels) {
      max_pixels = px;
    }
  }
  const float spread = std::sqrt(static_cast<float>(views_.size()));
  const float voxels_per_ray = std::ceil(4.0f * trunc_factor_);
  uint64_t count_est = static_cast<uint64_t>(static_cast<float>(max_pixels) *
                                             voxels_per_ray * spread);
  count_est = std::max<uint64_t>(count_est, 1u << 20);
  uint32_t count_cap =
      static_cast<uint32_t>(std::min<uint64_t>(count_est, 1u << 28));

  integrator.InitializeCounting(count_cap);

  for (size_t i = 0; i < views_.size(); ++i) {
    const auto& sv = views_[i];
    const int rows = static_cast<int>(sv.depth.rows());
    const int cols = static_cast<int>(sv.depth.cols());
    const Mat3f Kf = sv.K.cast<float>();
    const Mat3f Rf = sv.R.cast<float>();
    const Vec3f tf = sv.t.cast<float>();
    const uint8_t* mask_ptr = (sv.mask.size() > 0) ? sv.mask.data() : nullptr;
    integrator.Count(Kf, Rf, tf, sv.depth.data(), rows, cols, mask_ptr,
                     voxel_size_, trunc_dist, bbox_min_ptr, bbox_max_ptr);
  }

  uint32_t n_unique = integrator.GetUniqueCount();
  std::cerr << "[SVOFuser] Counted " << n_unique << " unique voxels\n";
  last_voxel_count_ = n_unique;
  return n_unique;
}

void SVOFuser::Fuse(std::vector<Vec3f>* fused_points,
                    std::vector<Vec3f>* fused_normals,
                    std::vector<Vec3<uint8_t>>* fused_colors) {
  fused_points->clear();
  fused_normals->clear();
  fused_colors->clear();

  if (views_.empty()) {
    return;
  }

  if (last_voxel_count_ == 0) {
    throw std::runtime_error(
        "SVOFuser::Fuse: CountVoxels() must be called before Fuse()");
  }

  if (!opencl::CLContext::Instance().IsAvailable()) {
    throw std::runtime_error(
        "SVOFuser: GPU (OpenCL) is required but not available");
  }

  const float trunc_dist = trunc_factor_ * voxel_size_;

  // Convert world-space bbox to voxel integer coordinates once.
  const float inv_vs = 1.0f / voxel_size_;
  Eigen::Vector3i bbox_min_v, bbox_max_v;
  if (has_bbox_) {
    bbox_min_v =
        (bbox_min_world_ * inv_vs).array().floor().cast<int>().matrix();
    bbox_max_v = (bbox_max_world_ * inv_vs).array().ceil().cast<int>().matrix();
  }
  const Eigen::Vector3i* bbox_min_ptr = has_bbox_ ? &bbox_min_v : nullptr;
  const Eigen::Vector3i* bbox_max_ptr = has_bbox_ ? &bbox_max_v : nullptr;

  // Allocate at 2× the counted size for ~50% load factor.
  uint32_t capacity = static_cast<uint32_t>(
      std::min<uint64_t>(static_cast<uint64_t>(last_voxel_count_) * 2,
                         std::numeric_limits<uint32_t>::max()));
  capacity = std::max<uint32_t>(capacity, 1u << 20);
  std::cerr << "[SVOFuser] Voxel count " << last_voxel_count_ << " → capacity "
            << capacity << "\n";

  SVOIntegratorCL integrator(device_idx_);
  integrator.Initialize(capacity);

  for (size_t i = 0; i < views_.size(); ++i) {
    const auto& sv = views_[i];
    const int rows = static_cast<int>(sv.depth.rows());
    const int cols = static_cast<int>(sv.depth.cols());
    const Mat3f Kf = sv.K.cast<float>();
    const Mat3f Rf = sv.R.cast<float>();
    const Vec3f tf = sv.t.cast<float>();
    const float* normal_ptr =
        (sv.normal.size() > 0) ? sv.normal.data() : nullptr;
    const uint8_t* color_ptr =
        (sv.color.size() > 0) ? sv.color.data() : nullptr;
    const uint8_t* mask_ptr = (sv.mask.size() > 0) ? sv.mask.data() : nullptr;
    const float* weight_ptr =
        (sv.weight.size() > 0) ? sv.weight.data() : nullptr;

    integrator.Integrate(Kf, Rf, tf, sv.depth.data(), rows, cols, normal_ptr,
                         color_ptr, mask_ptr, weight_ptr, voxel_size_,
                         trunc_dist, bbox_min_ptr, bbox_max_ptr);
  }

  {
    integrator.ExtractPoints(min_weight_, voxel_size_, fused_points,
                             fused_normals, fused_colors);
    std::cerr << "[SVOFuser] GPU ExtractPoints: " << fused_points->size()
              << " surface points\n";
  }
}

}  // namespace dense
