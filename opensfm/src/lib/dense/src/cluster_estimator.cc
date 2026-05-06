#include <dense/cluster.h>

#ifdef OPENSFM_HAVE_OPENCL

#include <dense/fuser.h>
#include <dense/opencl_kernels.h>

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace dense {

// =====================================================================
// DepthmapClusterEstimator implementation.
// =====================================================================

DepthmapClusterEstimator::DepthmapClusterEstimator() = default;
DepthmapClusterEstimator::~DepthmapClusterEstimator() = default;

void DepthmapClusterEstimator::SetParams(const DepthmapParams& params) {
  params_ = params;
}

void DepthmapClusterEstimator::SetGeomConsistencyWeight(float weight) {
  geom_weight_ = weight;
}

void DepthmapClusterEstimator::SetDevice(int device_idx) {
  device_idx_ = device_idx;
}

int DepthmapClusterEstimator::NumRefViews() const {
  return static_cast<int>(refs_.size());
}

int DepthmapClusterEstimator::BeginRefView(const Mat3d& K, const Mat3d& R,
                                           const Vec3d& t, const ImageU8& image,
                                           float depth_min, float depth_max) {
  refs_.emplace_back();
  auto& entry = refs_.back();
  entry.depth_min = depth_min;
  entry.depth_max = depth_max;
  entry.K = K;
  entry.R = R;
  entry.t = t;
  entry.estimator.AddView(K, R, t, image);
  return static_cast<int>(refs_.size()) - 1;
}

void DepthmapClusterEstimator::AddSourceView(const Mat3d& K, const Mat3d& R,
                                             const Vec3d& t,
                                             const ImageU8& image) {
  if (refs_.empty()) {
    throw std::runtime_error("DepthmapClusterEstimator: no ref view started");
  }
  refs_.back().estimator.AddView(K, R, t, image);
}

void DepthmapClusterEstimator::SetSfMPoints(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points) {
  if (refs_.empty()) {
    throw std::runtime_error("DepthmapClusterEstimator: no ref view started");
  }
  refs_.back().estimator.SetSfMPoints(points);
}

void DepthmapClusterEstimator::AddGeomLink(int ref_idx, int source_view_idx,
                                           int from_ref_idx) {
  if (ref_idx < 0 || ref_idx >= static_cast<int>(refs_.size())) {
    throw std::runtime_error(
        "DepthmapClusterEstimator::AddGeomLink: invalid ref_idx");
  }
  refs_[ref_idx].geom_links.emplace_back(source_view_idx, from_ref_idx);
}

void DepthmapClusterEstimator::Run(std::vector<DepthmapResult>* results) {
  const int K = static_cast<int>(refs_.size());
  if (K == 0) {
    throw std::runtime_error("DepthmapClusterEstimator: no reference views");
  }

  results->resize(K);

  // Prepare all estimators.
  int n_levels = 0;
  for (auto& entry : refs_) {
    DepthmapParams p = params_;
    p.depth_min = entry.depth_min;
    p.depth_max = entry.depth_max;
    entry.estimator.SetDevice(device_idx_);
    entry.estimator.SetParams(p);
    entry.estimator.SetGeomConsistencyWeight(geom_weight_);
    int nl = entry.estimator.Prepare();
    n_levels = std::max(n_levels, nl);
  }

  std::cerr << "[DepthmapClusterEstimator] " << K << " ref views, " << n_levels
            << " levels, geom_weight=" << geom_weight_ << "\n";

  // prev_depths[i] = intermediate depth from previous level for ref i.
  std::vector<ImageF> prev_depths(K);

  for (int level = 0; level < n_levels; ++level) {
    for (int i = 0; i < K; ++i) {
      auto& entry = refs_[i];

      // Feed previous-level depths from other cluster members.
      if (level > 0 && geom_weight_ > 0.0f && !entry.geom_links.empty()) {
        std::vector<std::pair<int, ImageF>> pd_entries;
        for (const auto& link : entry.geom_links) {
          int src_idx = link.first;
          int from_idx = link.second;
          if (from_idx >= 0 && from_idx < K &&
              prev_depths[from_idx].size() > 0) {
            pd_entries.emplace_back(src_idx, prev_depths[from_idx]);
          }
        }
        if (!pd_entries.empty()) {
          entry.estimator.SetPreviousDepths(pd_entries);
        } else {
          entry.estimator.ClearPreviousDepths();
        }
      } else {
        entry.estimator.ClearPreviousDepths();
      }

      entry.estimator.RunLevel(level, n_levels, &(*results)[i]);

      // Save intermediate depth for next level.
      if (level < n_levels - 1) {
        prev_depths[i] = (*results)[i].depth;
        entry.estimator.ReleaseGpuBuffers();
      } else {
        entry.estimator.ReleaseBuffers();
      }
    }
  }

  // Free intermediate depth maps no longer needed.
  prev_depths.clear();
  prev_depths.shrink_to_fit();
}

void DepthmapClusterEstimator::Clear() { refs_.clear(); }

size_t DepthmapClusterEstimator::GpuMemoryBytes() {
  auto& ctx = opencl::CLContext::Instance();
  if (!ctx.IsAvailable()) {
    return 0;
  }
  size_t max_mem = 0;
  for (int i = 0; i < ctx.NumDevices(); ++i) {
    max_mem = std::max(max_mem, ctx.Device(i).GlobalMemSize());
  }
  return max_mem;
}

}  // namespace dense

#endif  // OPENSFM_HAVE_OPENCL
