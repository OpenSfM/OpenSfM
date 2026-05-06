#include <dense/fuser.h>
#include <foundation/interpolation.h>
#include <vl/kdtree.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dense {

// =====================================================================
// DepthmapFuser
// =====================================================================

namespace {

template <typename T>
T VectorMedian(std::vector<T>& v) {
  if (v.empty()) {
    return T(0);
  }
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + n, v.end());
  return v[n];
}

}  // namespace

DepthmapFuser::DepthmapFuser()
    : min_num_consistent_(3),
      max_reproj_error_sq_(2.0f * 2.0f),
      max_depth_error_(0.01f),
      min_cos_normal_error_(
          std::cos(10.0f * static_cast<float>(M_PI) / 180.0f)),
      border_margin_(5),
      num_threads_(4),
      sor_knn_(12),
      sor_stddev_factor_(2.5f),
      behind_depth_factor_(0.3f) {}

void DepthmapFuser::SetMinNumConsistent(int n) { min_num_consistent_ = n; }

void DepthmapFuser::SetMaxReprojError(float px) {
  max_reproj_error_sq_ = px * px;
}

void DepthmapFuser::SetMaxDepthError(float ratio) { max_depth_error_ = ratio; }

void DepthmapFuser::SetMaxNormalError(float degrees) {
  min_cos_normal_error_ = std::cos(degrees * static_cast<float>(M_PI) / 180.0f);
}

void DepthmapFuser::SetBorderMargin(int px) {
  border_margin_ = std::max(0, px);
}

void DepthmapFuser::SetNumThreads(int n) { num_threads_ = std::max(1, n); }

void DepthmapFuser::SetSORParams(int knn, float stddev_factor) {
  sor_knn_ = std::max(0, knn);
  sor_stddev_factor_ = stddev_factor;
}

void DepthmapFuser::SetBehindDepthFactor(float f) {
  behind_depth_factor_ = std::clamp(f, 0.0f, 1.0f);
}

void DepthmapFuser::AddView(const Mat3d& K, const Mat3d& R, const Vec3d& t,
                            const ImageF& depth, const PixelData3f& normal,
                            const PixelData3u8& color, const ImageU8& mask,
                            const std::vector<int>& neighbor_ids,
                            bool primary) {
  View v;
  v.K = K;
  v.R = R;
  v.t = t;
  v.depth = depth;
  v.normal = normal;
  v.color = color;
  v.mask = mask;
  v.fused = ImageU8::Zero(depth.rows(), depth.cols());
  v.neighbors = neighbor_ids;
  v.primary = primary;
  views_.push_back(std::move(v));
}

// BilinearDepth: bilinear interpolation on a depthmap, returning -1 if any
// of the four supporting samples is non-positive (i.e. invalid depth).
static float BilinearDepth(const ImageF& depth, float u, float v) {
  int x0 = static_cast<int>(std::floor(u));
  int y0 = static_cast<int>(std::floor(v));
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  if (x0 < 0 || y0 < 0 || x1 >= depth.cols() || y1 >= depth.rows()) {
    return -1.0f;
  }

  if (depth(y0, x0) <= 0.0f || depth(y0, x1) <= 0.0f || depth(y1, x0) <= 0.0f ||
      depth(y1, x1) <= 0.0f) {
    return -1.0f;
  }

  return foundation::BilinearInterpolation(depth, v, u, -1.0f);
}

void DepthmapFuser::Fuse(std::vector<Vec3f>* fused_points,
                         std::vector<Vec3f>* fused_normals,
                         std::vector<Vec3<uint8_t>>* fused_colors) {
  fused_points->clear();
  fused_normals->clear();
  fused_colors->clear();

  const int num_views = static_cast<int>(views_.size());
  if (num_views == 0) {
    return;
  }

  const int nthreads = std::min(num_threads_, num_views);

  struct ThreadOutput {
    std::vector<Vec3f> pts;
    std::vector<Vec3f> nrm;
    std::vector<Vec3<uint8_t>> clr;
  };
  std::vector<ThreadOutput> thread_outputs(nthreads);

  std::vector<int> primary_indices;
  primary_indices.reserve(num_views);
  for (int i = 0; i < num_views; ++i) {
    if (views_[i].primary) {
      primary_indices.push_back(i);
    }
  }
  if (primary_indices.empty()) {
    return;
  }

  std::vector<std::vector<int>> thread_views(nthreads);
  for (int i = 0; i < static_cast<int>(primary_indices.size()); ++i) {
    thread_views[i % nthreads].push_back(primary_indices[i]);
  }

  std::atomic<size_t> global_count{0};

  auto worker = [&](int tid) {
    auto& out = thread_outputs[tid];

    for (int ref_idx : thread_views[tid]) {
      View& ref = views_[ref_idx];
      const int rows = ref.depth.rows();
      const int cols = ref.depth.cols();
      const int bm = border_margin_;

      // Precompute float-precision camera data.
      const Mat3f Kf = ref.K.cast<float>();
      const Mat3f Rf = ref.R.cast<float>();
      const Vec3f tf = ref.t.cast<float>();
      const Mat3f Kinvf = Kf.inverse();
      const Mat3f Rinvf = Rf.transpose();

      for (int r = bm; r < rows - bm; ++r) {
        for (int c = bm; c < cols - bm; ++c) {
          if (ref.fused(r, c)) {
            continue;
          }

          // Skip pixels outside the valid undistortion region.
          if (ref.mask.size() > 0 && ref.mask(r, c) == 0) {
            continue;
          }

          float ref_depth = ref.depth(r, c);
          if (ref_depth <= 0.0f) {
            continue;
          }

          // Back-project reference pixel to 3D world.
          Vec3f ref_point =
              Rinvf * (ref_depth * Kinvf * Vec3f(c, r, 1.0f) - tf);

          // Reference normal in world frame.
          int ref_px = r * cols + c;
          Vec3f ref_normal_cam = ref.normal.col(ref_px);
          float nlen = ref_normal_cam.norm();
          if (nlen < 1e-6f) {
            continue;
          }
          ref_normal_cam /= nlen;
          Vec3f ref_normal_world = Rinvf * ref_normal_cam;

          // Accumulate consistent observations.
          std::vector<Vec3f> pts, nrm;
          std::vector<Vec3<uint8_t>> clr;

          pts.push_back(ref_point);
          nrm.push_back(ref_normal_world);
          auto ref_color = ref.color.col(ref_px);
          clr.push_back(
              Vec3<uint8_t>(ref_color(0), ref_color(1), ref_color(2)));

          struct UsedPixel {
            int view, row, col;
          };
          std::vector<UsedPixel> used_pixels;

          for (int ngb_idx : ref.neighbors) {
            if (ngb_idx < 0 || ngb_idx >= num_views) {
              continue;
            }
            View& src = views_[ngb_idx];
            const int src_cols = src.depth.cols();
            const Mat3f src_Kf = src.K.cast<float>();
            const Mat3f src_Rf = src.R.cast<float>();
            const Vec3f src_tf = src.t.cast<float>();

            // Project reference 3D point into source view.
            Vec3f proj = src_Kf * (src_Rf * ref_point + src_tf);
            if (proj.z() < 1e-6f) {
              continue;
            }

            float src_u = proj.x() / proj.z();
            float src_v = proj.y() / proj.z();
            int src_c = static_cast<int>(std::round(src_u));
            int src_r = static_cast<int>(std::round(src_v));

            if (src_c < bm || src_c >= src.depth.cols() - bm || src_r < bm ||
                src_r >= src.depth.rows() - bm) {
              continue;
            }

            // Skip source pixels outside the valid undistortion region.
            if (src.mask.size() > 0 && src.mask(src_r, src_c) == 0) {
              continue;
            }

            if (src.fused(src_r, src_c)) {
              continue;
            }

            float src_depth = BilinearDepth(src.depth, src_u, src_v);
            if (src_depth <= 0.0f) {
              src_depth = src.depth(src_r, src_c);
              if (src_depth <= 0.0f) {
                continue;
              }
            }

            // Asymmetric depth consistency: use a tighter tolerance
            // when the candidate point lies BEHIND the source surface
            // (free-space violation direction) to suppress double walls.
            float depth_of_proj = proj.z();
            float front_threshold = max_depth_error_ * depth_of_proj;
            float behind_threshold = front_threshold * behind_depth_factor_;
            float signed_diff = src_depth - depth_of_proj;
            // signed_diff > 0: candidate is in front of source (occluder)
            // signed_diff < 0: candidate is behind source (free-space side)
            if (signed_diff > front_threshold ||
                signed_diff < -behind_threshold) {
              continue;
            }

            // Reprojection consistency (forward-backward).
            Mat3f src_Kinvf = src_Kf.inverse();
            Mat3f src_Rinvf = src_Rf.transpose();
            Vec3f src_point =
                src_Rinvf *
                (src_depth * src_Kinvf * Vec3f(src_c, src_r, 1.0f) - src_tf);
            Vec3f reproj = Kf * (Rf * src_point + tf);
            if (reproj.z() < 1e-6f) {
              continue;
            }
            Vec2f reproj_px(reproj.x() / reproj.z(), reproj.y() / reproj.z());
            float reproj_err_sq = (reproj_px - Vec2f(c, r)).squaredNorm();
            if (reproj_err_sq > max_reproj_error_sq_) {
              continue;
            }

            // Normal consistency.
            int src_px = src_r * src_cols + src_c;
            Vec3f src_normal_cam = src.normal.col(src_px);
            float src_nlen = src_normal_cam.norm();
            if (src_nlen < 1e-6f) {
              continue;
            }
            src_normal_cam /= src_nlen;
            Vec3f src_normal_world = src_Rinvf * src_normal_cam;
            float cos_angle = ref_normal_world.dot(src_normal_world);
            if (cos_angle < min_cos_normal_error_) {
              continue;
            }

            // Consistent — accumulate.
            pts.push_back(src_point);
            nrm.push_back(src_normal_world);
            auto src_color = src.color.col(src_px);
            clr.push_back(
                Vec3<uint8_t>(src_color(0), src_color(1), src_color(2)));
            used_pixels.push_back({ngb_idx, src_r, src_c});
          }

          int num_consistent = static_cast<int>(pts.size());
          if (num_consistent >= min_num_consistent_) {
            // Compute component-wise medians.
            std::vector<float> comp(num_consistent);
            Vec3f median_pt;
            for (int d = 0; d < 3; ++d) {
              for (int k = 0; k < num_consistent; ++k) {
                comp[k] = pts[k](d);
              }
              median_pt(d) = VectorMedian(comp);
            }
            out.pts.push_back(median_pt);

            Vec3f median_nrm;
            for (int d = 0; d < 3; ++d) {
              for (int k = 0; k < num_consistent; ++k) {
                comp[k] = nrm[k](d);
              }
              median_nrm(d) = VectorMedian(comp);
            }
            float mn_len = median_nrm.norm();
            if (mn_len > 1e-6f) {
              median_nrm /= mn_len;
            }
            out.nrm.push_back(median_nrm);

            std::vector<uint8_t> comp_u8(num_consistent);
            Vec3<uint8_t> median_clr;
            for (int d = 0; d < 3; ++d) {
              for (int k = 0; k < num_consistent; ++k) {
                comp_u8[k] = clr[k](d);
              }
              median_clr(d) = VectorMedian(comp_u8);
            }
            out.clr.push_back(median_clr);

            ref.fused(r, c) = 1;
            for (const auto& up : used_pixels) {
              views_[up.view].fused(up.row, up.col) = 1;
            }
          }
        }
      }

      size_t current = global_count.fetch_add(1) + 1;
      std::cerr << "[Fusion] Image " << current << "/" << num_views
                << " done (thread " << tid << "), " << out.pts.size()
                << " pts so far\n";
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < nthreads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& th : threads) {
    th.join();
  }

  size_t total_pts = 0;
  for (auto& o : thread_outputs) {
    total_pts += o.pts.size();
  }

  fused_points->reserve(total_pts);
  fused_normals->reserve(total_pts);
  fused_colors->reserve(total_pts);

  for (auto& o : thread_outputs) {
    fused_points->insert(fused_points->end(), o.pts.begin(), o.pts.end());
    fused_normals->insert(fused_normals->end(), o.nrm.begin(), o.nrm.end());
    fused_colors->insert(fused_colors->end(), o.clr.begin(), o.clr.end());
  }

  std::cerr << "[Fusion] Total fused points before SOR: " << total_pts << "\n";

  // ---- Statistical Outlier Removal (SOR) via vlfeat KD-tree ----
  if (sor_knn_ > 0 && total_pts > static_cast<size_t>(sor_knn_ + 1)) {
    std::cerr << "[Fusion] Running SOR (k=" << sor_knn_
              << ", factor=" << sor_stddev_factor_ << ") on " << total_pts
              << " points ...\n";

    const int K = sor_knn_;
    const int N = static_cast<int>(total_pts);

    // vlfeat requires a contiguous float[N*3] buffer.
    std::vector<float> flat_pts(N * 3);
    for (int i = 0; i < N; ++i) {
      Eigen::Map<Vec3f>(flat_pts.data() + i * 3) = (*fused_points)[i];
    }

    VlKDForest* forest = vl_kdforest_new(VL_TYPE_FLOAT, 3, 1, VlDistanceL2);
    vl_kdforest_build(forest, static_cast<vl_size>(N), flat_pts.data());

    std::vector<float> mean_knn_dist(N);

    auto sor_worker = [&](int start, int end, VlKDForestSearcher* searcher) {
      std::vector<VlKDForestNeighbor> nbrs(K + 1);
      for (int i = start; i < end; ++i) {
        const float* q = &flat_pts[i * 3];
        vl_size found = vl_kdforestsearcher_query(
            searcher, nbrs.data(), static_cast<vl_size>(K + 1), q);

        float sum = 0.0f;
        int count = 0;
        for (vl_size j = 0; j < found && count < K; ++j) {
          if (nbrs[j].index == static_cast<vl_uindex>(i)) {
            continue;
          }
          sum += std::sqrt(static_cast<float>(nbrs[j].distance));
          ++count;
        }
        mean_knn_dist[i] = (count > 0) ? sum / count : 1e30f;
      }
    };

    {
      std::vector<std::thread> sor_threads;
      std::vector<VlKDForestSearcher*> searchers;
      int chunk = (N + nthreads - 1) / nthreads;
      for (int t = 0; t < nthreads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, N);
        if (start < end) {
          VlKDForestSearcher* s = vl_kdforest_new_searcher(forest);
          searchers.push_back(s);
          sor_threads.emplace_back(sor_worker, start, end, s);
        }
      }
      for (auto& th : sor_threads) {
        th.join();
      }
      for (auto* s : searchers) {
        vl_kdforestsearcher_delete(s);
      }
    }

    vl_kdforest_delete(forest);

    double sum_d = 0, sum_d2 = 0;
    for (int i = 0; i < N; ++i) {
      double d = mean_knn_dist[i];
      if (d < 1e20f) {
        sum_d += d;
        sum_d2 += d * d;
      }
    }
    double mean_d = sum_d / N;
    double var_d = sum_d2 / N - mean_d * mean_d;
    double std_d = std::sqrt(std::max(0.0, var_d));
    float threshold = static_cast<float>(mean_d + sor_stddev_factor_ * std_d);

    std::vector<Vec3f> filtered_pts, filtered_nrm;
    std::vector<Vec3<uint8_t>> filtered_clr;
    int kept = 0;
    for (int i = 0; i < N; ++i) {
      if (mean_knn_dist[i] <= threshold) {
        filtered_pts.push_back((*fused_points)[i]);
        filtered_nrm.push_back((*fused_normals)[i]);
        filtered_clr.push_back((*fused_colors)[i]);
        kept++;
      }
    }

    *fused_points = std::move(filtered_pts);
    *fused_normals = std::move(filtered_nrm);
    *fused_colors = std::move(filtered_clr);

    std::cerr << "[Fusion] SOR: kept " << kept << " / " << N
              << " points (threshold=" << threshold << ", mean=" << mean_d
              << ", std=" << std_d << ")\n";
  }
}

}  // namespace dense
