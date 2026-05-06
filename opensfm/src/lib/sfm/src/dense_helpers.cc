#include <foundation/union_find.h>
#include <sfm/dense_helpers.h>
#include <vl/kdtree.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sfm::dense_helpers {

// =====================================================================
// BuildCovisibilityGraph
// =====================================================================

CovisibilityGraph BuildCovisibilityGraph(
    const map::TracksManager& tracks_manager, const map::Map& reconstruction,
    const std::vector<map::ShotId>& processable, int fuse_knn,
    double fuse_radius_factor) {
  CovisibilityGraph result;
  result.shot_order = processable;

  const int N = static_cast<int>(processable.size());
  if (N <= 0) {
    return result;
  }

  // Shot ID → integer index for edge construction.
  std::unordered_map<map::ShotId, int> sid_to_idx;
  sid_to_idx.reserve(N);
  for (int i = 0; i < N; ++i) {
    sid_to_idx[processable[i]] = i;
  }
  std::unordered_set<map::ShotId> processable_set(processable.begin(),
                                                  processable.end());

  // ─ 1. Collect reconstructed points visible by ≥ 2 processable shots ─
  const auto& landmarks = reconstruction.GetLandmarks();

  struct RawPoint {
    map::TrackId track_id;
    Vec3d coord;
    std::vector<int> observers;  // indices into processable
  };

  std::vector<RawPoint> raw_points;
  raw_points.reserve(landmarks.size());

  for (const auto& [tid, lm] : landmarks) {
    const auto obs = tracks_manager.GetTrackObservations(tid);
    std::vector<int> observers;
    observers.reserve(obs.size());
    for (const auto& [sid, _] : obs) {
      auto it = sid_to_idx.find(sid);
      if (it != sid_to_idx.end()) {
        observers.push_back(it->second);
      }
    }
    if (observers.size() < 2) {
      continue;
    }
    std::sort(observers.begin(), observers.end());

    raw_points.push_back({tid, lm.GetGlobalPos(), std::move(observers)});
  }

  const int n_pts = static_cast<int>(raw_points.size());
  std::cerr << "[BuildCovisibilityGraph] " << n_pts
            << " points visible by >= 2 processable shots\n";

  if (n_pts == 0) {
    return result;
  }

  // ─ 2. Scene scale via kNN (vlfeat KD-tree) ──────────────────────────
  const int k = std::max(1, std::min(fuse_knn, n_pts - 1));

  // Flatten coordinates for vlfeat (row-major, 3×N layout expected by vl).
  std::vector<float> flat_coords(static_cast<size_t>(n_pts) * 3);
  for (int i = 0; i < n_pts; ++i) {
    flat_coords[i * 3 + 0] = static_cast<float>(raw_points[i].coord.x());
    flat_coords[i * 3 + 1] = static_cast<float>(raw_points[i].coord.y());
    flat_coords[i * 3 + 2] = static_cast<float>(raw_points[i].coord.z());
  }

  VlKDForest* forest = vl_kdforest_new(VL_TYPE_FLOAT, 3, 1, VlDistanceL2);
  vl_kdforest_build(forest, static_cast<vl_size>(n_pts), flat_coords.data());

  // Query k+1 neighbors per point (includes self) to estimate scene scale.
  // Also save raw kNN indices + squared distances for the fusion step
  // (avoids re-querying vlfeat, whose internal state is fragile after
  //  parallel use).
  const int max_k = k;  // max neighbors per point, excluding self
  std::vector<float> knn_dists(static_cast<size_t>(n_pts));
  std::vector<int> knn_idx(static_cast<size_t>(n_pts) * max_k, -1);
  std::vector<float> knn_d2(static_cast<size_t>(n_pts) * max_k, 0.0f);

#ifdef _OPENMP
  const int n_threads = omp_get_max_threads();
#else
  const int n_threads = 1;
#endif

  // Each thread needs its own searcher.
  std::vector<VlKDForestSearcher*> searchers(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    searchers[t] = vl_kdforest_new_searcher(forest);
  }

#pragma omp parallel
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    std::vector<VlKDForestNeighbor> nbrs(k + 1);

#pragma omp for schedule(static)
    for (int i = 0; i < n_pts; ++i) {
      const float* q = &flat_coords[i * 3];
      vl_size found = vl_kdforestsearcher_query(searchers[tid], nbrs.data(),
                                                static_cast<vl_size>(k + 1), q);

      // Store kNN results (excluding self) and compute median distance.
      int slot = 0;
      std::vector<float> dists;
      dists.reserve(found);
      for (vl_size j = 0; j < found && slot < max_k; ++j) {
        int other = static_cast<int>(nbrs[j].index);
        if (other == i) {
          continue;
        }
        knn_idx[i * max_k + slot] = other;
        knn_d2[i * max_k + slot] = static_cast<float>(nbrs[j].distance);
        dists.push_back(std::sqrt(static_cast<float>(nbrs[j].distance)));
        ++slot;
      }
      if (!dists.empty()) {
        size_t mid = dists.size() / 2;
        std::nth_element(dists.begin(), dists.begin() + mid, dists.end());
        knn_dists[i] = dists[mid];
      } else {
        knn_dists[i] = 0.0f;
      }
    }
  }

  for (auto* s : searchers) {
    vl_kdforestsearcher_delete(s);
  }
  vl_kdforest_delete(forest);

  // Global scale = median of per-point median kNN distances.
  std::vector<float> all_dists = knn_dists;
  std::nth_element(all_dists.begin(), all_dists.begin() + n_pts / 2,
                   all_dists.end());
  const double scale = static_cast<double>(all_dists[n_pts / 2]);
  const double fuse_radius = fuse_radius_factor * scale;
  const float fuse_r2 = static_cast<float>(fuse_radius * fuse_radius);

  std::cerr << "[BuildCovisibilityGraph] scale=" << scale
            << ", fuse_radius=" << fuse_radius << "\n";

  // ─ 3. Fuse nearby points via Union-Find ──────────────────────────────
  //   Iterate saved kNN results and merge directly — no re-query needed.
  std::vector<std::unique_ptr<UnionFindElement<int>>> uf_elements;
  uf_elements.reserve(n_pts);
  for (int i = 0; i < n_pts; ++i) {
    uf_elements.push_back(std::make_unique<UnionFindElement<int>>(i));
  }

  size_t fused_count = 0;
  for (int i = 0; i < n_pts; ++i) {
    for (int s = 0; s < max_k; ++s) {
      int other = knn_idx[i * max_k + s];
      if (other < 0) {
        break;
      }
      if (other > i && knn_d2[i * max_k + s] <= fuse_r2) {
        Union(uf_elements[i].get(), uf_elements[other].get());
        ++fused_count;
      }
    }
  }
  knn_idx.clear();
  knn_d2.clear();

  std::cerr << "[BuildCovisibilityGraph] " << fused_count
            << " pairs merged in Union-Find\n";

  // ─ 4. Build super-points from UF clusters ─────────────────────────
  //   Group by root → merge visibility, centroid, track IDs.
  std::unordered_map<int, std::vector<int>> groups;  // root → member indices
  for (int i = 0; i < n_pts; ++i) {
    int root = Find(uf_elements[i].get())->data;
    groups[root].push_back(i);
  }
  uf_elements.clear();

  result.super_points.reserve(groups.size());
  for (const auto& [root, members] : groups) {
    SuperPoint sp;

    // Centroid.
    sp.coord = Vec3d::Zero();
    for (int idx : members) {
      sp.coord += raw_points[idx].coord;
    }
    sp.coord /= static_cast<double>(members.size());

    // Merge visibility (sorted, deduplicated via set).
    std::set<int> vis_indices;
    for (int idx : members) {
      for (int obs : raw_points[idx].observers) {
        vis_indices.insert(obs);
      }
    }
    sp.vis.reserve(vis_indices.size());
    for (int vi : vis_indices) {
      sp.vis.push_back(processable[vi]);
    }

    // Track IDs.
    sp.tracks.reserve(members.size());
    for (int idx : members) {
      sp.tracks.push_back(raw_points[idx].track_id);
    }

    result.super_points.push_back(std::move(sp));
  }
  groups.clear();
  raw_points.clear();

  std::cerr << "[BuildCovisibilityGraph] " << result.super_points.size()
            << " super-points\n";
  std::cerr.flush();

  // ─ 5. Build pairwise covisibility weights ──────────────────────────
  //   For each SP, increment weight for every pair of observers.
  //   Parallel: each thread accumulates into a thread-local hash map.
  //   Key: pack (min_idx, max_idx) into uint64_t.

  auto pack_pair = [](int a, int b) -> uint64_t {
    int lo = std::min(a, b);
    int hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
  };

  const int n_sp = static_cast<int>(result.super_points.size());
  std::vector<std::unordered_map<uint64_t, int>> thread_counts(n_threads);

#pragma omp parallel
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    auto& local = thread_counts[tid];

#pragma omp for schedule(dynamic, 64)
    for (int si = 0; si < n_sp; ++si) {
      const auto& sp = result.super_points[si];
      const int nv = static_cast<int>(sp.vis.size());
      for (int ci = 0; ci < nv; ++ci) {
        auto it_a = sid_to_idx.find(sp.vis[ci]);
        if (it_a == sid_to_idx.end()) {
          continue;
        }
        int a = it_a->second;
        for (int cj = ci + 1; cj < nv; ++cj) {
          auto it_b = sid_to_idx.find(sp.vis[cj]);
          if (it_b == sid_to_idx.end()) {
            continue;
          }
          int b = it_b->second;
          local[pack_pair(a, b)]++;
        }
      }
    }
  }

  // Merge thread-local maps.
  std::unordered_map<uint64_t, int> pair_count;
  for (auto& tc : thread_counts) {
    for (const auto& [key, cnt] : tc) {
      pair_count[key] += cnt;
    }
    tc.clear();
  }
  thread_counts.clear();

  result.edges.reserve(pair_count.size());
  result.weights.reserve(pair_count.size());
  for (const auto& [key, cnt] : pair_count) {
    if (cnt < 1) {
      continue;
    }
    int lo = static_cast<int>(key >> 32);
    int hi = static_cast<int>(key & 0xFFFFFFFF);
    result.edges.emplace_back(lo, hi);
    result.weights.push_back(static_cast<double>(cnt));
  }

  std::cerr << "[BuildCovisibilityGraph] " << result.edges.size()
            << " edges in covisibility graph\n";
  std::cerr.flush();

  return result;
}

// =====================================================================
// SelectNeighbors
// =====================================================================

NeighborResult SelectNeighbors(const map::TracksManager& /*tracks_manager*/,
                               const map::Map& reconstruction,
                               const std::vector<SuperPoint>& super_points,
                               const std::vector<map::ShotId>& processable,
                               int num_neighbors, int min_point_best,
                               int min_point_all, double theta_min_deg,
                               double theta_max_deg) {
  NeighborResult result;

  const int N = static_cast<int>(processable.size());
  if (N == 0) {
    return result;
  }

  const double theta_min = theta_min_deg * M_PI / 180.0;
  const double theta_max = theta_max_deg * M_PI / 180.0;
  const double cos_lo = std::cos(theta_max);  // lower cosine bound
  const double cos_hi = std::cos(theta_min);  // upper cosine bound

  std::unordered_map<map::ShotId, int> sid_to_idx;
  sid_to_idx.reserve(N);
  for (int i = 0; i < N; ++i) {
    sid_to_idx[processable[i]] = i;
  }

  // ─ 1. Compute per-pair scores directly from super-points ─────────
  //   For each SP, compute the viewing angle at the SP centroid for
  //   every camera pair, weighted by the number of tracks in the SP.
  //   Each pair accumulates only 3 numbers: count, score_best, sum_cos.

  // Pre-fetch camera origins (needed for angle computation).
  std::vector<Vec3d> origins(N);
  for (int i = 0; i < N; ++i) {
    const auto& shot = reconstruction.GetShot(processable[i]);
    origins[i] = shot.GetPose()->GetOrigin();
  }

  // Per-pair accumulator: only 20 bytes per pair instead of a vector.
  struct PairScore {
    int score_best = 0;  // tracks with angle in [theta_min, theta_max]
    int count = 0;       // total shared tracks
    double sum_cos = 0.0;
  };
  using PairMap = std::unordered_map<int, PairScore>;

  const int n_sp = static_cast<int>(super_points.size());

#ifdef _OPENMP
  const int n_threads = omp_get_max_threads();
#else
  const int n_threads = 1;
#endif

  // Thread-local per-shot score maps.
  std::vector<std::vector<PairMap>> thread_per_shot(n_threads,
                                                    std::vector<PairMap>(N));

  std::cerr << "[SelectNeighbors] scoring " << n_sp << " super-points, " << N
            << " shots, " << n_threads << " threads\n";
  std::cerr.flush();

#pragma omp parallel
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    auto& local = thread_per_shot[tid];

#pragma omp for schedule(dynamic, 64)
    for (int si = 0; si < n_sp; ++si) {
      const auto& sp = super_points[si];

      // Resolve shot indices for this SP's visibility.
      std::vector<int> cam_indices;
      cam_indices.reserve(sp.vis.size());
      for (const auto& sid : sp.vis) {
        auto it = sid_to_idx.find(sid);
        if (it != sid_to_idx.end()) {
          cam_indices.push_back(it->second);
        }
      }
      const int nc = static_cast<int>(cam_indices.size());
      if (nc < 2) {
        continue;
      }

      // Use SP centroid for angle computation (one angle per pair per SP
      // instead of per-track — ~10x faster, accurate for fused points).
      const Vec3d& P = sp.coord;
      const int n_tracks = static_cast<int>(sp.tracks.size());

      // Precompute camera-to-centroid vectors.
      struct CamVec {
        Vec3d v;
        double norm;
      };
      std::vector<CamVec> cvs(nc);
      for (int ci = 0; ci < nc; ++ci) {
        cvs[ci].v = origins[cam_indices[ci]] - P;
        cvs[ci].norm = cvs[ci].v.norm();
      }

      for (int ci = 0; ci < nc; ++ci) {
        if (cvs[ci].norm < 1e-12) {
          continue;
        }
        int a = cam_indices[ci];
        for (int cj = ci + 1; cj < nc; ++cj) {
          if (cvs[cj].norm < 1e-12) {
            continue;
          }
          int b = cam_indices[cj];
          double cos_theta = std::clamp(
              cvs[ci].v.dot(cvs[cj].v) / (cvs[ci].norm * cvs[cj].norm), -1.0,
              1.0);
          bool good = cos_theta > cos_lo && cos_theta < cos_hi;
          int good_count = good ? n_tracks : 0;

          auto& pa = local[a][b];
          pa.count += n_tracks;
          pa.sum_cos += cos_theta * n_tracks;
          pa.score_best += good_count;

          auto& pb = local[b][a];
          pb.count += n_tracks;
          pb.sum_cos += cos_theta * n_tracks;
          pb.score_best += good_count;
        }
      }
    }
  }

  // Merge thread-local maps.
  std::vector<PairMap> per_shot(N);
  for (int t = 0; t < n_threads; ++t) {
    for (int i = 0; i < N; ++i) {
      for (auto& [other, ps] : thread_per_shot[t][i]) {
        auto& dst = per_shot[i][other];
        dst.score_best += ps.score_best;
        dst.count += ps.count;
        dst.sum_cos += ps.sum_cos;
      }
    }
    thread_per_shot[t].clear();
  }
  thread_per_shot.clear();
  thread_per_shot.shrink_to_fit();

  std::cerr << "[SelectNeighbors] pair scoring done\n";
  std::cerr.flush();

  // ─ 2. Per-shot neighbor selection (parallel over shots) ────────────
  // Output arrays (one per shot, filled in parallel).
  std::vector<std::vector<map::ShotId>> out_best(N);
  std::vector<std::vector<map::ShotId>> out_all(N);

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < N; ++i) {
    struct BestCandidate {
      int idx;
      int score;
      double avg_theta;
    };
    struct AllCandidate {
      int idx;
      int score;
    };

    std::vector<BestCandidate> ns_best;
    std::vector<AllCandidate> ns_all;

    for (auto& [other_idx, ps] : per_shot[i]) {
      if (ps.count == 0) {
        continue;
      }

      // All-neighbors: any pair with enough shared tracks.
      if (ps.count > min_point_all) {
        ns_all.push_back({other_idx, ps.count});
      }
      if (ps.count <= min_point_best) {
        continue;
      }

      double avg_cos = ps.sum_cos / ps.count;
      double avg_theta = std::acos(std::clamp(avg_cos, -1.0, 1.0));
      if (avg_theta < theta_min || avg_theta > theta_max) {
        continue;
      }

      if (ps.score_best > min_point_best) {
        ns_best.push_back({other_idx, ps.score_best, avg_theta});
      }
    }

    // Sort best by score descending.
    std::sort(ns_best.begin(), ns_best.end(),
              [](const BestCandidate& a, const BestCandidate& b) {
                return a.score > b.score;
              });

    // Build output: self + top-K.
    auto& best = out_best[i];
    best.reserve(1 + num_neighbors);
    best.push_back(processable[i]);  // self first
    for (int j = 0;
         j < std::min(num_neighbors, static_cast<int>(ns_best.size())); ++j) {
      best.push_back(processable[ns_best[j].idx]);
    }

    // All = best ∪ all_candidates not already in best.
    std::unordered_set<int> best_set;
    best_set.insert(i);
    for (const auto& bc : ns_best) {
      if (static_cast<int>(best_set.size()) <= num_neighbors) {
        best_set.insert(bc.idx);
      }
    }

    auto& all = out_all[i];
    all = best;
    for (const auto& ac : ns_all) {
      if (best_set.find(ac.idx) == best_set.end()) {
        all.push_back(processable[ac.idx]);
      }
    }
  }

  // Pack into result.
  for (int i = 0; i < N; ++i) {
    result.best_neighbors[processable[i]] = std::move(out_best[i]);
    result.all_neighbors[processable[i]] = std::move(out_all[i]);
  }

  return result;
}

}  // namespace sfm::dense_helpers
