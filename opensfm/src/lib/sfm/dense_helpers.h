#pragma once

#include <map/defines.h>
#include <map/map.h>
#include <map/tracks_manager.h>

#include <Eigen/Core>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace sfm::dense_helpers {

// ─── Super-point: fused 3-D point with merged visibility ─────────────
struct SuperPoint {
  Vec3d coord;                       // centroid of merged points
  std::vector<map::ShotId> vis;      // sorted observing shot IDs
  std::vector<map::TrackId> tracks;  // underlying track IDs
};

// ─── Covisibility graph output ───────────────────────────────────────
struct CovisibilityGraph {
  std::vector<SuperPoint> super_points;    // all SPs (merged + singletons)
  std::vector<std::pair<int, int>> edges;  // (shot_idx_i, shot_idx_j), i < j
  std::vector<double> weights;             // shared-SP count per edge
  std::vector<map::ShotId> shot_order;     // index → shot ID mapping
};

/// Build the super-point covisibility graph for Leiden clustering.
///
/// 1. Collects all reconstructed 3-D landmarks visible by ≥ 2
///    processable shots from the tracks manager.
/// 2. Fuses nearby points (within `fuse_radius_factor * median_kNN_dist`)
///    into super-points via Union-Find, merging their visibility sets.
/// 3. Builds pairwise shot covisibility weights (= number of shared SPs).
///
/// Multithreaded via OpenMP (kNN query, UF accumulation, pair counting).
CovisibilityGraph BuildCovisibilityGraph(
    const map::TracksManager& tracks_manager, const map::Map& reconstruction,
    const std::vector<map::ShotId>& processable, int fuse_knn = 15,
    double fuse_radius_factor = 0.5);

// ─── Neighbor selection output ───────────────────────────────────────
struct NeighborResult {
  /// best_neighbors[shot_id] = [shot_id_self, top-K by angle score …]
  std::unordered_map<map::ShotId, std::vector<map::ShotId>> best_neighbors;
  /// all_neighbors[shot_id] = best_neighbors ∪ views with enough common SPs
  std::unordered_map<map::ShotId, std::vector<map::ShotId>> all_neighbors;
};

/// Select best & all neighbors for each shot from global covisibility.
///
/// For every pair of shots sharing super-points, computes a baseline-
/// angle score on their shared 3-D tracks.  Best neighbors are ranked
/// by the count of tracks with baseline angle in [theta_min, theta_max].
///
/// Multithreaded via OpenMP (parallel over shots).
NeighborResult SelectNeighbors(const map::TracksManager& tracks_manager,
                               const map::Map& reconstruction,
                               const std::vector<SuperPoint>& super_points,
                               const std::vector<map::ShotId>& processable,
                               int num_neighbors, int min_point_best = 20,
                               int min_point_all = 40,
                               double theta_min_deg = 3.0,
                               double theta_max_deg = 60.0);

}  // namespace sfm::dense_helpers
