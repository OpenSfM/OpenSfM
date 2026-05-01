#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <map/map.h>
#include <map/tracks_manager.h>
#include <pybind11/pybind11.h>
#include <robust/instanciations.h>
#include <sfm/ba_helpers.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
namespace sfm {

/// Rig assignment for a single shot: (instance_id, rig_camera_id, co-shots)
struct RigAssignment {
  map::RigInstanceId instance_id;
  map::RigCameraId rig_camera_id;
  std::vector<map::ShotId> instance_shots;
};

/// C++ implementation of the incremental reconstruction grow loop.
/// Replaces the Python while(True) loop in grow_reconstruction.
class ReconstructionGrower {
 public:
  /// Run the grow loop.  Returns a Python dict report.
  ///
  /// @param reconstruction   Python Reconstruction object (for align calls)
  /// @param shot_camera_map  shot_id -> camera_id (from exif)
  /// @param rig_assignments  shot_id -> RigAssignment (empty if no rigs)
  /// @param images           shots still available to add (consumed)
  /// @param data             Python DataSet object (for load_exif)
  static py::dict Grow(
      map::Map& map, const map::TracksManager& tracks_manager,
      const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
      const std::unordered_map<map::RigCameraId, map::RigCamera>&
          rig_camera_priors,
      const std::unordered_map<map::ShotId, map::CameraId>& shot_camera_map,
      const std::unordered_map<map::ShotId, RigAssignment>& rig_assignments,
      std::unordered_set<map::ShotId> images, py::object reconstruction,
      py::object data, const py::dict& config);

  /// Parse a Python exif dict directly into ShotMeasurements.
  static map::ShotMeasurements ParseExifDict(const py::dict& exif,
                                             bool use_altitude, double reflat,
                                             double reflon, double refalt);

  /// Triangulate tracks visible from given shots that aren't yet landmarks.
  /// Returns the set of newly created landmark (track) IDs.
  static std::unordered_set<map::TrackId> TriangulateNewTracks(
      map::Map& map, const map::TracksManager& tracks_manager,
      const std::unordered_set<map::ShotId>& shot_ids, const py::dict& config);

 private:
  // ---- Resection ----
  struct ResectionResult {
    bool success = false;
    std::unordered_set<map::ShotId> new_shots;
    std::unordered_set<map::TrackId> inlier_tracks;
    int num_common_points = 0;
    int num_inliers = 0;
  };

  static ResectionResult Resect(
      map::Map& map, const map::TracksManager& tracks_manager,
      const map::ShotId& shot_id, const map::CameraId& camera_id,
      double threshold, int min_inliers,
      const std::unordered_map<map::ShotId, RigAssignment>& rig_assignments,
      const py::dict& config);

  // ---- Resection candidates ----
  class ResectionCandidates {
   public:
    ResectionCandidates(const map::Map& map);

    void Add(const std::unordered_set<map::ShotId>& new_shots,
             const std::unordered_set<map::TrackId>& new_tracks,
             const map::Map& map, const map::TracksManager& tm);

    void Remove(
        const std::vector<std::pair<map::LandmarkId, map::ShotId>>& outliers,
        const std::vector<map::LandmarkId>& removed_tracks);

    void Update(const map::Map& map, const map::TracksManager& tm);

    std::vector<std::pair<map::ShotId, int>> GetCandidates(
        const std::unordered_set<map::ShotId>& images) const;

   private:
    absl::flat_hash_map<map::ShotId, absl::flat_hash_set<map::TrackId>>
        candidates_;
    absl::flat_hash_set<map::TrackId> tracks_;
  };

  // ---- Scheduling ----
  struct BundleScheduler {
    int interval = 100;
    double new_points_ratio = 1.2;
    size_t num_points_last = 0;
    size_t num_shots_last = 0;

    bool Should(size_t np, size_t ns) const {
      return np >= num_points_last * new_points_ratio ||
             ns >= num_shots_last + interval;
    }
    void Done(size_t np, size_t ns) {
      num_points_last = np;
      num_shots_last = ns;
    }
  };

  struct RetriangulateScheduler {
    bool active = false;
    double ratio = 1.25;
    size_t num_points_last = 0;

    bool Should(size_t np) const {
      return active && np > num_points_last * ratio;
    }
    void Done(size_t np) { num_points_last = np; }
  };

  // ---- Logging via Python logger ----
  static void LogInfo(const std::string& msg);
  static void LogWarning(const std::string& msg);
  static void LogDebug(const std::string& msg);
  static void LogBundleStats(const std::string& bundle_type,
                             const py::dict& report);

  // ---- Outlier removal helper ----
  static void RemoveOutliersAndUpdate(
      map::Map& map, const py::dict& config,
      const std::vector<map::LandmarkId>& point_ids,
      ResectionCandidates& candidates);

  // ---- Reusable buffers (avoid per-shot allocations) ----
  struct TriangulationBuffers {
    std::vector<double> thresholds;
    MatX3d origins;
    MatX3d bearings;
    std::vector<std::pair<map::Shot*, map::ObservationIndex>> track_obs;
  };
};

}  // namespace sfm
