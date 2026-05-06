#pragma once
#include <bundle/bundle_adjuster.h>
#include <bundle/data/parameter_mask.h>
#include <map/ground_control_points.h>
#include <map/map.h>
#include <pybind11/pybind11.h>

#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;
namespace sfm {
class GroundControlPoint;
class BAHelpers {
 public:
  struct TracksSelection {
    std::unordered_set<map::TrackId> selected_tracks;
    std::unordered_set<map::TrackId> other_tracks;
  };

  /// Native result from outlier removal (no Python objects).
  struct OutlierRemovalResult {
    std::vector<std::pair<map::LandmarkId, map::ShotId>> outliers;
    std::vector<map::LandmarkId> removed_tracks;
  };

  /// Native result from local bundle adjustment.
  struct BundleLocalResult {
    std::vector<map::LandmarkId> point_ids;
    py::dict report;
  };

  // ---- Native C++ APIs (no py:: return types) ----

  /// Remove outlier observations and landmarks with < 2 observations.
  /// If point_ids is empty, processes ALL landmarks in the map.
  static OutlierRemovalResult RemoveOutliers(
      map::Map& map, const py::dict& config,
      const std::vector<map::LandmarkId>& point_ids = {});

  /// Local bundle adjustment returning native result.
  static BundleLocalResult BundleLocal(
      map::Map& map,
      const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
      const std::unordered_map<map::RigCameraId, map::RigCamera>&
          rig_camera_priors,
      const AlignedVector<map::GroundControlPoint>& gcp,
      const map::ShotId& central_shot_id, int grid_size,
      const py::dict& config);

  // ---- Python wrappers (for pybind, return py:: objects) ----

  /// Python wrapper: returns py::make_tuple(py_outliers, py_removed).
  static py::tuple RemoveOutliersPython(
      map::Map& map, const py::dict& config,
      const std::vector<map::LandmarkId>& point_ids = {});

  /// Python wrapper: returns py::make_tuple(py_point_ids, report).
  static py::tuple BundleLocalPython(
      map::Map& map,
      const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
      const std::unordered_map<map::RigCameraId, map::RigCamera>&
          rig_camera_priors,
      const AlignedVector<map::GroundControlPoint>& gcp,
      const map::ShotId& central_shot_id, int grid_size,
      const py::dict& config);

  // ---- Other APIs ----

  static py::dict Bundle(
      map::Map& map,
      const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
      const std::unordered_map<map::RigCameraId, map::RigCamera>&
          rig_camera_priors,
      const AlignedVector<map::GroundControlPoint>& gcp, int grid_size,
      const py::dict& config);

  static py::dict BundleShotPoses(
      map::Map& map, const std::unordered_set<map::ShotId>& shot_ids,
      const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
      const std::unordered_map<map::RigCameraId, map::RigCamera>&
          rig_camera_priors,
      const py::dict& config);

  static void BundleToMap(const bundle::BundleAdjuster& bundle_adjuster,
                          map::Map& output_map, bool update_cameras);

  static std::pair<std::unordered_set<map::ShotId>,
                   std::unordered_set<map::ShotId>>
  ShotNeighborhoodIds(map::Map& map, const map::ShotId& central_shot_id,
                      size_t radius, size_t min_common_points,
                      size_t max_interior_size);
  static std::pair<std::unordered_set<map::Shot*>,
                   std::unordered_set<map::Shot*>>
  ShotNeighborhood(map::Map& map, const map::ShotId& central_shot_id,
                   size_t radius, size_t min_common_points,
                   size_t max_interior_size);
  static std::string DetectAlignmentConstraints(
      const map::Map& map, const py::dict& config,
      const AlignedVector<map::GroundControlPoint>& gcp);

  static size_t AddGCPToBundle(
      bundle::BundleAdjuster& ba, const map::Map& map,
      const AlignedVector<map::GroundControlPoint>& gcp, const py::dict& config,
      size_t num_ba_points, size_t num_ba_shots);

  static TracksSelection SelectTracksGrid(
      map::Map& map, const std::unordered_set<map::ShotId>& shot_ids,
      size_t grid_size);

 private:
  static std::unordered_set<map::Shot*> DirectShotNeighbors(
      map::Map& map, const std::unordered_set<map::Shot*>& shot_ids,
      const size_t min_common_points, const size_t max_neighbors);
  static bool TriangulateGCP(
      const map::GroundControlPoint& point,
      const std::unordered_map<map::ShotId, map::Shot>& shots,
      float reproj_threshold, Vec3d& coordinates);

  static void AlignmentConstraints(
      const map::Map& map, const py::dict& config,
      const AlignedVector<map::GroundControlPoint>& gcp, MatX3d& Xp, MatX3d& X);

  static bundle::SimilarityParameterMask DetermineGCPBiasParameters(
      const map::Map& map, const AlignedVector<map::GroundControlPoint>& gcp,
      const py::dict& config);
};
}  // namespace sfm
