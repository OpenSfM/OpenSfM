#include <geometry/triangulation.h>
#include <map/ground_control_points.h>
#include <pybind11/stl.h>
#include <sfm/reconstruction_grower.h>
#include <sfm/retriangulation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace py = pybind11;
namespace sfm {

namespace {

// Extract config values once to avoid repeated Python dict lookups.
struct MapperConfig {
  double resection_threshold;
  int resection_min_inliers;
  double resect_redundancy_threshold;
  bool retriangulation;
  double retriangulation_ratio;
  int bundle_interval;
  double bundle_new_points_ratio;
  double triangulation_threshold;
  double triangulation_min_ray_angle_deg;
  double triangulation_min_depth;
  int local_bundle_radius;
  int local_bundle_min_common_points;
  int local_bundle_max_shots;
  int local_bundle_grid;

  static MapperConfig FromPy(const py::dict& config) {
    MapperConfig c;
    c.resection_threshold = config["resection_threshold"].cast<double>();
    c.resection_min_inliers = config["resection_min_inliers"].cast<int>();
    c.resect_redundancy_threshold =
        config["resect_redundancy_threshold"].cast<double>();
    c.retriangulation = config["retriangulation"].cast<bool>();
    c.retriangulation_ratio = config["retriangulation_ratio"].cast<double>();
    c.bundle_interval = config["bundle_interval"].cast<int>();
    c.bundle_new_points_ratio =
        config["bundle_new_points_ratio"].cast<double>();
    c.triangulation_threshold =
        config["triangulation_threshold"].cast<double>();
    c.triangulation_min_ray_angle_deg =
        config["triangulation_min_ray_angle"].cast<double>();
    c.triangulation_min_depth =
        config["triangulation_min_depth"].cast<double>();
    c.local_bundle_radius = config["local_bundle_radius"].cast<int>();
    c.local_bundle_min_common_points =
        config["local_bundle_min_common_points"].cast<int>();
    c.local_bundle_max_shots = config["local_bundle_max_shots"].cast<int>();
    c.local_bundle_grid = config["local_bundle_grid"].cast<int>();
    return c;
  }
};

}  // namespace

// ---------- Logging via Python logger ----------

void ReconstructionGrower::LogInfo(const std::string& msg) {
  py::gil_scoped_acquire gil;
  static py::object logger =
      py::module::import("logging").attr("getLogger")("opensfm.reconstruction");
  logger.attr("info")(msg);
}

void ReconstructionGrower::LogWarning(const std::string& msg) {
  py::gil_scoped_acquire gil;
  static py::object logger =
      py::module::import("logging").attr("getLogger")("opensfm.reconstruction");
  logger.attr("warning")(msg);
}

void ReconstructionGrower::LogDebug(const std::string& msg) {
  py::gil_scoped_acquire gil;
  static py::object logger =
      py::module::import("logging").attr("getLogger")("opensfm.reconstruction");
  logger.attr("debug")(msg);
}

void ReconstructionGrower::LogBundleStats(const std::string& bundle_type,
                                          const py::dict& report) {
  auto times = report["wall_times"].cast<py::dict>();
  double time_secs =
      times["run"].cast<double>() + times["setup"].cast<double>() +
      times["teardown"].cast<double>() + times["triangulate"].cast<double>();
  int num_images = report["num_images"].cast<int>();
  int num_points = report["num_points"].cast<int>();
  int num_reprojections = report["num_reprojections"].cast<int>();

  std::string msg = "Ran " + bundle_type + " bundle in " +
                    std::to_string(time_secs).substr(
                        0, std::to_string(time_secs).find('.') + 3) +
                    " secs.";
  if (num_points > 0) {
    std::stringstream ss;
    ss << " with " << num_images << "/" << num_points << "/"
       << num_reprojections << " (" << std::fixed << std::setprecision(2)
       << static_cast<double>(num_reprojections) / num_points
       << ") shots/points/proj. (avg. length)";
    msg += ss.str();
  }
  LogInfo(msg);
  LogDebug(report["brief_report"].cast<std::string>());
  for (auto& line : report["irls_report"].cast<py::list>()) {
    LogDebug(line.cast<std::string>());
  }
}

// ---------- ParseExifDict ----------

map::ShotMeasurements ReconstructionGrower::ParseExifDict(const py::dict& exif,
                                                          bool use_altitude,
                                                          double reflat,
                                                          double reflon,
                                                          double refalt) {
  static constexpr double kMaximumAltitude = 1e4;
  static const Vec3d kDefaultGpsStd{5.0, 5.0, 15.0};

  map::ShotMeasurements m;

  // GPS
  if (exif.contains("gps")) {
    py::dict gps = exif["gps"].cast<py::dict>();
    if (gps.contains("latitude") && gps.contains("longitude")) {
      double lat = gps["latitude"].cast<double>();
      double lon = gps["longitude"].cast<double>();
      double alt = 2.0;
      if (use_altitude) {
        double raw_alt =
            gps.contains("altitude") ? gps["altitude"].cast<double>() : 2.0;
        alt = std::min(kMaximumAltitude, raw_alt);
      }
      m.gps_position_.SetValue(
          geo::TopocentricFromLla(lat, lon, alt, reflat, reflon, refalt));

      Vec3d gps_std;
      if (gps.contains("latitude_std") && gps.contains("longitude_std") &&
          gps.contains("altitude_std")) {
        gps_std = Vec3d(gps["longitude_std"].cast<double>(),
                        gps["latitude_std"].cast<double>(),
                        gps["altitude_std"].cast<double>());
      } else if (gps.contains("dop") && gps["dop"].cast<double>() > 0) {
        double dop = gps["dop"].cast<double>();
        gps_std = Vec3d(dop, dop, dop);
      } else {
        gps_std = kDefaultGpsStd;
      }
      m.gps_accuracy_.SetValue(gps_std.cwiseMax(1e-3));
    }
  }

  // OPK
  if (exif.contains("opk")) {
    py::dict opk = exif["opk"].cast<py::dict>();
    if (opk.contains("omega") && opk.contains("phi") && opk.contains("kappa")) {
      m.opk_angles_.SetValue(Vec3d(opk["omega"].cast<double>(),
                                   opk["phi"].cast<double>(),
                                   opk["kappa"].cast<double>()));
      m.opk_accuracy_.SetValue(
          opk.contains("accuracy") ? opk["accuracy"].cast<double>() : 1.0);
    }
  }

  // Orientation
  m.orientation_.SetValue(
      exif.contains("orientation") ? exif["orientation"].cast<int>() : 1);

  // Gravity down
  if (exif.contains("gravity_down")) {
    auto gd = exif["gravity_down"].cast<std::vector<double>>();
    if (gd.size() == 3) {
      m.gravity_down_.SetValue(Vec3d(gd[0], gd[1], gd[2]));
    }
  }

  // Compass
  if (exif.contains("compass")) {
    py::dict compass = exif["compass"].cast<py::dict>();
    if (compass.contains("angle")) {
      m.compass_angle_.SetValue(compass["angle"].cast<double>());
    }
    if (compass.contains("accuracy") && !compass["accuracy"].is_none()) {
      m.compass_accuracy_.SetValue(compass["accuracy"].cast<double>());
    }
  }

  // Capture time
  if (exif.contains("capture_time")) {
    m.capture_time_.SetValue(exif["capture_time"].cast<double>());
  }

  // Sequence key
  if (exif.contains("skey")) {
    m.sequence_key_.SetValue(exif["skey"].cast<std::string>());
  }

  // Relative altitude
  if (exif.contains("relative_altitude")) {
    m.relative_altitude_.SetValue(exif["relative_altitude"].cast<double>());
  }

  return m;
}

// ---------- RemoveOutliersAndUpdate ----------

void ReconstructionGrower::RemoveOutliersAndUpdate(
    map::Map& map, const py::dict& config, ResectionCandidates& candidates,
    const std::vector<map::LandmarkId>& point_ids) {
  auto result = BAHelpers::RemoveOutliers(map, config, point_ids);
  LogInfo("Removed outliers: " + std::to_string(result.outliers.size()));
  candidates.Remove(result.outliers, result.removed_tracks);
}

// ---------- ResectionCandidates ----------

ReconstructionGrower::ResectionCandidates::ResectionCandidates(
    const map::Map& map) {
  // Initialize with existing reconstructed tracks
  for (const auto& [track_id, lm] : map.GetLandmarks()) {
    tracks_.insert(track_id);
  }
}

void ReconstructionGrower::ResectionCandidates::Add(
    const std::unordered_set<map::ShotId>& new_shots,
    const std::unordered_set<map::TrackId>& new_tracks, const map::Map& map,
    const map::TracksManager& tm) {
  // Add new tracks to our set
  for (const auto& tid : new_tracks) {
    tracks_.insert(tid);
  }

  // For each new track, find unreconstructed shots that see it
  for (const auto& tid : new_tracks) {
    auto obs = tm.GetTrackObservationIndices(tid);
    for (const auto& [shot_id, idx] : obs) {
      if (!map.HasShot(shot_id)) {
        candidates_[shot_id].insert(tid);
      }
    }
  }
}

void ReconstructionGrower::ResectionCandidates::Remove(
    const std::vector<std::pair<map::LandmarkId, map::ShotId>>& outliers,
    const std::vector<map::LandmarkId>& removed_tracks) {
  // Remove outlier observations from candidates
  for (const auto& [track_id, shot_id] : outliers) {
    auto it = candidates_.find(shot_id);
    if (it != candidates_.end()) {
      it->second.erase(track_id);
      if (it->second.empty()) candidates_.erase(it);
    }
  }
  // Remove fully removed tracks
  for (const auto& tid : removed_tracks) {
    tracks_.erase(tid);
    // We don't bother iterating all candidates to remove -- they'll just not
    // match existing landmarks at resection time.
  }
}

void ReconstructionGrower::ResectionCandidates::Update(
    const map::Map& map, const map::TracksManager& tm) {
  // Remove shots that are now reconstructed
  absl::flat_hash_map<map::ShotId, absl::flat_hash_set<map::TrackId>> kept;
  kept.reserve(candidates_.size());
  for (auto& [sid, tracks] : candidates_) {
    if (!map.HasShot(sid)) {
      kept.emplace(sid, std::move(tracks));
    }
  }
  candidates_ = std::move(kept);
}

std::vector<std::pair<map::ShotId, int>>
ReconstructionGrower::ResectionCandidates::GetCandidates(
    const std::unordered_set<map::ShotId>& images) const {
  std::vector<std::pair<map::ShotId, int>> result;
  result.reserve(candidates_.size());
  for (const auto& [shot_id, track_set] : candidates_) {
    if (images.count(shot_id)) {
      result.emplace_back(shot_id, static_cast<int>(track_set.size()));
    }
  }
  // Sort descending by count
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  return result;
}

// ---------- TriangulateNewTracks ----------

std::unordered_set<map::TrackId> ReconstructionGrower::TriangulateNewTracks(
    map::Map& map, const map::TracksManager& tracks_manager,
    const std::unordered_set<map::ShotId>& shot_ids, const py::dict& config) {
  auto cfg = MapperConfig::FromPy(config);
  double min_angle = cfg.triangulation_min_ray_angle_deg * M_PI / 180.0;
  double min_depth = cfg.triangulation_min_depth;

  // Gather candidate tracks from the new shots
  std::unordered_set<map::TrackId> candidate_tracks;
  for (const auto& shot_id : shot_ids) {
    if (!tracks_manager.HasShotObservations(shot_id)) {
      continue;
    }
    auto obs = tracks_manager.GetShotObservations(shot_id);
    for (const auto& [track_id, ob] : obs) {
      if (!map.HasLandmark(track_id)) {
        candidate_tracks.insert(track_id);
      }
    }
  }

  // Triangulation buffers
  std::vector<double> thresholds;
  MatX3d origins;
  MatX3d bearings;
  std::vector<std::pair<map::Shot*, map::ObservationIndex>> track_obs;

  std::unordered_set<map::TrackId> created_tracks;

  for (const auto& track_id : candidate_tracks) {
    auto obs_indices = tracks_manager.GetTrackObservationIndices(track_id);

    track_obs.clear();
    for (const auto& [sid, obs_idx] : obs_indices) {
      if (map.HasShot(sid)) {
        track_obs.emplace_back(&map.GetShot(sid), obs_idx);
      }
    }
    if (track_obs.size() < 2) continue;

    const size_t n = track_obs.size();
    origins.resize(n, 3);
    bearings.resize(n, 3);
    thresholds.resize(n);

    for (size_t i = 0; i < n; ++i) {
      auto* shot = track_obs[i].first;
      auto obs_idx = track_obs[i].second;
      const auto& obs = map.GetObservationPool()->Get(obs_idx);

      origins.row(i) = shot->GetPose()->GetOrigin();
      bearings.row(i) = shot->Bearing(obs.point);
      thresholds[i] = cfg.triangulation_threshold;
    }

    auto [success, point] = geometry::TriangulateBearingsMidpoint(
        origins, bearings, thresholds, min_angle, min_depth);

    if (!success) continue;

    // Refine
    Vec3d refined = geometry::PointRefinement(origins, bearings, point, 10);

    // Create landmark and add observations
    auto& lm = map.CreateLandmark(track_id, refined);
    for (const auto& [shot_ptr, obs_idx] : track_obs) {
      map.AddObservationByIndex(shot_ptr, &lm, obs_idx);
    }
    created_tracks.insert(track_id);
  }

  return created_tracks;
}

// ---------- Resect ----------

ReconstructionGrower::ResectionResult ReconstructionGrower::Resect(
    map::Map& map, const map::TracksManager& tracks_manager,
    const map::ShotId& shot_id, const map::CameraId& camera_id,
    double threshold, int min_inliers,
    const std::unordered_map<map::ShotId, RigAssignment>& rig_assignments,
    const std::unordered_map<map::ShotId, map::CameraId>& shot_camera_map,
    const py::dict& config) {
  ResectionResult result;

  // Get observations for this shot
  auto obs_map = tracks_manager.GetShotObservations(shot_id);

  // Gather 2D-3D correspondences
  std::vector<map::TrackId> track_ids;
  track_ids.reserve(obs_map.size());
  int n_correspondences = 0;
  for (const auto& [track_id, obs] : obs_map) {
    if (map.HasLandmark(track_id)) {
      track_ids.push_back(track_id);
      ++n_correspondences;
    }
  }

  result.num_common_points = n_correspondences;
  if (n_correspondences < min_inliers) {
    return result;
  }

  // Build bearing and point matrices
  const auto& cam = map.GetCamera(camera_id);
  MatX3d bs(n_correspondences, 3);
  MatX3d Xs(n_correspondences, 3);

  int idx = 0;
  for (const auto& tid : track_ids) {
    const auto& obs = obs_map.at(tid);
    bs.row(idx) = cam.Bearing(obs.point).transpose();
    Xs.row(idx) = map.GetLandmark(tid).GetGlobalPos().transpose();
    ++idx;
  }

  // RANSAC
  RobustEstimatorParams params;
  params.iterations = 1000;
  auto ransac_result =
      robust::RANSACAbsolutePose(bs, Xs, threshold, params, RANSAC);

  result.num_inliers = static_cast<int>(ransac_result.inliers_indices.size());
  if (result.num_inliers < min_inliers) {
    return result;
  }

  LogInfo(shot_id +
          " resection inliers: " + std::to_string(result.num_inliers) + " / " +
          std::to_string(result.num_common_points));

  // Extract pose from the 3x4 result
  Eigen::Matrix<double, 3, 4> Rt = ransac_result.lo_model;
  Mat3d R = Rt.block<3, 3>(0, 0);
  Vec3d t = Rt.col(3);
  geometry::Pose pose(R, t);

  // Handle rig vs non-rig
  auto rig_it = rig_assignments.find(shot_id);
  if (rig_it == rig_assignments.end()) {
    // Non-rig: create shot directly
    // Ensure rig camera exists (identity, same as camera_id)
    if (!map.HasRigCamera(camera_id)) {
      map.CreateRigCamera(map::RigCamera(geometry::Pose(), camera_id));
    }
    if (!map.HasRigInstance(shot_id)) {
      map.CreateRigInstance(shot_id);
    }
    map.CreateShot(shot_id, camera_id, camera_id, shot_id, pose);
    result.new_shots.insert(shot_id);
  } else {
    // Rig: create the entire instance
    const auto& assignment = rig_it->second;
    const auto& instance_id = assignment.instance_id;

    if (!map.HasRigInstance(instance_id)) {
      map.CreateRigInstance(instance_id);
    }
    auto& rig_instance = map.GetRigInstance(instance_id);

    // Create the resected shot
    if (!map.HasRigCamera(assignment.rig_camera_id)) {
      // TODO : shouldn't we initialize with the real rig camera ?
      map.CreateRigCamera(
          map::RigCamera(geometry::Pose(), assignment.rig_camera_id));
    }
    map.CreateShot(shot_id, camera_id, assignment.rig_camera_id, instance_id,
                   pose);
    result.new_shots.insert(shot_id);

    // Add other shots in the instance
    for (const auto& member_shot_id : assignment.instance_shots) {
      if (member_shot_id == shot_id) {
        continue;
      }
      if (map.HasShot(member_shot_id)) {
        continue;
      }

      auto member_it = rig_assignments.find(member_shot_id);
      if (member_it == rig_assignments.end()) {
        continue;
      }
      const auto& member_rcid = member_it->second.rig_camera_id;

      if (!map.HasRigCamera(member_rcid)) {
        map.CreateRigCamera(map::RigCamera(geometry::Pose(), member_rcid));
      }

      // Create the member shot in the map
      auto member_cam_it = shot_camera_map.find(member_shot_id);
      if (member_cam_it != shot_camera_map.end()) {
        map.CreateShot(member_shot_id, member_cam_it->second, member_rcid,
                       instance_id, geometry::Pose());
        result.new_shots.insert(member_shot_id);
      }
    }

    // Update instance pose from the resected shot
    rig_instance.UpdateInstancePoseWithShot(shot_id, pose);
  }

  // Complete tracks :
  // - For non-rigs, just add inliers
  // - For rigs, in addtion, run triangulation over resected shots

  // Triangulate first ...
  if (rig_assignments.find(shot_id) != rig_assignments.end()) {
    auto new_tracks =
        TriangulateNewTracks(map, tracks_manager, result.new_shots, config);
  }

  // ... add inliers second
  auto* shot_ptr = &map.GetShot(shot_id);
  for (int inlier_idx : ransac_result.inliers_indices) {
    const auto& tid = track_ids[inlier_idx];
    auto obs_idx = tracks_manager.GetObservationIndex(shot_id, tid);
    auto* lm_ptr = &map.GetLandmark(tid);
    map.AddObservationByIndex(shot_ptr, lm_ptr, obs_idx);
    result.inlier_tracks.insert(tid);
  }

  result.success = true;
  return result;
}

// ---------- TriangulationReconstruction ----------

py::dict ReconstructionGrower::TriangulationReconstruction(
    map::Map& map, const map::TracksManager& tracks_manager,
    const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
    const std::unordered_map<map::RigCameraId, map::RigCamera>&
        rig_camera_priors,
    int grid_size, py::object reconstruction, const py::dict& config,
    int outer_iterations, int inner_iterations) {
  py::object py_align =
      py::module::import("opensfm.reconstruction").attr("align_reconstruction");

  AlignedVector<map::GroundControlPoint> empty_gcp;

  for (int i = 0; i < outer_iterations; ++i) {
    // Robust retriangulation: clear all points and re-triangulate
    map.ClearObservationsAndLandmarks();
    sfm::retriangulation::ReconstructFromTracksManager(map, tracks_manager,
                                                       config,
                                                       /*use_robust=*/true);

    for (int j = 0; j < inner_iterations; ++j) {
      py_align(reconstruction, py::list(), config);

      auto brep = BAHelpers::Bundle(map, camera_priors, rig_camera_priors,
                                    empty_gcp, grid_size, config);
      LogBundleStats("GLOBAL", brep);

      auto outlier_result = BAHelpers::RemoveOutliers(map, config);
      LogInfo("Removed outliers: " +
              std::to_string(outlier_result.outliers.size()));
    }
  }

  py::dict report;
  report["num_points"] = py::cast(map.NumberOfLandmarks());
  return report;
}

// ---------- Grow ----------

py::dict ReconstructionGrower::Grow(
    map::Map& map, const map::TracksManager& tracks_manager,
    const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
    const std::unordered_map<map::RigCameraId, map::RigCamera>&
        rig_camera_priors,
    const std::unordered_map<map::ShotId, map::CameraId>& shot_camera_map,
    const std::unordered_map<map::ShotId, RigAssignment>& rig_assignments,
    std::unordered_set<map::ShotId> images, py::object reconstruction,
    py::object data, const py::dict& config) {
  auto cfg = MapperConfig::FromPy(config);

  bool use_altitude = config["use_altitude_tag"].cast<bool>();
  const auto& reference = map.GetTopocentricConverter();

  // Python align_reconstruction function (called via interop)
  py::object py_align =
      py::module::import("opensfm.reconstruction").attr("align_reconstruction");

  AlignedVector<map::GroundControlPoint> empty_gcp;

  ResectionCandidates candidates(map);
  BundleScheduler bundle_sched;
  bundle_sched.interval = cfg.bundle_interval;
  bundle_sched.new_points_ratio = cfg.bundle_new_points_ratio;
  bundle_sched.num_points_last = map.NumberOfLandmarks();
  bundle_sched.num_shots_last = map.NumberOfShots();

  RetriangulateScheduler retri_sched;
  retri_sched.active = cfg.retriangulation;
  retri_sched.ratio = cfg.retriangulation_ratio;
  retri_sched.num_points_last = map.NumberOfLandmarks();

  // Initial candidate population from existing shots
  {
    std::unordered_set<map::ShotId> initial_shots;
    for (const auto& [sid, shot] : map.GetShots()) {
      initial_shots.insert(sid);
    }
    std::unordered_set<map::TrackId> initial_tracks;
    for (const auto& [tid, lm] : map.GetLandmarks()) {
      initial_tracks.insert(tid);
    }
    candidates.Add(initial_shots, initial_tracks, map, tracks_manager);
  }

  int total_resected = 0;
  int total_triangulated = 0;
  int iterations = 0;
  std::unordered_set<map::ShotId> resected_shots;

  // Redundancy fallback state (Fix 5)
  std::unordered_set<map::ShotId> redundant_shots;
  double ratio_redundant = cfg.resect_redundancy_threshold;
  int local_ba_radius = cfg.local_bundle_radius;

  // Max shots limit
  int max_shots_count = config.contains("incremental_max_shots_count")
                            ? config["incremental_max_shots_count"].cast<int>()
                            : 0;

  std::stringstream ss;
  while (true) {
    LogInfo("--------------------------------------------------------");
    auto sorted_candidates = candidates.GetCandidates(images);
    bool any_success = false;

    for (const auto& [shot_id, count] : sorted_candidates) {
      if (!images.count(shot_id)) {
        continue;
      }

      // Check redundancy
      auto shot_obs = tracks_manager.GetShotObservations(shot_id);
      size_t total_tracks_for_shot = shot_obs.size();
      if (total_tracks_for_shot > 0) {
        double ratio = static_cast<double>(count) /
                       static_cast<double>(total_tracks_for_shot);

        ss.str("");
        ss << "Ratio of resected tracks in " << shot_id << ": " << std::fixed
           << std::setprecision(2) << ratio;
        LogInfo(ss.str());

        if (ratio > ratio_redundant) {
          ss.str("");
          ss << "Skipping " << shot_id << " due to high redundancy ("
             << std::fixed << std::setprecision(2) << ratio << " > "
             << ratio_redundant << ")";
          LogInfo(ss.str());

          redundant_shots.insert(shot_id);
          images.erase(shot_id);
          continue;
        }
      }

      // Look up camera for this shot
      auto cam_it = shot_camera_map.find(shot_id);
      if (cam_it == shot_camera_map.end()) {
        continue;
      }
      const auto& cam_id = cam_it->second;

      auto resection = Resect(
          map, tracks_manager, shot_id, cam_id, cfg.resection_threshold,
          cfg.resection_min_inliers, rig_assignments, shot_camera_map, config);

      if (!resection.success) {
        continue;
      }

      any_success = true;
      ++total_resected;

      for (const auto& ns : resection.new_shots) {
        images.erase(ns);
        resected_shots.insert(ns);
        redundant_shots.erase(ns);
      }

      // Set metadata from exif for all newly created shots
      {
        py::gil_scoped_acquire gil;
        for (const auto& ns : resection.new_shots) {
          if (!map.HasShot(ns)) {
            continue;
          }
          py::dict exif = data.attr("load_exif")(ns).cast<py::dict>();
          auto metadata = ParseExifDict(exif, use_altitude, reference.lat_,
                                        reference.long_, reference.alt_);
          map.GetShot(ns).SetShotMeasurements(metadata);
        }
      }

      // Bundle shot poses for the new shots
      {
        std::unordered_set<map::ShotId> new_shot_set(
            resection.new_shots.begin(), resection.new_shots.end());
        BAHelpers::BundleShotPoses(map, new_shot_set, camera_priors,
                                   rig_camera_priors, config);
      }

      // Add resected tracks to candidates
      candidates.Add(resection.new_shots, resection.inlier_tracks, map,
                     tracks_manager);

      {
        std::string shot_names;
        for (const auto& ns : resection.new_shots) {
          if (!shot_names.empty()) {
            shot_names += " and ";
          }
          shot_names += ns;
        }
        LogInfo("Adding " + shot_names + " to the reconstruction");
      }

      // Triangulate new tracks (Fix 2: returns track set)
      auto new_tri_tracks = TriangulateNewTracks(map, tracks_manager,
                                                 resection.new_shots, config);
      total_triangulated += static_cast<int>(new_tri_tracks.size());

      // Add triangulated tracks to candidates (Fix 2)
      {
        std::unordered_set<map::ShotId> empty_shots;
        candidates.Add(empty_shots, new_tri_tracks, map, tracks_manager);
      }
      size_t np = map.NumberOfLandmarks();
      size_t ns = map.NumberOfShots();

      if (retri_sched.Should(np)) {
        LogInfo("Re-triangulating");
        py_align(reconstruction, py::list(), config);
        {
          auto brep =
              BAHelpers::Bundle(map, camera_priors, rig_camera_priors,
                                empty_gcp, cfg.local_bundle_grid, config);
          LogBundleStats("GLOBAL", brep);
        }
        sfm::retriangulation::ReconstructFromTracksManager(map, tracks_manager,
                                                           config);
        candidates.Update(map, tracks_manager);
        {
          auto brep =
              BAHelpers::Bundle(map, camera_priors, rig_camera_priors,
                                empty_gcp, cfg.local_bundle_grid, config);
          LogBundleStats("GLOBAL", brep);
        }

        // Remove outliers after retriangulation
        RemoveOutliersAndUpdate(map, config, candidates);

        retri_sched.Done(map.NumberOfLandmarks());
        bundle_sched.Done(map.NumberOfLandmarks(), map.NumberOfShots());

      } else if (bundle_sched.Should(np, ns)) {
        py_align(reconstruction, py::list(), config);
        {
          auto brep =
              BAHelpers::Bundle(map, camera_priors, rig_camera_priors,
                                empty_gcp, cfg.local_bundle_grid, config);
          LogBundleStats("GLOBAL", brep);
        }

        // Remove outliers after global bundle
        RemoveOutliersAndUpdate(map, config, candidates);

        bundle_sched.Done(map.NumberOfLandmarks(), map.NumberOfShots());

      } else if (local_ba_radius > 0) {
        // Local bundle around newest shot
        auto local_result = BAHelpers::BundleLocal(
            map, camera_priors, rig_camera_priors, empty_gcp, shot_id,
            cfg.local_bundle_grid, config);
        LogBundleStats("LOCAL", local_result.report);
        RemoveOutliersAndUpdate(map, config, candidates,
                                local_result.point_ids);
      }

      LogInfo("Reconstruction now has " + std::to_string(map.NumberOfShots()) +
              " shots.");

      // Check max shots count
      if (max_shots_count > 0 &&
          static_cast<int>(map.NumberOfShots()) >= max_shots_count) {
        LogInfo("Reached maximum number of shots: " +
                std::to_string(max_shots_count));
        any_success = false;  // Force exit after this iteration
      }

      break;  // Process one shot per iteration, re-sort candidates
    }

    if (!any_success) {
      if (!redundant_shots.empty()) {
        images.insert(redundant_shots.begin(), redundant_shots.end());
        redundant_shots.clear();
        ratio_redundant = 1.0;
        local_ba_radius = 0;
      } else {
        LogInfo("Some images can not be added");
        break;
      }
    }
    ++iterations;

    // Check max shots after iteration
    if (max_shots_count > 0 &&
        static_cast<int>(map.NumberOfShots()) >= max_shots_count) {
      break;
    }
  }

  py::dict report;
  report["num_resected"] = total_resected;
  report["num_triangulated"] = total_triangulated;
  report["num_iterations"] = iterations;
  py::list resected_list;
  for (const auto& s : resected_shots) {
    resected_list.append(s);
  }
  report["resected_shots"] = resected_list;
  return report;
}

}  // namespace sfm
