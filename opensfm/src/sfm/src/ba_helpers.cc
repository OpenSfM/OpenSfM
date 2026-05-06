#include <bundle/bundle_adjuster.h>
#include <foundation/types.h>
#include <geometry/triangulation.h>
#include <map/ground_control_points.h>
#include <map/map.h>
#include <map/observation_pool.h>
#include <sfm/ba_helpers.h>
#include <sfm/retriangulation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "geo/geo.h"
#include "map/defines.h"

namespace {
// Returns true if the rows of X (Nx3) are approximately collinear.
// Requires X.rows() >= 3.
bool ArePointsCollinear(const MatX3d& X) {
  const Vec3d X_mean = X.colwise().mean();
  const MatX3d X_zero = X.rowwise() - X_mean.transpose();
  const Mat3d input = X_zero.transpose() * X_zero;
  Eigen::SelfAdjointEigenSolver<MatXd> ses(input, Eigen::EigenvaluesOnly);
  const Vec3d evals = ses.eigenvalues();
  const auto ratio_1st_2nd = std::abs(evals[2] / evals[1]);
  constexpr double epsilon_abs = 1e-10;
  constexpr double epsilon_ratio = 5e3;
  int cond1 = 0;
  for (int i = 0; i < 3; ++i) {
    cond1 += (evals[i] < epsilon_abs) ? 1 : 0;
  }
  return cond1 > 1 || ratio_1st_2nd > epsilon_ratio;
}
}  // namespace

namespace sfm {
std::pair<std::unordered_set<map::ShotId>, std::unordered_set<map::ShotId>>
BAHelpers::ShotNeighborhoodIds(map::Map& map,
                               const map::ShotId& central_shot_id,
                               size_t radius, size_t min_common_points,
                               size_t max_interior_size) {
  auto res = ShotNeighborhood(map, central_shot_id, radius, min_common_points,
                              max_interior_size);
  std::unordered_set<map::ShotId> interior;
  for (map::Shot* shot : res.first) {
    interior.insert(shot->GetId());
  }
  std::unordered_set<map::ShotId> boundary;
  for (map::Shot* shot : res.second) {
    boundary.insert(shot->GetId());
  }
  return std::make_pair(interior, boundary);
}

/**Reconstructed shots near a given shot.

Returns:
    a tuple with interior and boundary:
    - interior: the list of shots at distance smaller than radius
    - boundary: shots sharing at least on point with the interior

Central shot is at distance 0.  Shots at distance n + 1 share at least
min_common_points points with shots at distance n.
*/
std::pair<std::unordered_set<map::Shot*>, std::unordered_set<map::Shot*>>
BAHelpers::ShotNeighborhood(map::Map& map, const map::ShotId& central_shot_id,
                            size_t radius, size_t min_common_points,
                            size_t max_interior_size) {
  constexpr size_t MaxBoundarySize{1000000};
  std::unordered_set<map::Shot*> interior;
  auto& central_shot = map.GetShot(central_shot_id);
  const auto instance_shot =
      map.GetRigInstance(central_shot.GetRigInstanceId()).GetShotIDs();
  for (const auto& s : instance_shot) {
    interior.insert(&map.GetShot(s));
  }
  interior.insert(&central_shot);
  for (size_t distance = 1;
       distance < radius && interior.size() < max_interior_size; ++distance) {
    const auto remaining = max_interior_size - interior.size();
    const auto neighbors =
        DirectShotNeighbors(map, interior, min_common_points, remaining);
    interior.insert(neighbors.begin(), neighbors.end());
  }

  const auto boundary = DirectShotNeighbors(map, interior, 1, MaxBoundarySize);
  return std::make_pair(interior, boundary);
}

std::unordered_set<map::Shot*> BAHelpers::DirectShotNeighbors(
    map::Map& map, const std::unordered_set<map::Shot*>& shot_ids,
    const size_t min_common_points, const size_t max_neighbors) {
  std::unordered_set<map::Landmark*> points;
  for (auto* shot : shot_ids) {
    for (const auto& lm_obs : shot->GetLandmarkObservations()) {
      points.insert(lm_obs.first);
    }
  }

  std::unordered_map<map::Shot*, size_t> common_points;
  for (auto* pt : points) {
    for (const auto& neighbor_p : pt->GetObservations()) {
      auto* shot = neighbor_p.first;
      if (shot_ids.find(shot) == shot_ids.end()) {
        ++common_points[shot];
      }
    }
  }

  std::vector<std::pair<map::Shot*, size_t>> pairs(common_points.begin(),
                                                   common_points.end());
  std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<map::Shot*, size_t>& val1,
               const std::pair<map::Shot*, size_t>& val2) {
              return val1.second > val2.second;
            });

  const size_t max_n = std::min(max_neighbors, pairs.size());
  std::unordered_set<map::Shot*> neighbors;
  size_t idx = 0;
  for (auto& p : pairs) {
    if (p.second >= min_common_points && idx < max_n) {
      const auto instance_shots =
          map.GetRigInstance(p.first->GetRigInstanceId()).GetShotIDs();
      for (const auto& s : instance_shots) {
        neighbors.insert(&map.GetShot(s));
      }
    } else {
      break;
    }
    ++idx;
  }
  return neighbors;
}

BAHelpers::BundleLocalResult BAHelpers::BundleLocal(
    map::Map& map,
    const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
    const std::unordered_map<map::RigCameraId, map::RigCamera>&
        rig_camera_priors,
    const AlignedVector<map::GroundControlPoint>& gcp,
    const map::ShotId& central_shot_id, int grid_size, const py::dict& config) {
  BundleLocalResult result;
  py::dict& report = result.report;

  const auto timer_neighborhood_start =
      std::chrono::high_resolution_clock::now();
  auto neighborhood = ShotNeighborhood(
      map, central_shot_id, config["local_bundle_radius"].cast<size_t>(),
      config["local_bundle_min_common_points"].cast<size_t>(),
      config["local_bundle_max_shots"].cast<size_t>());
  const auto timer_neighborhood_end = std::chrono::high_resolution_clock::now();
  auto& interior = neighborhood.first;
  auto& boundary = neighborhood.second;

  // Convert subset to set for fast lookup
  std::unordered_set<map::ShotId> all_shots_interior;
  for (auto* shot : interior) {
    all_shots_interior.insert(shot->GetId());
  }
  const auto timer_grid_start = std::chrono::high_resolution_clock::now();
  auto selection = SelectTracksGrid(map, all_shots_interior, grid_size);
  const auto& subset = selection.selected_tracks;
  auto& to_retriangulate = selection.other_tracks;
  const auto timer_grid_end = std::chrono::high_resolution_clock::now();

  // set up BA
  const auto start = std::chrono::high_resolution_clock::now();
  auto ba = bundle::BundleAdjuster();
  ba.SetUseAnalyticDerivatives(
      config["bundle_analytic_derivatives"].cast<bool>());

  for (const auto& cam_pair : map.GetCameras()) {
    const auto& cam = cam_pair.second;
    const auto& cam_prior = camera_priors.at(cam.id);
    constexpr bool fix_cameras{true};
    ba.AddCamera(cam.id, cam, cam_prior, fix_cameras);
  }
  // combine the sets
  std::unordered_set<map::Shot*> int_and_bound(interior.cbegin(),
                                               interior.cend());
  int_and_bound.insert(boundary.cbegin(), boundary.cend());

  constexpr bool point_constant{false};
  constexpr bool rig_camera_constant{true};

  // gather required rig data to setup
  std::unordered_set<map::RigCameraId> rig_cameras_ids;
  std::unordered_set<map::RigInstanceId> rig_instances_ids;
  for (auto* shot : int_and_bound) {
    rig_cameras_ids.insert(shot->GetRigCameraId());
    rig_instances_ids.insert(shot->GetRigInstanceId());
  }

  // rig cameras are going to be fixed
  for (const auto& rig_camera_id : rig_cameras_ids) {
    const auto& rig_camera = map.GetRigCamera(rig_camera_id);
    ba.AddRigCamera(rig_camera_id, rig_camera.pose,
                    rig_camera_priors.at(rig_camera_id).pose,
                    rig_camera_constant);
  }

  // add rig instances shots
  const std::string gps_scale_group = "dummy";  // unused for now
  for (const auto& rig_instance_id : rig_instances_ids) {
    auto& instance = map.GetRigInstance(rig_instance_id);
    std::unordered_map<std::string, std::string> shot_cameras, shot_rig_cameras;

    // we're going to assign GPS constraint to the instance itself
    // by averaging its shot's GPS values (and std dev.)
    Vec3d average_position = Vec3d::Zero();
    Vec3d average_std = Vec3d::Zero();
    int gps_count = 0;

    // if any instance's shot is in boundary
    // then the entire instance will be fixed
    bool fix_instance = false;
    for (const auto& shot_n_rig_camera : instance.GetRigCameras()) {
      const auto shot_id = shot_n_rig_camera.first;
      auto& shot = map.GetShot(shot_id);
      shot_cameras[shot_id] = shot.GetCamera()->id;
      shot_rig_cameras[shot_id] = shot_n_rig_camera.second->id;

      const auto is_boundary = boundary.find(&shot) != boundary.end();
      const auto is_interior = !is_boundary;

      if (is_interior) {
        const auto& measurements = shot.GetShotMeasurements();
        if (config["bundle_use_gps"].cast<bool>() &&
            measurements.gps_position_.HasValue()) {
          average_position += measurements.gps_position_.Value();
          average_std += measurements.gps_accuracy_.Value();
          ++gps_count;
        }
      } else {
        fix_instance = true;
      }
    }

    ba.AddRigInstance(rig_instance_id, instance.GetPose(), shot_cameras,
                      shot_rig_cameras, fix_instance);

    // only add averaged rig position constraints to moving instances
    if (!fix_instance && gps_count > 0) {
      average_position /= gps_count;
      average_std /= gps_count;
      ba.AddRigInstancePositionPrior(rig_instance_id, average_position,
                                     average_std, gps_scale_group);
    }
  }

  double t_projections = 0;
  const auto t_pts_start = std::chrono::high_resolution_clock::now();

  // Retrieve a mapping between map shots and bundle shots we're just created
  std::unordered_map<map::Shot*, bundle::Shot*> shot_lookup;
  shot_lookup.reserve(interior.size() + boundary.size());
  for (auto* shot : interior) {
    shot_lookup[shot] = ba.GetShotRaw(shot->id_);
  }
  for (auto* shot : boundary) {
    shot_lookup[shot] = ba.GetShotRaw(shot->id_);
  }

  // Run over selected tracks only and add all their observations
  std::unordered_set<map::Landmark*> points;
  size_t added_landmarks = 0;
  size_t added_reprojections = 0;
  for (const auto& selected_track_id : subset) {
    auto& lm = map.GetLandmark(selected_track_id);

    auto* ba_point = ba.AddPoint(lm.id_, lm.GetGlobalPos(), point_constant);

    points.insert(&lm);
    result.point_ids.push_back(lm.id_);
    ++added_landmarks;

    for (const auto& obs_pair : lm.GetObservations()) {
      auto* shot = obs_pair.first;
      const auto& obs = map.GetObservationPool()->Get(obs_pair.second);

      auto s_it = shot_lookup.find(shot);
      if (s_it == shot_lookup.end()) {
        throw std::runtime_error("Shot " + shot->id_ +
                                 " not found in bundle adjuster");
      }
      ba.AddPointProjectionObservationRaw(s_it->second, ba_point, obs.point,
                                          obs.scale, false, std::nullopt);
      ++added_reprojections;
    }
  }

  t_projections += std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now() - t_pts_start)
                       .count() /
                   1000000.0;

  if (config["bundle_use_gcp"].cast<bool>() && !gcp.empty()) {
    const auto t_gcp_start = std::chrono::high_resolution_clock::now();
    AddGCPToBundle(ba, map, gcp, config, added_landmarks,
                   interior.size() + boundary.size());
    t_projections +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t_gcp_start)
            .count() /
        1000000.0;
  }

  ba.SetInternalParametersPriorSD(
      config["exif_focal_sd"].cast<double>(),
      config["aspect_ratio_sd"].cast<double>(),
      config["principal_point_sd"].cast<double>(),
      config["radial_distortion_k1_sd"].cast<double>(),
      config["radial_distortion_k2_sd"].cast<double>(),
      config["tangential_distortion_p1_sd"].cast<double>(),
      config["tangential_distortion_p2_sd"].cast<double>(),
      config["radial_distortion_k3_sd"].cast<double>(),
      config["radial_distortion_k4_sd"].cast<double>());
  ba.SetRigParametersPriorSD(config["rig_translation_sd"].cast<double>(),
                             config["rig_rotation_sd"].cast<double>());

  ba.SetDefaultDensityRatio(config["bundle_irls_density_ratio"].cast<double>());
  ba.SetGroupDensityRatio(
      "GCP2D", config["bundle_irls_gcp_density_ratio"].cast<double>());

  ba.SetNumThreads(config["processes"].cast<int>());
  ba.SetMaxNumIterations(10);
  ba.SetLinearSolverType("SPARSE_SCHUR");
  const auto timer_setup = std::chrono::high_resolution_clock::now();

  {
    py::gil_scoped_release release;
    ba.Run();
  }

  const auto timer_run = std::chrono::high_resolution_clock::now();
  for (const auto& rig_instance_id : rig_instances_ids) {
    auto& instance = map.GetRigInstance(rig_instance_id);
    auto i = ba.GetRigInstance(rig_instance_id);
    instance.SetPose(i.GetValue());
  }

  // Update points
  for (auto* point : points) {
    auto pt = ba.GetPoint(point->id_);
    if (!pt.GetValue().allFinite()) {
      // set large reprojection errors
      for (auto& proj_error : pt.reprojection_errors) {
        proj_error.second.setConstant(1.0);
      }
      for (auto& proj_weight : pt.reprojection_weights) {
        proj_weight.second = 0.0;
      }
    }
    point->SetGlobalPos(pt.GetValue());
    point->SetReprojectionErrors(pt.reprojection_errors);
    point->SetReprojectionWeights(pt.reprojection_weights);
  }

  const auto timer_teardown = std::chrono::high_resolution_clock::now();
  sfm::retriangulation::Triangulate(
      map, to_retriangulate, config["triangulation_threshold"].cast<float>(),
      config["triangulation_min_ray_angle"].cast<float>(),
      config["triangulation_min_depth"].cast<float>(),
      config["processes"].cast<int>());
  const auto timer_triangulate = std::chrono::high_resolution_clock::now();

  report["brief_report"] = ba.BriefReport();
  report["full_report"] = ba.FullReport();
  report["irls_report"] = ba.IRLSReport();
  report["is_solution_usable"] = ba.CeresSolverSummary().IsSolutionUsable();
  report["termination_type"] =
      ceres::TerminationTypeToString(ba.CeresSolverSummary().termination_type);
  report["wall_times"] = py::dict();
  report["wall_times"]["neighborhood"] =
      std::chrono::duration_cast<std::chrono::microseconds>(
          timer_neighborhood_end - timer_neighborhood_start)
          .count() /
      1000000.0;
  report["wall_times"]["setup"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_setup - start)
          .count() /
      1000000.0;
  report["wall_times"]["setup_projections"] = t_projections;
  report["wall_times"]["setup_other"] =
      report["wall_times"]["setup"].cast<double>() - t_projections;
  report["wall_times"]["grid"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_grid_end -
                                                            timer_grid_start)
          .count() /
      1000000.0;
  report["wall_times"]["run"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_run -
                                                            timer_setup)
          .count() /
      1000000.0;
  report["wall_times"]["teardown"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_teardown -
                                                            timer_run)
          .count() /
      1000000.0;
  report["wall_times"]["triangulate"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_triangulate -
                                                            timer_teardown)
          .count() /
      1000000.0;
  report["num_images"] = interior.size();
  report["num_interior_images"] = interior.size();
  report["num_boundary_images"] = boundary.size();
  report["num_other_images"] =
      map.NumberOfShots() - interior.size() - boundary.size();
  report["num_points"] = added_landmarks;
  report["num_reprojections"] = added_reprojections;
  return result;
}

py::tuple BAHelpers::BundleLocalPython(
    map::Map& map,
    const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
    const std::unordered_map<map::RigCameraId, map::RigCamera>&
        rig_camera_priors,
    const AlignedVector<map::GroundControlPoint>& gcp,
    const map::ShotId& central_shot_id, int grid_size, const py::dict& config) {
  auto result = BundleLocal(map, camera_priors, rig_camera_priors, gcp,
                            central_shot_id, grid_size, config);
  py::list pt_ids;
  for (const auto& id : result.point_ids) {
    pt_ids.append(id);
  }
  return py::make_tuple(pt_ids, result.report);
}

bool BAHelpers::TriangulateGCP(
    const map::GroundControlPoint& point,
    const std::unordered_map<map::ShotId, map::Shot>& shots,
    float reproj_threshold, Vec3d& coordinates) {
  constexpr auto min_ray_angle = 0.1 * M_PI / 180.0;
  constexpr auto min_depth = 1e-3;  // Assume GCPs 1mm+ away from the camera
  MatX3d os, bs;
  size_t added = 0;
  coordinates = Vec3d::Zero();
  bs.conservativeResize(point.observations_.size(), Eigen::NoChange);
  os.conservativeResize(point.observations_.size(), Eigen::NoChange);
  for (const auto& obs : point.observations_) {
    const auto shot_it = shots.find(obs.shot_id_);
    if (shot_it != shots.end()) {
      const auto& shot = (shot_it->second);
      const Vec3d bearing = shot.GetCamera()->Bearing(obs.projection_);
      const auto& shot_pose = shot.GetPose();
      bs.row(added) = shot_pose->RotationCameraToWorld() * bearing;
      os.row(added) = shot_pose->GetOrigin();
      ++added;
    }
  }
  bs.conservativeResize(added, Eigen::NoChange);
  os.conservativeResize(added, Eigen::NoChange);
  if (added >= 2) {
    const std::vector<double> thresholds(added, reproj_threshold);
    const auto& res = geometry::TriangulateBearingsMidpoint(
        os, bs, thresholds, min_ray_angle, min_depth);
    coordinates = res.second;
    return res.first;
  }
  return false;
}

// Add Ground Control Points constraints to the bundle problem
size_t BAHelpers::AddGCPToBundle(
    bundle::BundleAdjuster& ba, const map::Map& map,
    const AlignedVector<map::GroundControlPoint>& gcp, const py::dict& config,
    size_t num_ba_points, size_t num_ba_shots) {
  const auto& reference = map.GetTopocentricConverter();
  const auto& shots = map.GetShots();

  const float reproj_threshold =
      config["gcp_reprojection_error_threshold"].cast<float>();
  const double gcp_global_weight = config["gcp_global_weight"].cast<double>();
  const double gcp_observation_sd = config["gcp_observation_sd"].cast<double>();

  const double tracks_per_shots =
      static_cast<double>(num_ba_points) /
      std::max(static_cast<size_t>(1), num_ba_shots);
  const double weight_factor = std::sqrt(std::max(1.0, tracks_per_shots));

  size_t added_gcp_observations = 0;
  for (const auto& point : gcp) {
    const auto point_id = "gcp-" + point.id_;
    Vec3d coordinates;
    if (!TriangulateGCP(point, shots, reproj_threshold, coordinates)) {
      continue;
    }

    int valid_shots = 0;
    for (const auto& obs : point.observations_) {
      if (shots.count(obs.shot_id_) > 0) {
        ++valid_shots;
      }
    }

    if (!valid_shots) {
      continue;
    }

    const double prior_weight = gcp_global_weight * weight_factor;
    constexpr auto point_constant{false};
    ba.AddPoint(point_id, coordinates, point_constant);
    if (!point.lla_.empty()) {
      const auto point_std = Vec3d(config["gcp_horizontal_sd"].cast<double>(),
                                   config["gcp_horizontal_sd"].cast<double>(),
                                   config["gcp_vertical_sd"].cast<double>());
      ba.AddPointPrior(point_id, reference.ToTopocentric(point.GetLlaVec3d()),
                       point_std / prior_weight, point.has_altitude_);
    }

    const double obs_weight = gcp_global_weight * weight_factor;
    for (const auto& obs : point.observations_) {
      const auto& shot_id = obs.shot_id_;
      if (shots.count(shot_id) > 0) {
        ba.AddPointProjectionObservation(shot_id, point_id, obs.projection_,
                                         gcp_observation_sd / obs_weight, true,
                                         std::nullopt);
        ++added_gcp_observations;
      }
    }
  }
  return added_gcp_observations;
}

BAHelpers::TracksSelection BAHelpers::SelectTracksGrid(
    map::Map& map, const std::unordered_set<map::ShotId>& shot_ids,
    size_t grid_size) {
  TracksSelection selection;
  if (shot_ids.empty() || grid_size <= 1) {
    return selection;
  }

  const auto default_num_tracks = grid_size * grid_size * shot_ids.size() / 2;
  auto& set_selected_tracks = selection.selected_tracks;
  set_selected_tracks.reserve(default_num_tracks);
  auto& set_other_tracks = selection.other_tracks;
  set_other_tracks.reserve(default_num_tracks * 4);

  // Prepare grid cells: each cell holds the longest track (by observation
  // count)
  std::vector<std::pair<map::TrackId, size_t>> grid(grid_size * grid_size,
                                                    {"", 0});

  // For each shot (image)
  for (const auto& shot_id : shot_ids) {
    const auto& shot = map.GetShot(shot_id);
    const int width = shot.GetCamera()->width;
    const int height = shot.GetCamera()->height;
    if (width <= 0 || height <= 0) {
      continue;
    }

    std::fill(grid.begin(), grid.end(),
              std::make_pair<map::TrackId, size_t>("", 0));

    // For each observation in the shot
    for (const auto& lm_obs : shot.GetLandmarkObservations()) {
      auto* lm = lm_obs.first;
      const auto& obs = shot.GetObservationPool()->Get(lm_obs.second);
      set_other_tracks.insert(lm->id_);

      // Get normalized coordinates [0,1]
      const auto normalize = std::max(width, height);
      double x = (obs.point(0) * normalize + width * 0.5) / width;
      double y = (obs.point(1) * normalize + height * 0.5) / height;
      // Clamp to [0,1]
      x = std::max(0.0, std::min(1.0, x));
      y = std::max(0.0, std::min(1.0, y));
      // Compute grid cell
      const int gx = std::min(static_cast<int>(x * grid_size),
                              static_cast<int>(grid_size - 1));
      const int gy = std::min(static_cast<int>(y * grid_size),
                              static_cast<int>(grid_size - 1));
      // Track length = number of observations
      const size_t track_len = lm->GetObservations().size();
      // Keep the longest track in this cell
      if (track_len > grid[gx + gy * grid_size].second &&
          set_selected_tracks.count(lm->id_) == 0) {
        grid[gx + gy * grid_size] = {lm->id_, track_len};
      }
    }

    // Add selected tracks for this shot
    for (int i = 0; i < static_cast<int>(grid_size * grid_size); ++i) {
      const auto& track_id = grid[i].first;
      if (!track_id.empty()) {
        set_selected_tracks.insert(track_id);
      }
    }
  }
  for (const auto& selected_track : set_selected_tracks) {
    set_other_tracks.erase(selected_track);
  }

  return selection;
}

py::dict BAHelpers::BundleShotPoses(
    map::Map& map, const std::unordered_set<map::ShotId>& shot_ids,
    const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
    const std::unordered_map<map::RigCameraId, map::RigCamera>&
        rig_camera_priors,
    const py::dict& config) {
  py::dict report;

  constexpr auto fix_cameras = true;
  constexpr auto fix_points = true;
  constexpr auto fix_rig_camera = true;

  auto ba = bundle::BundleAdjuster();
  ba.SetUseAnalyticDerivatives(
      config["bundle_analytic_derivatives"].cast<bool>());
  const auto start = std::chrono::high_resolution_clock::now();

  // gather required rig data to setup
  std::unordered_set<map::RigInstanceId> rig_instances_ids;
  for (const auto& shot_id : shot_ids) {
    const auto& shot = map.GetShot(shot_id);
    rig_instances_ids.insert(shot.GetRigInstanceId());
  }
  std::unordered_set<map::RigCameraId> rig_cameras_ids;
  std::unordered_set<map::CameraId> cameras_ids;
  for (const auto& rig_instance_id : rig_instances_ids) {
    auto& instance = map.GetRigInstance(rig_instance_id);
    for (const auto& shot_n_rig_camera : instance.GetRigCameras()) {
      const auto rig_camera_id = shot_n_rig_camera.second->id;
      rig_cameras_ids.insert(rig_camera_id);

      const auto shot_id = shot_n_rig_camera.first;
      const auto camera_id = map.GetShot(shot_id).GetCamera()->id;
      cameras_ids.insert(camera_id);
    }
  }

  // rig cameras are going to be fixed
  for (const auto& rig_camera_id : rig_cameras_ids) {
    const auto& rig_camera = map.GetRigCamera(rig_camera_id);
    ba.AddRigCamera(rig_camera_id, rig_camera.pose,
                    rig_camera_priors.at(rig_camera_id).pose, fix_rig_camera);
  }

  for (const auto& camera_id : cameras_ids) {
    const auto& cam = map.GetCamera(camera_id);
    const auto& cam_prior = camera_priors.at(camera_id);
    ba.AddCamera(camera_id, cam, cam_prior, fix_cameras);
  }

  std::unordered_set<map::Landmark*> landmarks;
  for (const auto& shot_id : shot_ids) {
    const auto& shot = map.GetShot(shot_id);
    for (const auto& lm_obs : shot.GetLandmarkObservations()) {
      landmarks.insert(lm_obs.first);
    }
  }
  for (const auto& landmark : landmarks) {
    ba.AddPoint(landmark->id_, landmark->GetGlobalPos(), fix_points);
  }

  // add rig instances shots
  const std::string gps_scale_group = "dummy";  // unused for now
  for (const auto& rig_instance_id : rig_instances_ids) {
    auto& instance = map.GetRigInstance(rig_instance_id);
    std::unordered_map<std::string, std::string> shot_cameras, shot_rig_cameras;

    // we're going to assign GPS constraint to the instance itself
    // by averaging its shot's GPS values (and std dev.)
    Vec3d average_position = Vec3d::Zero();
    Vec3d average_std = Vec3d::Zero();
    int gps_count = 0;

    // if any instance's shot is in boundary
    // then the entire instance will be fixed
    bool fix_instance = false;

    for (const auto& shot_n_rig_camera : instance.GetRigCameras()) {
      const auto shot_id = shot_n_rig_camera.first;
      auto& shot = map.GetShot(shot_id);
      shot_cameras[shot_id] = shot.GetCamera()->id;
      shot_rig_cameras[shot_id] = shot_n_rig_camera.second->id;

      const auto is_fixed = shot_ids.find(shot_id) == shot_ids.end();
      if (!is_fixed) {
        if (config["bundle_use_gps"].cast<bool>()) {
          const auto pos = shot.GetShotMeasurements().gps_position_;
          const auto acc = shot.GetShotMeasurements().gps_accuracy_;
          if (pos.HasValue() && acc.HasValue()) {
            average_position += pos.Value();
            average_std += acc.Value();
            ++gps_count;
          }
        }
      } else {
        fix_instance = true;
      }

      ba.AddRigInstance(rig_instance_id, instance.GetPose(), shot_cameras,
                        shot_rig_cameras, fix_instance);

      // only add averaged rig position constraints to moving instances
      if (!fix_instance && gps_count > 0) {
        average_position /= gps_count;
        average_std /= gps_count;
        ba.AddRigInstancePositionPrior(rig_instance_id, average_position,
                                       average_std, gps_scale_group);
      }
    }
  }

  // add observations
  int added_reprojections = 0;
  const auto t_projections_start = std::chrono::high_resolution_clock::now();
  for (const auto& shot_id : shot_ids) {
    const auto& shot = map.GetShot(shot_id);
    for (const auto& lm_obs : shot.GetLandmarkObservations()) {
      const auto& obs = shot.GetObservationPool()->Get(lm_obs.second);
      ba.AddPointProjectionObservation(shot.id_, lm_obs.first->id_, obs.point,
                                       obs.scale, false, std::nullopt);
      ++added_reprojections;
    }
  }
  const double t_projections =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - t_projections_start)
          .count() /
      1000000.0;

  ba.SetInternalParametersPriorSD(
      config["exif_focal_sd"].cast<double>(),
      config["aspect_ratio_sd"].cast<double>(),
      config["principal_point_sd"].cast<double>(),
      config["radial_distortion_k1_sd"].cast<double>(),
      config["radial_distortion_k2_sd"].cast<double>(),
      config["tangential_distortion_p1_sd"].cast<double>(),
      config["tangential_distortion_p2_sd"].cast<double>(),
      config["radial_distortion_k3_sd"].cast<double>(),
      config["radial_distortion_k4_sd"].cast<double>());
  ba.SetRigParametersPriorSD(config["rig_translation_sd"].cast<double>(),
                             config["rig_rotation_sd"].cast<double>());

  ba.SetDefaultDensityRatio(config["bundle_irls_density_ratio"].cast<double>());
  ba.SetGroupDensityRatio(
      "GCP2D", config["bundle_irls_gcp_density_ratio"].cast<double>());

  ba.SetNumThreads(config["processes"].cast<int>());
  ba.SetMaxNumIterations(10);
  ba.SetLinearSolverType("DENSE_QR");
  const auto timer_setup = std::chrono::high_resolution_clock::now();

  {
    py::gil_scoped_release release;
    ba.Run();
  }

  const auto timer_run = std::chrono::high_resolution_clock::now();

  for (const auto& rig_instance_id : rig_instances_ids) {
    auto& instance = map.GetRigInstance(rig_instance_id);
    auto i = ba.GetRigInstance(rig_instance_id);
    instance.SetPose(i.GetValue());
  }

  const auto timer_teardown = std::chrono::high_resolution_clock::now();
  report["brief_report"] = ba.BriefReport();
  report["full_report"] = ba.FullReport();
  report["irls_report"] = ba.IRLSReport();
  report["is_solution_usable"] = ba.CeresSolverSummary().IsSolutionUsable();
  report["termination_type"] =
      ceres::TerminationTypeToString(ba.CeresSolverSummary().termination_type);
  report["wall_times"] = py::dict();
  report["wall_times"]["setup"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_setup - start)
          .count() /
      1000000.0;
  report["wall_times"]["setup_projections"] = t_projections;
  report["wall_times"]["setup_other"] =
      report["wall_times"]["setup"].cast<double>() - t_projections;
  report["wall_times"]["run"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_run -
                                                            timer_setup)
          .count() /
      1000000.0;
  report["wall_times"]["triangulate"] = 0.0;  // not done in this function
  report["wall_times"]["teardown"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_teardown -
                                                            timer_run)
          .count() /
      1000000.0;
  report["num_images"] = rig_instances_ids.size();
  report["num_points"] = landmarks.size();
  report["num_reprojections"] = added_reprojections;
  return report;
}

py::dict BAHelpers::Bundle(
    map::Map& map,
    const std::unordered_map<map::CameraId, geometry::Camera>& camera_priors,
    const std::unordered_map<map::RigCameraId, map::RigCamera>&
        rig_camera_priors,
    const AlignedVector<map::GroundControlPoint>& gcp, int grid_size,
    const py::dict& config) {
  py::dict report;

  // Get shot ids from the map
  const auto& all_shots = map.GetShots();
  std::unordered_set<map::ShotId> shot_ids;
  for (const auto& shot_pair : all_shots) {
    shot_ids.insert(shot_pair.first);
  }
  const auto timer_grid_start = std::chrono::high_resolution_clock::now();
  auto selection = SelectTracksGrid(map, shot_ids, grid_size);
  const auto& subset = selection.selected_tracks;
  auto& to_retriangulate = selection.other_tracks;
  const auto timer_grid_end = std::chrono::high_resolution_clock::now();

  auto ba = bundle::BundleAdjuster();
  const bool fix_cameras = !config["optimize_camera_parameters"].cast<bool>();
  ba.SetUseAnalyticDerivatives(
      config["bundle_analytic_derivatives"].cast<bool>());
  const auto start = std::chrono::high_resolution_clock::now();

  const auto& all_cameras = map.GetCameras();
  for (const auto& cam_pair : all_cameras) {
    const auto& cam = cam_pair.second;
    const auto& cam_prior = camera_priors.at(cam.id);
    ba.AddCamera(cam.id, cam, cam_prior, fix_cameras);
  }

  // Only add points in the subset
  std::unordered_map<const map::Landmark*, bundle::Point*> landmark_lookup;
  landmark_lookup.reserve(subset.empty() ? map.GetLandmarks().size()
                                         : subset.size());

  // Two different - yet similar - loops to avoid
  // one dummy structure allocation
  if (!subset.empty()) {
    for (const auto& track_id : subset) {
      const auto& pt = map.GetLandmark(track_id);
      ba.AddPoint(pt.id_, pt.GetGlobalPos(), false);
      landmark_lookup[&pt] = ba.GetPointRaw(pt.id_);
    }
  } else {
    for (const auto& lm_pair : map.GetLandmarks()) {
      const auto& pt = lm_pair.second;
      ba.AddPoint(pt.id_, pt.GetGlobalPos(), false);
      landmark_lookup[&pt] = ba.GetPointRaw(pt.id_);
    }
  }

  auto align_method = config["align_method"].cast<std::string>();
  if (align_method.compare("auto") == 0) {
    align_method = DetectAlignmentConstraints(map, config, gcp);
  }
  bool do_add_align_vector = false;
  Vec3d up_vector = Vec3d::Zero();
  if (align_method.compare("orientation_prior") == 0) {
    const std::string align_orientation_prior =
        config["align_orientation_prior"].cast<std::string>();
    if (align_orientation_prior.compare("vertical") == 0) {
      do_add_align_vector = true;
      up_vector = Vec3d(0, 0, -1);
    } else if (align_orientation_prior.compare("horizontal") == 0) {
      do_add_align_vector = true;
      up_vector = Vec3d(0, -1, 0);
    }
  }

  // Setup rig cameras
  constexpr size_t kMinRigInstanceForAdjust{10};
  const size_t shots_per_rig_cameras =
      map.GetRigCameras().size() > 0
          ? static_cast<size_t>(map.GetShots().size() /
                                map.GetRigCameras().size())
          : 1;
  float avg_rig_cameras_per_instance = 0;
  for (auto instance_pair : map.GetRigInstances()) {
    int num_rig_cameras = instance_pair.second.GetRigCameras().size();
    avg_rig_cameras_per_instance += num_rig_cameras;
  }
  avg_rig_cameras_per_instance /=
      std::max(static_cast<size_t>(1), map.GetRigInstances().size());

  // Whatever happens, never adjust rig cameras if there's not enough
  // instances (can sometimes be unstable in incremental SfM)
  const auto force_lock_rig_camera =
      shots_per_rig_cameras <= kMinRigInstanceForAdjust;

  // Controlled by the user : do we actually adjust rigs at all ?
  bool adjust_rig_cameras = config["optimize_rig_parameters"].cast<bool>();

  // Safety check : in  case of no-rigs, if the user asked for rig camera
  // optimization, we disable it to avoid unintended consequences (all cameras
  // will be rig cameras and optimization will be unstable)
  const bool is_no_rig =
      (std::fabs(avg_rig_cameras_per_instance -
                 float(map.GetRigInstances().size())) < 1e-6);
  if (adjust_rig_cameras && is_no_rig) {
    adjust_rig_cameras = false;
  }

  for (const auto& camera_pair : map.GetRigCameras()) {
    ba.AddRigCamera(camera_pair.first, camera_pair.second.pose,
                    rig_camera_priors.at(camera_pair.first).pose,
                    !adjust_rig_cameras || force_lock_rig_camera);
  }

  // Setup rig instances
  const std::string gps_scale_group = "dummy";  // unused for now
  for (auto instance_pair : map.GetRigInstances()) {
    auto& instance = instance_pair.second;

    Vec3d average_position = Vec3d::Zero();
    Vec3d average_std = Vec3d::Zero();
    int gps_count = 0;

    // average GPS and assign GPS constraint to the instance
    std::unordered_map<std::string, std::string> shot_cameras, shot_rig_cameras;
    for (const auto& shot_n_rig_camera : instance.GetRigCameras()) {
      const auto shot_id = shot_n_rig_camera.first;
      const auto& shot = map.GetShot(shot_id);
      shot_cameras[shot_id] = shot.GetCamera()->id;
      shot_rig_cameras[shot_id] = shot_n_rig_camera.second->id;

      if (config["bundle_use_gps"].cast<bool>()) {
        const auto pos = shot.GetShotMeasurements().gps_position_;
        const auto acc = shot.GetShotMeasurements().gps_accuracy_;
        if (pos.HasValue() && acc.HasValue()) {
          if ((acc.Value().array() <= 0).any()) {
            throw std::runtime_error(
                "Shot " + shot.GetId() +
                " has an accuracy component <= 0."
                " Try modifying "
                "your input parser to filter such values.");
          }
          average_position += pos.Value();
          average_std += acc.Value();
          ++gps_count;
        }
      }
    }

    ba.AddRigInstance(instance_pair.first, instance.GetPose(), shot_cameras,
                      shot_rig_cameras, false);

    if (config["bundle_use_gps"].cast<bool>() && gps_count > 0) {
      average_position /= gps_count;
      average_std /= gps_count;
      ba.AddRigInstancePositionPrior(instance_pair.first, average_position,
                                     average_std, gps_scale_group);
    }
  }

  double t_projections = 0;
  const auto t_obs_start = std::chrono::high_resolution_clock::now();
  size_t added_reprojections = 0;

  std::unordered_map<const map::Shot*, bundle::Shot*> shot_lookup;
  shot_lookup.reserve(map.GetShots().size());

  for (const auto& shot_pair : map.GetShots()) {
    const auto& shot = shot_pair.second;

    if (do_add_align_vector) {
      constexpr double std_dev = 1e-3;
      ba.AddAbsoluteUpVector(shot.id_, up_vector, std_dev);
    }
    shot_lookup[&shot] = ba.GetShotRaw(shot.id_);
  }

  for (const auto& lm_pair : landmark_lookup) {
    const map::Landmark* lm = lm_pair.first;
    bundle::Point* bp = lm_pair.second;

    for (const auto& obs_entry : lm->GetObservations()) {
      map::Shot* shot = obs_entry.first;
      const auto& obs = shot->GetObservationPool()->Get(obs_entry.second);

      auto s_it = shot_lookup.find(shot);
      if (s_it != shot_lookup.end()) {
        ba.AddPointProjectionObservationRaw(s_it->second, bp, obs.point,
                                            obs.scale, false, std::nullopt);
        ++added_reprojections;
      }
    }
  }
  t_projections += std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now() - t_obs_start)
                       .count() /
                   1000000.0;

  if (config["bundle_use_gcp"].cast<bool>() && !gcp.empty()) {
    const auto t_gcp_start = std::chrono::high_resolution_clock::now();
    AddGCPToBundle(ba, map, gcp, config, landmark_lookup.size(),
                   map.GetShots().size());
    t_projections +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t_gcp_start)
            .count() /
        1000000.0;
  }

  if (config["bundle_compensate_gps_bias"].cast<bool>() && !gcp.empty()) {
    const auto& biases = map.GetBiases();
    const auto bias_mask = DetermineGCPBiasParameters(map, gcp, config);
    const auto bias_indices = bundle::SimilarityMaskToIndices(bias_mask);
    for (const auto& camera : map.GetCameras()) {
      ba.SetCameraBias(camera.first, biases.at(camera.first), bias_indices);
    }
  }

  ba.SetInternalParametersPriorSD(
      config["exif_focal_sd"].cast<double>(),
      config["aspect_ratio_sd"].cast<double>(),
      config["principal_point_sd"].cast<double>(),
      config["radial_distortion_k1_sd"].cast<double>(),
      config["radial_distortion_k2_sd"].cast<double>(),
      config["tangential_distortion_p1_sd"].cast<double>(),
      config["tangential_distortion_p2_sd"].cast<double>(),
      config["radial_distortion_k3_sd"].cast<double>(),
      config["radial_distortion_k4_sd"].cast<double>());
  ba.SetRigParametersPriorSD(config["rig_translation_sd"].cast<double>(),
                             config["rig_rotation_sd"].cast<double>());

  ba.SetDefaultDensityRatio(config["bundle_irls_density_ratio"].cast<double>());
  ba.SetGroupDensityRatio(
      "GCP2D", config["bundle_irls_gcp_density_ratio"].cast<double>());

  ba.SetNumThreads(config["processes"].cast<int>());
  ba.SetMaxNumIterations(config["bundle_max_iterations"].cast<int>());
  ba.SetLinearSolverType("SPARSE_SCHUR");
  const auto timer_setup = std::chrono::high_resolution_clock::now();

  {
    py::gil_scoped_release release;
    ba.Run();
  }

  const auto timer_run = std::chrono::high_resolution_clock::now();

  BundleToMap(ba, map, !fix_cameras);
  const auto timer_teardown = std::chrono::high_resolution_clock::now();

  if (!subset.empty()) {
    sfm::retriangulation::Triangulate(
        map, to_retriangulate, config["triangulation_threshold"].cast<float>(),
        config["triangulation_min_ray_angle"].cast<float>(),
        config["triangulation_min_depth"].cast<float>(),
        config["processes"].cast<int>());
  }
  const auto timer_triangulate = std::chrono::high_resolution_clock::now();

  report["brief_report"] = ba.BriefReport();
  report["irls_report"] = ba.IRLSReport();
  report["full_report"] = ba.FullReport();
  report["is_solution_usable"] = ba.CeresSolverSummary().IsSolutionUsable();
  report["termination_type"] =
      ceres::TerminationTypeToString(ba.CeresSolverSummary().termination_type);
  report["wall_times"] = py::dict();
  report["wall_times"]["setup"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_setup - start)
          .count() /
      1000000.0;
  report["wall_times"]["setup_projections"] = t_projections;
  report["wall_times"]["setup_other"] =
      report["wall_times"]["setup"].cast<double>() - t_projections;
  report["wall_times"]["grid"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_grid_end -
                                                            timer_grid_start)
          .count() /
      1000000.0;
  report["wall_times"]["run"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_run -
                                                            timer_setup)
          .count() /
      1000000.0;
  report["wall_times"]["teardown"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_teardown -
                                                            timer_run)
          .count() /
      1000000.0;
  report["wall_times"]["triangulate"] =
      std::chrono::duration_cast<std::chrono::microseconds>(timer_triangulate -
                                                            timer_teardown)
          .count() /
      1000000.0;
  report["num_images"] = map.GetShots().size();
  report["num_points"] =
      subset.empty() ? map.GetLandmarks().size() : subset.size();
  report["num_reprojections"] = added_reprojections;
  return report;
}

void BAHelpers::BundleToMap(const bundle::BundleAdjuster& bundle_adjuster,
                            map::Map& output_map, bool update_cameras) {
  // update cameras
  if (update_cameras) {
    for (auto& cam : output_map.GetCameras()) {
      const auto& ba_cam = bundle_adjuster.GetCamera(cam.first);
      for (const auto& p : ba_cam.GetParametersMap()) {
        cam.second.SetParameterValue(p.first, p.second);
      }
    }
  }

  // Update bias
  for (auto& bias : output_map.GetBiases()) {
    const auto& new_bias = bundle_adjuster.GetBias(bias.first);
    if (!new_bias.IsValid()) {
      throw std::runtime_error("Bias " + bias.first +
                               " has either NaN or INF values.");
    }
    bias.second = new_bias;
  }

  // Update rig instances
  for (auto& instance : output_map.GetRigInstances()) {
    const auto new_instance =
        bundle_adjuster.GetRigInstance(instance.first).GetValue();
    if (!new_instance.IsValid()) {
      throw std::runtime_error("Rig Instance " + instance.first +
                               " has either NaN or INF values.");
    }
    instance.second.SetPose(new_instance);
  }

  // Update rig cameras
  for (auto& rig_camera : output_map.GetRigCameras()) {
    const auto new_rig_camera =
        bundle_adjuster.GetRigCamera(rig_camera.first).GetValue();
    if (!new_rig_camera.IsValid()) {
      throw std::runtime_error("Rig Camera " + rig_camera.first +
                               " has either NaN or INF values.");
    }
    rig_camera.second.pose = new_rig_camera;
  }

  // Update points
  for (auto& point : output_map.GetLandmarks()) {
    if (!bundle_adjuster.HasPoint(point.first)) {
      continue;
    }
    auto pt = bundle_adjuster.GetPoint(point.first);
    if (!pt.GetValue().allFinite()) {
      // set large reprojection errors
      for (auto& proj_error : pt.reprojection_errors) {
        proj_error.second.setConstant(1.0);
      }
      for (auto& proj_weight : pt.reprojection_weights) {
        proj_weight.second = 0.0;
      }
    }
    point.second.SetGlobalPos(pt.GetValue());
    point.second.SetReprojectionErrors(pt.reprojection_errors);
    point.second.SetReprojectionWeights(pt.reprojection_weights);
  }
}

void BAHelpers::AlignmentConstraints(
    const map::Map& map, const py::dict& config,
    const AlignedVector<map::GroundControlPoint>& gcp, MatX3d& Xp, MatX3d& X) {
  size_t reserve_size = 0;
  const auto& shots = map.GetShots();
  if (!gcp.empty() && config["bundle_use_gcp"].cast<bool>()) {
    reserve_size += gcp.size();
  }
  if (config["bundle_use_gps"].cast<bool>()) {
    for (const auto& shot_p : shots) {
      const auto& shot = shot_p.second;
      if (shot.GetShotMeasurements().gps_position_.HasValue()) {
        reserve_size += 1;
      }
    }
  }
  Xp.conservativeResize(reserve_size, Eigen::NoChange);
  X.conservativeResize(reserve_size, Eigen::NoChange);
  const auto& topocentricConverter = map.GetTopocentricConverter();
  size_t idx = 0;
  // Triangulated vs measured points
  if (!gcp.empty() && config["bundle_use_gcp"].cast<bool>()) {
    for (const auto& point : gcp) {
      if (point.lla_.empty()) {
        continue;
      }
      Vec3d coordinates;
      if (TriangulateGCP(
              point, shots,
              config["gcp_reprojection_error_threshold"].cast<float>(),
              coordinates)) {
        Xp.row(idx) = topocentricConverter.ToTopocentric(point.GetLlaVec3d());
        X.row(idx) = coordinates;
        ++idx;
      }
    }
  }
  if (config["bundle_use_gps"].cast<bool>()) {
    for (const auto& shot_p : shots) {
      const auto& shot = shot_p.second;
      const auto pos = shot.GetShotMeasurements().gps_position_;
      if (pos.HasValue()) {
        Xp.row(idx) = pos.Value();
        X.row(idx) = shot.GetPose()->GetOrigin();
        ++idx;
      }
    }
  }
}

std::string BAHelpers::DetectAlignmentConstraints(
    const map::Map& map, const py::dict& config,
    const AlignedVector<map::GroundControlPoint>& gcp) {
  MatX3d X, Xp;
  AlignmentConstraints(map, config, gcp, Xp, X);
  if (X.rows() < 3) {
    return "orientation_prior";
  }
  if (ArePointsCollinear(X)) {
    return "orientation_prior";
  }

  return "naive";
}

bundle::SimilarityParameterMask BAHelpers::DetermineGCPBiasParameters(
    const map::Map& map, const AlignedVector<map::GroundControlPoint>& gcp,
    const py::dict& config) {
  using Mask = bundle::SimilarityParameterMask;

  const auto& shots = map.GetShots();
  const float reproj_threshold =
      config["gcp_reprojection_error_threshold"].cast<float>();

  // Triangulate GCPs and collect their 3D positions
  std::vector<Vec3d> gcp_positions;
  for (const auto& point : gcp) {
    Vec3d coordinates;
    if (TriangulateGCP(point, shots, reproj_threshold, coordinates)) {
      gcp_positions.push_back(coordinates);
    }
  }

  const size_t n = gcp_positions.size();
  if (n == 0) {
    return Mask::All;
  }
  if (n == 1) {
    return Mask::Translation;
  }
  if (n == 2) {
    return Mask::Translation | Mask::Scale;
  }

  // 3+ GCPs: check spatial conditioning
  MatX3d X(n, 3);
  for (size_t i = 0; i < n; ++i) {
    X.row(i) = gcp_positions[i];
  }
  if (ArePointsCollinear(X)) {
    return Mask::Translation | Mask::Scale;
  }

  return Mask::All;
}

BAHelpers::OutlierRemovalResult BAHelpers::RemoveOutliers(
    map::Map& map, const py::dict& config,
    const std::vector<map::LandmarkId>& point_ids) {
  OutlierRemovalResult result;
  const auto& all_landmarks = map.GetLandmarks();

  // Generic iteration: avoid building a temporary vector of pointers.
  // for_each_point calls fn(const Landmark&) on the relevant set.
  auto for_each_point = [&](auto&& fn) {
    if (point_ids.empty()) {
      for (const auto& [id, lm] : all_landmarks) {
        fn(lm);
      }
    } else {
      for (const auto& pid : point_ids) {
        auto it = all_landmarks.find(pid);
        if (it != all_landmarks.end()) {
          fn(it->second);
        }
      }
    }
  };

  // Compute threshold
  const std::string filter_type =
      config["bundle_outlier_filtering_type"].cast<std::string>();
  double threshold = 1.0;
  if (filter_type == "FIXED") {
    threshold = config["bundle_outlier_fixed_threshold"].cast<double>();
  } else if (filter_type == "AUTO") {
    // Compute robust stats over ALL landmarks (not just the subset)
    std::vector<Eigen::Vector2d> all_errors;
    for (const auto& [id, lm] : all_landmarks) {
      for (const auto& [shot_id, err] : lm.GetReprojectionErrors()) {
        all_errors.emplace_back(err[0], err[1]);
      }
    }
    if (!all_errors.empty()) {
      // Median of errors (componentwise)
      std::vector<double> xs, ys;
      xs.reserve(all_errors.size());
      ys.reserve(all_errors.size());
      for (const auto& e : all_errors) {
        xs.push_back(e[0]);
        ys.push_back(e[1]);
      }
      std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
      std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
      double median_x = xs[xs.size() / 2];
      double median_y = ys[ys.size() / 2];

      // MAD-based robust std: 1.486 * median(|error - median|)
      std::vector<double> deviations;
      deviations.reserve(all_errors.size());
      for (const auto& e : all_errors) {
        double dx = e[0] - median_x;
        double dy = e[1] - median_y;
        deviations.push_back(std::sqrt(dx * dx + dy * dy));
      }
      std::nth_element(deviations.begin(),
                       deviations.begin() + deviations.size() / 2,
                       deviations.end());
      double robust_std = 1.486 * deviations[deviations.size() / 2];

      double mean_norm =
          std::sqrt((median_x + robust_std) * (median_x + robust_std) +
                    (median_y + robust_std) * (median_y + robust_std));
      // Actually match numpy: norm(mean + std) where mean is 2D vector and std
      // is scalar-ish. The Python does: np.linalg.norm(robust_mean +
      // robust_std) where robust_mean is 2D, robust_std is scalar. This adds
      // std to each component, then takes norm.
      mean_norm = std::sqrt((median_x + robust_std) * (median_x + robust_std) +
                            (median_y + robust_std) * (median_y + robust_std));

      double auto_ratio = config["bundle_outlier_auto_ratio"].cast<double>();
      threshold = auto_ratio * mean_norm;
    }
  }

  const double threshold_sqr = threshold * threshold;
  const double weight_threshold =
      config.contains("bundle_outlier_weight_threshold")
          ? config["bundle_outlier_weight_threshold"].cast<double>()
          : 0.5;

  // Collect outliers
  for_each_point([&](const map::Landmark& lm) {
    for (const auto& [shot_id, error] : lm.GetReprojectionErrors()) {
      double err_sqr = error[0] * error[0] + error[1] * error[1];
      if (err_sqr > threshold_sqr) {
        result.outliers.emplace_back(lm.id_, shot_id);
      }
    }
    for (const auto& [shot_id, weight] : lm.GetReprojectionWeights()) {
      if (weight < weight_threshold) {
        result.outliers.emplace_back(lm.id_, shot_id);
      }
    }
  });

  // Deduplicate
  std::sort(result.outliers.begin(), result.outliers.end());
  result.outliers.erase(
      std::unique(result.outliers.begin(), result.outliers.end()),
      result.outliers.end());

  // Remove observations and collect affected tracks
  std::unordered_set<map::LandmarkId> affected_tracks;
  for (const auto& [track_id, shot_id] : result.outliers) {
    if (all_landmarks.find(track_id) != all_landmarks.end()) {
      map.RemoveObservation(shot_id, track_id);
      affected_tracks.insert(track_id);
    }
  }

  // Remove landmarks with < 2 observations
  for (const auto& track_id : affected_tracks) {
    auto it = map.GetLandmarks().find(track_id);
    if (it != map.GetLandmarks().end()) {
      if (it->second.NumberOfObservations() < 2) {
        map.RemoveLandmark(track_id);
        result.removed_tracks.push_back(track_id);
      }
    }
  }

  // Clear reproj errors and weights on ALL landmarks to free memory
  for (auto& [id, lm] : map.GetLandmarks()) {
    lm.SetReprojectionErrors({});
    lm.SetReprojectionWeights({});
  }

  return result;
}

py::tuple BAHelpers::RemoveOutliersPython(
    map::Map& map, const py::dict& config,
    const std::vector<map::LandmarkId>& point_ids) {
  auto result = RemoveOutliers(map, config, point_ids);

  py::list py_outliers;
  for (const auto& [track_id, shot_id] : result.outliers) {
    py_outliers.append(py::make_tuple(track_id, shot_id));
  }
  py::set py_removed;
  for (const auto& tid : result.removed_tracks) {
    py_removed.add(py::cast(tid));
  }
  return py::make_tuple(py_outliers, py_removed);
}
}  // namespace sfm
