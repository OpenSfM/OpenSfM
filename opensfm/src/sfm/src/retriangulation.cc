#include <foundation/memory.h>
#include <foundation/threading.h>
#include <geometry/triangulation.h>
#include <map/defines.h>
#include <map/map.h>
#include <map/tracks_manager.h>
#include <omp.h>
#include <sfm/retriangulation.h>
#include <sfm/tracks_helpers.h>

#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <queue>
#include <random>

namespace {
struct TriangulationResult {
  map::TrackId track_id;
  Vec3d position;
  std::vector<std::pair<map::Shot*, map::ObservationIndex>> observations;
};
}  // namespace

namespace sfm::retriangulation {
void RealignMaps(const map::Map& map_from, map::Map& map_to,
                 bool update_points) {
  const auto& map_from_shots = map_from.GetShots();

  const auto& from_ref = map_from.GetTopocentricConverter();
  const auto& to_ref = map_to.GetTopocentricConverter();
  const auto& from_to_offset = to_ref.ToTopocentric(from_ref.GetLlaRef());

  // first, record transforms that remap points of 'to'
  std::unordered_map<map::ShotId, geometry::Similarity> from_to_transforms;
  for (const auto& shot_to : map_to.GetShots()) {
    if (!map_from.HasShot(shot_to.first)) {
      continue;
    }
    const auto& shot_from = map_from.GetShot(shot_to.first);
    auto shot_from_pose = *shot_from.GetPose();
    const auto shot_to_pose = *shot_to.second.GetPose();

    // put 'from' in LLA of 'to'
    shot_from_pose.SetOrigin(shot_from_pose.GetOrigin() + from_to_offset);

    // store the transform that map relative position in 'from' to 'to' :
    //
    // X_to' = 1 / scale_from * Rcw_from * Rwc_to * ( X_to - Oc_to ) +
    // Oc_from_to
    //

    const double scale = shot_from.scale != 0. ? (1.0 / shot_from.scale) : 1.0;
    const Mat3d R_to_from = shot_from_pose.RotationCameraToWorld() *
                            shot_to_pose.RotationWorldToCamera();
    const Vec3d t_from_to = -scale * R_to_from * shot_to_pose.GetOrigin() +
                            shot_from_pose.GetOrigin();

    from_to_transforms[shot_to.first] =
        geometry::Similarity(R_to_from, t_from_to, scale);
  }

  // remap points of 'to' using the computed transforms if needed
  if (update_points) {
    constexpr auto max_dbl = std::numeric_limits<double>::max();
    for (auto& lm : map_to.GetLandmarks()) {
      const auto point = lm.second.GetGlobalPos();
      std::pair<double, map::ShotId> best_shot = std::make_pair(max_dbl, "");
      for (const auto& shot_n_obs : lm.second.GetObservations()) {
        const auto shot = shot_n_obs.first;
        if (map_from_shots.find(shot->GetId()) == map_from_shots.end()) {
          continue;
        }
        const Vec3d ray = point - shot->GetPose()->GetOrigin();
        const double dist2 = ray.squaredNorm();
        if (dist2 < best_shot.first) {
          best_shot = std::make_pair(dist2, shot->GetId());
        }
      }

      if (best_shot.first == max_dbl) {
        continue;
      }
      const auto reference_shot = best_shot.second;
      if (from_to_transforms.find(reference_shot) == from_to_transforms.end()) {
        continue;
      }
      const auto transform = from_to_transforms.at(reference_shot);
      lm.second.SetGlobalPos(transform.Transform(lm.second.GetGlobalPos()));
    }
  }

  // finally, map shots and instances
  std::unordered_set<map::ShotId> to_delete;
  for (auto& shot_to : map_to.GetShots()) {
    // remember any shot not in 'from' but in 'to' for further deletion
    if (!map_from.HasShot(shot_to.first)) {
      to_delete.insert(shot_to.first);
      continue;
    }

    // copy cameras and some metadata
    const auto& shot_from = map_from.GetShot(shot_to.first);
    auto& camera_to = map_to.GetCamera(shot_to.second.GetCamera()->id);
    camera_to.SetParametersValues(shot_from.GetCamera()->GetParametersValues());

    shot_to.second.scale = shot_from.scale;
    shot_to.second.merge_cc = shot_from.merge_cc;
  }

  // only map rig instances (assuming rig camera didn't change)
  for (auto& rig_instance_to : map_to.GetRigInstances()) {
    for (const auto& any_shot : rig_instance_to.second.GetShots()) {
      if (map_from_shots.find(any_shot.first) != map_from_shots.end()) {
        const auto& any_shot_from = map_from_shots.at(any_shot.first);
        auto& to_pose = rig_instance_to.second.GetPose();

        // assign 'from' pose
        to_pose = any_shot_from.GetRigInstance()->GetPose();

        // put 'from' to 'to' LLA
        to_pose.SetOrigin(to_pose.GetOrigin() + from_to_offset);
        break;
      }
    }
  }

  // delete any shot not in 'from' but in 'to'
  for (const auto& shot_id : to_delete) {
    map_to.RemoveShot(shot_id);
  }
}

void ReconstructFromTracksManager(map::Map& map,
                                  const map::TracksManager& tracks_manager,
                                  const py::dict& config) {
  foundation::ScopedMallocArena scoped_malloc_arena;

  const float triangulation_threshold =
      config["triangulation_threshold"].cast<float>();
  const float min_angle = config["triangulation_min_ray_angle"].cast<float>();
  const float min_depth = config["triangulation_min_depth"].cast<float>();
  const int refinement_iterations =
      config["triangulation_refinement_iterations"].cast<int>();
  int processes = config["processes"].cast<int>();
  const float min_angle_rad = min_angle * M_PI / 180.0;

  // Clear existing observations and landmarks
  map.ClearObservationsAndLandmarks();

  // Share TracksManager's observation pool with the Map (immutable, zero-copy)
  map.SetObservationPool(tracks_manager.GetObservationPool());
  const auto* pool = tracks_manager.GetObservationPool().get();

  auto& shots = map.GetShots();
  auto track_ids = tracks_manager.GetTrackIds();

  // Randomize track ids to avoid threads to only get invalid tracks
  std::shuffle(track_ids.begin(), track_ids.end(),
               std::mt19937{std::random_device{}()});

  // Helper lambda to process a single track
  auto process_track =
      [&](const map::TrackId& track_id,
          std::vector<std::pair<map::Shot*, map::ObservationIndex>>& track_obs,
          std::vector<double>& thresholds, MatX3d& origins,
          MatX3d& bearings) -> std::pair<bool, Vec3d> {
    const auto idx_dict = tracks_manager.GetTrackObservationIndices(track_id);

    track_obs.clear();
    track_obs.reserve(idx_dict.size());

    for (const auto& kv : idx_dict) {
      const auto it = shots.find(kv.first);
      if (it != shots.end()) {
        track_obs.emplace_back(&it->second, kv.second);
      }
    }

    if (track_obs.size() < 2) {
      return {false, Vec3d::Zero()};
    }

    const size_t track_size = track_obs.size();
    if (thresholds.size() < track_size) {
      thresholds.resize(track_size, triangulation_threshold);
    }

    if (static_cast<size_t>(origins.rows()) < track_size) {
      origins.resize(track_size, 3);
      bearings.resize(track_size, 3);
    }

    for (size_t j = 0; j < track_size; ++j) {
      origins.row(j) = track_obs[j].first->GetPose()->GetOrigin();
      bearings.row(j) =
          track_obs[j].first->Bearing(pool->Get(track_obs[j].second).point);
    }

    auto res = geometry::TriangulateBearingsMidpoint(
        origins.topRows(track_size), bearings.topRows(track_size), thresholds,
        min_angle_rad, min_depth);

    if (res.first) {
      Vec3d refined = geometry::PointRefinement(
          origins.topRows(track_size), bearings.topRows(track_size), res.second,
          refinement_iterations);
      return {true, refined};
    }
    return {false, Vec3d::Zero()};
  };

  foundation::ConcurrentQueue<TriangulationResult> queue;
  std::atomic<int> producers_active{0};

  // 0 - Parallel track triangulation and valid observation collection per shot
  // (per thread, then re-aggregated)
#pragma omp parallel num_threads(processes)
  {
    const int thread_id = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();

    // Reusable buffers per thread
    std::vector<double> thresholds;
    MatX3d origins, bearings;
    std::vector<std::pair<map::Shot*, map::ObservationIndex>> track_obs;

    if (num_threads == 1) {
      // Sequential fallback
      for (const auto& track_id : track_ids) {
        auto res =
            process_track(track_id, track_obs, thresholds, origins, bearings);
        if (res.first) {
          auto& landmark = map.CreateLandmark(track_id, res.second);
          for (const auto& shot_n_obs : track_obs) {
            map.AddObservationByIndex(shot_n_obs.first, &landmark,
                                      shot_n_obs.second);
          }
        }
      }
    } else {
      // Producer-Consumer Pattern
#pragma omp single
      producers_active = num_threads - 1;

      if (thread_id == 0) {
        // Consumer: Thread 0
        TriangulationResult res;
        while (queue.Pop(res)) {
          auto& landmark = map.CreateLandmark(res.track_id, res.position);
          for (const auto& obs_pair : res.observations) {
            map.AddObservationByIndex(obs_pair.first, &landmark,
                                      obs_pair.second);
          }
        }
      } else {
        // Producers: Threads 1..N-1
        // Manual loop distribution with stride
        for (size_t i = thread_id - 1; i < track_ids.size();
             i += (num_threads - 1)) {
          const auto& track_id = track_ids[i];
          auto res =
              process_track(track_id, track_obs, thresholds, origins, bearings);
          if (res.first) {
            queue.Push({track_id, res.second, track_obs});
          }
        }

        if (--producers_active == 0) {
          queue.Finish();
        }
      }
    }
  }
}

int Triangulate(map::Map& map,
                const std::unordered_set<map::TrackId>& track_ids,
                float reproj_threshold, float min_angle, float min_depth,
                int processing_threads) {
  foundation::ScopedMallocArena scoped_malloc_arena;

  constexpr size_t kDefaultObservations = 10;
  const float min_angle_rad = min_angle * M_PI / 180.0;

  int count = 0;
  std::vector<map::TrackId> track_ids_vec(track_ids.begin(), track_ids.end());
  std::vector<map::TrackId> to_delete;
#pragma omp parallel num_threads(processing_threads)
  {
    std::vector<map::TrackId> thread_to_delete;

    std::vector<double> threshold_list(kDefaultObservations, reproj_threshold);
    MatX3d origins(kDefaultObservations, 3);
    MatX3d bearings(kDefaultObservations, 3);
#pragma omp for schedule(static)
    for (int i = 0; i < static_cast<int>(track_ids_vec.size()); ++i) {
      const auto& track_id = track_ids_vec.at(i);
      if (!map.HasLandmark(track_id)) {
        continue;
      }
      auto& landmark = map.GetLandmark(track_id);
      const auto& observations = landmark.GetObservations();
      const size_t num_observations = observations.size();
      if (num_observations < 2) {
        thread_to_delete.push_back(track_id);
        continue;
      }

      if (threshold_list.size() != num_observations) {
        threshold_list.resize(num_observations, reproj_threshold);
      }
      if (origins.rows() != num_observations) {
        origins.resize(num_observations, 3);
      }
      if (bearings.rows() != num_observations) {
        bearings.resize(num_observations, 3);
      }

      int idx = 0;
      for (const auto& obs_pair : observations) {
        const map::Shot* shot = obs_pair.first;
        origins.row(idx) = shot->GetPose()->GetOrigin();
        bearings.row(idx) = shot->LandmarkBearing(&landmark);
        ++idx;
      }

      // Triangulate using midpoint method
      const auto ok_pos = geometry::TriangulateBearingsMidpoint(
          origins, bearings, threshold_list, min_angle_rad, min_depth);
      if (!ok_pos.first) {
        thread_to_delete.push_back(track_id);
      } else {
        landmark.SetGlobalPos(ok_pos.second);

        // Refine the triangulated point
        const auto pos_refined = geometry::PointRefinement(
            origins, bearings, landmark.GetGlobalPos(), 20);
        landmark.SetGlobalPos(pos_refined);
      }
    }
#pragma omp critical
    {
      to_delete.insert(to_delete.end(), thread_to_delete.begin(),
                       thread_to_delete.end());
    }
  }

  // Remove landmarks that were not triangulated
  for (const auto& track_id : to_delete) {
    if (map.HasLandmark(track_id)) {
      map.RemoveLandmark(track_id);
    }
  }
  return count;
}

}  // namespace sfm::retriangulation
