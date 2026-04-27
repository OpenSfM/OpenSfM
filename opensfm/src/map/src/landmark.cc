#include <map/landmark.h>
#include <map/observation_pool.h>
#include <map/shot.h>

#include <algorithm>

namespace map {

Landmark::Landmark(const LandmarkId& lm_id, const Vec3d& global_pos)
    : id_(lm_id), global_pos_(global_pos), color_(255, 0, 0) {}

void Landmark::SetReprojectionErrors(
    const std::map<ShotId, Eigen::VectorXd>& reproj_errors) {
  reproj_errors_ = reproj_errors;
}

void Landmark::RemoveObservation(Shot* shot) {
  // Remove reprojection errors if present
  RemoveReprojectionError(shot->id_);
  observations_.erase(shot);
}

FeatureId Landmark::GetObservationIdInShot(Shot* shot) const {
  auto obs_it = observations_.find(shot);
  if (obs_it == observations_.end()) {
    throw std::runtime_error("Accessing with invalid shot ptr!");
  }
  return pool_->Get(obs_it->second).feature_id;
}

const Observation& Landmark::GetObservationInShot(Shot* shot) const {
  auto obs_it = observations_.find(shot);
  if (obs_it == observations_.end()) {
    throw std::runtime_error("Accessing with invalid shot ptr!");
  }
  return pool_->Get(obs_it->second);
}

void Landmark::AddObservation(Shot* shot, ObservationIndex obs_idx,
                              ObservationPool* pool) {
  pool_ = pool;
  observations_[shot] = obs_idx;
}
const std::map<Shot*, ObservationIndex, KeyCompare>& Landmark::GetObservations()
    const {
  return observations_;
}

std::map<ShotId, Eigen::VectorXd> Landmark::GetReprojectionErrors() const {
  return reproj_errors_;
}
void Landmark::RemoveReprojectionError(const ShotId& shot_id) {
  auto it = reproj_errors_.find(shot_id);
  if (it != reproj_errors_.end()) {
    reproj_errors_.erase(it);
  }
}

size_t Landmark::NumberOfObservations() const { return observations_.size(); }

};  // namespace map
