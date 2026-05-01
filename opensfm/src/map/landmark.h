#pragma once
#include <map/defines.h>
#include <map/observation_pool.h>

#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
namespace map {
class Shot;
struct Observation;

class Landmark {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Landmark(const LandmarkId& lm_id, const Vec3d& global_pos);

  // Getters and Setters
  Vec3d GetGlobalPos() const { return global_pos_; }
  void SetGlobalPos(const Vec3d& global_pos) { global_pos_ = global_pos; }
  Vec3i GetColor() const { return color_; }
  void SetColor(const Vec3i& color) { color_ = color; }

  // Utility functions
  void AddObservation(Shot* shot, ObservationIndex obs_idx,
                      ObservationPool* pool);
  void RemoveObservation(Shot* shot);
  size_t NumberOfObservations() const;
  FeatureId GetObservationIdInShot(Shot* shot) const;
  const Observation& GetObservationInShot(Shot* shot) const;
  const ObservationMap<Shot*>& GetObservations() const;
  void ClearObservations() { observations_.clear(); }

  // Comparisons
  bool operator==(const Landmark& lm) const { return id_ == lm.id_; }
  bool operator!=(const Landmark& lm) const { return !(*this == lm); }
  bool operator<(const Landmark& lm) const { return id_ < lm.id_; }
  bool operator<=(const Landmark& lm) const { return id_ <= lm.id_; }
  bool operator>(const Landmark& lm) const { return id_ > lm.id_; }
  bool operator>=(const Landmark& lm) const { return id_ >= lm.id_; }

  // Reprojection Errors
  void SetReprojectionErrors(
      const std::map<ShotId, Eigen::VectorXd>& reproj_errors);
  std::map<ShotId, Eigen::VectorXd> GetReprojectionErrors() const;
  void RemoveReprojectionError(const ShotId& shot_id);

  void SetReprojectionWeights(const std::map<ShotId, double>& weights);
  std::map<ShotId, double> GetReprojectionWeights() const;
  void RemoveReprojectionWeight(const ShotId& shot_id);

 public:
  const LandmarkId id_;

 private:
  Vec3d global_pos_;  // point in global
  ObservationMap<Shot*> observations_;
  Vec3i color_;
  std::map<ShotId, Eigen::VectorXd> reproj_errors_;
  std::map<ShotId, double> reproj_weights_;

  // Non-owning pointer to shared observation pool (set by Map)
  ObservationPool* pool_{nullptr};
};
}  // namespace map
