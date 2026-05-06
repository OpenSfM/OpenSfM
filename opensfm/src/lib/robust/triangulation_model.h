#pragma once

#include <foundation/numeric.h>
#include <geometry/triangulation.h>

#include "model.h"

/// RANSAC model for robust triangulation of a 3D point from bearing rays.
/// Data = (origin, bearing) pairs. Model = Vec3d (the 3D point).
/// Error metric: angular error between bearing and direction to point.
class Triangulation : public Model<Triangulation, 1, 1> {
 public:
  using Error = typename Model<Triangulation, 1, 1>::Error;
  using Type = Eigen::Vector3d;
  using Data =
      std::pair<Eigen::Vector3d, Eigen::Vector3d>;  // (origin, bearing)
  static const int MINIMAL_SAMPLES = 2;

  /// Threshold is already an angle (radians), no conversion needed.
  static double ThresholdAdapter(const double threshold_angle) {
    return threshold_angle;
  }

  /// Minimal estimation: triangulate from exactly 2 rays using midpoint.
  template <class IT>
  static int Estimate(IT begin, IT end, Type* models) {
    const int n = static_cast<int>(std::distance(begin, end));
    if (n < 2) return 0;

    Eigen::Matrix<double, 2, 3> centers;
    Eigen::Matrix<double, 2, 3> bearings;
    centers.row(0) = begin->first.transpose();
    bearings.row(0) = begin->second.normalized().transpose();
    centers.row(1) = (begin + 1)->first.transpose();
    bearings.row(1) = (begin + 1)->second.normalized().transpose();

    auto result =
        geometry::TriangulateTwoBearingsMidpointSolve(centers, bearings);
    if (!result.first) return 0;
    models[0] = result.second;
    return 1;
  }

  /// Non-minimal estimation: midpoint solve from N rays.
  template <class IT>
  static int EstimateNonMinimal(IT begin, IT end, Type* models) {
    const int n = static_cast<int>(std::distance(begin, end));
    if (n < 2) return 0;

    Eigen::Matrix<double, Eigen::Dynamic, 3> centers(n, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 3> bearings(n, 3);
    int i = 0;
    for (auto it = begin; it != end; ++it, ++i) {
      centers.row(i) = it->first.transpose();
      bearings.row(i) = it->second.normalized().transpose();
    }

    models[0] = geometry::TriangulateBearingsMidpointSolve(centers, bearings);
    return 1;
  }

  /// Evaluate: angular error between bearing and direction to the point.
  static Error Evaluate(const Type& model, const Data& d) {
    const Eigen::Vector3d direction = (model - d.first).normalized();
    const Eigen::Vector3d bearing = d.second.normalized();
    Error e;
    e[0] = geometry::AngleBetweenVectors(direction, bearing);
    return e;
  }
};
