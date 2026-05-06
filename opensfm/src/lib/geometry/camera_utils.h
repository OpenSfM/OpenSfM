#pragma once

/// @file camera_utils.h
/// @brief Bearing-vector utilities for wide-angle and fisheye cameras.

#include <geometry/camera.h>

#include <Eigen/Core>

namespace geometry {

/// Maximum half-angle (from optical axis) we consider reliable for fisheye
/// camera models.  Beyond this the distortion inversion (Newton-Raphson) is
/// prone to divergence, producing garbage bearing directions.
constexpr double kFisheyeMaxHalfAngle = 85.0 * M_PI / 180.0;
constexpr double kEpsSmall = 1e-10;

template <class T>
Eigen::Matrix<T, 3, 1> ClampBearingAngle(const Eigen::Matrix<T, 3, 1>& bearing,
                                         T maxAngle) {
  const T xy = bearing.template head<2>().norm();
  if (xy < T(kEpsSmall)) {
    return bearing;
  }

  const T angle = std::atan2(xy, std::max(bearing.z(), T(0)));
  if (angle <= maxAngle) {
    return bearing;
  }

  Eigen::Matrix<T, 3, 1> clamped;
  clamped.template head<2>() =
      bearing.template head<2>() * (std::sin(maxAngle) / xy);
  clamped.z() = std::cos(maxAngle);
  return clamped;
}

template <class T>
Eigen::Matrix<T, 3, 1> SanitizeFisheyeBearing(
    const Eigen::Matrix<T, 3, 1>& bearing, const Eigen::Matrix<T, 2, 1>& pixel,
    T maxHalfAngle) {
  constexpr T kHalf = T(0.5);

  const T cosMax = std::cos(maxHalfAngle);
  const T sinMax = std::sin(maxHalfAngle);
  const T normR = pixel.norm();
  const T bxy = bearing.template head<2>().norm();

  // 1. Bearing must be forward-facing.
  bool valid = bearing.z() > cosMax;

  // 2. Its xy-direction must agree with the pixel's radial direction.
  //    Skip for near-centre pixels where direction is undefined.
  if (valid && normR > T(kEpsSmall) && bxy > T(kEpsSmall)) {
    valid = bearing.template head<2>().dot(pixel) / (bxy * normR) > kHalf;
  }
  if (valid) {
    // Bearing direction is trustworthy — just clamp the angle.
    return ClampBearingAngle(bearing, maxHalfAngle);
  }

  // Synthesise from the pixel's radial direction (always correct).
  if (normR > T(kEpsSmall)) {
    Eigen::Matrix<T, 3, 1> b;
    b.template head<2>() = pixel.normalized() * sinMax;
    b.z() = cosMax;
    return b;
  }
  return Eigen::Matrix<T, 3, 1>::UnitZ();
}

template <typename T>
Eigen::Matrix<T, 3, 1> ComputeNiceBearing(const Eigen::Matrix<T, 2, 1>& pixel,
                                          const Camera& camera) {
  const auto projType = camera.GetProjectionType();
  const bool fisheye = geometry::IsFisheye(projType);
  const bool spherical = projType == geometry::ProjectionType::SPHERICAL;

  const Eigen::Vector2d norm =
      camera.PixelToNormalizedCoordinates(pixel.template cast<double>());
  Eigen::Vector3d bearing = camera.Bearing(norm);

  if (fisheye) {
    bearing = geometry::SanitizeFisheyeBearing(bearing.normalized(), norm,
                                               kFisheyeMaxHalfAngle);
    // Spherical : unit bearing
  } else if (spherical) {
    bearing.normalize();
    // Perspective : z-normalised bearing
  } else if (std::abs(bearing.z()) > T(kEpsSmall)) {
    bearing /= bearing.z();
    // Tiny Z : prevent extreme scaling by normalising as well
  } else {
    bearing = Eigen::Vector3d::UnitZ();
  }

  return bearing.template cast<T>();
}
}  // namespace geometry
