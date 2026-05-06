#include <geometry/rotation.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

namespace geometry {

Mat3d RotationFromOpk(double omega, double phi, double kappa) {
  // Single-axis rotations (negated) via AngleAxisd.
  const Mat3d Rw = Eigen::AngleAxisd(-omega, Vec3d::UnitX()).toRotationMatrix();
  const Mat3d Rp = Eigen::AngleAxisd(-phi, Vec3d::UnitY()).toRotationMatrix();
  const Mat3d Rk = Eigen::AngleAxisd(-kappa, Vec3d::UnitZ()).toRotationMatrix();

  // OpenSfM camera convention: z forward, y down, x right.
  Mat3d Rc;
  Rc << 1, 0, 0, 0, -1, 0, 0, 0, -1;

  return Rc * Rk * Rp * Rw;
}

Vec3d OpkFromRotation(const Mat3d& rotation_matrix) {
  // Undo camera convention: R_wc = Rc · Rk · Rp · Rw
  // so  Rk · Rp · Rw = Rc^T · R_wc   (Rc is its own inverse).
  // Following the Python convention:  R = rotation_matrix.T @ Rc
  Mat3d Rc;
  Rc << 1, 0, 0, 0, -1, 0, 0, 0, -1;

  const Mat3d R = rotation_matrix.transpose() * Rc;

  const double omega = std::atan2(-R(1, 2), R(2, 2));
  const double phi = std::asin(std::clamp(R(0, 2), -1.0, 1.0));
  const double kappa = std::atan2(-R(0, 1), R(0, 0));

  return Vec3d(omega, phi, kappa);
}

}  // namespace geometry
