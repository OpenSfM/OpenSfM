#pragma once

#include <foundation/types.h>

namespace geometry {

// ---------------------------------------------------------------------------
// OPK <-> rotation matrix conversions.
//
// OPK (Omega, Phi, Kappa) are photogrammetric Euler angles:
//   Omega (ω): rotation about the X (East) axis
//   Phi   (φ): rotation about the Y (North) axis
//   Kappa (κ): rotation about the Z (Up) axis
//
// RotationFromOpk builds the world-to-camera rotation as:
//   R_wc = Rc · R_κ · R_φ · R_ω
// where Rc = diag(1, -1, -1) is the OpenSfM camera convention flip
// (z forward, y down, x right).
//
// OpkFromRotation extracts (ω, φ, κ) from a world-to-camera rotation.
// ---------------------------------------------------------------------------

/// Build the world-to-camera rotation matrix from OPK angles (in radians).
Mat3d RotationFromOpk(double omega, double phi, double kappa);

/// Extract OPK angles (in radians) from a world-to-camera rotation matrix.
/// Returns Vec3d(omega, phi, kappa).
Vec3d OpkFromRotation(const Mat3d& rotation_matrix);

}  // namespace geometry
