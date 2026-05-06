#pragma once

#include <cmath>

namespace foundation {

/// Bilinear interpolation on a 2D Eigen matrix.
///
/// Samples the matrix at sub-pixel coordinates (y, x) using bilinear weights.
/// Returns `default_val` when the 2×2 neighbourhood falls outside the valid
/// range [0, rows-1) × [0, cols-1).
///
/// Works with any Eigen matrix type that supports `operator()(row, col)`,
/// `.rows()`, and `.cols()`.
template <typename MatType>
float BilinearInterpolation(const MatType& image, float y, float x,
                            float default_val = 0.0f) {
  if (x < 0.0f || x >= image.cols() - 1 || y < 0.0f || y >= image.rows() - 1) {
    return default_val;
  }
  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);
  float dx = x - ix;
  float dy = y - iy;
  float v00 = static_cast<float>(image(iy, ix));
  float v01 = static_cast<float>(image(iy, ix + 1));
  float v10 = static_cast<float>(image(iy + 1, ix));
  float v11 = static_cast<float>(image(iy + 1, ix + 1));
  float v0 = (1 - dx) * v00 + dx * v01;
  float v1 = (1 - dx) * v10 + dx * v11;
  return (1 - dy) * v0 + dy * v1;
}

}  // namespace foundation
