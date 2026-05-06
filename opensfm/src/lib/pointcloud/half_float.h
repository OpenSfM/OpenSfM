#pragma once

#include <cmath>
#include <cstdint>

namespace pointcloud {

/// Convert a 32-bit float to a 16-bit half-float bit pattern.
/// Handles ±0, normals, denormals, ±Inf, NaN.
inline uint16_t floatToHalf(float f) {
  union {
    float f;
    uint32_t u;
  } conv;
  conv.f = f;
  uint32_t bits = conv.u;

  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = ((bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = bits & 0x7FFFFF;

  if (exp > 15) {
    // Overflow → ±Inf (or NaN)
    if (exp == 128 && mantissa != 0) {
      return static_cast<uint16_t>(sign | 0x7E00);  // NaN
    }
    return static_cast<uint16_t>(sign | 0x7C00);  // ±Inf
  }
  if (exp < -14) {
    // Denorm or zero
    if (exp < -24) {
      return static_cast<uint16_t>(sign);  // ±0
    }
    mantissa |= 0x800000;
    int shift = -1 - exp;
    uint16_t half = static_cast<uint16_t>(sign | (mantissa >> (shift + 13)));
    return half;
  }

  return static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mantissa >> 13));
}

/// Convert a 16-bit half-float bit pattern back to a 32-bit float.
inline float halfToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;

  if (exp == 0) {
    if (mantissa == 0) {
      union {
        uint32_t u;
        float f;
      } conv;
      conv.u = sign;
      return conv.f;  // ±0
    }
    // Denormalized → normalize
    while ((mantissa & 0x400) == 0) {
      mantissa <<= 1;
      exp--;
    }
    exp++;
    mantissa &= 0x3FF;
  } else if (exp == 31) {
    union {
      uint32_t u;
      float f;
    } conv;
    conv.u = sign | 0x7F800000 | (mantissa << 13);
    return conv.f;  // ±Inf or NaN
  }

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = sign | ((exp + 112) << 23) | (mantissa << 13);
  return conv.f;
}

}  // namespace pointcloud
