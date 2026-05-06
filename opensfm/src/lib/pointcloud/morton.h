#pragma once

/// Morton codes interleave the bits of discrete 3D coordinates (x, y, z)
/// into a single 64-bit key so that spatially close points are close in
/// the 1-D ordering.  This enables cache-friendly octree construction via
/// a simple sort.
///
/// Reference: Schütz et al., "Rendering Large Point Clouds in Web Browsers"
///            (2020), §3.1 — Potree-style Morton ordering.

#include <cstdint>

namespace pointcloud {

/// Spread the lower 21 bits of v across every third bit position.
/// E.g. 0b...cba → 0b...c00b00a00   (21 input bits → 63 output bits)
inline uint64_t spreadBits3(uint64_t v) {
  v &= 0x1FFFFF;  // keep lower 21 bits
  v = (v | (v << 32)) & 0x1F00000000FFFF;
  v = (v | (v << 16)) & 0x1F0000FF0000FF;
  v = (v | (v << 8)) & 0x100F00F00F00F00F;
  v = (v | (v << 4)) & 0x10C30C30C30C30C3;
  v = (v | (v << 2)) & 0x1249249249249249;
  return v;
}

/// Compact every third bit back into contiguous lower 21 bits.
inline uint64_t compactBits3(uint64_t v) {
  v &= 0x1249249249249249;
  v = (v | (v >> 2)) & 0x10C30C30C30C30C3;
  v = (v | (v >> 4)) & 0x100F00F00F00F00F;
  v = (v | (v >> 8)) & 0x1F0000FF0000FF;
  v = (v | (v >> 16)) & 0x1F00000000FFFF;
  v = (v | (v >> 32)) & 0x1FFFFF;
  return v;
}

/// Encode discrete 3D coordinates (each 0..2^21-1) into a 63-bit Morton code.
/// Bit layout: z20 y20 x20 z19 y19 x19 … z0 y0 x0
inline uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) {
  return spreadBits3(x) | (spreadBits3(y) << 1) | (spreadBits3(z) << 2);
}

/// Decode a 63-bit Morton code back into 21-bit (x, y, z) coordinates.
inline void mortonDecode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
  x = static_cast<uint32_t>(compactBits3(code));
  y = static_cast<uint32_t>(compactBits3(code >> 1));
  z = static_cast<uint32_t>(compactBits3(code >> 2));
}

/// Maximum grid resolution: 2^21 = 2097152 cells per axis.
constexpr uint32_t kMortonMaxCoord = (1u << 21) - 1;
constexpr int kMortonBits = 21;

/// Quantise a floating-point coordinate in [minVal, maxVal] to [0, 2^21-1].
inline uint32_t quantise(float val, float minVal, float rangeInv) {
  float t = (val - minVal) * rangeInv;
  if (t < 0.0f) {
    t = 0.0f;
  }
  if (t > 1.0f) {
    t = 1.0f;
  }
  return static_cast<uint32_t>(t * kMortonMaxCoord);
}

}  // namespace pointcloud
