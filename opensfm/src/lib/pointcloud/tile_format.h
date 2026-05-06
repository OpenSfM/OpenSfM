#pragma once


/// Directory layout on disk:
///   point_cloud/
///     metadata.json          — AABB, point count, depth, etc.
///     r.bin                  — root tile
///     r0.bin … r7.bin        — depth-1 children
///     r00.bin … r77.bin      — depth-2 children
///     …
///
/// Each tile file is a compact binary blob:
///   [TileHeader]  (fixed-size, 48 bytes)
///   [PointRecord × numPoints]  (20 bytes each)
///
/// The octree key encodes the path from root to the node:
///   'r' = root, then each digit 0-7 is a child octant index.
///   Potree convention: octant index = 4*z + 2*y + x (each bit from the
///   current depth's bit of the Morton code).
///
/// Per-point attributes are designed for high-quality splatting:
///   - position  (3 × float16, relative to tile AABB min)
///   - normal    (3 × int8, snorm [-1,1] mapped from [-127,127])
///   - color     (3 × uint8, RGB)
///   - radius    (float16, world-space ellipsoid semi-major)
///   - padding to 20 bytes (nice alignment)
///
/// Reference: Schütz 2020, MPC paper §3.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pointcloud {

// ── Per-point record (20 bytes, tightly packed) ─────────────────────────────

#pragma pack(push, 1)

struct PointRecord {
  // Position: half-float relative to the tile AABB origin.
  // Stored as uint16_t (IEEE 754 binary16 bit pattern).
  uint16_t px, py, pz;

  // Normal: signed normalised [-127, 127] → [-1, 1].
  int8_t nx, ny, nz;

  // Color: sRGB 0-255.
  uint8_t r, g, b;

  // Splat radius (half-float, world-space units).
  uint16_t radius;

  // Padding to reach 20 bytes.  Reserved for future use (e.g. class label).
  uint8_t _pad[6];
};

static_assert(sizeof(PointRecord) == 20, "PointRecord must be 20 bytes");

#pragma pack(pop)

// ── Tile header (64 bytes) ──────────────────────────────────────────────────

#pragma pack(push, 1)

struct TileHeader {
  uint32_t magic;      // 'OSPT' = 0x5450534F
  uint32_t version;    // 1
  uint32_t numPoints;  // number of points in this tile
  uint32_t childMask;  // 8-bit mask: which of the 8 children exist on disk

  // Tile AABB (world-space). Points are stored relative to aabbMin.
  float aabbMin[3];
  float aabbMax[3];

  // Average point spacing in this tile.
  float spacing;
  // Depth level in the octree (root = 0).
  uint32_t depth;
};

static_assert(sizeof(TileHeader) == 48, "TileHeader must be 48 bytes");

#pragma pack(pop)

// ── Magic / version constants ───────────────────────────────────────────────

constexpr uint32_t kTileMagic = 0x5450534F;  // 'OSPT' little-endian
constexpr uint32_t kTileVersion = 1;

// ── In-memory octree node ───────────────────────────────────────────────────

/// Represents one node in the octree hierarchy (used during construction
/// and at runtime for traversal).
struct OctreeNode {
  std::string key;  // e.g. "r", "r0", "r03", …
  int depth{0};
  uint32_t childMask{0};  // which of 8 children are non-empty

  // World-space AABB.
  std::array<float, 3> aabbMin{};
  std::array<float, 3> aabbMax{};

  float spacing{0.0f};

  // Points assigned to this node (indices into the full point array).
  std::vector<uint32_t> pointIndices;

  // Children (null = absent).  Index by octant (0-7).
  OctreeNode* children[8]{};

  ~OctreeNode() {
    for (auto*& c : children) {
      delete c;
      c = nullptr;
    }
  }
};

// ── Metadata JSON ───────────────────────────────────────────────────────────

/// Top-level metadata written alongside the tile files.
struct OctreeMetadata {
  uint64_t totalPoints{0};
  int maxDepth{0};
  float rootSpacing{0.0f};

  // Global AABB (world-space).
  std::array<float, 3> aabbMin{};
  std::array<float, 3> aabbMax{};

  // Maximum points per tile (used during construction).
  uint32_t maxPointsPerTile{0};
};

}  // namespace pointcloud
