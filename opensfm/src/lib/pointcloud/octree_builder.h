#pragma once

/// The builder accepts raw point data (positions, normals, colors, radii),
/// sorts by Morton code, recursively subdivides into an octree, and writes
/// each node as a binary tile file.
///
/// Algorithm (Potree-style, Schütz 2020 §3):
///   1. Compute global AABB; cube-pad to make it isotropic.
///   2. Quantise positions to 21-bit Morton grid; compute Morton codes.
///   3. Radix-sort points by Morton code.
///   4. Build octree top-down: assign points to the deepest node that
///      keeps |points| ≤ maxPointsPerTile, or stop at maxDepth.
///      Each level also keeps a subsample of its children for LOD.
///   5. Write each node as a binary tile file.
///
/// Thread-safety: a single builder instance is NOT thread-safe.  Create
/// one per export.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <pointcloud/tile_format.h>

namespace pointcloud {

/// Configuration for the octree builder.
struct OctreeBuilderConfig {
  /// Maximum number of points per leaf tile.
  /// Tiles with more points are subdivided further.
  uint32_t maxPointsPerTile{50000};

  /// Maximum octree depth (root = 0).  Limits subdivision.
  int maxDepth{15};

  /// Number of LOD subsample points per inner node.
  /// These represent the node for far-away viewing.
  uint32_t lodSampleCount{10000};

  /// Output directory for the tile files.  Must exist.
  std::string outputDir;
};

/// Input point cloud data for the builder (SOA layout, NumPy-friendly).
struct BuilderInput {
  /// Interleaved [N*3] XYZ positions (float32).
  const float* positions{nullptr};

  /// Interleaved [N*3] normals (float32).  May be nullptr (defaults to 0).
  const float* normals{nullptr};

  /// Interleaved [N*3] RGB colors (uint8).  May be nullptr (defaults to 0).
  const uint8_t* colors{nullptr};

  /// Per-point splat radius (float32, N elements).  May be nullptr
  /// (defaults to the tile spacing).
  const float* radii{nullptr};

  /// Number of points.
  uint64_t numPoints{0};
};

/// Progress callback: called with (nodesProcessed, estimatedTotal).
using ProgressCallback = std::function<void(int nodesProcessed, int total)>;

/// Build the octree tile set from the given point cloud.
///
/// @param input   Point cloud data (positions required; normals, colors,
///                radii optional).
/// @param config  Builder configuration.
/// @param progress  Optional progress callback.
/// @return The metadata describing the generated octree.
OctreeMetadata buildOctree(const BuilderInput& input,
                           const OctreeBuilderConfig& config,
                           ProgressCallback progress = nullptr);

}  // namespace pointcloud
