#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include <pointcloud/octree_builder.h>
#include <pointcloud/half_float.h>
#include <pointcloud/morton.h>
#include <pointcloud/tile_io.h>

namespace pointcloud {

namespace {

// ── Helper: compute AABB ────────────────────────────────────────────────────

struct AABB {
  float min[3]{1e30f, 1e30f, 1e30f};
  float max[3]{-1e30f, -1e30f, -1e30f};

  void expand(float x, float y, float z) {
    if (x < min[0]) {
      min[0] = x;
    }
    if (y < min[1]) {
      min[1] = y;
    }
    if (z < min[2]) {
      min[2] = z;
    }
    if (x > max[0]) {
      max[0] = x;
    }
    if (y > max[1]) {
      max[1] = y;
    }
    if (z > max[2]) {
      max[2] = z;
    }
  }

  float extent(int axis) const { return max[axis] - min[axis]; }

  float maxExtent() const {
    return std::max({extent(0), extent(1), extent(2)});
  }

  void cubePad() {
    float e = maxExtent();
    for (int a = 0; a < 3; ++a) {
      float diff = e - extent(a);
      min[a] -= diff * 0.5f;
      max[a] += diff * 0.5f;
    }
  }
};

// ── Sort point indices by Morton code ───────────────────────────────────────

struct SortEntry {
  uint64_t morton;
  uint32_t index;
};

// ── Recursive octree construction ───────────────────────────────────────────

/// Convert a point to a PointRecord (compressed format for the tile).
PointRecord makePointRecord(uint32_t idx, const BuilderInput& input,
                            const float tileMin[3], const float tileMax[3],
                            float defaultRadius) {
  PointRecord rec{};
  const float* p = input.positions + idx * 3;

  // Relative position within the tile AABB → [0,1] → half-float.
  float rangeX = tileMax[0] - tileMin[0];
  float rangeY = tileMax[1] - tileMin[1];
  float rangeZ = tileMax[2] - tileMin[2];
  if (rangeX < 1e-12f) {
    rangeX = 1.0f;
  }
  if (rangeY < 1e-12f) {
    rangeY = 1.0f;
  }
  if (rangeZ < 1e-12f) {
    rangeZ = 1.0f;
  }

  rec.px = floatToHalf((p[0] - tileMin[0]) / rangeX);
  rec.py = floatToHalf((p[1] - tileMin[1]) / rangeY);
  rec.pz = floatToHalf((p[2] - tileMin[2]) / rangeZ);

  // Normal (snorm int8).
  if (input.normals) {
    const float* n = input.normals + idx * 3;
    auto clampSnorm = [](float v) -> int8_t {
      return static_cast<int8_t>(
          std::max(-127.0f, std::min(127.0f, v * 127.0f)));
    };
    rec.nx = clampSnorm(n[0]);
    rec.ny = clampSnorm(n[1]);
    rec.nz = clampSnorm(n[2]);
  }

  // Color.
  if (input.colors) {
    const uint8_t* c = input.colors + idx * 3;
    rec.r = c[0];
    rec.g = c[1];
    rec.b = c[2];
  }

  // Radius.
  float r = (input.radii) ? input.radii[idx] : defaultRadius;
  rec.radius = floatToHalf(r);

  std::memset(rec._pad, 0, sizeof(rec._pad));
  return rec;
}

/// Recursively build the octree and write tiles.
///
/// @param key        Octree key for this node (e.g. "r", "r0", "r03").
/// @param depth      Current depth (root = 0).
/// @param indices    Point indices belonging to this node.
/// @param aabbMin/Max  AABB for this node.
/// @param input      Full input point cloud.
/// @param config     Builder config.
/// @param mortons    Pre-computed Morton codes for all points.
/// @param meta       Accumulating metadata.
/// @param nodesWritten  Counter for progress.
/// @param progress   Optional progress callback.
void buildNode(const std::string& key, int depth,
               std::vector<uint32_t>& indices, const float aabbMin[3],
               const float aabbMax[3], const BuilderInput& input,
               const OctreeBuilderConfig& config,
               const std::vector<uint64_t>& mortons, OctreeMetadata& meta,
               int& nodesWritten, const ProgressCallback& progress) {
  if (indices.empty()) {
    return;
  }

  float extent = std::max({aabbMax[0] - aabbMin[0], aabbMax[1] - aabbMin[1],
                           aabbMax[2] - aabbMin[2]});

  // Determine if we should subdivide further.
  bool isLeaf =
      (indices.size() <= config.maxPointsPerTile) || (depth >= config.maxDepth);

  if (isLeaf) {
    // Write all points into this tile.
    // Spacing reflects the actual content of the tile.
    float spacing =
        extent /
        std::sqrt(static_cast<float>(std::max(size_t(1), indices.size())));
    TileHeader hdr{};
    hdr.magic = kTileMagic;
    hdr.version = kTileVersion;
    hdr.numPoints = static_cast<uint32_t>(indices.size());
    hdr.childMask = 0;
    std::copy_n(aabbMin, 3, hdr.aabbMin);
    std::copy_n(aabbMax, 3, hdr.aabbMax);
    hdr.spacing = spacing;
    hdr.depth = static_cast<uint32_t>(depth);

    std::vector<PointRecord> records;
    records.reserve(indices.size());
    for (uint32_t idx : indices) {
      records.push_back(makePointRecord(idx, input, aabbMin, aabbMax, spacing));
    }

    writeTile(config.outputDir, key, hdr, records);
    meta.totalPoints += indices.size();
    if (depth > meta.maxDepth) {
      meta.maxDepth = depth;
    }

    ++nodesWritten;
    if (progress) {
      progress(nodesWritten, 0);
    }
    return;
  }

  // ── Inner node: subdivide into 8 octants ──

  // Partition points into 8 children by their position relative to center.
  float cx = (aabbMin[0] + aabbMax[0]) * 0.5f;
  float cy = (aabbMin[1] + aabbMax[1]) * 0.5f;
  float cz = (aabbMin[2] + aabbMax[2]) * 0.5f;

  std::vector<uint32_t> childIndices[8];
  for (uint32_t idx : indices) {
    const float* p = input.positions + idx * 3;
    int octant = 0;
    if (p[0] >= cx) {
      octant |= 1;
    }
    if (p[1] >= cy) {
      octant |= 2;
    }
    if (p[2] >= cz) {
      octant |= 4;
    }
    childIndices[octant].push_back(idx);
  }

  // Select LOD subsample for this inner node.
  // Use spatial subsampling: pick every N-th point from the Morton-sorted
  // order.  This ensures spatial uniformity.
  uint32_t lodCount =
      std::min(config.lodSampleCount, static_cast<uint32_t>(indices.size()));
  std::vector<uint32_t> lodIndices;
  lodIndices.reserve(lodCount);
  if (lodCount > 0 && lodCount < indices.size()) {
    // Stride-based subsample.
    float stride = static_cast<float>(indices.size()) / lodCount;
    for (uint32_t i = 0; i < lodCount; ++i) {
      lodIndices.push_back(indices[static_cast<size_t>(i * stride)]);
    }
  } else {
    lodIndices = indices;
  }

  // Build child mask and recurse.
  uint32_t childMask = 0;
  for (int oct = 0; oct < 8; ++oct) {
    if (!childIndices[oct].empty()) {
      childMask |= (1u << oct);
    }
  }

  // Write this inner node's LOD tile.
  // Spacing must reflect the LOD subsample stored in this tile, NOT the
  // full point set.  The traversal uses this to decide whether children
  // are needed for more detail.
  float spacing =
      extent /
      std::sqrt(static_cast<float>(std::max(size_t(1), lodIndices.size())));
  TileHeader hdr{};
  hdr.magic = kTileMagic;
  hdr.version = kTileVersion;
  hdr.numPoints = static_cast<uint32_t>(lodIndices.size());
  hdr.childMask = childMask;
  std::copy_n(aabbMin, 3, hdr.aabbMin);
  std::copy_n(aabbMax, 3, hdr.aabbMax);
  hdr.spacing = spacing;
  hdr.depth = static_cast<uint32_t>(depth);

  std::vector<PointRecord> records;
  records.reserve(lodIndices.size());
  for (uint32_t idx : lodIndices) {
    records.push_back(makePointRecord(idx, input, aabbMin, aabbMax, spacing));
  }

  writeTile(config.outputDir, key, hdr, records);
  meta.totalPoints += lodIndices.size();
  if (depth > meta.maxDepth) {
    meta.maxDepth = depth;
  }

  ++nodesWritten;
  if (progress) {
    progress(nodesWritten, 0);
  }

  // Recurse into children.
  for (int oct = 0; oct < 8; ++oct) {
    if (childIndices[oct].empty()) {
      continue;
    }

    float childMin[3], childMax[3];
    childMin[0] = (oct & 1) ? cx : aabbMin[0];
    childMin[1] = (oct & 2) ? cy : aabbMin[1];
    childMin[2] = (oct & 4) ? cz : aabbMin[2];
    childMax[0] = (oct & 1) ? aabbMax[0] : cx;
    childMax[1] = (oct & 2) ? aabbMax[1] : cy;
    childMax[2] = (oct & 4) ? aabbMax[2] : cz;

    std::string childKey = key + std::to_string(oct);

    buildNode(childKey, depth + 1, childIndices[oct], childMin, childMax, input,
              config, mortons, meta, nodesWritten, progress);
  }
}

}  // namespace

// ── Public API ──────────────────────────────────────────────────────────────

OctreeMetadata buildOctree(const BuilderInput& input,
                           const OctreeBuilderConfig& config,
                           ProgressCallback progress) {
  OctreeMetadata meta{};

  if (input.numPoints == 0 || input.positions == nullptr) {
    return meta;
  }

  // 1. Compute global AABB and cube-pad.
  AABB aabb;
  for (uint64_t i = 0; i < input.numPoints; ++i) {
    const float* p = input.positions + i * 3;
    aabb.expand(p[0], p[1], p[2]);
  }
  aabb.cubePad();

  meta.aabbMin = {aabb.min[0], aabb.min[1], aabb.min[2]};
  meta.aabbMax = {aabb.max[0], aabb.max[1], aabb.max[2]};
  meta.maxPointsPerTile = config.maxPointsPerTile;

  float rootExtent = aabb.maxExtent();
  // Root spacing reflects the LOD subsample at the root level.
  // This is the effective average point spacing when viewing from far away.
  uint64_t rootLodCount =
      std::min(static_cast<uint64_t>(config.lodSampleCount), input.numPoints);
  meta.rootSpacing =
      rootExtent /
      std::sqrt(static_cast<float>(std::max(rootLodCount, uint64_t(1))));

  // 2. Compute Morton codes for all points.
  float rangeInv[3];
  for (int a = 0; a < 3; ++a) {
    float ext = aabb.max[a] - aabb.min[a];
    rangeInv[a] = (ext > 1e-12f) ? (1.0f / ext) : 0.0f;
  }

  std::vector<SortEntry> entries(input.numPoints);
  for (uint64_t i = 0; i < input.numPoints; ++i) {
    const float* p = input.positions + i * 3;
    uint32_t qx = quantise(p[0], aabb.min[0], rangeInv[0]);
    uint32_t qy = quantise(p[1], aabb.min[1], rangeInv[1]);
    uint32_t qz = quantise(p[2], aabb.min[2], rangeInv[2]);
    entries[i].morton = mortonEncode(qx, qy, qz);
    entries[i].index = static_cast<uint32_t>(i);
  }

  // 3. Sort by Morton code (standard sort; radix sort would be faster for
  //    very large clouds but std::sort is simpler and still O(N log N)).
  std::sort(entries.begin(), entries.end(),
            [](const SortEntry& a, const SortEntry& b) {
              return a.morton < b.morton;
            });

  // Build sorted index array and Morton array.
  std::vector<uint32_t> sortedIndices(input.numPoints);
  std::vector<uint64_t> mortons(input.numPoints);
  for (uint64_t i = 0; i < input.numPoints; ++i) {
    sortedIndices[i] = entries[i].index;
    mortons[i] = entries[i].morton;
  }
  entries.clear();  // free memory

  // 4. Build octree recursively and write tiles.
  int nodesWritten = 0;
  buildNode("r", 0, sortedIndices, aabb.min, aabb.max, input, config, mortons,
            meta, nodesWritten, progress);

  // 5. Write metadata.
  writeMetadata(config.outputDir, meta);

  return meta;
}

}  // namespace pointcloud
