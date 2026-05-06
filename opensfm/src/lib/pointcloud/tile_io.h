#pragma once

#include <string>
#include <vector>

#include <pointcloud/tile_format.h>

namespace pointcloud {

// ── Tile writing ────────────────────────────────────────────────────────────

/// Write a single tile file to disk.
/// @param dir     Directory to write into (e.g. ".../point_cloud/").
/// @param key     Octree key (e.g. "r", "r03").
/// @param header  Tile header (numPoints, AABB, childMask, etc.).
/// @param points  Point records to write (size must match header.numPoints).
/// @return true on success.
bool writeTile(const std::string& dir, const std::string& key,
               const TileHeader& header,
               const std::vector<PointRecord>& points);

/// Write the metadata.json file.
bool writeMetadata(const std::string& dir, const OctreeMetadata& meta);

// ── Tile reading ────────────────────────────────────────────────────────────

/// Read a single tile file from disk.
/// @param dir    Directory containing tiles.
/// @param key    Octree key.
/// @param[out] header  Filled on success.
/// @param[out] points  Filled on success (resized to header.numPoints).
/// @return true on success.
bool readTile(const std::string& dir, const std::string& key,
              TileHeader& header, std::vector<PointRecord>& points);

/// Read just the tile header (without loading point data).
bool readTileHeader(const std::string& dir, const std::string& key,
                    TileHeader& header);

/// Read the metadata.json file.
bool readMetadata(const std::string& dir, OctreeMetadata& meta);

/// Return the file path for a tile key within a directory.
std::string tileFilePath(const std::string& dir, const std::string& key);

}  // namespace pointcloud
