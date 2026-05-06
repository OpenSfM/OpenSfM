#pragma once
/// @file gcp_io.h
/// @brief Read/write ground control points in JSON and gcp_list.txt formats.

#include <map/ground_control_points.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace map {

/// Read ground control points from a ground_control_points.json string.
///
/// Mirrors opensfm/io.py::read_ground_control_points.
/// Observations are expected in normalized image coordinates.
std::vector<GroundControlPoint> ReadGcpJson(const std::string& content);

/// Write ground control points to a ground_control_points.json string.
///
/// Mirrors opensfm/io.py::write_ground_control_points.
std::string WriteGcpJson(const std::vector<GroundControlPoint>& gcps);

/// Read ground control points from a gcp_list.txt string.
///
/// @param content     The full text content of the gcp_list.txt file.
/// @param imageWidths Map from shot_id → (width, height). Shots not present
///                    in this map are skipped. Pixel coordinates are normalized
///                    using these dimensions.
/// @param crsName     If non-null, receives the raw CRS/projection string
///                    from the first line.
///
/// CRS transformation is handled internally via PROJ:
///   - WGS84 / EPSG:4326 → identity (lat, lon already in the file)
///   - UTM, EPSG:XXXX, +proj=... → transformed to WGS-84 lat/lon
/// The resulting lla_ always contains WGS-84 latitude/longitude/altitude.
/// The raw file columns are preserved in coordinates_.
std::vector<GroundControlPoint> ReadGcpList(
    const std::string& content,
    const std::unordered_map<std::string, std::pair<int, int>>& imageWidths,
    std::string* crsName = nullptr);

/// Write ground control points to a gcp_list.txt string.
///
/// @param gcps    The GCPs to write. Their lla_ must contain latitude,
///               longitude and altitude.
/// @param crs    The CRS header line (e.g. "WGS84"). Written as first line.
/// @param imageWidths Map from shot_id → (width, height). Needed to
///                    denormalize observation pixel coordinates.
///
/// @note Pixel coordinates are denormalized from the normalized form:
///       pixel_x = proj_x * max(w,h) + w/2 - 0.5
///       pixel_y = proj_y * max(w,h) + h/2 - 0.5
std::string WriteGcpList(
    const std::vector<GroundControlPoint>& gcps, const std::string& crs,
    const std::unordered_map<std::string, std::pair<int, int>>& imageWidths);

}  // namespace map
