#pragma once
/// @file crs_transform.h
/// @brief CRS coordinate transformation using PROJ.

#include <foundation/types.h>

#include <memory>
#include <string>

namespace geo {

/// Parse a GCP-list projection string into a PROJ-compatible CRS definition.
///
/// Handles:
///   - "WGS84" or "EPSG:4326" → returns empty string (identity, no transform)
///   - "WGS84 UTM 32N" → "+proj=utm +zone=32 +north +ellps=WGS84 ..."
///   - "+proj=..." or "EPSG:XXXX" → pass-through
///
/// @param line  The raw CRS string from the first line of gcp_list.txt.
/// @return      A PROJ-compatible string, or empty if already WGS-84.
std::string ParseGcpProjectionString(const std::string& line);

/// RAII wrapper around a PROJ coordinate transformation (source CRS → WGS-84).
///
/// Transforms (easting, northing) in the source CRS to (latitude, longitude)
/// in WGS-84.
class CrsTransform {
 public:
  /// Create a transform from the given CRS definition to WGS-84.
  /// If projString is empty, creates an identity (no-op) transform.
  explicit CrsTransform(const std::string& projString);
  ~CrsTransform();

  CrsTransform(const CrsTransform&) = delete;
  CrsTransform& operator=(const CrsTransform&) = delete;
  CrsTransform(CrsTransform&&) noexcept;
  CrsTransform& operator=(CrsTransform&&) noexcept;

  /// Returns true if this is an identity transform (no CRS conversion needed).
  bool isIdentity() const;

  /// Returns true if the transform was successfully created.
  bool isValid() const;

  /// Transform (easting, northing) → (latitude, longitude).
  /// Returns false on failure.
  bool transform(double easting, double northing, double& lat,
                 double& lon) const;

  /// Inverse transform: (latitude, longitude) → (easting, northing).
  /// Returns false on failure.
  bool inverseTransform(double lat, double lon, double& easting,
                        double& northing) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}  // namespace geo
