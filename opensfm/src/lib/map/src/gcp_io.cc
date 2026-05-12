#include <geo/crs_transform.h>
#include <map/gcp_io.h>
#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {
static constexpr int kPrecision = 15;
}
namespace map {

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string_view trim(std::string_view sv) {
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
    sv.remove_prefix(1);
  }
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
    sv.remove_suffix(1);
  }
  return sv;
}

static bool isValidGcpLine(std::string_view line) {
  auto trimmed = trim(line);
  return !trimmed.empty() && trimmed[0] != '#';
}

/// Normalized image coordinates (mirrors features.py):
///   x_norm = (pixel_x + 0.5 - width/2.0) / max(width, height)
///   y_norm = (pixel_y + 0.5 - height/2.0) / max(width, height)
static Vec2d normalizePixel(double px, double py, int width, int height) {
  const double size = static_cast<double>(std::max(width, height));
  return {(px + 0.5 - width / 2.0) / size, (py + 0.5 - height / 2.0) / size};
}

/// Denormalize: inverse of normalizePixel.
static void denormalizePixel(double nx, double ny, int width, int height,
                             double& px, double& py) {
  const double size = static_cast<double>(std::max(width, height));
  px = nx * size + width / 2.0 - 0.5;
  py = ny * size + height / 2.0 - 0.5;
}

// Escape a string for JSON output (handles backslash and double-quote).
static std::string jsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '"') {
      out += "\\\"";
    } else if (c == '\\') {
      out += "\\\\";
    } else {
      out += c;
    }
  }
  return out;
}

// ── ReadGcpJson ──────────────────────────────────────────────────────────────

std::vector<GroundControlPoint> ReadGcpJson(const std::string& content) {
  simdjson::ondemand::parser parser;
  simdjson::padded_string padded(content);
  simdjson::ondemand::document doc;
  if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) {
    return {};
  }

  simdjson::ondemand::array points;
  if (doc["points"].get_array().get(points) != simdjson::SUCCESS) {
    return {};
  }

  std::vector<GroundControlPoint> result;

  for (auto pointVal : points) {
    simdjson::ondemand::object pointObj;
    if (pointVal.get_object().get(pointObj) != simdjson::SUCCESS) {
      continue;
    }

    GroundControlPoint gcp;
    gcp.role_ = OPTIMIZATION;

    // id
    std::string_view idSv;
    if (pointObj["id"].get_string().get(idSv) == simdjson::SUCCESS) {
      gcp.id_ = std::string(idSv);
    }

    // position → lla
    simdjson::ondemand::object posObj;
    if (pointObj["position"].get_object().get(posObj) == simdjson::SUCCESS) {
      double lat = 0, lon = 0, alt = 0;
      posObj["latitude"].get_double().get(lat);
      posObj["longitude"].get_double().get(lon);
      gcp.lla_["latitude"] = lat;
      gcp.lla_["longitude"] = lon;
      if (posObj["altitude"].get_double().get(alt) == simdjson::SUCCESS) {
        gcp.lla_["altitude"] = alt;
        gcp.has_altitude_ = true;
      } else {
        gcp.lla_["altitude"] = 0.0;
        gcp.has_altitude_ = false;
      }
    }

    // coordinates (topocentric)
    simdjson::ondemand::array coordArr;
    if (pointObj["coordinates"].get_array().get(coordArr) ==
        simdjson::SUCCESS) {
      double c[3] = {0, 0, 0};
      int ci = 0;
      for (auto elem : coordArr) {
        if (ci < 3) {
          elem.get_double().get(c[ci]);
          ++ci;
        }
      }
      gcp.coordinates_ = Vec3d(c[0], c[1], c[2]);
    }

    // observations
    simdjson::ondemand::array obsArr;
    if (pointObj["observations"].get_array().get(obsArr) == simdjson::SUCCESS) {
      for (auto obsVal : obsArr) {
        simdjson::ondemand::object obsObj;
        if (obsVal.get_object().get(obsObj) != simdjson::SUCCESS) {
          continue;
        }

        GroundControlPointObservation obs;
        std::string_view shotSv;
        if (obsObj["shot_id"].get_string().get(shotSv) == simdjson::SUCCESS) {
          obs.shot_id_ = std::string(shotSv);
        }

        simdjson::ondemand::array projArr;
        if (obsObj["projection"].get_array().get(projArr) ==
            simdjson::SUCCESS) {
          double p[2] = {0, 0};
          int pi = 0;
          for (auto elem : projArr) {
            if (pi < 2) {
              elem.get_double().get(p[pi]);
              ++pi;
            }
          }
          obs.projection_ = Vec2d(p[0], p[1]);
        }

        gcp.observations_.push_back(std::move(obs));
      }
    }

    result.push_back(std::move(gcp));
  }

  return result;
}

// ── WriteGcpJson ─────────────────────────────────────────────────────────────

std::string WriteGcpJson(const std::vector<GroundControlPoint>& gcps) {
  std::ostringstream out;
  out << std::setprecision(kPrecision);
  out << "{\n  \"points\": [\n";

  for (size_t i = 0; i < gcps.size(); ++i) {
    const auto& gcp = gcps[i];
    if (i > 0) {
      out << ",\n";
    }
    out << "    {\n";
    out << "      \"id\": \"" << jsonEscape(gcp.id_) << "\",\n";

    // position (LLA)
    auto latIt = gcp.lla_.find("latitude");
    auto lonIt = gcp.lla_.find("longitude");
    auto altIt = gcp.lla_.find("altitude");
    if (latIt != gcp.lla_.end() && lonIt != gcp.lla_.end()) {
      out << "      \"position\": {\n";
      out << "        \"latitude\": " << latIt->second << ",\n";
      out << "        \"longitude\": " << lonIt->second;
      if (gcp.has_altitude_ && altIt != gcp.lla_.end()) {
        out << ",\n        \"altitude\": " << altIt->second << "\n";
      } else {
        out << "\n";
      }
      out << "      },\n";
    }

    // observations
    out << "      \"observations\": [\n";
    for (size_t j = 0; j < gcp.observations_.size(); ++j) {
      const auto& obs = gcp.observations_[j];
      if (j > 0) {
        out << ",\n";
      }
      out << "        {\n";
      out << "          \"shot_id\": \"" << jsonEscape(obs.shot_id_) << "\",\n";
      out << "          \"projection\": [" << obs.projection_.x() << ", "
          << obs.projection_.y() << "]\n";
      out << "        }";
    }
    out << "\n      ]\n";
    out << "    }";
  }

  out << "\n  ]\n}\n";
  return out.str();
}

// ── ReadGcpList ──────────────────────────────────────────────────────────────

std::vector<GroundControlPoint> ReadGcpList(
    const std::string& content,
    const std::unordered_map<std::string, std::pair<int, int>>& imageWidths,
    std::string* crsName) {
  // Split content into lines and filter valid ones.
  std::vector<std::string> validLines;
  {
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
      if (isValidGcpLine(line)) {
        validLines.push_back(std::move(line));
      }
    }
  }

  if (validLines.empty()) {
    return {};
  }

  // First valid line is the CRS/projection string.
  std::string crs(trim(validLines[0]));
  if (crsName) {
    *crsName = crs;
  }

  // Build CRS transform (identity for WGS-84, PROJ-based otherwise).
  std::string projString = geo::ParseGcpProjectionString(crs);
  geo::CrsTransform ct(projString);

  // Parse data lines (starting from index 1).
  // Key: GCP name when provided, otherwise encoded coordinates.
  // Use a lookup map + result vector to preserve insertion order.
  std::map<std::string, size_t> keyToIndex;
  std::vector<GroundControlPoint> result;

  for (size_t i = 1; i < validLines.size(); ++i) {
    std::istringstream iss(validLines[i]);

    // File columns: col1 col2 alt pixelX pixelY shotId [gcpName]
    // For WGS84: col1=longitude, col2=latitude
    // For projected CRS (UTM etc.): col1=easting, col2=northing
    // gcpName is optional: when present, lines sharing the same name are
    // grouped into one GCP; when absent, grouping is by 3-D position.
    double col1, col2, alt, pixelX, pixelY;
    std::string shotId;
    if (!(iss >> col1 >> col2 >> alt >> pixelX >> pixelY >> shotId)) {
      continue;
    }
    // Optional remainder of line is the GCP name (may contain spaces).
    std::string gcpName;
    std::getline(iss, gcpName);
    gcpName = std::string(trim(gcpName));

    // Skip shots not in imageWidths.
    auto wit = imageWidths.find(shotId);

    // Build deduplication key.
    std::string dedupKey;
    if (!gcpName.empty()) {
      dedupKey = gcpName;
    } else {
      std::ostringstream ks;
      ks << std::setprecision(kPrecision) << col1 << "|" << col2 << "|" << alt;
      dedupKey = ks.str();
    }

    auto [mapIt, inserted] = keyToIndex.emplace(dedupKey, result.size());
    if (inserted) {
      bool hasAltitude = !std::isnan(alt);
      if (!hasAltitude) {
        alt = 0.0;
      }

      // Resolve WGS-84 lat/lon.
      double lat = 0, lon = 0;
      if (ct.isIdentity()) {
        // WGS84 format: col1=longitude, col2=latitude
        lon = col1;
        lat = col2;
      } else {
        // Projected CRS: col1=easting, col2=northing → transform to lat/lon
        if (!ct.transform(col1, col2, lat, lon)) {
          keyToIndex.erase(mapIt);
          continue;
        }
      }
      GroundControlPoint gcp;
      gcp.id_ = gcpName.empty() ? ("unnamed-" + std::to_string(result.size()))
                                : gcpName;
      gcp.has_altitude_ = hasAltitude;
      gcp.role_ = OPTIMIZATION;
      gcp.lla_["latitude"] = lat;
      gcp.lla_["longitude"] = lon;
      gcp.lla_["altitude"] = alt;
      gcp.coordinates_ = Eigen::Vector3d(col1, col2, alt);

      result.push_back(std::move(gcp));
    }

    // Normalize pixel coordinates if image dimensions are known;
    // otherwise store raw pixel coords to avoid dropping observations
    // for shots that aren't in the reconstruction.
    Vec2d proj;
    if (wit != imageWidths.end()) {
      const auto& [w, h] = wit->second;
      proj = normalizePixel(pixelX, pixelY, w, h);
    } else {
      proj = Vec2d(pixelX, pixelY);
    }

    GroundControlPointObservation obs;
    obs.shot_id_ = shotId;
    obs.projection_ = proj;
    result[mapIt->second].AddObservation(obs);
  }

  return result;
}

// ── WriteGcpList ─────────────────────────────────────────────────────────────

std::string WriteGcpList(
    const std::vector<GroundControlPoint>& gcps, const std::string& crs,
    const std::unordered_map<std::string, std::pair<int, int>>& imageWidths) {
  std::ostringstream out;
  out << std::setprecision(kPrecision);

  geo::CrsTransform ct(crs);

  // Header line: CRS
  out << (crs.empty() ? "WGS84" : crs) << "\n";

  for (const auto& gcp : gcps) {
    double lat = 0, lon = 0, alt = 0;
    if (auto it = gcp.lla_.find("latitude"); it != gcp.lla_.end()) {
      lat = it->second;
    }
    if (auto it = gcp.lla_.find("longitude"); it != gcp.lla_.end()) {
      lon = it->second;
    }
    if (auto it = gcp.lla_.find("altitude"); it != gcp.lla_.end()) {
      alt = it->second;
    }

    if (ct.isIdentity()) {
      // WGS84 format: col1=longitude, col2=latitude
    } else {
      // Projected CRS: transform lat/lon to easting/northing for output
      double easting = 0, northing = 0;
      if (!ct.inverseTransform(lat, lon, easting, northing)) {
        continue;
      }
      lon = easting;
      lat = northing;
    }

    // One line per observation: lon lat alt px py shot_id gcp_id
    for (const auto& obs : gcp.observations_) {
      auto wit = imageWidths.find(obs.shot_id_);
      if (wit != imageWidths.end()) {
        const auto& [w, h] = wit->second;
        double px, py;
        denormalizePixel(obs.projection_.x(), obs.projection_.y(), w, h, px,
                         py);
        out << lon << "\t" << lat << "\t" << alt << "\t" << px << "\t" << py
            << "\t" << obs.shot_id_ << "\t" << gcp.id_ << "\n";
      } else {
        // No image dimensions — write normalized coords directly.
        out << lon << "\t" << lat << "\t" << alt << "\t" << obs.projection_.x()
            << "\t" << obs.projection_.y() << "\t" << obs.shot_id_ << "\t"
            << gcp.id_ << "\n";
      }
    }
  }

  return out.str();
}

}  // namespace map
