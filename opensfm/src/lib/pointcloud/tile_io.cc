#include <pointcloud/tile_io.h>

#include <cstring>
#include <fstream>
#include <sstream>

namespace pointcloud {

std::string tileFilePath(const std::string& dir, const std::string& key) {
  return dir + "/" + key + ".bin";
}

// ── Writing ─────────────────────────────────────────────────────────────────

bool writeTile(const std::string& dir, const std::string& key,
               const TileHeader& header,
               const std::vector<PointRecord>& points) {
  std::string path = tileFilePath(dir, key);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }

  out.write(reinterpret_cast<const char*>(&header), sizeof(TileHeader));
  if (header.numPoints > 0) {
    out.write(
        reinterpret_cast<const char*>(points.data()),
        static_cast<std::streamsize>(header.numPoints) * sizeof(PointRecord));
  }
  return out.good();
}

bool writeMetadata(const std::string& dir, const OctreeMetadata& meta) {
  std::string path = dir + "/metadata.json";
  std::ofstream out(path);
  if (!out) {
    return false;
  }

  // Minimal JSON — no external dependency needed.
  out << "{\n";
  out << "  \"version\": 1,\n";
  out << "  \"totalPoints\": " << meta.totalPoints << ",\n";
  out << "  \"maxDepth\": " << meta.maxDepth << ",\n";
  out << "  \"rootSpacing\": " << meta.rootSpacing << ",\n";
  out << "  \"maxPointsPerTile\": " << meta.maxPointsPerTile << ",\n";
  out << "  \"aabb\": {\n";
  out << "    \"min\": [" << meta.aabbMin[0] << ", " << meta.aabbMin[1] << ", "
      << meta.aabbMin[2] << "],\n";
  out << "    \"max\": [" << meta.aabbMax[0] << ", " << meta.aabbMax[1] << ", "
      << meta.aabbMax[2] << "]\n";
  out << "  }\n";
  out << "}\n";
  return out.good();
}

// ── Reading ─────────────────────────────────────────────────────────────────

bool readTile(const std::string& dir, const std::string& key,
              TileHeader& header, std::vector<PointRecord>& points) {
  std::string path = tileFilePath(dir, key);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }

  in.read(reinterpret_cast<char*>(&header), sizeof(TileHeader));
  if (!in || header.magic != kTileMagic || header.version != kTileVersion) {
    return false;
  }

  points.resize(header.numPoints);
  if (header.numPoints > 0) {
    in.read(
        reinterpret_cast<char*>(points.data()),
        static_cast<std::streamsize>(header.numPoints) * sizeof(PointRecord));
  }
  return in.good();
}

bool readTileHeader(const std::string& dir, const std::string& key,
                    TileHeader& header) {
  std::string path = tileFilePath(dir, key);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }

  in.read(reinterpret_cast<char*>(&header), sizeof(TileHeader));
  return in.good() && header.magic == kTileMagic &&
         header.version == kTileVersion;
}

bool readMetadata(const std::string& dir, OctreeMetadata& meta) {
  std::string path = dir + "/metadata.json";
  std::ifstream in(path);
  if (!in) {
    return false;
  }

  // Minimal JSON parser — expects our exact output format.
  // For robustness in production, we'd use simdjson, but this keeps
  // the pointcloud library dependency-free.
  std::string content((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());

  auto extractUint64 = [&](const std::string& key) -> uint64_t {
    auto pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) {
      return 0;
    }
    pos = content.find(':', pos);
    if (pos == std::string::npos) {
      return 0;
    }
    return std::stoull(content.substr(pos + 1));
  };
  auto extractInt = [&](const std::string& key) -> int {
    auto pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) {
      return 0;
    }
    pos = content.find(':', pos);
    if (pos == std::string::npos) {
      return 0;
    }
    return std::stoi(content.substr(pos + 1));
  };
  auto extractFloat = [&](const std::string& key) -> float {
    auto pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) {
      return 0.0f;
    }
    pos = content.find(':', pos);
    if (pos == std::string::npos) {
      return 0.0f;
    }
    return std::stof(content.substr(pos + 1));
  };

  meta.totalPoints = extractUint64("totalPoints");
  meta.maxDepth = extractInt("maxDepth");
  meta.rootSpacing = extractFloat("rootSpacing");
  meta.maxPointsPerTile = static_cast<uint32_t>(extractInt("maxPointsPerTile"));

  // Parse AABB min/max arrays.
  auto extractArray = [&](const std::string& key, std::array<float, 3>& out) {
    auto pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) {
      return;
    }
    pos = content.find('[', pos);
    if (pos == std::string::npos) {
      return;
    }
    auto end = content.find(']', pos);
    if (end == std::string::npos) {
      return;
    }
    std::string arr = content.substr(pos + 1, end - pos - 1);
    std::istringstream ss(arr);
    char comma;
    ss >> out[0] >> comma >> out[1] >> comma >> out[2];
  };
  extractArray("min", meta.aabbMin);
  extractArray("max", meta.aabbMax);

  return true;
}

}  // namespace pointcloud
