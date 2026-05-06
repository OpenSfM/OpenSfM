#include <gtest/gtest.h>
#include <map/gcp_io.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

using ImageWidths = std::unordered_map<std::string, std::pair<int, int>>;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a single GCP with known values for roundtrip testing.
map::GroundControlPoint MakeTestGcp(const std::string& id, double lat,
                                    double lon, double alt, bool hasAlt) {
  map::GroundControlPoint gcp;
  gcp.id_ = id;
  gcp.lla_["latitude"] = lat;
  gcp.lla_["longitude"] = lon;
  gcp.lla_["altitude"] = alt;
  gcp.has_altitude_ = hasAlt;
  gcp.role_ = map::OPTIMIZATION;
  gcp.coordinates_ = Vec3d(lat, lon, alt);
  return gcp;
}

void AddObservation(map::GroundControlPoint& gcp, const std::string& shotId,
                    double px, double py) {
  map::GroundControlPointObservation obs;
  obs.shot_id_ = shotId;
  obs.projection_ = Vec2d(px, py);
  gcp.AddObservation(obs);
}

// ── JSON Tests ───────────────────────────────────────────────────────────────

TEST(GcpIo, ReadJsonBasic) {
  const std::string json = R"({
    "points": [
      {
        "id": "gcp1",
        "position": {
          "latitude": 52.5,
          "longitude": 13.4,
          "altitude": 35.0
        },
        "observations": [
          {"shot_id": "img001.jpg", "projection": [0.1, -0.05]}
        ]
      }
    ]
  })";

  auto gcps = map::ReadGcpJson(json);
  ASSERT_EQ(gcps.size(), 1u);
  EXPECT_EQ(gcps[0].id_, "gcp1");
  EXPECT_DOUBLE_EQ(gcps[0].lla_.at("latitude"), 52.5);
  EXPECT_DOUBLE_EQ(gcps[0].lla_.at("longitude"), 13.4);
  EXPECT_DOUBLE_EQ(gcps[0].lla_.at("altitude"), 35.0);
  EXPECT_TRUE(gcps[0].has_altitude_);
  ASSERT_EQ(gcps[0].observations_.size(), 1u);
  EXPECT_EQ(gcps[0].observations_[0].shot_id_, "img001.jpg");
  EXPECT_NEAR(gcps[0].observations_[0].projection_.x(), 0.1, 1e-12);
  EXPECT_NEAR(gcps[0].observations_[0].projection_.y(), -0.05, 1e-12);
}

TEST(GcpIo, ReadJsonNoAltitude) {
  const std::string json = R"({
    "points": [
      {
        "id": "gcp_no_alt",
        "position": { "latitude": 48.0, "longitude": 2.0 },
        "observations": []
      }
    ]
  })";

  auto gcps = map::ReadGcpJson(json);
  ASSERT_EQ(gcps.size(), 1u);
  EXPECT_FALSE(gcps[0].has_altitude_);
  EXPECT_DOUBLE_EQ(gcps[0].lla_.at("altitude"), 0.0);
}

TEST(GcpIo, ReadJsonEmpty) {
  EXPECT_TRUE(map::ReadGcpJson("").empty());
  EXPECT_TRUE(map::ReadGcpJson("{}").empty());
  EXPECT_TRUE(map::ReadGcpJson("{\"points\": []}").empty());
}

TEST(GcpIo, JsonRoundtrip) {
  // Build GCPs with known data.
  std::vector<map::GroundControlPoint> original;
  {
    auto gcp = MakeTestGcp("point-A", 52.520008, 13.404954, 34.5, true);
    AddObservation(gcp, "img001.jpg", 0.12, -0.03);
    AddObservation(gcp, "img002.jpg", -0.05, 0.11);
    original.push_back(std::move(gcp));
  }
  {
    auto gcp = MakeTestGcp("point-B", 48.856613, 2.352222, 65.0, true);
    AddObservation(gcp, "img003.jpg", 0.0, 0.0);
    original.push_back(std::move(gcp));
  }

  // Write → Read roundtrip.
  std::string jsonStr = map::WriteGcpJson(original);
  auto loaded = map::ReadGcpJson(jsonStr);

  ASSERT_EQ(loaded.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    const auto& orig = original[i];
    const auto& load = loaded[i];

    EXPECT_EQ(load.id_, orig.id_);
    EXPECT_NEAR(load.lla_.at("latitude"), orig.lla_.at("latitude"), 1e-9);
    EXPECT_NEAR(load.lla_.at("longitude"), orig.lla_.at("longitude"), 1e-9);
    EXPECT_NEAR(load.lla_.at("altitude"), orig.lla_.at("altitude"), 1e-9);
    EXPECT_EQ(load.has_altitude_, orig.has_altitude_);

    ASSERT_EQ(load.observations_.size(), orig.observations_.size());
    for (size_t j = 0; j < orig.observations_.size(); ++j) {
      EXPECT_EQ(load.observations_[j].shot_id_, orig.observations_[j].shot_id_);
      EXPECT_NEAR(load.observations_[j].projection_.x(),
                  orig.observations_[j].projection_.x(), 1e-9);
      EXPECT_NEAR(load.observations_[j].projection_.y(),
                  orig.observations_[j].projection_.y(), 1e-9);
    }
  }
}

TEST(GcpIo, JsonSpecialCharacters) {
  auto gcp = MakeTestGcp("gcp\"with\\special", 0.0, 0.0, 0.0, true);
  AddObservation(gcp, "shot\"1.jpg", 0.0, 0.0);

  std::string json = map::WriteGcpJson({gcp});
  auto loaded = map::ReadGcpJson(json);

  ASSERT_EQ(loaded.size(), 1u);
  EXPECT_EQ(loaded[0].id_, "gcp\"with\\special");
  ASSERT_EQ(loaded[0].observations_.size(), 1u);
  EXPECT_EQ(loaded[0].observations_[0].shot_id_, "shot\"1.jpg");
}

// ── GCP List (WGS84) Tests ───────────────────────────────────────────────────

TEST(GcpIo, ReadListWgs84Basic) {
  // WGS84 format: lon lat alt px py shot_id
  const std::string content =
      "WGS84\n"
      "13.4 52.5 35.0 500.0 300.0 img001.jpg\n";

  ImageWidths widths = {{"img001.jpg", {1000, 600}}};
  std::string crs;
  auto gcps = map::ReadGcpList(content, widths, &crs);

  EXPECT_EQ(crs, "WGS84");
  ASSERT_EQ(gcps.size(), 1u);
  // Verify lat/lon are NOT swapped: lat should be 52.5, lon should be 13.4
  EXPECT_NEAR(gcps[0].lla_.at("latitude"), 52.5, 1e-9);
  EXPECT_NEAR(gcps[0].lla_.at("longitude"), 13.4, 1e-9);
  EXPECT_NEAR(gcps[0].lla_.at("altitude"), 35.0, 1e-9);
  EXPECT_TRUE(gcps[0].has_altitude_);
}

TEST(GcpIo, ReadListCommentsAndBlanks) {
  const std::string content =
      "# This is a header comment\n"
      "\n"
      "WGS84\n"
      "# Another comment\n"
      "\n"
      "2.0 48.0 100.0 100.0 200.0 img002.jpg\n";

  ImageWidths widths = {{"img002.jpg", {800, 600}}};
  auto gcps = map::ReadGcpList(content, widths);

  ASSERT_EQ(gcps.size(), 1u);
  EXPECT_NEAR(gcps[0].lla_.at("latitude"), 48.0, 1e-9);
  EXPECT_NEAR(gcps[0].lla_.at("longitude"), 2.0, 1e-9);
}

TEST(GcpIo, ReadListPreservesUnknownShots) {
  // Observations for shots not in imageWidths should be kept with raw pixel
  // coords (not normalized), so that GCPs are never silently dropped.
  const std::string content =
      "WGS84\n"
      "13.4 52.5 35.0 100 200 known.jpg\n"
      "13.4 52.5 35.0 300 400 unknown.jpg\n";

  ImageWidths widths = {{"known.jpg", {1000, 1000}}};
  auto gcps = map::ReadGcpList(content, widths);

  ASSERT_EQ(gcps.size(), 1u);
  // Both observations should be present.
  ASSERT_EQ(gcps[0].observations_.size(), 2u);
  EXPECT_EQ(gcps[0].observations_[0].shot_id_, "known.jpg");
  EXPECT_EQ(gcps[0].observations_[1].shot_id_, "unknown.jpg");
  // unknown.jpg observation should have raw pixel coords (not normalized).
  EXPECT_NEAR(gcps[0].observations_[1].projection_.x(), 300.0, 1e-9);
  EXPECT_NEAR(gcps[0].observations_[1].projection_.y(), 400.0, 1e-9);
}

TEST(GcpIo, ReadListGcpNotDroppedWhenAllShotsUnknown) {
  // A GCP whose observations ALL reference non-reconstructed shots should
  // still survive (not be silently dropped).
  const std::string content =
      "WGS84\n"
      "13.4 52.5 35.0 100 200 unreconstructed1.jpg\n"
      "13.4 52.5 35.0 300 400 unreconstructed2.jpg\n";

  ImageWidths widths;  // empty — no reconstructed shots
  auto gcps = map::ReadGcpList(content, widths);

  ASSERT_EQ(gcps.size(), 1u);
  EXPECT_EQ(gcps[0].observations_.size(), 2u);
  EXPECT_NEAR(gcps[0].lla_.at("latitude"), 52.5, 1e-9);
  EXPECT_NEAR(gcps[0].lla_.at("longitude"), 13.4, 1e-9);
}

TEST(GcpIo, ReadListDeduplicatesPoints) {
  // Same 3D position, two different shots → one GCP with two observations.
  const std::string content =
      "WGS84\n"
      "13.4 52.5 35.0 100 200 img001.jpg\n"
      "13.4 52.5 35.0 400 300 img002.jpg\n";

  ImageWidths widths = {{"img001.jpg", {1000, 1000}},
                        {"img002.jpg", {1000, 1000}}};
  auto gcps = map::ReadGcpList(content, widths);

  ASSERT_EQ(gcps.size(), 1u);
  EXPECT_EQ(gcps[0].observations_.size(), 2u);
}

TEST(GcpIo, ReadListEmpty) {
  EXPECT_TRUE(map::ReadGcpList("", {}).empty());
  EXPECT_TRUE(map::ReadGcpList("# only a comment\n", {}).empty());
}

TEST(GcpIo, ReadListNamedGcps) {
  // Optional name suffix (may contain spaces): groups observations by name.
  const std::string content =
      "WGS84\n"
      "13.4 52.5 35.0 100 200 img001.jpg Ground Control A\n"
      "13.4 52.5 35.0 300 400 img002.jpg Ground Control A\n"
      "2.0 48.0 100.0 500 600 img003.jpg gcp-B\n";

  ImageWidths widths = {{"img001.jpg", {1000, 1000}},
                        {"img002.jpg", {1000, 1000}},
                        {"img003.jpg", {1000, 1000}}};
  auto gcps = map::ReadGcpList(content, widths);

  ASSERT_EQ(gcps.size(), 2u);
  EXPECT_EQ(gcps[0].id_, "Ground Control A");
  EXPECT_EQ(gcps[0].observations_.size(), 2u);
  EXPECT_EQ(gcps[1].id_, "gcp-B");
  EXPECT_EQ(gcps[1].observations_.size(), 1u);
}

TEST(GcpIo, ListRoundtripWithNames) {
  // Named GCPs should survive a Write→Read roundtrip with their names intact.
  std::vector<map::GroundControlPoint> original;
  {
    auto gcp = MakeTestGcp("station-1", 52.520008, 13.404954, 34.5, true);
    AddObservation(gcp, "img001.jpg", 0.12, -0.03);
    original.push_back(std::move(gcp));
  }
  {
    auto gcp = MakeTestGcp("station-2", 48.856613, 2.352222, 65.0, true);
    AddObservation(gcp, "img002.jpg", 0.0, 0.0);
    original.push_back(std::move(gcp));
  }

  ImageWidths widths = {{"img001.jpg", {4000, 3000}},
                        {"img002.jpg", {4000, 3000}}};

  std::string listStr = map::WriteGcpList(original, "WGS84", widths);
  auto loaded = map::ReadGcpList(listStr, widths);

  ASSERT_EQ(loaded.size(), 2u);
  EXPECT_EQ(loaded[0].id_, "station-1");
  EXPECT_EQ(loaded[1].id_, "station-2");
  EXPECT_NEAR(loaded[0].lla_.at("latitude"), 52.520008, 1e-6);
  EXPECT_NEAR(loaded[1].lla_.at("latitude"), 48.856613, 1e-6);
}

TEST(GcpIo, ListRoundtripWgs84) {
  // Build GCPs programmatically.
  std::vector<map::GroundControlPoint> original;
  {
    auto gcp = MakeTestGcp("gcp-1", 52.520008, 13.404954, 34.5, true);
    AddObservation(gcp, "img001.jpg", 0.12, -0.03);
    AddObservation(gcp, "img002.jpg", -0.05, 0.11);
    original.push_back(std::move(gcp));
  }
  {
    auto gcp = MakeTestGcp("gcp-2", 48.856613, 2.352222, 65.0, true);
    AddObservation(gcp, "img003.jpg", 0.0, 0.0);
    original.push_back(std::move(gcp));
  }

  ImageWidths widths = {
      {"img001.jpg", {4000, 3000}},
      {"img002.jpg", {4000, 3000}},
      {"img003.jpg", {6000, 4000}},
  };

  // Write → Read roundtrip.
  std::string listStr = map::WriteGcpList(original, "WGS84", widths);
  auto loaded = map::ReadGcpList(listStr, widths);

  ASSERT_EQ(loaded.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    const auto& orig = original[i];
    const auto& load = loaded[i];

    // Lat/lon/alt should survive the roundtrip.
    EXPECT_NEAR(load.lla_.at("latitude"), orig.lla_.at("latitude"), 1e-6)
        << "Lat mismatch for GCP " << i;
    EXPECT_NEAR(load.lla_.at("longitude"), orig.lla_.at("longitude"), 1e-6)
        << "Lon mismatch for GCP " << i;
    EXPECT_NEAR(load.lla_.at("altitude"), orig.lla_.at("altitude"), 1e-6)
        << "Alt mismatch for GCP " << i;
    EXPECT_EQ(load.has_altitude_, orig.has_altitude_);

    // Observations should match (after pixel normalization roundtrip).
    ASSERT_EQ(load.observations_.size(), orig.observations_.size());
    for (size_t j = 0; j < orig.observations_.size(); ++j) {
      EXPECT_EQ(load.observations_[j].shot_id_, orig.observations_[j].shot_id_);
      // Pixel normalization roundtrip may introduce small floating-point error.
      EXPECT_NEAR(load.observations_[j].projection_.x(),
                  orig.observations_[j].projection_.x(), 1e-9)
          << "Proj x mismatch for GCP " << i << " obs " << j;
      EXPECT_NEAR(load.observations_[j].projection_.y(),
                  orig.observations_[j].projection_.y(), 1e-9)
          << "Proj y mismatch for GCP " << i << " obs " << j;
    }
  }
}

TEST(GcpIo, PixelNormalizationRoundtrip) {
  // Verify that pixel coords survive Write→Read with exact inversion.
  // Use a single GCP with a known pixel position.
  auto gcp = MakeTestGcp("px-test", 40.0, -74.0, 10.0, true);
  // Observation with specific normalized coords.
  AddObservation(gcp, "shot.jpg", 0.25, -0.125);

  ImageWidths widths = {{"shot.jpg", {4000, 3000}}};

  std::string text = map::WriteGcpList({gcp}, "WGS84", widths);
  auto loaded = map::ReadGcpList(text, widths);

  ASSERT_EQ(loaded.size(), 1u);
  ASSERT_EQ(loaded[0].observations_.size(), 1u);
  EXPECT_NEAR(loaded[0].observations_[0].projection_.x(), 0.25, 1e-9);
  EXPECT_NEAR(loaded[0].observations_[0].projection_.y(), -0.125, 1e-9);
}

// ── GCP List (UTM / PROJ) Tests ─────────────────────────────────────────────

TEST(GcpIo, ReadListUtm) {
  // UTM zone 33N: Berlin area.
  // Known point: lat≈52.52, lon≈13.405 → UTM33N easting≈389262,
  // northing≈5819838
  const std::string content =
      "WGS84 UTM 33N\n"
      "389262 5819838 34.5 500 300 img.jpg\n";

  ImageWidths widths = {{"img.jpg", {1000, 600}}};
  std::string crs;
  auto gcps = map::ReadGcpList(content, widths, &crs);

  EXPECT_EQ(crs, "WGS84 UTM 33N");
  ASSERT_EQ(gcps.size(), 1u);
  // Known inverse: UTM33N (389262, 5819838) → lat≈52.5174, lon≈13.3680
  EXPECT_NEAR(gcps[0].lla_.at("latitude"), 52.5174, 0.001);
  EXPECT_NEAR(gcps[0].lla_.at("longitude"), 13.3680, 0.001);
  EXPECT_NEAR(gcps[0].lla_.at("altitude"), 34.5, 1e-9);
  // Raw coordinates should be the UTM values.
  EXPECT_NEAR(gcps[0].coordinates_.x(), 389262, 1e-3);
  EXPECT_NEAR(gcps[0].coordinates_.y(), 5819838, 1e-3);
}

TEST(GcpIo, ReadListEpsg) {
  // EPSG:32633 = UTM zone 33N — same test with EPSG notation.
  const std::string content =
      "EPSG:32633\n"
      "389262 5819838 34.5 500 300 img.jpg\n";

  ImageWidths widths = {{"img.jpg", {1000, 600}}};
  std::string crs;
  auto gcps = map::ReadGcpList(content, widths, &crs);

  EXPECT_EQ(crs, "EPSG:32633");
  ASSERT_EQ(gcps.size(), 1u);
  EXPECT_NEAR(gcps[0].lla_.at("latitude"), 52.5174, 0.001);
  EXPECT_NEAR(gcps[0].lla_.at("longitude"), 13.3680, 0.001);
}

// ── Cross-format consistency ─────────────────────────────────────────────────

TEST(GcpIo, JsonAndListProduceSameLatLon) {
  // A GCP written via JSON and via gcp_list.txt (WGS84) should have identical
  // lat/lon when read back.
  const double lat = 52.520008, lon = 13.404954, alt = 34.5;
  auto gcp = MakeTestGcp("cross-fmt", lat, lon, alt, true);
  AddObservation(gcp, "img.jpg", 0.1, -0.05);

  ImageWidths widths = {{"img.jpg", {4000, 3000}}};

  // JSON roundtrip.
  std::string json = map::WriteGcpJson({gcp});
  auto fromJson = map::ReadGcpJson(json);

  // List roundtrip.
  std::string list = map::WriteGcpList({gcp}, "WGS84", widths);
  auto fromList = map::ReadGcpList(list, widths);

  ASSERT_EQ(fromJson.size(), 1u);
  ASSERT_EQ(fromList.size(), 1u);

  EXPECT_NEAR(fromJson[0].lla_.at("latitude"), fromList[0].lla_.at("latitude"),
              1e-6);
  EXPECT_NEAR(fromJson[0].lla_.at("longitude"),
              fromList[0].lla_.at("longitude"), 1e-6);
  EXPECT_NEAR(fromJson[0].lla_.at("altitude"), fromList[0].lla_.at("altitude"),
              1e-6);

  // Observation projections should also match.
  ASSERT_EQ(fromJson[0].observations_.size(), 1u);
  ASSERT_EQ(fromList[0].observations_.size(), 1u);
  EXPECT_NEAR(fromJson[0].observations_[0].projection_.x(),
              fromList[0].observations_[0].projection_.x(), 1e-9);
  EXPECT_NEAR(fromJson[0].observations_[0].projection_.y(),
              fromList[0].observations_[0].projection_.y(), 1e-9);
}

}  // namespace
