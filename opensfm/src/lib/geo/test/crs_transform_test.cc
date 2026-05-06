#include <geo/crs_transform.h>
#include <gtest/gtest.h>

#include <string>

namespace {

// ── ParseGcpProjectionString ─────────────────────────────────────────────────

TEST(CrsTransform, ParseWgs84ReturnsEmpty) {
  EXPECT_TRUE(geo::ParseGcpProjectionString("WGS84").empty());
}

TEST(CrsTransform, ParseEpsg4326ReturnsEmpty) {
  EXPECT_TRUE(geo::ParseGcpProjectionString("EPSG:4326").empty());
}

TEST(CrsTransform, ParseUtmNorth) {
  std::string proj = geo::ParseGcpProjectionString("WGS84 UTM 32N");
  EXPECT_NE(proj.find("+proj=utm"), std::string::npos);
  EXPECT_NE(proj.find("+zone=32"), std::string::npos);
  EXPECT_NE(proj.find("+north"), std::string::npos);
}

TEST(CrsTransform, ParseUtmSouth) {
  std::string proj = geo::ParseGcpProjectionString("WGS84 UTM 56S");
  EXPECT_NE(proj.find("+proj=utm"), std::string::npos);
  EXPECT_NE(proj.find("+zone=56"), std::string::npos);
  EXPECT_NE(proj.find("+south"), std::string::npos);
}

TEST(CrsTransform, ParseUtmNoHemisphere) {
  // Zone number without N/S suffix defaults to north.
  std::string proj = geo::ParseGcpProjectionString("WGS84 UTM 10");
  EXPECT_NE(proj.find("+zone=10"), std::string::npos);
  EXPECT_NE(proj.find("+north"), std::string::npos);
}

TEST(CrsTransform, ParseEpsgPassthrough) {
  std::string proj = geo::ParseGcpProjectionString("EPSG:32633");
  EXPECT_EQ(proj, "EPSG:32633");
}

TEST(CrsTransform, ParseProj4Passthrough) {
  std::string input = "+proj=utm +zone=33 +north +datum=WGS84";
  EXPECT_EQ(geo::ParseGcpProjectionString(input), input);
}

TEST(CrsTransform, ParseUnknownReturnsEmpty) {
  // Unknown CRS strings return empty (treated as identity).
  EXPECT_TRUE(geo::ParseGcpProjectionString("SomeRandomCRS").empty());
}

TEST(CrsTransform, ParseTrimsWhitespace) {
  EXPECT_TRUE(geo::ParseGcpProjectionString("  WGS84  ").empty());
}

// ── CrsTransform identity ────────────────────────────────────────────────────

TEST(CrsTransform, IdentityFromEmptyString) {
  geo::CrsTransform ct("");
  EXPECT_TRUE(ct.isIdentity());
  EXPECT_TRUE(ct.isValid());
}

TEST(CrsTransform, IdentityTransformPassesThrough) {
  geo::CrsTransform ct("");
  double lat = 0, lon = 0;
  ASSERT_TRUE(ct.transform(13.4, 52.5, lat, lon));
  // Identity: easting→lon, northing→lat.
  EXPECT_DOUBLE_EQ(lon, 13.4);
  EXPECT_DOUBLE_EQ(lat, 52.5);
}

// ── CrsTransform with PROJ ──────────────────────────────────────────────────

TEST(CrsTransform, UtmZone33NForward) {
  // UTM zone 33N: known Berlin point.
  std::string proj = geo::ParseGcpProjectionString("WGS84 UTM 33N");
  geo::CrsTransform ct(proj);
  EXPECT_FALSE(ct.isIdentity());
  EXPECT_TRUE(ct.isValid());

  // UTM33N easting=389262, northing=5819838 → lat≈52.5174, lon≈13.3680
  double lat = 0, lon = 0;
  ASSERT_TRUE(ct.transform(389262, 5819838, lat, lon));
  EXPECT_NEAR(lat, 52.5174, 0.001);
  EXPECT_NEAR(lon, 13.3680, 0.001);
}

TEST(CrsTransform, EpsgUtmEquivalence) {
  // EPSG:32633 = UTM zone 33N — should give the same result.
  geo::CrsTransform ctUtm(geo::ParseGcpProjectionString("WGS84 UTM 33N"));
  geo::CrsTransform ctEpsg(geo::ParseGcpProjectionString("EPSG:32633"));

  double lat1 = 0, lon1 = 0, lat2 = 0, lon2 = 0;
  ASSERT_TRUE(ctUtm.transform(389262, 5819838, lat1, lon1));
  ASSERT_TRUE(ctEpsg.transform(389262, 5819838, lat2, lon2));

  EXPECT_NEAR(lat1, lat2, 1e-6);
  EXPECT_NEAR(lon1, lon2, 1e-6);
}

TEST(CrsTransform, InvalidProjStringNotValid) {
  // A clearly invalid projection string should create an invalid transform.
  geo::CrsTransform ct("+proj=invalid_proj_that_does_not_exist");
  EXPECT_FALSE(ct.isIdentity());
  EXPECT_FALSE(ct.isValid());

  double lat = 0, lon = 0;
  EXPECT_FALSE(ct.transform(0, 0, lat, lon));
}

TEST(CrsTransform, MoveSemantics) {
  geo::CrsTransform ct1(geo::ParseGcpProjectionString("WGS84 UTM 33N"));
  EXPECT_TRUE(ct1.isValid());

  geo::CrsTransform ct2(std::move(ct1));
  EXPECT_TRUE(ct2.isValid());

  double lat = 0, lon = 0;
  ASSERT_TRUE(ct2.transform(389262, 5819838, lat, lon));
  EXPECT_NEAR(lat, 52.52, 0.01);
}

}  // namespace
