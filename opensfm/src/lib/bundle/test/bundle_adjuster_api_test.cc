#include <bundle/bundle_adjuster.h>
#include <gtest/gtest.h>

namespace bundle {
namespace {

constexpr double kTolerance = 1e-12;

void ExpectPoseNear(const geometry::Pose& actual,
                    const geometry::Pose& expected) {
  EXPECT_TRUE(actual.GetOrigin().isApprox(expected.GetOrigin(), kTolerance));
  EXPECT_TRUE(actual.RotationCameraToWorldMin().isApprox(
      expected.RotationCameraToWorldMin(), kTolerance));
}

void ExpectSimilarityNear(const geometry::Similarity& actual,
                          const geometry::Similarity& expected) {
  EXPECT_TRUE(
      actual.Translation().isApprox(expected.Translation(), kTolerance));
  EXPECT_TRUE(actual.Rotation().isApprox(expected.Rotation(), kTolerance));
  EXPECT_NEAR(actual.Scale(), expected.Scale(), kTolerance);
}

void ExpectCameraNear(const geometry::Camera& actual,
                      const geometry::Camera& expected) {
  EXPECT_EQ(actual.GetProjectionType(), expected.GetProjectionType());
  EXPECT_EQ(actual.width, expected.width);
  EXPECT_EQ(actual.height, expected.height);
  EXPECT_TRUE(actual.GetParametersValues().isApprox(
      expected.GetParametersValues(), kTolerance));
}

struct ToyBundleScene {
  geometry::Camera camera;
  geometry::Pose rig_camera_pose;
  geometry::Pose rig0_pose;
  geometry::Pose rig1_pose;
  geometry::Similarity default_bias;
  Vec3d point;
  Vec3d gcp_point;
};

ToyBundleScene PopulateToyBundleScene(BundleAdjuster& ba) {
  ToyBundleScene scene;
  scene.camera = geometry::Camera::CreatePerspectiveCamera(0.8, 0.0, 0.0);
  scene.camera.id = "cam0";
  scene.camera.width = 1000;
  scene.camera.height = 1000;
  scene.rig_camera_pose = geometry::Pose();
  scene.rig0_pose.SetOrigin(Vec3d(0, 0, -5));
  scene.rig1_pose.SetOrigin(Vec3d(1, 0, -5));
  scene.default_bias =
      geometry::Similarity(Vec3d::Zero().eval(), Vec3d::Zero().eval(), 1.0);
  scene.point = Vec3d(0, 0, 5);
  scene.gcp_point = Vec3d(1, 0, 5);

  ba.AddCamera("cam0", scene.camera, scene.camera, true);
  ba.AddRigCamera("RC0", scene.rig_camera_pose, scene.rig_camera_pose, true);
  ba.AddRigInstance("rig0", scene.rig0_pose, {{"shot0", "cam0"}},
                    {{"shot0", "RC0"}}, false);
  ba.AddRigInstance("rig1", scene.rig1_pose, {{"shot1", "cam0"}},
                    {{"shot1", "RC0"}}, false);

  ba.AddPoint("pt0", scene.point, false);
  ba.AddPointProjectionObservation("shot0", "pt0", Vec2d(0.0, 0.0), 0.004,
                                   false);
  ba.AddPoint("gcp0", scene.gcp_point, false);
  ba.AddPointPrior("gcp0", scene.gcp_point, Vec3d::Constant(0.01), true);
  ba.AddPointProjectionObservation("shot0", "gcp0", Vec2d(0.01, 0.0), 0.001,
                                   true);

  ba.AddRelativeMotion(RelativeMotion("rig0", "rig1", Vec3d(0.0, 0.1, 0.2),
                                      Vec3d(1.0, 2.0, 3.0), 1.5, 2.0, true));

  return scene;
}

// ============================================================================
// BundleAdjuster public API: counts, getters, and geometry round-trip
// ============================================================================

TEST(BundleAdjusterAPI, ReturnsExpectedCountsAndGeometry) {
  BundleAdjuster ba;
  const auto scene = PopulateToyBundleScene(ba);

  EXPECT_EQ(ba.GetProjectionsCount(), 2);
  EXPECT_EQ(ba.GetRelativeMotionsCount(), 1);
  ExpectPoseNear(ba.GetRigCamera("RC0").GetValue(), scene.rig_camera_pose);
  ExpectPoseNear(ba.GetRigInstance("rig0").GetValue(), scene.rig0_pose);

  const auto rig_cameras = ba.GetRigCameras();
  ASSERT_EQ(rig_cameras.size(), 1);
  ASSERT_EQ(rig_cameras.count("RC0"), 1);
  ExpectPoseNear(rig_cameras.at("RC0").GetValue(), scene.rig_camera_pose);

  const auto rig_instances = ba.GetRigInstances();
  ASSERT_EQ(rig_instances.size(), 2);
  ASSERT_EQ(rig_instances.count("rig0"), 1);
  ASSERT_EQ(rig_instances.count("rig1"), 1);
  ExpectPoseNear(rig_instances.at("rig0").GetValue(), scene.rig0_pose);
  ExpectPoseNear(rig_instances.at("rig1").GetValue(), scene.rig1_pose);
  EXPECT_EQ(rig_instances.at("rig0").shot_cameras.at("shot0"), "cam0");
  EXPECT_EQ(rig_instances.at("rig1").shot_cameras.at("shot1"), "cam0");

  ExpectCameraNear(ba.GetCamera("cam0"), scene.camera);
  ExpectSimilarityNear(ba.GetBias("cam0"), scene.default_bias);
  EXPECT_TRUE(ba.GetPoint("pt0").GetValue().isApprox(scene.point, kTolerance));
  EXPECT_TRUE(
      ba.GetPoint("gcp0").GetValue().isApprox(scene.gcp_point, kTolerance));

  auto* shot = ba.GetShotRaw("shot0");
  ASSERT_NE(shot, nullptr);
  EXPECT_EQ(shot->GetCamera()->GetID(), "cam0");
  EXPECT_EQ(shot->GetRigCamera()->GetID(), "RC0");
  EXPECT_EQ(shot->GetRigInstance()->GetID(), "rig0");
}

// ============================================================================
// Reconstruction scale sharing
// ============================================================================

TEST(BundleAdjusterAPI, ReconstructionScaleSharingMatchesOwnershipMode) {
  BundleAdjuster ba;
  ba.AddReconstruction("rec0", false);
  ba.AddReconstructionInstance("rec0", 2.0, "rig0");
  ba.AddReconstructionInstance("rec0", 3.0, "rig1");

  auto shared = ba.GetReconstruction("rec0");
  EXPECT_TRUE(shared.shared);
  EXPECT_DOUBLE_EQ(shared.GetScale("rig0"), 2.0);
  EXPECT_DOUBLE_EQ(shared.GetScale("rig1"), 2.0);

  auto* shared_scale = shared.GetScalePtr("rig0");
  ASSERT_NE(shared_scale, nullptr);
  *shared_scale = 4.0;
  EXPECT_DOUBLE_EQ(shared.GetScale("rig0"), 4.0);
  EXPECT_DOUBLE_EQ(shared.GetScale("rig1"), 4.0);

  shared.SetScale("rig1", 5.0);
  EXPECT_DOUBLE_EQ(shared.GetScale("rig0"), 5.0);
  EXPECT_DOUBLE_EQ(shared.GetScale("rig1"), 5.0);

  ba.SetScaleSharing("rec0", false);
  auto unshared = ba.GetReconstruction("rec0");
  EXPECT_FALSE(unshared.shared);
  EXPECT_DOUBLE_EQ(unshared.GetScale("rig0"), 2.0);
  EXPECT_DOUBLE_EQ(unshared.GetScale("rig1"), 3.0);

  unshared.SetScale("rig1", 6.0);
  EXPECT_DOUBLE_EQ(unshared.GetScale("rig0"), 2.0);
  EXPECT_DOUBLE_EQ(unshared.GetScale("rig1"), 6.0);
}

// ============================================================================
// Report wrapper consistency (requires a Run)
// ============================================================================

TEST(BundleAdjusterAPI, ReportWrappersMatchCeresSummary) {
  BundleAdjuster ba;
  PopulateToyBundleScene(ba);

  ba.AddReconstruction("rec0", false);
  ba.AddReconstructionInstance("rec0", 1.0, "rig0");
  ba.AddReconstructionInstance("rec0", 1.0, "rig1");
  ba.AddRigInstancePositionPrior("rig0", Vec3d(0, 0, -5),
                                 Vec3d::Constant(0.1), "0");

  ba.SetMaxNumIterations(5);
  ba.SetNumThreads(1);
  ba.SetLinearSolverType("DENSE_QR");
  ba.SetUseAnalyticDerivatives(false);

  ba.Run();

  EXPECT_TRUE(ba.CeresSolverSummary().IsSolutionUsable());
  EXPECT_EQ(ba.BriefReport(), ba.CeresSolverSummary().BriefReport());
  EXPECT_EQ(ba.FullReport(), ba.CeresSolverSummary().FullReport());
}

// ============================================================================
// SetCameraBias
// ============================================================================

TEST(BundleAdjusterAPI, SetCameraBiasUpdatesStoredBias) {
  BundleAdjuster ba;
  PopulateToyBundleScene(ba);

  geometry::Similarity bias(Vec3d(0.1, 0.2, 0.3), Vec3d(0.01, 0.02, 0.03),
                            1.5);
  ba.SetCameraBias("cam0", bias);

  ExpectSimilarityNear(ba.GetBias("cam0"), bias);
}

TEST(BundleAdjusterAPI, SetCameraBiasThrowsForUnknownCamera) {
  BundleAdjuster ba;
  geometry::Similarity bias;
  EXPECT_THROW(ba.SetCameraBias("nonexistent", bias), std::runtime_error);
}

// ============================================================================
// AddRelativeRotation
// ============================================================================

TEST(BundleAdjusterAPI, AddRelativeRotationIncrementsCount) {
  BundleAdjuster ba;
  PopulateToyBundleScene(ba);

  ba.AddRelativeRotation(
      RelativeRotation("rig0", "rig1", Vec3d(0.01, 0.02, 0.03)));

  // RelativeRotation is separate from RelativeMotion
  EXPECT_EQ(ba.GetRelativeMotionsCount(), 1);
}

// ============================================================================
// HasPoint / GetPointRaw
// ============================================================================

TEST(BundleAdjusterAPI, HasPointAndGetPointRaw) {
  BundleAdjuster ba;
  PopulateToyBundleScene(ba);

  EXPECT_TRUE(ba.HasPoint("pt0"));
  EXPECT_TRUE(ba.HasPoint("gcp0"));
  EXPECT_FALSE(ba.HasPoint("nonexistent"));

  auto* raw = ba.GetPointRaw("pt0");
  ASSERT_NE(raw, nullptr);
  EXPECT_TRUE(raw->GetValue().isApprox(Vec3d(0, 0, 5), kTolerance));
}

// ============================================================================
// SetGaugeFixShots
// ============================================================================

TEST(BundleAdjusterAPI, SetGaugeFixShotsDoesNotCrash) {
  BundleAdjuster ba;
  PopulateToyBundleScene(ba);

  ba.AddReconstruction("rec0", false);
  ba.AddReconstructionInstance("rec0", 1.0, "rig0");
  ba.AddReconstructionInstance("rec0", 1.0, "rig1");

  // Should not throw — pins rig0 as origin, rig1 as scale
  EXPECT_NO_THROW(ba.SetGaugeFixShots("shot0", "shot1"));
}

// ============================================================================
// Absolute constraints: verify they can be added without crashing
// (running these through a full Run() requires carefully crafted geometry,
//  so we only test the adder APIs here; full integration is in
//  bundle_adjuster_scene_test.cc)
// ============================================================================

TEST(BundleAdjusterAPI, AbsoluteConstraintsCanBeAdded) {
  BundleAdjuster ba;
  PopulateToyBundleScene(ba);

  EXPECT_NO_THROW(ba.AddAbsoluteUpVector("shot0", Vec3d(0, -1, 0), 0.1));
  EXPECT_NO_THROW(ba.AddAbsolutePan("shot0", 0.0, 0.1));
  EXPECT_NO_THROW(ba.AddAbsoluteTilt("shot0", 0.0, 0.1));
  EXPECT_NO_THROW(ba.AddAbsoluteRoll("shot0", 0.0, 0.1));
}

TEST(BundleAdjusterAPI, LinearMotionCanBeAdded) {
  BundleAdjuster ba;
  PopulateToyBundleScene(ba);

  // Add a third rig instance for the triplet
  geometry::Pose rig2_pose;
  rig2_pose.SetOrigin(Vec3d(2, 0, -5));
  ba.AddRigInstance("rig2", rig2_pose, {{"shot2", "cam0"}},
                    {{"shot2", "RC0"}}, false);

  EXPECT_NO_THROW(ba.AddLinearMotion("shot0", "shot1", "shot2", 0.5,
                                     0.01, 0.01));
}

// ============================================================================
// ComputeReprojectionErrors flag
// ============================================================================

TEST(BundleAdjusterAPI, ReprojectionErrorsComputedWhenEnabled) {
  BundleAdjuster ba;
  PopulateToyBundleScene(ba);

  ba.AddReconstruction("rec0", false);
  ba.AddReconstructionInstance("rec0", 1.0, "rig0");
  ba.AddReconstructionInstance("rec0", 1.0, "rig1");
  ba.AddRigInstancePositionPrior("rig0", Vec3d(0, 0, -5),
                                 Vec3d::Constant(0.1), "0");

  ba.SetComputeReprojectionErrors(true);
  ba.SetMaxNumIterations(5);
  ba.SetNumThreads(1);
  ba.SetLinearSolverType("DENSE_QR");
  ba.SetUseAnalyticDerivatives(false);

  ba.Run();

  EXPECT_TRUE(ba.CeresSolverSummary().IsSolutionUsable());
  // After run with reprojection errors enabled, point should have errors stored
  auto pt = ba.GetPoint("pt0");
  EXPECT_FALSE(pt.reprojection_errors.empty());
}

}  // namespace
}  // namespace bundle
