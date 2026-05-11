#include <bundle/bundle_adjuster.h>
#include <bundle/irls_solver.h>
#include <bundle/mixture_reweighting.h>
#include <gtest/gtest.h>

namespace bundle {
namespace {

// ============================================================================
// BundleAdjuster density ratio import/export consistency
// ============================================================================

TEST(DensityRatio, BundleAdjuster_DefaultDensityRatio_InitialValue) {
  BundleAdjuster ba;
  EXPECT_DOUBLE_EQ(ba.GetDefaultDensityRatio(), 0.001);
}

TEST(DensityRatio, BundleAdjuster_SetGetDefaultDensityRatio) {
  BundleAdjuster ba;
  ba.SetDefaultDensityRatio(0.05);
  EXPECT_DOUBLE_EQ(ba.GetDefaultDensityRatio(), 0.05);
}

TEST(DensityRatio, BundleAdjuster_SetGetGroupDensityRatio) {
  BundleAdjuster ba;
  ba.SetGroupDensityRatio("GCP2D", 0.1);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("GCP2D"), 0.1);
}

TEST(DensityRatio, BundleAdjuster_GroupFallsBackToDefault) {
  BundleAdjuster ba;
  ba.SetDefaultDensityRatio(0.05);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("PROJ"), 0.05);
}

TEST(DensityRatio, BundleAdjuster_GroupOverrideBeatsDefault) {
  BundleAdjuster ba;
  ba.SetDefaultDensityRatio(0.05);
  ba.SetGroupDensityRatio("GCP2D", 0.2);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("GCP2D"), 0.2);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("PROJ"), 0.05);
}

TEST(DensityRatio, BundleAdjuster_MultipleGroupOverrides) {
  BundleAdjuster ba;
  ba.SetDefaultDensityRatio(0.01);
  ba.SetGroupDensityRatio("GCP2D", 0.1);
  ba.SetGroupDensityRatio("GCP3D", 0.2);
  ba.SetGroupDensityRatio("PROJ", 0.03);

  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("GCP2D"), 0.1);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("GCP3D"), 0.2);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("PROJ"), 0.03);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("GPS"), 0.01);
}

TEST(DensityRatio, BundleAdjuster_OverwriteGroupDensityRatio) {
  BundleAdjuster ba;
  ba.SetGroupDensityRatio("GCP2D", 0.1);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("GCP2D"), 0.1);
  ba.SetGroupDensityRatio("GCP2D", 0.5);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("GCP2D"), 0.5);
}

// ============================================================================
// IRLSSolver density ratio propagation
// ============================================================================

class DummyCostFunction : public ceres::CostFunction {
 public:
  explicit DummyCostFunction(int num_residuals) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(1);
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 0.0;
    }
    return true;
  }
};

TEST(DensityRatio, IRLSSolver_SetGroupDensityRatio) {
  IRLSSolver solver;
  double param = 1.0;
  solver.AddParameterBlock(&param, 1);

  auto* cost = new DummyCostFunction(2);
  solver.AddResidualBlock(cost, nullptr, "PROJ", &param);
  solver.SetGroupDensityRatio("PROJ", 0.05);

  auto* cost2 = new DummyCostFunction(2);
  solver.AddResidualBlock(cost2, nullptr, "GCP2D", &param);
  solver.SetGroupDensityRatio("GCP2D", 0.2);

  auto results = solver.ComputeWeights();
  ASSERT_EQ(results.size(), 2);
}

TEST(DensityRatio, IRLSSolver_SetAllGroupsDensityRatio) {
  IRLSSolver solver;
  double param = 1.0;
  solver.AddParameterBlock(&param, 1);

  auto* cost1 = new DummyCostFunction(2);
  solver.AddResidualBlock(cost1, nullptr, "PROJ", &param);

  auto* cost2 = new DummyCostFunction(2);
  solver.AddResidualBlock(cost2, nullptr, "GCP2D", &param);

  auto* cost3 = new DummyCostFunction(2);
  solver.AddResidualBlock(cost3, nullptr, "GPS", &param);

  solver.SetAllGroupsDensityRatio(0.07);
  solver.SetGroupDensityRatio("GCP2D", 0.3);

  auto results = solver.ComputeWeights();
  ASSERT_EQ(results.size(), 3);
}

// ============================================================================
// End-to-end: BundleAdjuster density ratio reaches IRLSSolver
// ============================================================================

TEST(DensityRatio, BundleAdjuster_EndToEnd_DensityRatioInRun) {
  BundleAdjuster ba;

  geometry::Camera camera =
      geometry::Camera::CreatePerspectiveCamera(0.8, 0.0, 0.0);
  camera.id = "cam0";
  camera.width = 1000;
  camera.height = 1000;
  ba.AddCamera("cam0", camera, camera, true);

  geometry::Pose pose;
  pose.SetOrigin(Vec3d(0, 0, -5));

  ba.AddRigCamera("RC0", geometry::Pose(), geometry::Pose(), true);
  ba.AddRigInstance("rig0", pose, {{"shot0", "cam0"}}, {{"shot0", "RC0"}},
                    false);
  ba.AddRigInstancePositionPrior("rig0", Vec3d(0, 0, -5), Vec3d::Constant(0.1),
                                 "0");

  ba.AddPoint("pt0", Vec3d(0, 0, 5), false);
  ba.AddPointProjectionObservation("shot0", "pt0", Vec2d(0.0, 0.0), 0.004,
                                   false);

  ba.AddPoint("gcp0", Vec3d(1, 0, 5), false);
  ba.AddPointPrior("gcp0", Vec3d(1, 0, 5), Vec3d::Constant(0.01), true);
  ba.AddPointProjectionObservation("shot0", "gcp0", Vec2d(0.01, 0.0), 0.001,
                                   true);

  ba.SetDefaultDensityRatio(0.002);
  ba.SetGroupDensityRatio("GCP2D", 0.5);

  EXPECT_DOUBLE_EQ(ba.GetDefaultDensityRatio(), 0.002);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("GCP2D"), 0.5);
  EXPECT_DOUBLE_EQ(ba.GetGroupDensityRatio("PROJ"), 0.002);

  ba.SetMaxNumIterations(5);
  ba.SetNumThreads(1);
  ba.SetLinearSolverType("DENSE_QR");
  ba.SetUseAnalyticDerivatives(false);

  ba.Run();

  EXPECT_TRUE(ba.CeresSolverSummary().IsSolutionUsable());
}

}  // namespace
}  // namespace bundle
