#include <bundle/classic_sigma_reweighting.h>
#include <bundle/irls_solver.h>
#include <bundle/mad_sigma_reweighting.h>
#include <bundle/mixture_reweighting.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

namespace bundle {
namespace {

// Dummy cost function that claims to have N residuals.
// We only need num_residuals() to return the correct dimension.
class DummyCostFunction : public ceres::CostFunction {
 public:
  DummyCostFunction(int num_residuals) {
    set_num_residuals(num_residuals);
    // Add a dummy parameter block of size 1 so Ceres is happy if we ever
    // inspected it
    mutable_parameter_block_sizes()->push_back(1);
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    return true;
  }
};

class ReweightingTest : public ::testing::Test {
 protected:
  // Helper to create an ErrorGroup with N residuals of dimension dim
  // The ResidualInfo structs are populated with dummy cost functions.
  ErrorGroup CreateErrorGroup(int n, int dim) {
    ErrorGroup group;
    // We need to keep the cost functions alive
    cost_functions_.emplace_back(std::make_unique<DummyCostFunction>(dim));

    for (int i = 0; i < n; ++i) {
      ResidualInfo info;
      info.id = nullptr;  // Mocking ID is hard, hopefully not used
      info.cost_function = cost_functions_.back().get();
      info.weight = new double(1.0);
      group.residuals_info.push_back(info);
    }
    return group;
  }

  // Storage for cost functions to keep them valid during test
  std::vector<std::unique_ptr<ceres::CostFunction>> cost_functions_;
};

TEST_F(ReweightingTest, ClassicSigmaReweighting_LogLikelihood_And_Weights) {
  ClassicSigmaReweighting strategy;
  strategy.sigma_threshold = 3.0;

  int n_inliers = 100;
  int n_outliers = 10;
  int dim = 1;
  ErrorGroup group = CreateErrorGroup(n_inliers + n_outliers, dim);

  std::vector<double> residuals;
  residuals.reserve(n_inliers + n_outliers);
  for (int i = 0; i < n_inliers; ++i) {
    residuals.push_back(0.5);  // safely inside 3*sigma if sigma~0.5
  }
  for (int i = 0; i < n_outliers; ++i) {
    residuals.push_back(100.0);
  }

  GroupWeightResult result = strategy.ComputeGroupWeights(residuals, group);

  EXPECT_NE(result.log_likelihood, 0.0);

  EXPECT_GT(*group.residuals_info[0].weight, 0.0);
  EXPECT_LE(*group.residuals_info[0].weight, 1.0);
}

TEST_F(ReweightingTest, MADSigmaReweighting_RejectOutliers) {
  MADSigmaReweighting strategy;

  int n_inliers = 100;
  int n_outliers = 50;
  int dim = 1;
  ErrorGroup group = CreateErrorGroup(n_inliers + n_outliers, dim);

  std::vector<double> residuals;
  for (int i = 0; i < n_inliers; ++i) {
    residuals.push_back(1.0 + (i % 2 == 0 ? 0.1 : -0.1));
  }

  for (int i = 0; i < n_outliers; ++i) {
    residuals.push_back(1000.0);
  }

  GroupWeightResult result = strategy.ComputeGroupWeights(residuals, group);
  EXPECT_NE(result.log_likelihood, 0.0);

  EXPECT_GT(*group.residuals_info[0].weight, 0.8);
  EXPECT_LT(*group.residuals_info[n_inliers].weight, 0.1);
}

TEST_F(ReweightingTest, MADSigmaReweighting_RejectOutliers_Fixed) {
  MADSigmaReweighting strategy;
  strategy.sigma_threshold = 3.0;

  int n_inliers = 100;
  int n_outliers = 50;
  int dim = 1;
  ErrorGroup group = CreateErrorGroup(n_inliers + n_outliers, dim);

  std::vector<double> residuals;
  // Inliers: centered at 0 with some noise. Norms are absolute values.
  for (int i = 0; i < n_inliers; ++i) {
    residuals.push_back(0.1 + (i % 2 == 0 ? 0.05 : -0.05));
  }
  // Outliers: 1000.0
  for (int i = 0; i < n_outliers; ++i) {
    residuals.push_back(1000.0);
  }

  strategy.ComputeGroupWeights(residuals, group);

  EXPECT_NEAR(*group.residuals_info[0].weight, 1.0, 0.1);
  EXPECT_NEAR(*group.residuals_info[n_inliers].weight, 0.0, 1e-5);
}

TEST_F(ReweightingTest, MixtureReweighting_SeparatesMixture) {
  MixtureReweighting strategy;

  int dim = 2;
  int n_inliers = 200;
  int n_outliers = 20;
  ErrorGroup group = CreateErrorGroup(n_inliers + n_outliers, dim);

  group.outlier_density = 0.1;

  std::vector<double> residuals;
  for (int i = 0; i < n_inliers; ++i) {
    residuals.push_back(0.1);
    residuals.push_back(0.1);
  }
  for (int i = 0; i < n_outliers; ++i) {
    residuals.push_back(10.0);
    residuals.push_back(10.0);

    // Initialize weights to small values to simulate a good initialization
    // This allows EM to compute a clean covariance in the first step
    *group.residuals_info[n_inliers + i].weight = 0.01;
  }

  GroupWeightResult result = strategy.ComputeGroupWeights(residuals, group);
  EXPECT_NE(result.log_likelihood, 0.0);

  EXPECT_GT(*group.residuals_info[0].weight, 0.8);
  EXPECT_LT(*group.residuals_info[n_inliers].weight, 0.1);
}

}  // namespace
}  // namespace bundle
