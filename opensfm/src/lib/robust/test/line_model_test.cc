#include <gtest/gtest.h>
#include <robust/line_model.h>
#include <robust/robust_estimator.h>
#include <robust/scorer.h>

#include <vector>

namespace {

// ============================================================================
// Line model: Estimate / Evaluate
// ============================================================================

TEST(LineModel, EstimateFromTwoPoints) {
  // y = 2x + 1  =>  model = (2, 1)
  Eigen::Vector2d p1(1.0, 3.0);
  Eigen::Vector2d p2(3.0, 7.0);

  std::vector<Eigen::Vector2d> pts = {p1, p2};
  Line::Type models[Line::MAX_MODELS];
  int n = Line::Estimate(pts.begin(), pts.end(), models);

  ASSERT_EQ(n, 1);
  EXPECT_NEAR(models[0][0], 2.0, 1e-12);
  EXPECT_NEAR(models[0][1], 1.0, 1e-12);
}

TEST(LineModel, EstimateVerticalLineProducesNonFiniteModel) {
  // Vertical line: x1[0] == x2[0] → division by zero → inf coefficients.
  // Estimate returns 1 because inf is not NaN, but the model is degenerate.
  Eigen::Vector2d p1(2.0, 1.0);
  Eigen::Vector2d p2(2.0, 5.0);

  std::vector<Eigen::Vector2d> pts = {p1, p2};
  Line::Type models[Line::MAX_MODELS];
  int n = Line::Estimate(pts.begin(), pts.end(), models);

  EXPECT_EQ(n, 1);
  EXPECT_FALSE(std::isfinite(models[0][0]));
  EXPECT_FALSE(std::isfinite(models[0][1]));
}

TEST(LineModel, EstimateWithOriginPointReturnsZero) {
  Eigen::Vector2d p1(0.0, 0.0);
  Eigen::Vector2d p2(1.0, 1.0);

  std::vector<Eigen::Vector2d> pts = {p1, p2};
  Line::Type models[Line::MAX_MODELS];
  int n = Line::Estimate(pts.begin(), pts.end(), models);

  EXPECT_EQ(n, 0);
}

TEST(LineModel, EvaluateReturnsResidual) {
  Eigen::Vector2d model(2.0, 1.0);  // y = 2x + 1

  auto e1 = Line::Evaluate(model, Eigen::Vector2d(3.0, 7.0));
  EXPECT_NEAR(e1[0], 0.0, 1e-12);

  auto e2 = Line::Evaluate(model, Eigen::Vector2d(3.0, 7.5));
  EXPECT_NEAR(e2[0], 0.5, 1e-12);

  auto e3 = Line::Evaluate(model, Eigen::Vector2d(3.0, 6.0));
  EXPECT_NEAR(e3[0], -1.0, 1e-12);
}

TEST(LineModel, EvaluateModelReturnsAllErrors) {
  Eigen::Vector2d model(2.0, 1.0);
  std::vector<Eigen::Vector2d> pts = {
      {1.0, 3.0},
      {2.0, 5.5},
      {0.5, 1.0},
  };

  auto errors = Line::EvaluateModel(model, pts.begin(), pts.end());
  ASSERT_EQ(errors.size(), 3);
  EXPECT_NEAR(errors[0][0], 0.0, 1e-12);
  EXPECT_NEAR(errors[1][0], 0.5, 1e-12);
  EXPECT_NEAR(errors[2][0], -1.0, 1e-12);
}

// ============================================================================
// End-to-end RANSAC on a line with outliers
// ============================================================================

TEST(RobustEstimatorLine, RANSACFindsLineAmidOutliers) {
  const double a_true = 3.0;
  const double b_true = 2.0;

  std::vector<Eigen::Vector2d> samples;
  for (int i = 1; i <= 20; ++i) {
    double x = 0.1 * i;
    double y = a_true * x + b_true;
    samples.push_back(Eigen::Vector2d(x, y));
  }
  samples.push_back(Eigen::Vector2d(0.5, 100.0));
  samples.push_back(Eigen::Vector2d(1.0, -50.0));
  samples.push_back(Eigen::Vector2d(1.5, 200.0));
  samples.push_back(Eigen::Vector2d(0.3, -80.0));
  samples.push_back(Eigen::Vector2d(0.7, 150.0));

  RansacScoring scorer(0.5);
  RobustEstimatorParams params;
  params.iterations = 200;
  params.probability = 0.999;

  auto result = Estimate<RansacScoring, Line>(samples, scorer, params);

  EXPECT_NEAR(result.lo_model[0], a_true, 0.1);
  EXPECT_NEAR(result.lo_model[1], b_true, 0.1);
  EXPECT_GE(result.inliers_indices.size(), 18u);
}

TEST(RobustEstimatorLine, MSACFindsLineAmidOutliers) {
  const double a_true = -1.0;
  const double b_true = 5.0;

  std::vector<Eigen::Vector2d> samples;
  for (int i = 1; i <= 15; ++i) {
    double x = 0.2 * i;
    double y = a_true * x + b_true;
    samples.push_back(Eigen::Vector2d(x, y));
  }
  samples.push_back(Eigen::Vector2d(0.5, 50.0));
  samples.push_back(Eigen::Vector2d(1.0, -30.0));
  samples.push_back(Eigen::Vector2d(2.0, 40.0));
  samples.push_back(Eigen::Vector2d(0.8, -60.0));

  MSacScoring scorer(0.5);
  RobustEstimatorParams params;
  params.iterations = 200;

  auto result = Estimate<MSacScoring, Line>(samples, scorer, params);

  EXPECT_NEAR(result.lo_model[0], a_true, 0.1);
  EXPECT_NEAR(result.lo_model[1], b_true, 0.1);
  EXPECT_GE(result.inliers_indices.size(), 13u);
}

// ============================================================================
// ShouldStop adaptive iteration count
// ============================================================================

TEST(RobustEstimatorLine, ShouldStopWhenHighInlierRatio) {
  RobustEstimatorParams params;
  params.probability = 0.99;
  params.use_iteration_reduction = true;

  ScoreInfo<Line::Type> score;
  for (int i = 0; i < 90; ++i) score.inliers_indices.push_back(i);

  EXPECT_TRUE(ShouldStop<Line>(params, score, 100, 50));
}

TEST(RobustEstimatorLine, ShouldNotStopWhenLowInlierRatio) {
  RobustEstimatorParams params;
  params.probability = 0.99;
  params.use_iteration_reduction = true;

  ScoreInfo<Line::Type> score;
  for (int i = 0; i < 10; ++i) score.inliers_indices.push_back(i);

  EXPECT_FALSE(ShouldStop<Line>(params, score, 100, 5));
}

TEST(RobustEstimatorLine, DisabledIterationReductionNeverStops) {
  RobustEstimatorParams params;
  params.use_iteration_reduction = false;

  ScoreInfo<Line::Type> score;
  for (int i = 0; i < 99; ++i) score.inliers_indices.push_back(i);

  EXPECT_FALSE(ShouldStop<Line>(params, score, 100, 1000));
}

// ============================================================================
// RandomSamplesGenerator
// ============================================================================

TEST(RandomSampler, GeneratesCorrectSampleSize) {
  RandomSamplesGenerator<std::mt19937> gen(42);
  std::vector<Eigen::Vector2d> data;
  for (int i = 0; i < 50; ++i) data.push_back(Eigen::Vector2d(i, i));

  auto samples = gen.GetRandomSamples<Line>(data, Line::MINIMAL_SAMPLES);
  EXPECT_EQ(samples.size(), static_cast<size_t>(Line::MINIMAL_SAMPLES));
}

TEST(RandomSampler, DeterministicWithSameSeed) {
  std::vector<Eigen::Vector2d> data;
  for (int i = 0; i < 50; ++i) data.push_back(Eigen::Vector2d(i, i * 2));

  RandomSamplesGenerator<std::mt19937> gen1(123);
  auto s1 = gen1.GetRandomSamples<Line>(data, 2);

  RandomSamplesGenerator<std::mt19937> gen2(123);
  auto s2 = gen2.GetRandomSamples<Line>(data, 2);

  ASSERT_EQ(s1.size(), s2.size());
  for (size_t i = 0; i < s1.size(); ++i) {
    EXPECT_TRUE(s1[i].isApprox(s2[i]));
  }
}

}  // namespace
