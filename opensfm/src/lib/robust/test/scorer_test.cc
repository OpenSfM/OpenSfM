#include <gtest/gtest.h>
#include <robust/scorer.h>

#include <Eigen/Dense>
#include <vector>

namespace {

// ============================================================================
// RansacScoring
// ============================================================================

TEST(RansacScoring, CountsInliersCorrectly) {
  RansacScoring scorer(0.5);
  using E = Eigen::Matrix<double, 1, 1>;
  std::vector<E> errors;
  errors.push_back(E::Constant(0.1));   // inlier
  errors.push_back(E::Constant(0.4));   // inlier
  errors.push_back(E::Constant(0.6));   // outlier
  errors.push_back(E::Constant(0.05));  // inlier

  ScoreInfo<int> best;
  auto result = scorer.Score(errors.begin(), errors.end(), best);

  EXPECT_DOUBLE_EQ(result.score, 3.0);
  ASSERT_EQ(result.inliers_indices.size(), 3);
  EXPECT_EQ(result.inliers_indices[0], 0);
  EXPECT_EQ(result.inliers_indices[1], 1);
  EXPECT_EQ(result.inliers_indices[2], 3);
}

TEST(RansacScoring, AllBelowThresholdAreInliers) {
  RansacScoring scorer(10.0);
  using E = Eigen::Matrix<double, 1, 1>;
  std::vector<E> errors = {E::Constant(1.0), E::Constant(2.0),
                           E::Constant(3.0)};

  ScoreInfo<int> best;
  auto result = scorer.Score(errors.begin(), errors.end(), best);

  EXPECT_DOUBLE_EQ(result.score, 3.0);
  EXPECT_EQ(result.inliers_indices.size(), 3);
}

TEST(RansacScoring, NoInliers) {
  RansacScoring scorer(0.01);
  using E = Eigen::Matrix<double, 1, 1>;
  std::vector<E> errors = {E::Constant(1.0), E::Constant(2.0)};

  ScoreInfo<int> best;
  auto result = scorer.Score(errors.begin(), errors.end(), best);

  EXPECT_DOUBLE_EQ(result.score, 0.0);
  EXPECT_TRUE(result.inliers_indices.empty());
}

// ============================================================================
// MSacScoring
// ============================================================================

TEST(MSacScoring, PenalizesOutliers) {
  MSacScoring scorer(1.0);
  using E = Eigen::Matrix<double, 1, 1>;
  std::vector<E> errors;
  errors.push_back(E::Constant(0.5));  // inlier: contributes 0.25
  errors.push_back(E::Constant(2.0));  // outlier: contributes 1.0

  ScoreInfo<int> best;
  auto result = scorer.Score(errors.begin(), errors.end(), best);

  // score = 1 / (0.25 + 1.0 + eps)
  EXPECT_NEAR(result.score, 1.0 / (1.25 + 1e-8), 1e-6);
  ASSERT_EQ(result.inliers_indices.size(), 1);
  EXPECT_EQ(result.inliers_indices[0], 0);
}

TEST(MSacScoring, AllInliers) {
  MSacScoring scorer(2.0);
  using E = Eigen::Matrix<double, 1, 1>;
  std::vector<E> errors = {E::Constant(0.5), E::Constant(1.0)};

  ScoreInfo<int> best;
  auto result = scorer.Score(errors.begin(), errors.end(), best);

  EXPECT_NEAR(result.score, 1.0 / (1.25 + 1e-8), 1e-6);
  EXPECT_EQ(result.inliers_indices.size(), 2);
}

// ============================================================================
// LMedSScoring
// ============================================================================

TEST(LMedSScoring, ComputesMedianBasedThreshold) {
  LMedSScoring scorer(2.5);
  using E = Eigen::Matrix<double, 1, 1>;

  // 4 errors: norms = [0.1, 0.2, 0.5, 3.0]
  // median index = floor(4 * 0.5) = 2 → nth_element picks 0.5
  // mad = 1.4826 * 0.5 = 0.7413
  // threshold = 2.5 * 0.7413 = 1.85325
  std::vector<E> errors = {E::Constant(0.1), E::Constant(0.2),
                           E::Constant(0.5), E::Constant(3.0)};

  ScoreInfo<int> best;
  auto result = scorer.Score(errors.begin(), errors.end(), best);

  // Inliers: 0.1, 0.2, 0.5 (all < 1.85325); 3.0 is outlier
  EXPECT_EQ(result.inliers_indices.size(), 3);
  // Score = 1 / (median + eps) = 1 / (0.5 + 1e-8)
  EXPECT_NEAR(result.score, 1.0 / (0.5 + 1e-8), 1e-4);
}

// ============================================================================
// ScoreInfo ordering
// ============================================================================

TEST(ScoreInfo, HigherScoreIsGreater) {
  ScoreInfo<int> a, b;
  a.score = 5.0;
  b.score = 3.0;

  EXPECT_TRUE(b < a);
  EXPECT_FALSE(a < b);
}

}  // namespace
