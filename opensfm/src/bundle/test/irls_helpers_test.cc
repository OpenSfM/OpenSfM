#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <bundle/irls_helpers.h>

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace bundle {
namespace {

TEST(IRLSHelpersTest, ComputeKForGroup_Mixture) {
  // Synthesize data: 
  // 60% inliers from Gaussian with sigma = 1.0 (variance in each dim)
  // 40% outliers from Uniform in volume V
  
  int N = 1000;
  int n_inliers = 600;
  int n_outliers = 400;
  double sigma = 0.5; // Small sigma
  double sigma2 = sigma * sigma;
  double V = 10000.0; // Large search volume
  
  std::vector<double> d2_residuals;
  d2_residuals.reserve(N);
  
  std::mt19937 rng(42);
  std::normal_distribution<double> gaussian(0.0, sigma);
  std::uniform_real_distribution<double> uniform_outlier(0.0, std::sqrt(V)); 
  
  // Inliers: d2 = x^2 + y^2 where x,y ~ N(0, sigma)
  for (int i = 0; i < n_inliers; ++i) {
      double x = gaussian(rng);
      double y = gaussian(rng);
      d2_residuals.push_back(x*x + y*y);
  }
  
  // Outliers: uniformly distributed large residuals
  std::uniform_real_distribution<double> uniform_large(50.0, 500.0);
  for(int i = 0; i < n_outliers; ++i) {
      d2_residuals.push_back(uniform_large(rng));
  }
  
  double K = ComputeKForGroup(d2_residuals, V);
  
  // Expected calculation:
  // K ~ (Outlier Density) / (Peak Inlier Density)
  // Gamma=0.6, V=10000, Sigma=0.5 -> K ~ 1e-4
  
  EXPECT_GT(K, 0.0);
  EXPECT_LT(K, 1e-3); // Should be small
  
  // Check approximate value order of magnitude
  double expected_K = (0.4 / V) / (0.6 / (2.0 * M_PI * sigma2));
  // Allow factor of 2-3 error due to estimation noise
  EXPECT_NEAR(K, expected_K, expected_K * 2.0);
}

TEST(IRLSHelpersTest, ComputeKForGroup_FallbackSmallData) {
    std::vector<double> d2_residuals = {0.1, 0.2, 0.3}; // N=3 < 10
    double K = ComputeKForGroup(d2_residuals, 100.0);
    EXPECT_EQ(K, 1e-3);
}

} // namespace
} // namespace bundle