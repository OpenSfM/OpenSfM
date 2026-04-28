#pragma once

#include <bundle/irls_solver.h>

#include <memory>
#include <vector>

namespace bundle {

class MixtureReweighting : public IReweightingStrategy {
 public:
  GroupWeightResult ComputeGroupWeights(
      const std::vector<double>& group_residuals,
      ErrorGroup& error_group) override;

  constexpr static double kMinDeterminant = 1e-3;
  constexpr static double kMinVariance = 0.025;
  constexpr static double kMinWeight = 1e-10;
  constexpr static double kMinSampleSize = 10;

  constexpr static int kMaxInnerIterations = 5;
  static constexpr double kConvergenceThreshold = 1e-3;
};

}  // namespace bundle
