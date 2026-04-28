#pragma once

#include <bundle/irls_solver.h>

#include <memory>
#include <vector>

namespace bundle {

class ClassicSigmaReweighting : public IReweightingStrategy {
 public:
  explicit ClassicSigmaReweighting(double sigma_threshold = 3.0)
      : sigma_threshold(sigma_threshold) {}

  GroupWeightResult ComputeGroupWeights(
      const std::vector<double>& group_residuals,
      ErrorGroup& error_group) override;

  double sigma_threshold;
};

}  // namespace bundle
