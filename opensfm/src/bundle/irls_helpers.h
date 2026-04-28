#pragma once

#include <ceres/ceres.h>
#include <vector>
#include <bundle/irls_solver.h>

namespace bundle {

// Forward declaration
struct ErrorGroup;

// Compute squared norms of residuals for a group, filtering by dimension
// Expected to be used after EvaluateResiduals
std::vector<double> ComputeSquaredNorms(
    const std::vector<double>& all_residuals,
    const ErrorGroup& error_group,
    int expected_residual_dim);

// Estimate outlier ratio and scales for a group of squared residuals
double ComputeKForGroup(const std::vector<double>& d2_residuals,
                        double search_volume);

// Returns true if covariance was successfully computed (sum of weights > epsilon)
bool ComputeWhitening(const std::vector<double>& all_residuals,
                      ErrorGroup& error_group);

}  // namespace bundle