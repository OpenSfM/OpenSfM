#include <bundle/irls_helpers.h>
#include <bundle/mad_sigma_reweighting.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace bundle {

GroupWeightResult MADSigmaReweighting::ComputeGroupWeights(
    const std::vector<double>& group_residuals, ErrorGroup& error_group) {
  GroupWeightResult result;
  result.group_id = error_group.group_id;
  result.log_likelihood = 0.0;
  result.num_outliers = 0;
  result.num_total = error_group.residuals_info.size();

  if (error_group.residuals_info.empty()) {
    return result;
  }
  int residual_dim =
      error_group.residuals_info[0].cost_function->num_residuals();

  // Evaluate residuals and compute norms
  std::vector<double> squared_norms =
      ComputeSquaredNorms(group_residuals, error_group, residual_dim);
  if (squared_norms.empty()) {
    return result;
  }

  std::vector<double> norms;
  norms.reserve(squared_norms.size());
  for (double sq : squared_norms) {
    norms.push_back(std::sqrt(sq));
  }

  std::nth_element(norms.begin(), norms.begin() + norms.size() / 2,
                   norms.end());
  double median = norms[norms.size() / 2];

  // Compute absolute deviations from median
  std::vector<double> abs_deviations;
  abs_deviations.reserve(norms.size());
  for (double norm : norms) {
    abs_deviations.push_back(std::abs(norm - median));
  }

  // Find MAD (median of absolute deviations)
  std::nth_element(abs_deviations.begin(),
                   abs_deviations.begin() + abs_deviations.size() / 2,
                   abs_deviations.end());
  double mad = abs_deviations[abs_deviations.size() / 2];

  // Convert MAD to sigma estimate (for normal distribution: sigma ≈ 1.4826 *
  // MAD)
  double sigma = 1.4826 * mad;
  double threshold = sigma_threshold * sigma;

  // Prevent division by zero validation
  double ll_sigma = std::max(1e-5, sigma);
  double inv_sigma2 = 1.0 / (ll_sigma * ll_sigma);
  double log_norm_const = -std::log(std::sqrt(2 * M_PI) * ll_sigma);
  double log_outlier_prob = std::log(error_group.outlier_density);
  double log_likelihood = 0.0;

  // Assign weights: 0 if norm deviation > threshold, 1 otherwise
  int outliers = 0;
  size_t sq_norm_idx = 0;
  for (size_t i = 0; i < error_group.residuals_info.size(); ++i) {
    auto& res_info = error_group.residuals_info[i];
    int dim = res_info.cost_function->num_residuals();
    if (dim != residual_dim) {
      continue;
    }

    double norm = norms[sq_norm_idx++];
    double deviation = std::abs(norm - median);

    if (deviation > threshold) {
      *res_info.weight = 0.0;
      log_likelihood += log_outlier_prob;
      outliers++;
    } else {
      *res_info.weight = 1.0;
      // Inlier probability
      log_likelihood +=
          log_norm_const - 0.5 * deviation * deviation * inv_sigma2;
    }
  }

  ComputeWhitening(group_residuals, error_group);
  result.log_likelihood = log_likelihood;
  result.num_outliers = outliers;
  return result;
}

}  // namespace bundle