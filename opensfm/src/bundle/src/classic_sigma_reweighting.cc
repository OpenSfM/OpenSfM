#include <bundle/classic_sigma_reweighting.h>
#include <bundle/irls_helpers.h>

#include <cmath>
#include <vector>

namespace bundle {

GroupWeightResult ClassicSigmaReweighting::ComputeGroupWeights(
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

  // Compute mean and standard deviation of squared norms
  double mean = std::accumulate(
      squared_norms.begin(), squared_norms.end(), 0.0,
      [](double acc, double val) { return acc + std::sqrt(val); });
  mean /= squared_norms.size();

  double variance = std::accumulate(squared_norms.begin(), squared_norms.end(),
                                    0.0, [mean](double acc, double val) {
                                      return acc + (val - mean) * (val - mean);
                                    });
  variance /= squared_norms.size();

  double sigma = std::sqrt(variance);
  double threshold = sigma_threshold * sigma;

  // For Likelihood, use RMS as sigma (std dev of residuals)
  double ll_sigma = std::max(1e-5, std::sqrt(mean));
  double inv_ll_sigma2 = 1.0 / (ll_sigma * ll_sigma);
  double log_norm_const = -std::log(std::sqrt(2 * M_PI) * ll_sigma);
  double log_outlier_prob = std::log(error_group.outlier_density);
  double log_likelihood = 0.0;

  // Assign weights: 0 if residual > threshold, 1 otherwise
  int outliers = 0;
  size_t sq_norm_idx = 0;
  for (size_t i = 0; i < error_group.residuals_info.size(); ++i) {
    auto& res_info = error_group.residuals_info[i];

    int dim = res_info.cost_function->num_residuals();
    if (dim != residual_dim) {
      continue;
    }

    double deviation = std::abs(std::sqrt(squared_norms[sq_norm_idx]) - mean);
    if (deviation > threshold) {
      *res_info.weight = 0.0;
      log_likelihood += log_outlier_prob;
      outliers++;
    } else {
      *res_info.weight = 1.0;
      log_likelihood +=
          log_norm_const - 0.5 * deviation * deviation * inv_ll_sigma2;
    }
  }

  ComputeWhitening(group_residuals, error_group);
  result.log_likelihood = log_likelihood;
  result.num_outliers = outliers;
  return result;
}

}  // namespace bundle