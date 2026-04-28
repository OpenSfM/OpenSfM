#include <bundle/irls_helpers.h>
#include <bundle/mixture_reweighting.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

namespace bundle {

GroupWeightResult MixtureReweighting::ComputeGroupWeights(
    const std::vector<double>& group_residuals, ErrorGroup& error_group) {
  GroupWeightResult result;
  result.group_id = error_group.group_id;
  result.log_likelihood = 0.0;
  result.num_outliers = 0;
  result.num_total = error_group.residuals_info.size();

  if (error_group.residuals_info.empty()) {
    return result;
  }

  if (error_group.residuals_info.size() < kMinSampleSize) {
    return result;
  }

  int residual_dim =
      error_group.residuals_info[0].cost_function->num_residuals();

  int outliers_end = 0;
  double K_end = 0.0;
  double log_likelihood = 0.0;
  double prev_log_likelihood = std::numeric_limits<double>::max();
  for (int iter = 0; iter < kMaxInnerIterations; ++iter) {
    if (!ComputeWhitening(group_residuals, error_group)) {
      return result;
    }
    Eigen::MatrixXd covariance = error_group.last_covariance;
    covariance +=
        kMinVariance * Eigen::MatrixXd::Identity(residual_dim, residual_dim);

    // Regularization
    double det = covariance.determinant();
    if (det < kMinDeterminant) {
      det = kMinVariance * kMinVariance;
      covariance =
          kMinVariance * Eigen::MatrixXd::Identity(residual_dim, residual_dim);
    }
    Eigen::MatrixXd inv_cov = covariance.inverse();

    // Compute normalization factor for Gaussian PDF
    // normalizer = (2*pi)^k * det(Sigma)
    double normalizer = std::pow(2 * M_PI, residual_dim) * det;
    double sqrt_normalizer = std::sqrt(normalizer);

    // Estimate K (Uniform outlier density)
    // If density_ratio = p_out / p_in_max, and p_in_max = 1 / sqrt_normalizer
    // Then K = density_ratio / sqrt_normalizer.
    double K = error_group.density_ratio / sqrt_normalizer;

    log_likelihood = 0.0;

    int outliers = 0;
    int offset = 0;
    for (size_t i = 0; i < error_group.residuals_info.size(); ++i) {
      auto& res_info = error_group.residuals_info[i];

      int dim = res_info.cost_function->num_residuals();

      if (dim != residual_dim) {
        offset += dim;
        continue;
      }

      Eigen::Map<const Eigen::VectorXd> current_residual(
          &group_residuals[offset], dim);
      double mahalanobis_sq =
          current_residual.transpose() * inv_cov * current_residual;
      double p_in = std::exp(-0.5 * mahalanobis_sq) / sqrt_normalizer;

      log_likelihood += std::log(p_in + K);

      double new_weight = p_in / (p_in + K);
      new_weight = std::max(kMinWeight, std::min(new_weight, 1.0));

      *res_info.weight = new_weight;
      if (new_weight < 0.5) {
        outliers++;
      }

      offset += dim;
    }

    outliers_end = outliers;
    K_end = K;

    // Stop if log-likelihood didn't change significantly
    double ll_change = std::abs(log_likelihood - prev_log_likelihood) /
                       std::abs(prev_log_likelihood);
    if (ll_change < kConvergenceThreshold) {
      break;
    }
    prev_log_likelihood = log_likelihood;
  }

  result.log_likelihood = log_likelihood;
  result.num_outliers = outliers_end;

  return result;
}

}  // namespace bundle