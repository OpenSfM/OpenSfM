#include <bundle/irls_helpers.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

namespace bundle {

std::vector<double> ComputeSquaredNorms(
    const std::vector<double>& all_residuals, const ErrorGroup& error_group,
    int expected_residual_dim) {
  std::vector<double> squared_norms;
  squared_norms.reserve(error_group.residuals_info.size());

  int offset = 0;
  for (const auto& res : error_group.residuals_info) {
    int dim = res.cost_function->num_residuals();

    if (dim != expected_residual_dim) {
      offset += dim;
      continue;
    }

    double sq_norm = 0;
    for (int d = 0; d < dim; ++d) {
      double val = all_residuals[offset + d];
      sq_norm += val * val;
    }
    squared_norms.push_back(sq_norm);
    offset += dim;
  }
  return squared_norms;
}

double ComputeKForGroup(const std::vector<double>& d2_residuals,
                        double search_volume) {
  const int N = d2_residuals.size();
  if (N < 10) {
    return 1e-3;  // Fallback for small data
  }

  // 1. Sort residuals to see the cumulative distribution
  std::vector<double> sorted_d2 = d2_residuals;
  std::sort(sorted_d2.begin(), sorted_d2.end());

  double best_K = 1e-3;
  double max_likelihood = -std::numeric_limits<double>::max();

  // 2. Sweep potential split points (m)
  // We assume at least 40% are inliers and at most 98% are inliers
  for (int m = static_cast<int>(N * 0.4); m < static_cast<int>(N * 0.98); ++m) {
    // Estimate sigma^2 from the first m (inlier) candidates
    double sum_d2 = 0;
    for (int i = 0; i < m; ++i) {
      sum_d2 += sorted_d2[i];
    }
    double sigma2 = sum_d2 / (2.0 * m);

    // Calculate Inlier Ratio
    double gamma = static_cast<double>(m) / N;

    // 3. Compute Log-Likelihood of this specific GMM split
    // L = sum( log( inlier_part + outlier_part ) )
    double current_L = 0;
    double uniform_density = (1.0 - gamma) / search_volume;

    for (int i = 0; i < N; ++i) {
      double inlier_density = (gamma / (2.0 * M_PI * sigma2)) *
                              std::exp(-0.5 * sorted_d2[i] / sigma2);
      current_L += std::log(inlier_density + uniform_density + 1e-12);
    }

    if (current_L > max_likelihood) {
      max_likelihood = current_L;
      // K is the ratio of outlier-to-inlier density peaks
      // K = (Density_outlier) / (Density_inlier_peak)
      best_K = (uniform_density) / (gamma / (2.0 * M_PI * sigma2));
    }
  }

  return best_K;
}

bool ComputeWhitening(const std::vector<double>& all_residuals,
                      ErrorGroup& error_group) {
  if (error_group.residuals_info.empty()) {
    return false;
  }

  int residual_dim =
      error_group.residuals_info[0].cost_function->num_residuals();

  // Stats accumulation
  Eigen::MatrixXd sum_weighted_outer_product =
      Eigen::MatrixXd::Zero(residual_dim, residual_dim);
  double sum_weights = 0;

  int offset = 0;
  for (const auto& res : error_group.residuals_info) {
    int dim = res.cost_function->num_residuals();
    if (dim != residual_dim) {
      offset += dim;
      continue;
    }

    double w = *res.weight;  // Current weight
    Eigen::Map<const Eigen::VectorXd> current_residual(&all_residuals[offset],
                                                       dim);
    sum_weighted_outer_product +=
        w * current_residual * current_residual.transpose();
    sum_weights += w;

    offset += dim;
  }

  if (sum_weights < 1e-9) {
    return false;
  }

  Eigen::MatrixXd covariance = sum_weighted_outer_product / sum_weights;
  error_group.last_covariance = covariance;

  // Compute Cholesky decomposition for whitening if enabled
  if (error_group.enable_whitening) {
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if (llt.info() == Eigen::Success) {
      Eigen::MatrixXd L = llt.matrixL();
      error_group.covariance_sqrt_inv = L.inverse();
    } else {
      Eigen::MatrixXd regularized =
          covariance +
          Eigen::MatrixXd::Identity(residual_dim, residual_dim) * 1e-6;
      Eigen::LLT<Eigen::MatrixXd> llt_reg(regularized);
      if (llt_reg.info() == Eigen::Success) {
        Eigen::MatrixXd L = llt_reg.matrixL();
        error_group.covariance_sqrt_inv = L.inverse();
      }
    }
  }
  return true;
}
}  // namespace bundle