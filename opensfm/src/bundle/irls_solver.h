#pragma once

#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace bundle {

struct ResidualInfo {
  ceres::ResidualBlockId id;
  ceres::CostFunction* cost_function;  // Original unweighted cost function
  ceres::CostFunction* whitened_cost_function;  // Whitening wrapper (if used)
  ceres::LossFunction* irls_loss;  // Composed loss (original + IRLS weight)
  ceres::LossFunction*
      original_loss;  // Original loss passed by caller (not owned)
  std::string group_id;
  double* weight;  // IRLS weight
};

struct ErrorGroup {
  std::string group_id;

  // All residuals in this group
  std::vector<ResidualInfo> residuals_info;  // Residuals in this group
  std::vector<ceres::ResidualBlockId>
      residual_block_ids;  // For calls to Evaluate

  // Outlier model parameters
  double outlier_density{
      3.14e-6};                 // K parameter (p_out), updated each iteration
  double density_ratio{0.001};  // Ratio of outlier to inlier density peaks
  double temperature{1.0};      // Temperature for annealing of K
  double search_volume{1.0};    // Volume for uniform outlier distribution

  // Whitening/GMM data
  Eigen::MatrixXd last_covariance;  // Last computed covariance matrix
  Eigen::MatrixXd
      covariance_sqrt_inv;  // L^{-1} from Cholesky decomposition for whitening
  bool enable_whitening{false};  // Whether to apply Mahalanobis whitening

  ErrorGroup() : covariance_sqrt_inv(Eigen::Matrix2d::Identity()) {}
};

struct ResidualLocation {
  ceres::ResidualBlockId id;
  std::string group_id;
  size_t index;
};

// Forward declarations
class IRLSSolver;
class IReweightingStrategy;

struct GroupWeightResult {
  std::string group_id;
  double log_likelihood;
  int num_outliers;
  int num_total;
};

struct IRLSSummary : public ceres::Solver::Summary {
  std::vector<std::string> irls_iteration_summaries;
  std::string BriefReport() const;
  std::vector<std::string> IRLSReport() const {
    return irls_iteration_summaries;
  }
};

// Context containing all data needed by reweighting strategies
// This makes strategies testable in isolation
struct IRLSSolverContext {
  std::map<std::string, ErrorGroup>* error_groups;
};

// Abstract interface for reweighting strategies
// Strategies can be tested independently by creating test contexts
class IReweightingStrategy {
 public:
  virtual ~IReweightingStrategy() = default;

  // Compute weights for a single group, returns contribution to log-likelihood
  virtual GroupWeightResult ComputeGroupWeights(
      const std::vector<double>& residuals, ErrorGroup& error_group) = 0;
};

class IRLSSolver {
 public:
  IRLSSolver();
  explicit IRLSSolver(std::unique_ptr<IReweightingStrategy> strategy);
  ~IRLSSolver();

  void AddParameterBlock(double* values, int size);
  void SetParameterBlockConstant(double* values);
  void SetParameterLowerBound(double* values, int index, double lower_bound);
  void SetParameterUpperBound(double* values, int index, double upper_bound);

  ResidualLocation AddResidualBlock(
      ceres::CostFunction* cost_function, ceres::LossFunction* loss_function,
      const std::string& group_id,
      const std::vector<double*>& parameter_blocks);

  // Helper for variable arguments to match Ceres interface style, converting to
  // vector
  template <typename... Ts>
  ResidualLocation AddResidualBlock(ceres::CostFunction* cost_function,
                                    ceres::LossFunction* loss_function,
                                    const std::string& group_id, double* x0,
                                    Ts*... xs) {
    std::vector<double*> parameter_blocks{x0, xs...};
    return AddResidualBlock(cost_function, loss_function, group_id,
                            parameter_blocks);
  }

  void SetSolverOptions(const ceres::Solver::Options& options);

  double GetResidualWeight(const ResidualLocation& location) const;

  // Set or change the reweighting strategy
  void SetReweightingStrategy(std::unique_ptr<IReweightingStrategy> strategy);
  IReweightingStrategy* GetReweightingStrategy() {
    return reweighting_strategy_.get();
  }

  void SetGroupOutlierDensity(const std::string& group_id, double density);
  void SetGroupTemperature(const std::string& group_id, double temperature);
  void SetGroupWhitening(const std::string& group_id, bool enable);
  void SetGroupSearchVolume(const std::string& group_id, double search_volume);
  void SetGroupDensityRatio(const std::string& group_id, double ratio);
  void SetAllGroupsDensityRatio(double ratio);
  void InitializeKFromResiduals(bool enable);

  void Run();

  const IRLSSummary& GetSummary() const;
  ceres::Problem* GetProblem();

  std::vector<GroupWeightResult> ComputeWeights();
  double GetLogLikelihood() const { return current_log_likelihood_; }

  void AddIterationSummary(const std::string& summary);

 private:
  ceres::Problem problem_;
  ceres::Solver::Options solver_options_;
  IRLSSummary last_run_summary_;

  std::unique_ptr<IReweightingStrategy> reweighting_strategy_;
  IRLSSolverContext context_;  // Built once, reused for all weight computations

  int max_rounds_{10};  // Default rounds (used to compute interval if
                        // max_num_iterations is set)

  std::map<std::string, ErrorGroup> error_groups_;
  double current_log_likelihood_{-std::numeric_limits<double>::infinity()};

  bool auto_initialize_k_{false};
};

}  // namespace bundle