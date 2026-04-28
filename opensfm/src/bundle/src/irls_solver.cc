#include <bundle/irls_helpers.h>
#include <bundle/irls_solver.h>
#include <bundle/mixture_reweighting.h>

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace bundle {

namespace {

// Wrapper that applies Mahalanobis whitening using Cholesky inverse
class WhiteningCostFunction : public ceres::CostFunction {
 public:
  WhiteningCostFunction(ceres::CostFunction* original,
                        const Eigen::MatrixXd* sqrt_inv_ptr)
      : original_(original), sqrt_inv_ptr_(sqrt_inv_ptr) {
    const auto& parameter_block_sizes = original->parameter_block_sizes();
    mutable_parameter_block_sizes()->insert(
        mutable_parameter_block_sizes()->begin(), parameter_block_sizes.begin(),
        parameter_block_sizes.end());
    set_num_residuals(original->num_residuals());
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    if (!original_->Evaluate(parameters, residuals, jacobians)) {
      return false;
    }

    const int num_res = num_residuals();
    const Eigen::MatrixXd& L_inv = *sqrt_inv_ptr_;

    if (L_inv.rows() != num_res || L_inv.cols() != num_res) {
      return true;
    }

    Eigen::Map<Eigen::VectorXd> r(residuals, num_res);
    r = L_inv * r;

    if (jacobians) {
      const int num_params = parameter_block_sizes().size();
      for (int i = 0; i < num_params; ++i) {
        if (jacobians[i]) {
          int block_size = parameter_block_sizes()[i];
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
              J(jacobians[i], num_res, block_size);
          J = L_inv * J;
        }
      }
    }
    return true;
  }

 private:
  std::unique_ptr<ceres::CostFunction> original_;
  const Eigen::MatrixXd* sqrt_inv_ptr_;
};

// Composed loss: applies the original loss function first, then scales by
// the IRLS weight. When original_loss is nullptr, behaves as pure weighted L2.
// This preserves robust loss behavior for groups too small for IRLS
// reweighting.
class IRLSWeightLoss : public ceres::LossFunction {
 public:
  IRLSWeightLoss(const double* weight_ptr, ceres::LossFunction* original_loss)
      : weight_ptr_(weight_ptr), original_loss_(original_loss) {}

  void Evaluate(double s, double rho[3]) const override {
    const double w = *weight_ptr_;
    if (original_loss_) {
      double orig_rho[3];
      original_loss_->Evaluate(s, orig_rho);
      rho[0] = w * orig_rho[0];
      rho[1] = w * orig_rho[1];
      rho[2] = w * orig_rho[2];
    } else {
      rho[0] = w * s;
      rho[1] = w;
      rho[2] = 0.0;
    }
  }

 private:
  const double* weight_ptr_;
  ceres::LossFunction* original_loss_;  // Not owned
};

class IRLSIterationCallback : public ceres::IterationCallback {
 public:
  IRLSIterationCallback(IRLSSolver* solver, int interval, int max_rounds)
      : solver_(solver), interval_(interval), max_rounds_(max_rounds) {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    const double kFunctionTolerance = 1e-7;
    const double kGradientTolerance = 1e-10;
    const double kParameterTolerance = 1e-8;

    std::string conv_type = "None";
    bool inner_converged = false;
    if (summary.iteration > 0 && summary.step_is_successful) {
      if (std::abs(summary.cost_change) <= kFunctionTolerance * summary.cost) {
        conv_type = "Func";
        inner_converged = true;
      } else if (summary.gradient_max_norm <= kGradientTolerance) {
        conv_type = "Grad";
        inner_converged = true;
      } else {
        double x_norm = 0.0;
        std::vector<double*> parameter_blocks;
        solver_->GetProblem()->GetParameterBlocks(&parameter_blocks);
        for (double* block : parameter_blocks) {
          int block_size = solver_->GetProblem()->ParameterBlockSize(block);
          x_norm += std::accumulate(
              block, block + block_size, 0.0,
              [](double sum, double val) { return sum + val * val; });
        }
        x_norm = std::sqrt(x_norm);

        if (summary.step_norm <=
            kParameterTolerance * (x_norm + kParameterTolerance)) {
          conv_type = "Para";
          inner_converged = true;
        }
      }
    }

    if ((summary.iteration > 0 && (summary.iteration % interval_ == 0) &&
         summary.step_is_successful) ||
        inner_converged) {
      double prev_ll = solver_->GetLogLikelihood();
      const auto group_results = solver_->ComputeWeights();
      double new_ll = solver_->GetLogLikelihood();

      std::stringstream ss;
      ss << "IRLS " << current_round_ << "/" << max_rounds_ << "    "
         << conv_type << "Tol";

      for (const auto& res : group_results) {
        std::string label = res.group_id;
        if (label == "GCP_PROJECTION") {
          label = "GCP";
        } else if (label == "CAMERA_PRIOR") {
          label = "PRIOR";
        }
        ss << "    " << label << " " << res.num_outliers << "/"
           << res.num_total;
      }

      double rel_diff = 0.0;
      if (prev_ll > -std::numeric_limits<double>::infinity()) {
        double abs_diff = std::abs(new_ll - prev_ll);
        rel_diff = abs_diff / (std::abs(prev_ll) + 1e-20);
      }
      ss << "    Delta=" << rel_diff;

      solver_->AddIterationSummary(ss.str());

      current_round_++;
      if (prev_ll > -std::numeric_limits<double>::infinity()) {
        if (rel_diff < 1e-3) {
          return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        }
      }
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  IRLSSolver* solver_;
  int interval_;
  int max_rounds_;
  int current_round_ = 0;
};

ceres::Problem::Options ProblemOptions() {
  ceres::Problem::Options options;
  options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  return options;
}

}  // namespace

std::string IRLSSummary::BriefReport() const {
  std::stringstream ss;
  for (const auto& s : irls_iteration_summaries) {
    ss << s << "\n";
  }
  ss << ceres::Solver::Summary::BriefReport();
  return ss.str();
}

IRLSSolver::IRLSSolver()
    : problem_(ProblemOptions()),
      reweighting_strategy_(std::make_unique<MixtureReweighting>()) {}

IRLSSolver::IRLSSolver(std::unique_ptr<IReweightingStrategy> strategy)
    : problem_(ProblemOptions()), reweighting_strategy_(std::move(strategy)) {}

IRLSSolver::~IRLSSolver() {
  for (auto& [group_id, error_group] : error_groups_) {
    for (auto& res_info : error_group.residuals_info) {
      if (res_info.whitened_cost_function) {
        delete res_info.whitened_cost_function;
      } else {
        delete res_info.cost_function;
      }
      delete res_info.irls_loss;
      delete res_info.weight;
    }
  }
}

void IRLSSolver::AddParameterBlock(double* values, int size) {
  problem_.AddParameterBlock(values, size);
}

void IRLSSolver::SetParameterBlockConstant(double* values) {
  problem_.SetParameterBlockConstant(values);
}

void IRLSSolver::SetParameterLowerBound(double* values, int index,
                                        double lower_bound) {
  problem_.SetParameterLowerBound(values, index, lower_bound);
}

void IRLSSolver::SetParameterUpperBound(double* values, int index,
                                        double upper_bound) {
  problem_.SetParameterUpperBound(values, index, upper_bound);
}

ResidualLocation IRLSSolver::AddResidualBlock(
    ceres::CostFunction* cost_function, ceres::LossFunction* loss_function,
    const std::string& group_id, const std::vector<double*>& parameter_blocks) {
  auto& error_group = error_groups_[group_id];
  error_group.group_id = group_id;

  ResidualInfo res_info;
  res_info.id = nullptr;
  res_info.cost_function = cost_function;
  res_info.whitened_cost_function = nullptr;
  res_info.weight = new double(1.0);
  res_info.irls_loss = nullptr;
  res_info.original_loss = loss_function;

  error_group.residuals_info.push_back(res_info);
  auto& stored_res_info = error_group.residuals_info.back();

  stored_res_info.irls_loss =
      new IRLSWeightLoss(stored_res_info.weight, loss_function);

  ceres::CostFunction* final_cost_function = cost_function;
  if (error_group.enable_whitening) {
    auto* whitened = new WhiteningCostFunction(
        cost_function, &error_group.covariance_sqrt_inv);
    stored_res_info.whitened_cost_function = whitened;
    final_cost_function = whitened;
  }

  const auto id = problem_.AddResidualBlock(
      final_cost_function, stored_res_info.irls_loss, parameter_blocks);
  stored_res_info.id = id;
  error_group.residual_block_ids.push_back(id);
  return {id, group_id, error_group.residuals_info.size() - 1};
}

void IRLSSolver::SetSolverOptions(const ceres::Solver::Options& options) {
  solver_options_ = options;
}

double IRLSSolver::GetResidualWeight(const ResidualLocation& location) const {
  const auto& group = error_groups_.at(location.group_id);
  if (location.index >= group.residuals_info.size()) {
    return 0.0;
  }
  return *group.residuals_info[location.index].weight;
}

void IRLSSolver::SetReweightingStrategy(
    std::unique_ptr<IReweightingStrategy> strategy) {
  reweighting_strategy_ = std::move(strategy);
}

void IRLSSolver::SetGroupOutlierDensity(const std::string& group_id,
                                        double density) {
  error_groups_[group_id].outlier_density = density;
}

void IRLSSolver::SetGroupTemperature(const std::string& group_id,
                                     double temperature) {
  error_groups_[group_id].temperature = temperature;
}

void IRLSSolver::SetGroupWhitening(const std::string& group_id, bool enable) {
  error_groups_[group_id].enable_whitening = enable;
}

void IRLSSolver::SetGroupSearchVolume(const std::string& group_id,
                                      double search_volume) {
  error_groups_[group_id].search_volume = search_volume;
}

void IRLSSolver::SetGroupDensityRatio(const std::string& group_id,
                                      double ratio) {
  error_groups_[group_id].density_ratio = ratio;
}

void IRLSSolver::SetAllGroupsDensityRatio(double ratio) {
  for (auto& [group_id, error_group] : error_groups_) {
    error_group.density_ratio = ratio;
  }
}

void IRLSSolver::InitializeKFromResiduals(bool enable) {
  auto_initialize_k_ = enable;
}

std::vector<GroupWeightResult> IRLSSolver::ComputeWeights() {
  std::vector<GroupWeightResult> results;
  if (!reweighting_strategy_) {
    return results;
  }

  double total_log_likelihood = 0.0;
  bool computed_any = false;

  for (auto& group_id_error_group : error_groups_) {
    auto& group_id = group_id_error_group.first;
    auto& error_group = group_id_error_group.second;
    if (error_group.residual_block_ids.empty()) {
      continue;
    }

    const size_t num_blocks = error_group.residual_block_ids.size();
    std::vector<int> block_sizes(num_blocks);
    std::vector<int> block_offsets(num_blocks);
    size_t total_residuals = 0;

    // Pre-compute sizes and offsets
    for (size_t i = 0; i < num_blocks; ++i) {
      const ceres::ResidualBlockId block_id = error_group.residual_block_ids[i];
      const ceres::CostFunction* cost_function =
          problem_.GetCostFunctionForResidualBlock(block_id);
      int sz = cost_function->num_residuals();
      block_sizes[i] = sz;
      block_offsets[i] = total_residuals;
      total_residuals += sz;
    }

    std::vector<double> group_residuals_values(total_residuals);

    // Parallel residual evaluation
#pragma omp parallel
    {
      constexpr int kMinBlocksForParallel = 10;
      std::vector<double*> parameter_blocks;
      parameter_blocks.reserve(kMinBlocksForParallel);
#pragma omp for
      for (int i = 0; i < static_cast<int>(num_blocks); ++i) {
        const ceres::ResidualBlockId block_id =
            error_group.residual_block_ids[i];
        const ceres::CostFunction* cost_function =
            problem_.GetCostFunctionForResidualBlock(block_id);

        parameter_blocks.clear();
        problem_.GetParameterBlocksForResidualBlock(block_id,
                                                    &parameter_blocks);

        double* residual_ptr = &group_residuals_values[block_offsets[i]];
        cost_function->Evaluate(parameter_blocks.data(), residual_ptr, nullptr);
      }
    }

    // Update weights for this group
    auto res = reweighting_strategy_->ComputeGroupWeights(
        group_residuals_values, error_group);
    results.push_back(res);
    total_log_likelihood += res.log_likelihood;
    computed_any = true;

    // Update density with temperature
    error_group.outlier_density *= error_group.temperature;
  }

  if (computed_any) {
    current_log_likelihood_ = total_log_likelihood;
  }
  return results;
}

void IRLSSolver::Run() {
  context_.error_groups = &error_groups_;

  if (auto_initialize_k_) {
    std::vector<double> group_residuals_values;
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.apply_loss_function = false;

    for (auto& [group_id, error_group] : error_groups_) {
      if (error_group.residual_block_ids.empty()) {
        continue;
      }

      // Evaluate only this group's residuals
      std::cout << "Auto-initializing K for group " << group_id << "..."
                << std::endl;
      eval_options.residual_blocks = error_group.residual_block_ids;
      problem_.Evaluate(eval_options, nullptr, &group_residuals_values, nullptr,
                        nullptr);
      if (error_group.residuals_info.empty()) {
        continue;
      }

      int residual_dim =
          error_group.residuals_info[0].cost_function->num_residuals();

      std::vector<double> squared_norms = ComputeSquaredNorms(
          group_residuals_values, error_group, residual_dim);

      double search_volume =
          error_group.enable_whitening ? 1.0 : error_group.search_volume;
      std::cout << "Search volume: " << search_volume << std::endl;
      double estimated_K = ComputeKForGroup(squared_norms, search_volume);
      error_group.outlier_density = estimated_K;
      std::cout << "Auto-init K for group " << group_id << ": " << estimated_K
                << std::endl;
    }
  }

  int max_iterations = solver_options_.max_num_iterations;
  int interval = max_iterations / max_rounds_;
  if (interval < 1) {
    interval = 1;
  }

  IRLSIterationCallback callback(this, interval, max_rounds_);

  ceres::Solver::Options run_options = solver_options_;
  run_options.callbacks.push_back(&callback);
  run_options.update_state_every_iteration = true;
  run_options.use_nonmonotonic_steps = true;

  // Disable Ceres internal convergence checks to let IRLSIterationCallback
  // drive the process
  run_options.function_tolerance = 0.0;
  run_options.gradient_tolerance = 0.0;
  run_options.parameter_tolerance = 0.0;

  // Cap trust region to prevent lambda = 1/trust_region from dropping below
  // the minimum IRLS weight (~1e-10). Without this, the zero tolerances let
  // the trust region grow unboundedly, making J^T W J + lambda*I singular.
  run_options.max_trust_region_radius = 1e8;

  ceres::Solve(run_options, &problem_, &last_run_summary_);
}

const IRLSSummary& IRLSSolver::GetSummary() const { return last_run_summary_; }

void IRLSSolver::AddIterationSummary(const std::string& summary) {
  last_run_summary_.irls_iteration_summaries.push_back(summary);
}

ceres::Problem* IRLSSolver::GetProblem() { return &problem_; }

}  // namespace bundle