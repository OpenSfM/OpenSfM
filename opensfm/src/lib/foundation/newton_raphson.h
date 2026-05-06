#pragma once

#include <Eigen/Eigen>
#include <limits>

namespace foundation {

// Unfortunately we need these traits because we want to use
// straight double fo the scalar (N=1, M=1) case. Otherwise,
// wrapping in an Eigen object is killing the performance
template <int N, int M>
struct TypeTraits {
  using Jacobian = Eigen::Matrix<double, N, M>;
  using Values = Eigen::Matrix<double, N, 1>;
  static double Norm(const Values& x) { return x.norm(); }
};

template <>
struct TypeTraits<1, 1> {
  using Jacobian = double;
  using Values = double;
  static double Norm(const Values& x) { return std::fabs(x); }
};

// Class for computing the jacobian using finite differencing.
// Very slow and prone to non-convergence.
template <class F, int N, int M>
struct FiniteDiff {
  static typename TypeTraits<N, M>::Jacobian Derivative(
      const F& func, typename TypeTraits<N, M>::Values& x) {
    typename TypeTraits<N, M>::Jacobian jacobian;
    typename TypeTraits<N, M>::Values x_plus = x;
    constexpr auto eps = 1e-15;
    for (int i = 0; i < M; ++i) {
      x_plus[i] += eps;
      jacobian.col(i) = (func(x_plus) - func(x)) / eps;
      x_plus[i] -= eps;
    }
    return jacobian;
  }
};

template <class F>
struct FiniteDiff<F, 1, 1> {
  static typename TypeTraits<1, 1>::Jacobian Derivative(
      const F& func, typename TypeTraits<1, 1>::Values& x) {
    constexpr auto eps = 1e-15;
    return (func(x + eps) - func(x)) / eps;
  }
};

// Class when the client know how to compute the jacobian.
template <class F, int N, int M>
struct ManualDiff {
  static typename TypeTraits<N, M>::Jacobian Derivative(
      const F& func, const typename TypeTraits<N, M>::Values& x) {
    return func.derivative(x);
  }
};

// Again, we need to specialize that one for the scalar case.
template <int N, int M>
typename TypeTraits<N, M>::Values SolveDecr(
    const typename TypeTraits<N, M>::Jacobian& d,
    const typename TypeTraits<N, M>::Values& f) {
  return (d.transpose() * d).inverse() * d.transpose() * f;
}

template <>
typename TypeTraits<1, 1>::Values SolveDecr<1, 1>(
    const typename TypeTraits<1, 1>::Jacobian& d,
    const typename TypeTraits<1, 1>::Values& f);

template <class F, int N, int M, class D = FiniteDiff<F, N, M>>
typename TypeTraits<N, M>::Values NewtonRaphson(
    const F& func, const typename TypeTraits<N, M>::Values& initial_value,
    int iterations, double tol = 1e-6) {
  using Values = typename TypeTraits<N, M>::Values;
  Values current_value = initial_value;

  for (int i = 0; i < iterations; ++i) {
    const Values at_current_value = func(current_value);
    const double current_res_norm = TypeTraits<N, M>::Norm(at_current_value);
    if (current_res_norm < tol) {
      break;
    }

    const auto derivative = D::Derivative(func, current_value);
    const auto current_decr = SolveDecr<N, M>(derivative, at_current_value);
    const double current_decr_norm = TypeTraits<N, M>::Norm(current_decr);
    if (current_decr_norm < tol || !std::isfinite(current_decr_norm)) {
      break;
    }

    // Backtracking line search: try full step, then halve until
    // the residual decreases and stays finite.
    bool accepted = false;
    double lambda = 1.0;
    constexpr int kMaxBacktracks = 6;
    for (int j = 0; j < kMaxBacktracks; ++j) {
      Values trial = current_value - lambda * current_decr;
      const double trial_norm = TypeTraits<N, M>::Norm(func(trial));
      if (std::isfinite(trial_norm) && trial_norm < current_res_norm) {
        current_value = trial;
        accepted = true;
        break;
      }
      lambda *= 0.5;
    }
    if (!accepted) {
      break;  // Cannot reduce residual; return best estimate
    }
  }
  return current_value;
}
}  // namespace foundation
