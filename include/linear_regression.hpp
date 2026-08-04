#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <cstddef>
#include <vector>

// Multi-feature linear regression trained by batch gradient descent on MSE.
//
// Model:      y_hat = dot(w, x) + b
// Loss:       L = (1/N) * sum_i (y_hat_i - y_i)^2
// Gradients:  dL/dw_j = (2/N) * sum_i (y_hat_i - y_i) * x_ij
//             dL/db   = (2/N) * sum_i (y_hat_i - y_i)

class LinearRegression {
public:
  LinearRegression(std::size_t n_features, double lr = 0.01);

  // Single sample: x.size() must equal n_features.
  double predict(const std::vector<double> &x) const;

  // Batch: one prediction per row of X.
  std::vector<double> predict(const std::vector<std::vector<double>> &X) const;

  double loss(const std::vector<std::vector<double>> &X,
              const std::vector<double> &Y) const;

  // One full-batch gradient descent step.
  // Returns the pre-update loss (for logging / early stopping).
  double train_step(const std::vector<std::vector<double>> &X,
                    const std::vector<double> &Y);

  // Loops train_step. log_every=0 to disable logging.
  // tol=0 to disable early stopping.
  void fit(const std::vector<std::vector<double>> &X,
           const std::vector<double> &Y, int epochs, int log_every = 100,
           double tol = 0.0);

  const std::vector<double> &weights() const { return weights_; }
  double bias() const { return bias_; }

private:
  std::vector<double> weights_;
  double bias_;
  double lr_;
};

#endif
