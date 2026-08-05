#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <cstddef>
#include <vector>

// Binary logistic regression trained by batch gradient descent on
// binary cross-entropy loss.
//
// Model:   p = sigmoid(dot(w, x) + b)          in (0, 1), a probability
// Label:   1 if p >= 0.5 else 0
// Loss:    L = -(1/N) sum_i [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ]
// Grads:   dL/dw_j = (1/N) sum_i (p_i - y_i) * x_ij
//          dL/db   = (1/N) sum_i (p_i - y_i)

class LogisticRegression {

public:
  LogisticRegression(std::size_t n_features, double lr = 0.1);

  // Predicted PROBABILITY for a single sample: sigmoid(w * x + b).
  double predict(const std::vector<double> &x) const;

  // Predicted probabilities for a batch.
  std::vector<double> predict(const std::vector<std::vector<double>> &X) const;

  // Hard 0/1 label for a single sample (threshold at 0.5).
  int predict_label(const std::vector<double> &x, double threshold = 0.5) const;

  // Hard 0/1 labels for a batch.
  std::vector<int> predict_label(const std::vector<std::vector<double>> &X,
                                 double threshold = 0.5) const;

  // Binary cross-entropy over a dataset. Y entries must be 0.0 or 1.0.
  double loss(const std::vector<std::vector<double>> &X,
              const std::vector<double> &Y) const;

  // One full-batch gradient descent step. Returns pre-update loss.
  double train_step(const std::vector<std::vector<double>> &X,
                    const std::vector<double> &Y);

  // Loops train_step. log_every=0 disables logging.
  // tol=0 disables early stopping (stops when loss change < tol).
  void fit(const std::vector<std::vector<double>> &X,
           const std::vector<double> &Y, int epochs, int log_every = 100,
           double tol = 0.0);

  const std::vector<double> &weights() const { return weights_; }
  double bias() const { return bias_; }

private:
  std::vector<double> weights_;
  double bias_;
  double lr_;

  // Raw linear score w·x + b, before the sigmoid.
  double linear(const std::vector<double> &x) const;
};

#endif
