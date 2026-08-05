#include "../include/logistic_regression.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

// Numerically stable sigmoid.
// exp(-z) overflows to +inf for very negative z, which would make
// 1/(1+inf) = 0 — technically fine, but exp() of a huge number can also
// raise range errors. Branching on the sign keeps every exp() argument <= 0,
// where exp is well-behaved (between 0 and 1).
static double sigmoid(double z) {
  if (z >= 0.0) {
    return 1.0 / (1.0 + std::exp(-z));
  } else {
    double ez = std::exp(z); // z < 0, so ez in (0, 1)
    return ez / (1.0 + ez);
  }
}

LogisticRegression::LogisticRegression(std::size_t n_features, double lr)
    : weights_(n_features, 0.0), bias_(0.0), lr_(lr) {}

double LogisticRegression::linear(const std::vector<double> &x) const {
  assert(x.size() == weights_.size());
  double z = bias_;
  for (std::size_t j = 0; j < weights_.size(); ++j) {
    z += weights_[j] * x[j];
  }
  return z;
}

double LogisticRegression::predict(const std::vector<double> &x) const {
  return sigmoid(linear(x));
}

std::vector<double>
LogisticRegression::predict(const std::vector<std::vector<double>> &X) const {
  std::vector<double> out;
  out.reserve(X.size());
  for (const auto &x : X) {
    out.push_back(predict(x));
  }
  return out;
}

int LogisticRegression::predict_label(const std::vector<double> &x,
                                      double threshold) const {
  return predict(x) >= threshold ? 1 : 0;
}

std::vector<int>
LogisticRegression::predict_label(const std::vector<std::vector<double>> &X,
                                  double threshold) const {
  std::vector<int> out;
  out.reserve(X.size());
  for (const auto &x : X) {
    out.push_back(predict_label(x, threshold));
  }
  return out;
}

double LogisticRegression::loss(const std::vector<std::vector<double>> &X,
                                const std::vector<double> &Y) const {
  assert(X.size() == Y.size());
  const std::size_t N = X.size();

  // Clamp p away from 0 and 1: log(0) = -inf would poison the sum.
  const double eps = 1e-15;

  double sum = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    double p = predict(X[i]);
    p = std::min(std::max(p, eps), 1.0 - eps);
    sum += Y[i] * std::log(p) + (1.0 - Y[i]) * std::log(1.0 - p);
  }
  return -sum / static_cast<double>(N);
}

double LogisticRegression::train_step(const std::vector<std::vector<double>> &X,
                                      const std::vector<double> &Y) {
  assert(X.size() == Y.size());
  const std::size_t N = X.size();
  const std::size_t F = weights_.size();

  std::vector<double> grad_w(F, 0.0);
  double grad_b = 0.0;

  const double eps = 1e-15;
  double loss_sum = 0.0;

  // Single pass: accumulate gradients AND loss together.
  for (std::size_t i = 0; i < N; ++i) {
    assert(X[i].size() == F);
    double p = predict(X[i]); // sigmoid(w·x + b)
    double diff = p - Y[i];   // residual (same role as linear!)

    for (std::size_t j = 0; j < F; ++j) {
      grad_w[j] += diff * X[i][j];
    }
    grad_b += diff;

    double pc = std::min(std::max(p, eps), 1.0 - eps);
    loss_sum += Y[i] * std::log(pc) + (1.0 - Y[i]) * std::log(1.0 - pc);
  }

  // Note: NO factor of 2 here (unlike linear regression's MSE gradient).
  const double scale = 1.0 / static_cast<double>(N);
  for (std::size_t j = 0; j < F; ++j) {
    weights_[j] -= lr_ * scale * grad_w[j];
  }
  bias_ -= lr_ * scale * grad_b;

  return -loss_sum / static_cast<double>(N); // pre-update cross-entropy
}

void LogisticRegression::fit(const std::vector<std::vector<double>> &X,
                             const std::vector<double> &Y, int epochs,
                             int log_every, double tol) {
  double prev = std::numeric_limits<double>::infinity();
  for (int epoch = 0; epoch < epochs; ++epoch) {
    double l = train_step(X, Y);
    if (log_every > 0 && epoch % log_every == 0) {
      std::cout << "epoch: " << epoch << " loss: " << l << '\n';
    }
    if (tol > 0.0 && std::abs(prev - l) < tol) {
      std::cout << "converged at epoch " << epoch << " loss: " << l << '\n';
      return;
    }
    prev = l;
  }
}
