#include "../include/linear_regression.hpp"

#include <cassert>
#include <cinttypes>
#include <iostream>
#include <vector>

LinearRegression::LinearRegression(std::size_t n_features, double lr)
    : weights_(n_features, 0.0), bias_(0.0), lr_(lr) {}

double LinearRegression::predict(const std::vector<double> &x) const {
  assert(x.size() == weights_.size());
  double z = bias_;
  for (std::size_t j = 0; j < weights_.size(); ++j)
    z += weights_[j] * x[j];

  return z;
}

std::vector<double>
LinearRegression::predict(const std::vector<std::vector<double>> &X) const {
  std::vector<double> out;
  out.reserve(X.size());
  for (const auto &x : X)
    out.push_back(predict(x));

  return out;
}

double LinearRegression::loss(const std::vector<std::vector<double>> &X,
                              const std::vector<double> &Y) const {
  assert(X.size() == Y.size());
  const std::size_t N = X.size();
  double sum = 0.0;

  for (std::size_t i = 0; i < N; ++i) {
    double diff = predict(X[i]) - Y[i];
    sum += diff * diff;
  }

  return sum / static_cast<double>(N);
}

double LinearRegression::train_step(const std::vector<std::vector<double>> &X,
                                    const std::vector<double> &Y) {
  assert(X.size() == Y.size());
  const std::size_t N = X.size();
  const std::size_t F = weights_.size();

  // Single pass: accumulate loss AND gradients together.
  std::vector<double> grad_w(F, 0.0);
  double grad_b = 0.0;
  double sum_sq = 0.0;

  for (std::size_t i = 0; i < N; ++i) {
    assert(X[i].size() == F);
    double diff = predict(X[i]) - Y[i];
    sum_sq += diff * diff;

    for (std::size_t j = 0; j < F; ++j) {
      grad_w[j] += diff * X[i][j];
    }

    grad_b += diff;
  }

  const double scale = 2.0 / static_cast<double>(N);
  for (std::size_t j = 0; j < F; ++j)
    weights_[j] -= lr_ * scale * grad_w[j];

  bias_ -= lr_ * scale * grad_b;

  return sum_sq / static_cast<double>(N);
}

void LinearRegression::fit(const std::vector<std::vector<double>> &X,
                           const std::vector<double> &Y, int epochs,
                           int log_every, double tol) {

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double l = train_step(X, Y);
    if (log_every > 0 && epoch % log_every == 0) {
      std::cout << "epoch: " << epoch << " loss: " << l << '\n';
    }
    if (tol > 0.0 && l < tol) {
      std::cout << "converged at epoch " << epoch << " loss: " << l << '\n';
    }
  }
}
