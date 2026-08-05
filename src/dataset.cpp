#include "../include/dataset.hpp"

#include <cassert>
#include <cmath>
#include <vector>

void StandardScaler::fit(const std::vector<std::vector<double>> &X) {
  assert(!X.empty());
  const std::size_t N = X.size();
  const std::size_t F = X[0].size();

  mean_.assign(F, 0.0);
  std_.assign(F, 0.0);

  for (std::size_t i = 0; i < N; ++i) {
    assert(X[i].size() == F);
    for (std::size_t j = 0; j < F; ++j) {
      mean_[j] += X[i][j];
    }
  }

  for (std::size_t j = 0; j < F; ++j) {
    mean_[j] /= static_cast<double>(N);
  }

  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < F; ++j) {
      double d = X[i][j] - mean_[j];
      std_[j] += d * d;
    }
  }

  for (std::size_t j = 0; j < F; ++j) {
    std_[j] = std::sqrt(std_[j] / static_cast<double>(N));

    // Guard against zero-variance columns tp avoid div by 0.
    if (std_[j] == 0.0) {
      std_[j] = 1.0;
    }
  }

  fitted_ = true;
}

std::vector<std::vector<double>>
StandardScaler::transform(const std::vector<std::vector<double>> &X) const {
  assert(fitted_);
  const std::size_t N = X.size();
  const std::size_t F = mean_.size();

  std::vector<std::vector<double>> out(N, std::vector<double>(F, 0.0));
  for (std::size_t i = 0; i < N; ++i) {
    assert(X[i].size() == F);
    for (std::size_t j = 0; j < F; ++j) {
      out[i][j] = (X[i][j] - mean_[j]) / std_[j];
    }
  }

  return out;
}

std::vector<std::vector<double>>
StandardScaler::fit_transform(const std::vector<std::vector<double>> &X) {
  fit(X);
  return transform(X);
}
