#include "../include/metrics.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace metrics {

double mse(const std::vector<double> &y_true,
           const std::vector<double> &y_pred) {
  assert(y_true.size() == y_pred.size());
  const std::size_t N = y_true.size();
  double sum = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    double d = y_pred[i] - y_true[i];
    sum += d * d;
  }
  return sum / static_cast<double>(N);
}

double mae(const std::vector<double> &y_true,
           const std::vector<double> &y_pred) {
  assert(y_true.size() == y_pred.size());
  const std::size_t N = y_true.size();
  double sum = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    sum += std::abs(y_pred[i] - y_true[i]);
  }
  return sum / static_cast<double>(N);
}

double r2(const std::vector<double> &y_true,
          const std::vector<double> &y_pred) {
  assert(y_true.size() == y_pred.size());
  const std::size_t N = y_true.size();

  double mean = 0.0;
  for (std::size_t i = 0; i < N; ++i)
    mean += y_true[i];
  mean /= static_cast<double>(N);

  double ss_res = 0.0; // residual sum of squares
  double ss_tot = 0.0; // total sum of squares
  for (std::size_t i = 0; i < N; ++i) {
    double res = y_true[i] - y_pred[i];
    double tot = y_true[i] - mean;
    ss_res += res * res;
    ss_tot += tot * tot;
  }

  if (ss_tot == 0.0) {
    return (ss_res == 0.0) ? 1.0 : 0.0;
  }
  return 1.0 - ss_res / ss_tot;
}

// Count confusion-matrix cells for the positive class (label 1).
static void confusion(const std::vector<int> &y_true,
                      const std::vector<int> &y_pred, int &tp, int &fp,
                      int &fn) {
  assert(y_true.size() == y_pred.size());
  tp = fp = fn = 0;
  for (std::size_t i = 0; i < y_true.size(); ++i) {
    if (y_pred[i] == 1 && y_true[i] == 1)
      ++tp;
    else if (y_pred[i] == 1 && y_true[i] == 0)
      ++fp;
    else if (y_pred[i] == 0 && y_true[i] == 1)
      ++fn;
  }
}

double accuracy(const std::vector<int> &y_true,
                const std::vector<int> &y_pred) {
  assert(y_true.size() == y_pred.size());
  const std::size_t N = y_true.size();
  int correct = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (y_true[i] == y_pred[i])
      ++correct;
  }
  return static_cast<double>(correct) / static_cast<double>(N);
}

double precision(const std::vector<int> &y_true,
                 const std::vector<int> &y_pred) {
  int tp, fp, fn;
  confusion(y_true, y_pred, tp, fp, fn);
  if (tp + fp == 0)
    return 0.0;
  return static_cast<double>(tp) / (tp + fp);
}

double recall(const std::vector<int> &y_true, const std::vector<int> &y_pred) {
  int tp, fp, fn;
  confusion(y_true, y_pred, tp, fp, fn);
  if (tp + fn == 0)
    return 0.0;
  return static_cast<double>(tp) / (tp + fn);
}

} // namespace metrics
