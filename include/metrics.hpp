#ifndef METRICS_HPP
#define METRICS_HPP

#include <vector>

namespace metrics {

// ---Regression---

double mse(const std::vector<double> &y_true,
           const std::vector<double> &y_pred);

double mae(const std::vector<double> &y_true,
           const std::vector<double> &y_pred);

// Coefficeint of determination.

double r2(const std::vector<double> &y_true, const std::vector<double> &y_pred);

// ---Classification (labels are 0 or 1)---

double accuracy(const std::vector<int> &y_true, const std::vector<int> &y_pred);

double precision(const std::vector<int> &y_true,
                 const std::vector<int> &y_pred);

double recall(const std::vector<int> &y_true, const std::vector<int> &y_pred);

} // namespace metrics

#endif
