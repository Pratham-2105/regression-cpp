#include "../include/metrics.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Tiny test helper: check that two doubles are close, print PASS/FAIL.
static int g_failures = 0;

static void check_close(const std::string &name, double got, double expected,
                        double tol = 1e-9) {
  bool ok = std::abs(got - expected) < tol;
  std::cout << (ok ? "PASS" : "FAIL") << "  " << name << ": got " << got
            << ", expected " << expected << '\n';
  if (!ok)
    ++g_failures;
}

int main() {
  // ---- Regression metrics ----
  {
    std::vector<double> y_true = {3.0, -0.5, 2.0, 7.0};
    std::vector<double> y_pred = {2.5, 0.0, 2.0, 8.0};

    // MSE = mean of [0.25, 0.25, 0, 1] = 1.5 / 4 = 0.375
    check_close("mse", metrics::mse(y_true, y_pred), 0.375);

    // MAE = mean of [0.5, 0.5, 0, 1] = 2.0 / 4 = 0.5
    check_close("mae", metrics::mae(y_true, y_pred), 0.5);

    // R^2 = 1 - 1.5 / 29.1875
    check_close("r2", metrics::r2(y_true, y_pred), 1.0 - 1.5 / 29.1875);
  }

  // Perfect predictions -> R^2 == 1, errors == 0.
  {
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};
    check_close("mse perfect", metrics::mse(y, y), 0.0);
    check_close("mae perfect", metrics::mae(y, y), 0.0);
    check_close("r2 perfect", metrics::r2(y, y), 1.0);
  }

  // Predicting the mean everywhere -> R^2 == 0.
  {
    std::vector<double> y_true = {1.0, 2.0, 3.0, 4.0}; // mean = 2.5
    std::vector<double> y_pred = {2.5, 2.5, 2.5, 2.5};
    check_close("r2 mean-only", metrics::r2(y_true, y_pred), 0.0);
  }

  // ---- Classification metrics ----
  //   TP = 3, FP = 2, FN = 1, TN = 2
  {
    std::vector<int> y_true = {1, 0, 1, 1, 0, 1, 0, 0};
    std::vector<int> y_pred = {1, 0, 0, 1, 1, 1, 0, 1};

    check_close("accuracy", metrics::accuracy(y_true, y_pred), 5.0 / 8.0);
    check_close("precision", metrics::precision(y_true, y_pred), 3.0 / 5.0);
    check_close("recall", metrics::recall(y_true, y_pred), 3.0 / 4.0);
  }

  // Edge case: model predicts no positives at all.
  {
    std::vector<int> y_true = {1, 0, 1, 0};
    std::vector<int> y_pred = {0, 0, 0, 0};
    check_close("precision no-positives", metrics::precision(y_true, y_pred),
                0.0);
    check_close("recall no-positives", metrics::recall(y_true, y_pred), 0.0);
    check_close("accuracy no-positives", metrics::accuracy(y_true, y_pred),
                0.5);
  }

  // Perfect classifier.
  {
    std::vector<int> y_true = {1, 0, 1, 1, 0};
    std::vector<int> y_pred = {1, 0, 1, 1, 0};
    check_close("accuracy perfect", metrics::accuracy(y_true, y_pred), 1.0);
    check_close("precision perfect", metrics::precision(y_true, y_pred), 1.0);
    check_close("recall perfect", metrics::recall(y_true, y_pred), 1.0);
  }

  std::cout << '\n';
  if (g_failures == 0) {
    std::cout << "All tests passed.\n";
    return 0;
  }
  std::cout << g_failures << " test(s) FAILED.\n";
  return 1;
}
