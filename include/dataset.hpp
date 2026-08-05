#ifndef DATASET_HPP
#define DATASET_HPP

#include <cstddef>
#include <vector>

// Standardize features to zero mean and unit variance, per column;
//
// Usage:
//  StandardScaler;
//  scaler.fit(X_train);
//  auto X_train_scaled = scaler.transform(X_train);
//  auto X_test_scaled = scaler.transform(X_test);
//
//  Fit ONLY on training data. Transform train and test using the same
//  stats. FItting on test leaks information into preprocessing.

class StandardScaler {
public:
  // Compute per-column mean and std over X. X[i][j] = sample i, feature j.
  void fit(const std::vector<std::vector<double>> &X);

  // Return a new matrix where each column hhas been standardized using the
  // stats from fit().
  std::vector<std::vector<double>>
  transform(const std::vector<std::vector<double>> &X) const;

  // fit + transform in one call. Only for training data.
  std::vector<std::vector<double>>
  fit_transform(const std::vector<std::vector<double>> &X);

  const std::vector<double> &mean() const { return mean_; }
  const std::vector<double> &std() const { return std_; }

private:
  std::vector<double> mean_;
  std::vector<double> std_;
  bool fitted_ = false;
};

#endif
