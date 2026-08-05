#include "dataset.hpp"
#include "logistic_regression.hpp"
#include "metrics.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

int main() {
  // ---------------------------------------------------------------
  // 1. Generate a synthetic "exam pass" dataset.
  //
  //    Features:  x0 = hours studied  (roughly 0..10)
  //               x1 = hours slept    (roughly 3..9)
  //    True rule: a student passes if a weighted combination of study
  //               and sleep clears a threshold -- PLUS random noise, so
  //               the classes overlap near the boundary (realistic).
  // ---------------------------------------------------------------
  std::mt19937 rng(42); // fixed seed -> reproducible run
  std::uniform_real_distribution<double> study_dist(0.0, 10.0);
  std::uniform_real_distribution<double> sleep_dist(3.0, 9.0);
  std::normal_distribution<double> noise(0.0, 1.5); // the "overlap"

  const std::size_t N = 400;
  std::vector<std::vector<double>> X;
  std::vector<double> Y;
  X.reserve(N);
  Y.reserve(N);

  for (std::size_t i = 0; i < N; ++i) {
    double study = study_dist(rng);
    double sleep = sleep_dist(rng);

    // Hidden "true" score. Studying matters most, sleep helps, and we
    // subtract a threshold so roughly half the students pass.
    double score = 1.2 * study + 0.8 * sleep - 9.0 + noise(rng);
    double label = (score > 0.0) ? 1.0 : 0.0;

    X.push_back({study, sleep});
    Y.push_back(label);
  }

  // ---------------------------------------------------------------
  // 2. Shuffle, then split 75% train / 25% test.
  //
  //    Shuffling first matters: if the data had any ordering, an
  //    unshuffled split could put all of one class in the test set.
  // ---------------------------------------------------------------
  std::vector<std::size_t> idx(N);
  for (std::size_t i = 0; i < N; ++i)
    idx[i] = i;
  std::shuffle(idx.begin(), idx.end(), rng);

  const std::size_t n_train = static_cast<std::size_t>(0.75 * N);

  std::vector<std::vector<double>> X_train, X_test;
  std::vector<double> Y_train, Y_test;
  for (std::size_t k = 0; k < N; ++k) {
    std::size_t i = idx[k];
    if (k < n_train) {
      X_train.push_back(X[i]);
      Y_train.push_back(Y[i]);
    } else {
      X_test.push_back(X[i]);
      Y_test.push_back(Y[i]);
    }
  }

  std::cout << "train samples: " << X_train.size()
            << ", test samples: " << X_test.size() << "\n\n";

  // ---------------------------------------------------------------
  // 3. Standardize. Fit on TRAIN ONLY, transform both.
  // ---------------------------------------------------------------
  StandardScaler scaler;
  auto X_train_s = scaler.fit_transform(X_train);
  auto X_test_s = scaler.transform(X_test); // same stats as train

  std::cout << "feature means (train): ";
  for (double m : scaler.mean())
    std::cout << m << ' ';
  std::cout << "\nfeature stds  (train): ";
  for (double s : scaler.std())
    std::cout << s << ' ';
  std::cout << "\n\n";

  // ---------------------------------------------------------------
  // 4. Train the model on the training set.
  // ---------------------------------------------------------------
  LogisticRegression model(2, /*lr=*/0.1);
  model.fit(X_train_s, Y_train, /*epochs=*/2000, /*log_every=*/400,
            /*tol=*/1e-9);

  std::cout << "\nlearned weights: ";
  for (double w : model.weights())
    std::cout << w << ' ';
  std::cout << "\nlearned bias:    " << model.bias() << "\n\n";

  // ---------------------------------------------------------------
  // 5. Evaluate on BOTH sets. The gap between them is the story.
  // ---------------------------------------------------------------
  auto eval = [&](const std::string &name,
                  const std::vector<std::vector<double>> &Xs,
                  const std::vector<double> &Yd) {
    auto preds = model.predict_label(Xs);
    std::vector<int> yi(Yd.begin(), Yd.end());
    std::cout << name << " -> "
              << "accuracy: " << metrics::accuracy(yi, preds)
              << "  precision: " << metrics::precision(yi, preds)
              << "  recall: " << metrics::recall(yi, preds) << '\n';
  };

  eval("train", X_train_s, Y_train);
  eval("test ", X_test_s, Y_test);

  // ---------------------------------------------------------------
  // 6. Inspect a few individual test predictions with probabilities.
  // ---------------------------------------------------------------
  std::cout << "\nsample test predictions (raw features -> probability):\n";
  for (std::size_t k = 0; k < 6 && k < X_test.size(); ++k) {
    double p = model.predict(X_test_s[k]);
    std::cout << "  study=" << X_test[k][0] << "h sleep=" << X_test[k][1]
              << "h  ->  p(pass)=" << p
              << "  predicted=" << (p >= 0.5 ? "PASS" : "FAIL")
              << "  actual=" << (Y_test[k] > 0.5 ? "PASS" : "FAIL") << '\n';
  }

  return 0;
}
