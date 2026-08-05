#include "logistic_regression.hpp"
#include "metrics.hpp"

#include <iostream>
#include <vector>

int main() {
  // Two features. True boundary: class 1 iff (x1 + x2 > 0).
  // Points sampled on both sides, comfortably separable.
  std::vector<std::vector<double>> X = {
      {2.0, 3.0},   {1.0, 1.0},   {3.0, 0.5},   {0.5, 2.0},   // clearly class 1
      {-2.0, -1.0}, {-1.0, -3.0}, {-0.5, -2.0}, {-3.0, -0.5}, // clearly class 0
      {1.5, 0.2},   {0.2, 1.5},   // class 1, near-ish
      {-1.5, -0.2}, {-0.2, -1.5}, // class 0, near-ish
  };

  std::vector<double> Y;
  Y.reserve(X.size());
  for (const auto &row : X) {
    Y.push_back((row[0] + row[1] > 0.0) ? 1.0 : 0.0);
  }

  LogisticRegression model(2, /*lr=*/0.5);
  model.fit(X, Y, /*epochs=*/2000, /*log_every=*/200, /*tol=*/1e-9);

  std::cout << "\nlearned weights: ";
  for (double w : model.weights())
    std::cout << w << ' ';
  std::cout << "\nlearned bias:    " << model.bias() << '\n';

  // Evaluate on the training set.
  auto preds = model.predict_label(X);
  std::vector<int> y_int(Y.begin(), Y.end());
  std::cout << "\naccuracy:  " << metrics::accuracy(y_int, preds)
            << "\nprecision: " << metrics::precision(y_int, preds)
            << "\nrecall:    " << metrics::recall(y_int, preds) << '\n';

  // Probabilities on a few points to see confidence.
  std::cout << "\nsample probabilities:\n";
  std::vector<std::vector<double>> probe = {
      {3.0, 3.0}, {0.1, 0.0}, {-3.0, -3.0}};
  for (const auto &x : probe) {
    std::cout << "  x=(" << x[0] << ", " << x[1]
              << ") -> p=" << model.predict(x)
              << " label=" << model.predict_label(x) << '\n';
  }

  return 0;
}
