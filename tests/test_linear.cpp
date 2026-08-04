#include "../include/linear_regression.hpp"

#include <iostream>

int main() {
  // Truth: y = 2*x1 - 3*x2 + 0.5*x3 + 4
  // Weights we hope to recover: [2.0, -3.0, 0.5], bias 4.0.
  std::vector<std::vector<double>> X = {
      {1.0, 2.0, 3.0}, {2.0, 1.0, 0.0}, {0.0, 0.0, 5.0}, {3.0, 4.0, 1.0},
      {5.0, 2.0, 2.0}, {1.0, 5.0, 4.0}, {4.0, 3.0, 0.0}, {2.0, 2.0, 6.0},
  };

  std::vector<double> Y;
  Y.reserve(X.size());
  for (const auto &row : X) {
    double y = 2.0 * row[0] - 3.0 * row[1] + 0.5 * row[2] + 4.0;
    Y.push_back(y);
  }

  LinearRegression model(3, 0.01);
  model.fit(X, Y, /*epochs=*/5000, /*log_every=*/500, /*tol=*/1e-20);

  std::cout << "\nlearned weights: ";
  for (double w : model.weights())
    std::cout << w << ' ';
  std::cout << "\nlearned bias:    " << model.bias() << '\n';
  std::cout << "true weights:    2 -3 0.5\ntrue bias:       4\n";

  // Test on an unseen point.
  std::vector<double> x_new = {10.0, 5.0, 2.0};
  double truth = 2.0 * 10.0 - 3.0 * 5.0 + 0.5 * 2.0 + 4.0;
  std::cout << "\npredict on [10,5,2]: " << model.predict(x_new)
            << " (truth = " << truth << ")\n";

  return 0;
}
