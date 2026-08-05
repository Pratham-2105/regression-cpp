#include "../include/dataset.hpp"
#include "../include/linear_regression.hpp"

#include <iostream>

int main() {
  // Same truth: y = 2*x1 - 3*x2 + 0.5*x3 + 4
  // But x3 is on a huge scale now (thousands).
  std::vector<std::vector<double>> X = {
      {1.0, 2.0, 3000.0}, {2.0, 1.0, 0.0},    {0.0, 0.0, 5000.0},
      {3.0, 4.0, 1000.0}, {5.0, 2.0, 2000.0}, {1.0, 5.0, 4000.0},
      {4.0, 3.0, 0.0},    {2.0, 2.0, 6000.0},
  };

  std::vector<double> Y;
  Y.reserve(X.size());
  for (const auto &row : X) {
    Y.push_back(2.0 * row[0] - 3.0 * row[1] + 0.5 * row[2] + 4.0);
  }

  StandardScaler scaler;
  auto X_scaled = scaler.fit_transform(X);

  std::cout << "means: ";
  for (double m : scaler.mean())
    std::cout << m << ' ';
  std::cout << "\nstds:  ";
  for (double s : scaler.std())
    std::cout << s << ' ';
  std::cout << "\n\n";

  LinearRegression model(3, 0.01);
  model.fit(X_scaled, Y, /*epochs=*/5000, /*log_every=*/500, /*tol=*/1e-20);

  std::cout << "\nlearned weights (in scaled space): ";
  for (double w : model.weights())
    std::cout << w << ' ';
  std::cout << "\nlearned bias:                       " << model.bias() << '\n';

  // Predict on an unseen raw sample. MUST scale it first.
  std::vector<double> x_raw = {10.0, 5.0, 2000.0};
  auto x_scaled = scaler.transform({x_raw})[0];
  double truth = 2.0 * 10.0 - 3.0 * 5.0 + 0.5 * 2000.0 + 4.0;
  std::cout << "\npredict on [10, 5, 2000]: " << model.predict(x_scaled)
            << " (truth = " << truth << ")\n";

  return 0;
}
