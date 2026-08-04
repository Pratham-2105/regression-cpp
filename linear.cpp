#include <iostream>
#include <vector>

const double epsilon = 1e-5;

double predict(double x, double weight, double bias) {

  double z = weight * x + bias;

  return z;
}

double mse(const std::vector<double> &X, const std::vector<double> &Y,
           double weight, double bias) {
  size_t N = X.size();
  double sum = 0.0;

  for (size_t i = 0; i < N; ++i) {
    double z = predict(X[i], weight, bias);
    double target = z - Y[i];

    sum += (target * target);
  }

  double loss = sum / N;

  return loss;
}

double gradient_weight(const std::vector<double> &X,
                       const std::vector<double> &Y, double weight,
                       double bias) {
  size_t N = X.size();
  double sum = 0.0;

  for (size_t i = 0; i < N; ++i) {
    double z = predict(X[i], weight, bias);
    double difference = z - Y[i];

    sum += (difference * X[i]);
  }

  double grad = (2.0 / N) * sum;

  return grad;
}

double gradient_bias(const std::vector<double> &X, const std::vector<double> &Y,
                     double weight, double bias) {
  size_t N = X.size();
  double sum = 0.0;

  for (size_t i = 0; i < N; ++i) {
    double z = predict(X[i], weight, bias);
    double difference = z - Y[i];

    sum += difference;
  }

  double grad = (2.0 / N) * sum;

  return grad;
}

double numerical_gradient_weight(const std::vector<double> &X,
                                 const std::vector<double> &Y, double weight,
                                 double bias, double epsilon) {
  double L1 = mse(X, Y, weight + epsilon, bias);
  double L2 = mse(X, Y, weight - epsilon, bias);

  return (L1 - L2) / (2 * epsilon);
}

double numerical_gradient_bias(const std::vector<double> &X,
                               const std::vector<double> &Y, double weight,
                               double bias, double epsilon) {
  double L1 = mse(X, Y, weight, bias + epsilon);
  double L2 = mse(X, Y, weight, bias - epsilon);

  return (L1 - L2) / (2 * epsilon);
}

int main() {
  std::vector<double> X = {1, 2, 3, 4, 5};
  std::vector<double> Y = {3, 5, 7, 9, 11};

  double weight = 0.0;
  double bias = 0.0;
  /*
    for (auto it : X) {
      std::cout << "x: " << it << " prediction " << predict(it, weight, bias)
                << '\n';
    }
  */

  /*
    std::cout << "MSE: " << mse(X, Y, weight, bias) << '\n';
  */

  /*
  std::cout << "dL/dw: " << gradient_weight(X, Y, weight, bias) << '\n';

  std::cout << "dL/db: " << gradient_bias(X, Y, weight, bias) << '\n';
  */

  std::cout << "analytical dL/dw: " << gradient_weight(X, Y, weight, bias)
            << '\n';

  std::cout << "numerical  dL/dw: "
            << numerical_gradient_weight(X, Y, weight, bias, epsilon) << '\n';

  std::cout << "analytical dL/db: " << gradient_bias(X, Y, weight, bias)
            << '\n';

  std::cout << "numerical  dL/db: "
            << numerical_gradient_bias(X, Y, weight, bias, epsilon) << '\n';

  return 0;
}
