# regression-cpp

Linear and logistic regression implemented from scratch in C++17, with no ML
libraries — just `std::vector` and hand-derived gradients. Built as a learning
project to understand how these models work underneath the abstractions.

## What's inside

- **Scalar linear regression** (`linear.cpp`) — the simplest possible version,
  `y = wx + b`, trained by gradient descent. Kept as a readable reference that
  shows the math in its barest form.
- **Multi-feature linear regression** (`LinearRegression`) — arbitrary number
  of features via `y = wᵀx + b`, batch gradient descent on mean squared error.
- **Feature standardization** (`StandardScaler`) — scales features to zero mean
  and unit variance, following the fit/transform pattern. Necessary because
  gradient descent breaks when features are on very different scales.
- **Evaluation metrics** — MSE, MAE, R² for regression; accuracy, precision,
  recall for classification.
- **Logistic regression** (`LogisticRegression`) — *in progress.* Binary
  classification via the sigmoid and cross-entropy loss.

## Project layout

regression-cpp/
├── linear.cpp # scalar reference implementation
├── include/
│ ├── linear_regression.hpp # multi-feature linear model
│ ├── logistic_regression.hpp # (in progress)
│ ├── dataset.hpp # StandardScaler
│ └── metrics.hpp # evaluation metrics
├── src/
│ ├── linear_regression.cpp
│ ├── logistic_regression.cpp # (in progress)
│ ├── dataset.cpp
│ └── metrics.cpp
├── tests/
│ ├── test_linear_scaled.cpp # linear model + scaler end-to-end
│ ├── test_metrics.cpp # metric correctness checks
│ └── test_logistic.cpp # (in progress)
└── compile_flags.txt # include path for editor tooling


## Building

Compiled with g++ and C++17. All warnings enabled. From the project root:

```bash
# Scalar reference
g++ -std=c++17 -Wall -Wextra -Wpedantic linear.cpp -o linear
./linear

# Linear regression + standardization
g++ -std=c++17 -Wall -Wextra -Wpedantic -Iinclude \
    src/linear_regression.cpp src/dataset.cpp \
    tests/test_linear_scaled.cpp -o tests/test_linear_scaled
./tests/test_linear_scaled

# Metrics tests
g++ -std=c++17 -Wall -Wextra -Wpedantic -Iinclude \
    src/metrics.cpp tests/test_metrics.cpp -o tests/test_metrics
./tests/test_metrics
```

The `-Iinclude` flag tells the compiler where to find the headers, so source
files can `#include "linear_regression.hpp"` without hardcoded paths.

## Notes on the approach

Everything is derived and implemented by hand — the gradients are worked out
on paper, not computed by an autodiff engine. Where the code makes a design
choice (standardize before training, clamp probabilities before taking logs,
stop training when loss stops improving), the reasoning is in the comments.

Two things this project makes concrete:

- **Why feature scaling matters.** Without standardization, a feature on a large
  numeric scale produces large gradients, and a single learning rate can't serve
  features of different magnitudes — training diverges to NaN. Standardizing
  every feature to unit variance fixes it.
- **Why linear and logistic regression share the same training loop.** Despite
  using different loss functions (squared error vs cross-entropy), both yield a
  gradient of the form `(1/N) · Σ (prediction − target) · feature`. The training
  code carries over almost unchanged.

## Status

- [x] Scalar linear regression
- [x] Multi-feature linear regression
- [x] Feature standardization
- [x] Evaluation metrics
- [ ] Logistic regression
- [ ] Worked examples (house prices, binary classification)
- [ ] Makefile

## License

Personal learning project.
