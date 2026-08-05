# Compiler and flags, defined once so every rule uses the same settings.
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Iinclude

# Where object files go, so they don't clutter the source tree.
BUILD := build

# Library object files: the shared code every test/example links against.
OBJS := $(BUILD)/linear_regression.o \
        $(BUILD)/logistic_regression.o \
        $(BUILD)/dataset.o \
        $(BUILD)/metrics.o

# The executables we can build. `make all` builds every one.
BINS := tests/test_metrics \
        tests/test_linear_scaled \
        tests/test_logistic \
        examples/binary_classification

# ---- Default target: build everything ----
all: $(BINS)

# ---- Pattern rule: how to turn any src/*.cpp into build/*.o ----
# $< = the first prerequisite (the .cpp). $@ = the target (the .o).
# The | $(BUILD) means "build/ must exist first" (order-only prereq).
$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create the build directory if it doesn't exist.
$(BUILD):
	mkdir -p $(BUILD)

# ---- Link rules: each executable = its own .o + the library objects ----
# Since the test/example .cpp lives outside src/, we compile+link it directly.

tests/test_metrics: tests/test_metrics.cpp $(BUILD)/metrics.o
	$(CXX) $(CXXFLAGS) $^ -o $@

tests/test_linear_scaled: tests/test_linear_scaled.cpp \
                          $(BUILD)/linear_regression.o $(BUILD)/dataset.o
	$(CXX) $(CXXFLAGS) $^ -o $@

tests/test_logistic: tests/test_logistic.cpp \
                     $(BUILD)/logistic_regression.o $(BUILD)/metrics.o
	$(CXX) $(CXXFLAGS) $^ -o $@

examples/binary_classification: examples/binary_classification.cpp \
                                $(BUILD)/logistic_regression.o \
                                $(BUILD)/dataset.o $(BUILD)/metrics.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# ---- Convenience: build and run each test ----
test: tests/test_metrics tests/test_linear_scaled tests/test_logistic
	./tests/test_metrics
	./tests/test_linear_scaled
	./tests/test_logistic

# ---- Run the example ----
run-example: examples/binary_classification
	./examples/binary_classification

# ---- Clean up all build artifacts ----
clean:
	rm -rf $(BUILD) $(BINS)

# These targets aren't real files; declare them "phony" so Make always
# runs them instead of checking for a file of that name.
.PHONY: all test run-example clean
