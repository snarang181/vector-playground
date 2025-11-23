#include "vecplay/bench.hpp"
#include "vecplay/kernels.hpp"

#include <chrono>
#include <random>
#include <vector>

namespace vecplay {
BenchResult run_benchmark(const BenchConfig &config) {
  // Initialize input data
  std::vector<float> x(config.n); // Input vector x
  std::vector<float> y(config.n); // Input/output vector y

  // Fill x and y with random data
  std::mt19937                          rng(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (std::size_t i = 0; i < config.n; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
  }

  float alpha = 1.25f; // Scalar for SAXPY

  using clock = std::chrono::high_resolution_clock;
  auto start  = clock::now();

  float checksum = 0.0f; // For validation

  for (int iter = 0; iter < config.iterations; ++iter) {
    switch (config.kernel) {
    case KernelKind::Saxpy:
      switch (config.variant) {
      case VariantKind::Scalar:
        saxpy_scalar(y.data(), x.data(), alpha, config.n);
        break;
      case VariantKind::Auto:
        saxpy_auto(y.data(), x.data(), alpha, config.n);
        break;
      case VariantKind::Manual:
        saxpy_manual_unrolled(y.data(), x.data(), alpha, config.n, config.unroll_factor);
        break;
      default:
        throw std::runtime_error("Unknown variant");
      }
      break;
    case KernelKind::Dot:
      switch (config.variant) {
      case VariantKind::Scalar:
        checksum = dot_scalar(x.data(), y.data(), config.n);
        break;
      case VariantKind::Auto:
        checksum = dot_auto(x.data(), y.data(), config.n);
        break;
      case VariantKind::Manual:
        checksum = dot_manual(x.data(), y.data(), config.n);
        break;
      default:
        throw std::runtime_error("Unknown variant");
      }
      break;
    default:
      throw std::runtime_error("Unknown kernel");
    }
  }

  auto   end          = clock::now();
  double time_seconds = std::chrono::duration<double>(end - start).count();

  // FLOPS
  // SAXPY: Mul + add = 2 FLOPS per element
  // DOT: Mul + add = 2 FLOPS per element
  double flops_per_iter = 2.0 * static_cast<double>(config.n);
  double total_flops    = flops_per_iter * static_cast<double>(config.iterations);
  double gflops         = (total_flops / (time_seconds * 1e9)); // in GFLOPS

  // If kernel was SAXPY, compute checksum of y
  if (config.kernel == KernelKind::Saxpy) {
    checksum = 0.0f;
    for (float val : y) {
      checksum += val;
    }
  }
  return BenchResult{time_seconds, checksum, gflops};
}
} // namespace vecplay