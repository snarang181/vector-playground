#pragma once

#include "vecplay/kernels.hpp"
#include <cstddef>
#include <string>

namespace vecplay {
struct BenchConfig {
  KernelKind  kernel        = KernelKind::Saxpy;
  VariantKind variant       = VariantKind::Auto;
  std::size_t n             = 1 << 20; // Default to 1 million elements
  std::size_t iterations    = 10;      // Default to 10 iterations
  bool        csv           = false;   // Output in CSV format
  int         unroll_factor = 2;       // Unroll factor for manual variant
};

// Results of a benchmark run.
struct BenchResult {
  double time_seconds;   // total time for all iterations
  float  checksum;       // checksum of the result array (for validation)
  double gflops_per_sec; // performance in GFLOPS
};

// Run the benchmark with the given configuration.
BenchResult run_benchmark(const BenchConfig &config);
} // namespace vecplay