#include "vecplay/bench.hpp"
#include "vecplay/kernels.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace vecplay;

bool verify_checksum(float computed, float expected, float tol = 1e-5f) {
  return std::abs(computed - expected) <= tol;
}

struct Args {
  std::string kernel        = "saxpy";
  std::string variant       = "auto";
  std::size_t n             = 1 << 20; // 1 million elements
  std::size_t iterations    = 10;
  bool        csv           = false;
  int         unroll_factor = 2; // Default unroll factor for manual variant
};

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto get_val = [&](int &i) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for argument: " + arg);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--kernel") {
      args.kernel = get_val(i);
    } else if (arg == "--variant") {
      args.variant = get_val(i);
    } else if (arg == "--n") {
      args.n = std::stoull(get_val(i));
    } else if (arg == "--iters") {
      args.iterations = std::stoull(get_val(i));
    } else if (arg == "--csv") {
      args.csv = true;
    } else if (arg == "--unroll") {
      args.unroll_factor = std::stoi(get_val(i));
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return args;
}

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);

    BenchConfig config;
    config.kernel        = parseKernel(args.kernel);
    config.variant       = parseVariant(args.variant);
    config.n             = args.n;
    config.iterations    = args.iterations;
    config.csv           = args.csv;
    config.unroll_factor = args.unroll_factor;

    BenchResult result = run_benchmark(config);

    if (config.csv) {
      // CSV: kernel,variant,n,iters,unroll_factor,time_sec,gflops,checksum
      std::cout << args.kernel << "," << args.variant << "," << config.n << "," << config.iterations
                << "," << config.unroll_factor << "," << result.time_seconds << ","
                << result.gflops_per_sec << "," << result.checksum << "\n";
    } else {
      std::cout << "Benchmark Results:\n";
      std::cout << "  Kernel: " << args.kernel << "\n";
      std::cout << "  Variant: " << args.variant << "\n";
      std::cout << "  Size: " << config.n << "\n";
      std::cout << "  Iterations: " << config.iterations << "\n";
      std::cout << "  Unroll Factor: " << config.unroll_factor << "\n";

      std::cout << "  Total Time (s): " << result.time_seconds << "\n";
      std::cout << "  Performance (GFLOPS): " << result.gflops_per_sec << "\n";
      std::cout << "  Checksum: " << result.checksum << "\n";
    }
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}