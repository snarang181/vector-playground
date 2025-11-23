#include "vecplay/kernels.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#pragma message("Neon support detected.")
#include <arm_neon.h>
#define VECPLAY_HAS_NEON 1
#else
#define VECPLAY_HAS_NEON 0
#endif

namespace {
// UNROLL = how many 4-lane vectors to process per loop iteration.
// Effective computation per iteration = UNROLL * 4 elements.
template <int UNROLL>
static void saxpy_manual_impl(float *y, const float *x, float a, std::size_t n) {
#if VECPLAY_HAS_NEON
  static_assert(UNROLL >= 1, "Unroll factor must be at least 1");

  std::size_t i     = 0;
  float32x4_t a_vec = vdupq_n_f32(a); // Duplicate

  const std::size_t vec_width  = 4;
  const std::size_t step_elems = UNROLL * vec_width;

  std::size_t limit = n / step_elems * step_elems; // Process in chunks of UNROLL * 4
  for (; i < limit; i += step_elems) {
    float32x4_t x_vec[UNROLL];
    float32x4_t y_vec[UNROLL];

// Manual unrolling
#pragma clang loop unroll(disable) // Disable automatic unrolling
    for (int u = 0; u < UNROLL; ++u) {
      std::size_t base = i + u * vec_width; // Base index for this unroll
      x_vec[u]         = vld1q_f32(&x[base]);
      y_vec[u]         = vld1q_f32(&y[base]);
      // Source-destructive multiply-accumulate
      y_vec[u] = vmlaq_f32(y_vec[u], x_vec[u], a_vec);
      // Store results back to y
      vst1q_f32(&y[base], y_vec[u]);
    }
  }
  // Handle remaining elements
  for (; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
#else
  // Fallback to scalar if Neon is not available
  saxpy_scalar(y, x, a, n);
#endif
}
} // namespace

namespace vecplay {
KernelKind parseKernel(const std::string &name) {
  if (name == "saxpy")
    return KernelKind::Saxpy;
  if (name == "dot")
    return KernelKind::Dot;
  throw std::runtime_error("Unknown kernel name: " + name);
}

VariantKind parseVariant(const std::string &name) {
  if (name == "scalar")
    return VariantKind::Scalar;
  if (name == "auto")
    return VariantKind::Auto;
  if (name == "manual")
    return VariantKind::Manual;
  throw std::runtime_error("Unknown variant name: " + name);
}

// SAXPY Implementations

// Scalar implementation: y[i] = a * x[i] + y[i] - no vectorization
void saxpy_scalar(float *y, const float *x, float a, std::size_t n) { // clang-format off
#pragma clang loop vectorize(disable) // Disable auto-vectorization
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// Auto-vectorized implementation: rely on compiler optimizations
void saxpy_auto(float *y, const float *x, float a, std::size_t n) { 
#pragma clang loop vectorize(enable) // Enable auto-vectorization
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

void saxpy_manual_unrolled(float *y, const float *x, float a, std::size_t n, int unroll_factor) {
#if VECPLAY_HAS_NEON
    switch (unroll_factor) {
    case 1:
        saxpy_manual_impl<1>(y, x, a, n);
        break;
    case 2:
        saxpy_manual_impl<2>(y, x, a, n);
        break;
    case 4:
        saxpy_manual_impl<4>(y, x, a, n);
        break;
    case 8:
        saxpy_manual_impl<8>(y, x, a, n);
        break;
    default:
        // Fallback to unroll factor of 2 if unsupported
        saxpy_manual_impl<2>(y, x, a, n);
        break;
    }
#else
    // Fallback to scalar if Neon is not available
    (void)unroll_factor; // Suppress unused variable warning
    saxpy_scalar(y, x, a, n);
#endif
}

void saxpy_manual(float *y, const float *x, float a, std::size_t n) {
  // Default unroll factor of 2 -> process 8 elements per iteration
  saxpy_manual_unrolled(y, x, a, n, 2);
}

// DOT Product Implementations
float dot_scalar(const float *x, const float *y, std::size_t n) {
  float result = 0.0f;
#pragma clang loop vectorize(disable) // Disable auto-vectorization
  for (std::size_t i = 0; i < n; ++i) {
    result += x[i] * y[i];
  }
  return result;
} // clang-format on

float dot_auto(const float *x, const float *y, std::size_t n) {
  float result = 0.0f;
#pragma clang loop vectorize(enable) // Enable auto-vectorization
  for (std::size_t i = 0; i < n; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

float dot_manual(const float *x, const float *y, std::size_t n) {
#if VECPLAY_HAS_NEON
  std::size_t       i       = 0;
  const std::size_t step    = 4;                 // Process 4 floats at a time
  float32x4_t       sum_vec = vdupq_n_f32(0.0f); // Initialize sum vector to zero

  std::size_t n_vec = n - (n % step); // Handle multiples of 4
  for (; i < n_vec; i += step) {
    float32x4_t x_vec = vld1q_f32(&x[i]);
    float32x4_t y_vec = vld1q_f32(&y[i]);
    // Separate multiply and accumulate
    float32x4_t prod_vec = vmulq_f32(x_vec, y_vec);
    sum_vec              = vaddq_f32(sum_vec, prod_vec);
  }
  // Horizontal add to get the final sum from sum_vec
  float32x2_t sum_low  = vget_low_f32(sum_vec);                              // Get lower 2 floats
  float32x2_t sum_high = vget_high_f32(sum_vec);                             // Get higher 2 floats
  float32x2_t pair_sum = vadd_f32(sum_low, sum_high);                        // Add pairs
  float       acc = vget_lane_f32(pair_sum, 0) + vget_lane_f32(pair_sum, 1); // Final horizontal add

  // Handle remaining elements
  for (; i < n; ++i) {
    acc += x[i] * y[i];
  }
  return acc;
#else
  // Fallback to scalar if Neon is not available
  return dot_scalar(x, y, n);
#endif
}

} // namespace vecplay