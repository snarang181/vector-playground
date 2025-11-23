#pragma once

#include <cstddef>
#include <string>

namespace vecplay {
enum class KernelKind {
  Saxpy, // Single-precision A·X Plus Y
  Dot    // Dot Product
};

enum class VariantKind {
  Scalar, // Scalar implementation
  Auto,   // Auto-vectorized by the compiler
  Manual  // Neon intrinsics (if available)
};

// Convert CLI strings to enums.
KernelKind  parseKernel(const std::string &name);
VariantKind parseVariant(const std::string &name);

// Kernels all operate on float arrays.

//
// SAXPY: y[i] = a * x[i] + y[i]
void saxpy_scalar(float *y, const float *x, float a, std::size_t n);
void saxpy_auto(float *y, const float *x, float a, std::size_t n);
void saxpy_manual(float *y, const float *x, float a, std::size_t n);
void saxpy_manual_unrolled(float *y, const float *x, float a, std::size_t n, int unroll_factor);

//
// DOT PRODUCT: return sum of x[i] * y[i]
float dot_scalar(const float *x, const float *y, std::size_t n);
float dot_auto(const float *x, const float *y, std::size_t n);
float dot_manual(const float *x, const float *y, std::size_t n);
} // namespace vecplay