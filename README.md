# vector-playground

A tiny playground for experimenting with **vectorization** on Apple Silicon.

Right now it focuses on the classic **SAXPY** kernel (`y = a * x + y`) and a simple **dot product**, and compares:

- 🚶‍♂️ **Scalar** – vectorization explicitly disabled  
- 🤖 **Auto** – let Clang’s auto-vectorizer do its thing  
- 🛠 **Manual NEON** – hand-written NEON intrinsics with configurable loop unroll factors  

The goal is to *measure*, not guess: how close is the compiler’s auto-vectorized code to a carefully tuned NEON kernel on an M-series Mac?

---

## Highlights

- Apple Silicon–friendly C++ / CMake project.
- Benchmarks for:
  - `saxpy`: `y[i] = a * x[i] + y[i]`
  - `dot`: `sum += x[i] * x[i]` (or `x[i] * y[i]`, depending on config)
- Three variants:
  - `scalar` – vectorization disabled
  - `auto` – auto-vectorized by Clang
  - `manual` – NEON intrinsics with explicit unroll factors (`1`, `2`, `4`, …)
- Benchmark output includes:
  - Total time
  - GFLOP/s
  - Checksum (for sanity-checking correctness across variants)

If you’ve ever wondered *“can I beat the compiler?”* on simple kernels, this repo is a place to play with that question.

---

## Requirements

Tested on:

- Apple Silicon (M1 / M2 / M3…)  
- C++ compiler: **Clang** (Apple Clang) with NEON support
- CMake 3.16+

You’ll also need a standard command-line toolchain (`make`, `ninja`, etc., depending on your CMake generator).

---

## Building

From the repo root:

```bash
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS_RELEASE="-O3 -ffast-math -march=native" \
      ..
cmake --build . -j
