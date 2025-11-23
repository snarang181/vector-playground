#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
BIN="${REPO_ROOT}/build/vec_bench"

# Make sure the binary exists
if [[ ! -f "${BIN}" ]]; then
    echo "Benchmark binary not found at ${BIN}. Please build the project first."
    exit 1
fi

echo "Using benchmark binary at ${BIN}"

OUT="${REPO_ROOT}/saxpy_bench_results.csv"

# Overwrite and add CSV header
echo "kernel,variant,unroll,n,iters,time_sec,gflops,checksum" > "${OUT}"

# Problem sizes and iterations
SIZES=(200000 1000000 2000000)
ITERS=200

# Variants
VARIANTS=(scalar auto manual)

# Unroll factors to sweep for manual
MANUAL_UNROLLS=(1 2 4 8)

for N in "${SIZES[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        if [[ "${VARIANT}" == "manual" ]]; then
            for UNROLL in "${MANUAL_UNROLLS[@]}"; do
                echo "Running: kernel=saxpy, variant=${VARIANT}, N=${N}, unroll=${UNROLL}, iters=${ITERS}"

                OUTPUT=$("$BIN" \
                    --kernel saxpy \
                    --variant manual \
                    --n "${N}" \
                    --iters "${ITERS}" \
                    --unroll "${UNROLL}")

                # Parse fields from vec_bench output
                kernel=$(awk -F': ' '/Kernel:/ {print $2}' <<< "${OUTPUT}")
                variant=$(awk -F': ' '/Variant:/ {print $2}' <<< "${OUTPUT}")
                size=$(awk -F': ' '/Size:/ {print $2}' <<< "${OUTPUT}")
                iters=$(awk -F': ' '/Iterations:/ {print $2}' <<< "${OUTPUT}")
                time_sec=$(awk -F': ' '/Total Time \(s\):/ {print $2}' <<< "${OUTPUT}")
                gflops=$(awk -F': ' '/Performance \(GFLOPS\):/ {print $2}' <<< "${OUTPUT}")
                checksum=$(awk -F': ' '/Checksum:/ {print $2}' <<< "${OUTPUT}")

                echo "${kernel},${variant},${UNROLL},${size},${iters},${time_sec},${gflops},${checksum}" >> "${OUT}"
            done
        else
            echo "Running: kernel=saxpy, variant=${VARIANT}, N=${N}, iters=${ITERS}"

            OUTPUT=$("$BIN" \
                --kernel saxpy \
                --variant "${VARIANT}" \
                --n "${N}" \
                --iters "${ITERS}")

            kernel=$(awk -F': ' '/Kernel:/ {print $2}' <<< "${OUTPUT}")
            variant=$(awk -F': ' '/Variant:/ {print $2}' <<< "${OUTPUT}")
            size=$(awk -F': ' '/Size:/ {print $2}' <<< "${OUTPUT}")
            iters=$(awk -F': ' '/Iterations:/ {print $2}' <<< "${OUTPUT}")
            time_sec=$(awk -F': ' '/Total Time \(s\):/ {print $2}' <<< "${OUTPUT}")
            gflops=$(awk -F': ' '/Performance \(GFLOPS\):/ {print $2}' <<< "${OUTPUT}")
            checksum=$(awk -F': ' '/Checksum:/ {print $2}' <<< "${OUTPUT}")

            # No unroll factor for scalar/auto → use NA
            echo "${kernel},${variant},NA,${size},${iters},${time_sec},${gflops},${checksum}" >> "${OUT}"
        fi
    done
done

echo "Benchmarking complete. Results saved to ${OUT}."
