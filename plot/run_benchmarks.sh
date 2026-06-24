#!/usr/bin/env bash
#
# Regenerate the QR benchmark comparison plots end to end:
#   1. build the GPU benchmark + cuSOLVER reference benchmark targets
#   2. run both, emitting Google Benchmark JSON
#   3. render the comparison PNGs (the reference is overlaid as a dashed baseline)
#
# Override any of the variables below via the environment, e.g.
#   BUILD_DIR=/tmp/build BENCH_FILTER='/512/' ./plot/run_benchmarks.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-cuda}"
BENCH_FILTER="${BENCH_FILTER:-}" # optional --benchmark_filter regex

RESULTS_JSON="$OUT_DIR/results.json"
REFERENCE_JSON="$OUT_DIR/reference.json"
GPU_BIN="$BUILD_DIR/benchmarks/matdecomp_gpu_benchmarks"
REF_BIN="$BUILD_DIR/benchmarks/matdecomp_reference_benchmarks"

# ── activate the CUDA conda env (nvcc to build, cuSOLVER libs to run) ─────────
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  for base in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
      set +u # conda.sh trips over `set -u`
      # shellcheck disable=SC1091
      source "$base/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV"
      set -u
      break
    fi
  done
fi

# ── build ────────────────────────────────────────────────────────────────────
echo ">> building benchmark targets"
cmake --build "$BUILD_DIR" --target \
  matdecomp_gpu_benchmarks matdecomp_reference_benchmarks -j

# ── run, emitting JSON ───────────────────────────────────────────────────────
flags=(--benchmark_out_format=json)
[[ -n "$BENCH_FILTER" ]] && flags+=(--benchmark_filter="$BENCH_FILTER")

echo ">> running implementation benchmarks -> $RESULTS_JSON"
"$GPU_BIN" "${flags[@]}" --benchmark_out="$RESULTS_JSON"

json_args=("$RESULTS_JSON")
if [[ -x "$REF_BIN" ]]; then
  echo ">> running cuSOLVER reference benchmarks -> $REFERENCE_JSON"
  "$REF_BIN" "${flags[@]}" --benchmark_out="$REFERENCE_JSON"
  json_args+=("$REFERENCE_JSON")
else
  echo ">> WARNING: $REF_BIN not found;" \
    "reconfigure with -DMATDECOMP_REFERENCE_BENCHMARKS=ON to include it"
fi

# ── plot ─────────────────────────────────────────────────────────────────────
python="$SCRIPT_DIR/.venv/bin/python"
[[ -x "$python" ]] || python="${PYTHON:-python3}"

echo ">> rendering plots into $OUT_DIR"
"$python" "$SCRIPT_DIR/plot_benchmarks.py" "${json_args[@]}" --outdir "$OUT_DIR"
echo ">> done"
