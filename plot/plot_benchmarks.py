#!/usr/bin/env python3
"""Plot matdecomp QR benchmark results from Google Benchmark JSON.

Reads one or more Google Benchmark JSON files (as produced by
``matdecomp_gpu_benchmarks --benchmark_format=json --benchmark_out=...``)
and writes static PNG figures into the output directory.

Three figure types are produced:

* ``contour_<algo>_<dtype>.png`` -- filled contour of log-runtime over the
  matrix dimensions (M, N), one per algorithm.
* ``square_comparison.png`` -- runtime vs size for square matrices (M == N),
  one line per algorithm.
* ``skinny_comparison_<pct>.png`` -- runtime vs M for fixed aspect ratios
  N/M of 6.25%, 12.5% and 25%, one figure per ratio.

Algorithm names are discovered from the JSON, so adding more benchmarks needs
no change here.
"""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Skinny aspect ratios (N / M) to compare, with display labels.
SKINNY_RATIOS = ((0.0625, "6.25pct"), (0.125, "12.5pct"), (0.25, "25pct"))

# benchmarkGpuModifiedGramSchmidt<float>/512/2 -> algo, dtype, M, N
NAME_RE = re.compile(
    r"^(?P<algo>\w+)<(?P<dtype>\w+)>/(?P<M>\d+)/(?P<N>\d+)$"
)


def _clean_algo(name):
    """Strip the ``benchmark[Gpu|Cpu]`` prefix for a readable legend label."""
    return re.sub(r"^benchmark(Gpu|Cpu)?", "", name)


def load(paths):
    """Load benchmark records from the given JSON files.

    Returns a list of dicts with keys ``algo``, ``dtype``, ``M``, ``N`` and
    ``time_ms`` (per-iteration mean runtime in milliseconds). Only real
    iteration rows are kept; aggregate rows (mean/median/stddev) are skipped.
    """
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for entry in payload.get("benchmarks", []):
            if entry.get("run_type", "iteration") != "iteration":
                continue
            match = NAME_RE.match(entry["name"])
            if match is None:
                continue
            records.append(
                {
                    "algo": _clean_algo(match["algo"]),
                    "dtype": match["dtype"],
                    "M": int(match["M"]),
                    "N": int(match["N"]),
                    "time_ms": entry["real_time"] / 1e6,
                }
            )
    if not records:
        raise SystemExit("No benchmark records parsed from: " + ", ".join(paths))
    return records


def _algos(records):
    """Return the sorted set of algorithm labels present in the records."""
    return sorted({r["algo"] for r in records})


def plot_contour(records, outdir):
    """Filled log-runtime contour over (N, M), one figure per algorithm."""
    by_key = defaultdict(list)
    for rec in records:
        by_key[(rec["algo"], rec["dtype"])].append(rec)

    for (algo, dtype), recs in sorted(by_key.items()):
        if len(recs) < 3:
            continue  # tricontourf needs at least a triangle
        # Triangulate in log2 space so the power-of-two grid is evenly spread.
        x = np.log2([r["N"] for r in recs])
        y = np.log2([r["M"] for r in recs])
        z = np.log10([r["time_ms"] for r in recs])

        fig, ax = plt.subplots(figsize=(7, 6))
        contour = ax.tricontourf(x, y, z, levels=14, cmap="viridis")
        ax.tricontour(x, y, z, levels=14, colors="k", linewidths=0.3,
                      alpha=0.4)
        ax.plot(x, y, "k.", markersize=3, alpha=0.5)

        bar = fig.colorbar(contour, ax=ax)
        bar.set_label("runtime [log10 ms]")

        _log2_ticks(ax.set_xticks, ax.set_xticklabels, x)
        _log2_ticks(ax.set_yticks, ax.set_yticklabels, y)
        ax.set_xlabel("N (columns)")
        ax.set_ylabel("M (rows)")
        ax.set_title(f"{algo} <{dtype}> -- runtime over matrix dimensions")

        path = os.path.join(outdir, f"contour_{algo}_{dtype}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


def _log2_ticks(set_ticks, set_labels, values):
    """Place ticks at each integer power of two spanned by ``values``."""
    lo, hi = int(np.floor(values.min())), int(np.ceil(values.max()))
    ticks = list(range(lo, hi + 1))
    set_ticks(ticks)
    set_labels([str(2 ** t) for t in ticks])


def _series(recs):
    """Sort records by M and return parallel (M, time_ms) lists."""
    recs = sorted(recs, key=lambda r: r["M"])
    return [r["M"] for r in recs], [r["time_ms"] for r in recs]


def plot_square(records, outdir):
    """Runtime vs size for square matrices (M == N), one line per algorithm."""
    by_algo = defaultdict(list)
    for rec in records:
        if rec["M"] == rec["N"]:
            by_algo[rec["algo"]].append(rec)
    if not by_algo:
        print("no square (M == N) data; skipping square_comparison")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in _algos(records):
        if algo not in by_algo:
            continue
        sizes, times = _series(by_algo[algo])
        ax.plot(sizes, times, marker="o", label=algo)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("matrix size (M = N)")
    ax.set_ylabel("runtime [ms]")
    ax.set_title("Square-matrix QR: runtime by algorithm")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend()

    path = os.path.join(outdir, "square_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_skinny(records, outdir):
    """Runtime vs M at fixed N/M ratios, one figure per ratio."""
    for ratio, label in SKINNY_RATIOS:
        by_algo = defaultdict(list)
        for rec in records:
            if rec["N"] == round(ratio * rec["M"]):
                by_algo[rec["algo"]].append(rec)
        if not by_algo:
            print(f"no data for N/M = {label}; skipping")
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for algo in _algos(records):
            if algo not in by_algo:
                continue
            sizes, times = _series(by_algo[algo])
            ax.plot(sizes, times, marker="o", label=algo)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("M (rows)")
        ax.set_ylabel("runtime [ms]")
        ax.set_title(
            f"Skinny QR (N/M = {ratio * 100:g}%): runtime by algorithm"
        )
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
        ax.legend()

        path = os.path.join(outdir, f"skinny_comparison_{label}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json",
        nargs="*",
        default=["results.json"],
        help="benchmark JSON file(s) to plot (default: results.json)",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="directory to write PNGs into (default: this script's directory)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    records = load(args.json)
    plot_contour(records, args.outdir)
    plot_square(records, args.outdir)
    plot_skinny(records, args.outdir)


if __name__ == "__main__":
    main()
