#!/usr/bin/env python3
"""Plot matdecomp QR benchmark results from Google Benchmark JSON.

Reads one or more Google Benchmark JSON files (as produced by
``matdecomp_gpu_benchmarks --benchmark_format=json --benchmark_out=...``)
and writes static PNG figures into the output directory.

Three figure types are produced:

* ``contour_<algo>_<dtype>.png`` -- filled contour of log-runtime over the
  matrix dimensions (M, N), one per algorithm.
* ``contour_<algo>_vs_<ref>_<dtype>.png`` -- filled contour of the runtime
  ratio ``log10(algo / reference)`` over (M, N), one per algorithm, when a
  reference series (cuSOLVER/cuSPARSE) is present. Red = slower than reference.
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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Skinny aspect ratios (N / M) to compare, with display labels.
SKINNY_RATIOS = ((0.0625, "6.25pct"), (0.125, "12.5pct"), (0.25, "25pct"))

# Super Skinny widths
SUPER_SKINNY_NS = (2, 4, 8, 16, 32)

# Matches both our templated benchmarks and the reference, which has no dtype
# and a trailing ``/manual_time`` suffix (from UseManualTime):
#   benchmarkGpuModifiedGramSchmidt<float>/512/2 -> algo, dtype, M, N
#   benchmarkCusolver/512/2/manual_time           -> algo, (dtype defaults to float)
NAME_RE = re.compile(
    r"^(?P<algo>\w+)(?:<(?P<dtype>\w+)>)?/(?P<M>\d+)/(?P<N>\d+)(?:/\w+)*$"
)

# Reference/baseline series (e.g. cuSOLVER) are drawn as a distinct dashed line.
REFERENCE_RE = re.compile(r"cusolver|cusparse|reference", re.IGNORECASE)


def _clean_algo(name):
    """Strip the ``benchmark[Gpu|Cpu]`` prefix for a readable legend label."""
    return re.sub(r"^benchmark(Gpu|Cpu)?", "", name)


def _is_reference(algo):
    """True if ``algo`` is an external reference implementation, not ours."""
    return bool(REFERENCE_RE.search(algo))


def _draw_series(ax, sizes, times, algo):
    """Plot one algorithm's line; references stand out as a dashed black baseline."""
    if _is_reference(algo):
        ax.plot(
            sizes,
            times,
            color="black",
            linestyle="--",
            linewidth=2,
            marker="x",
            zorder=10,
            label=f"{algo} (ref)",
        )
    else:
        ax.plot(sizes, times, marker="o", label=algo)


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

            # Skip skipped/failed rows (e.g. OOM via SkipWithError): they carry
            # error_occurred and/or a zero runtime, and log10(0) = -inf would
            # break the contour triangulation.
            if entry.get("error_occurred") or entry["real_time"] <= 0.0:
                continue

            records.append(
                {
                    "algo": _clean_algo(match["algo"]),
                    "dtype": match["dtype"] or "float",
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
        ax.tricontour(x, y, z, levels=14, colors="k", linewidths=0.3, alpha=0.4)
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


def _by_key_times(records):
    """Group runtimes as ``{(algo, dtype): {(M, N): time_ms}}``."""
    grouped = defaultdict(dict)
    for rec in records:
        grouped[(rec["algo"], rec["dtype"])][(rec["M"], rec["N"])] = rec["time_ms"]
    return grouped


def plot_contour_vs_reference(records, outdir):
    """Fill contour of the runtime ratio (algo / reference) over (N, M).

    One figure per (our-algorithm, reference) pair, sharing the matrix sizes the
    two have in common. Colour is ``log10(time_algo / time_ref)`` on a diverging
    scale centred at 0: red where we are slower than the reference, blue where
    faster, white at parity. Skipped entirely when no reference series is loaded.
    """
    grouped = _by_key_times(records)
    refs = sorted(key for key in grouped if _is_reference(key[0]))
    if not refs:
        print("no reference series found; skipping reference-ratio contours")
        return

    for ref_algo, ref_dtype in refs:
        ref_times = grouped[(ref_algo, ref_dtype)]
        for (algo, dtype), times in sorted(grouped.items()):
            if _is_reference(algo) or dtype != ref_dtype:
                continue
            shared = [mn for mn in times if mn in ref_times]
            if len(shared) < 3:
                continue  # tricontourf needs at least a triangle
            x = np.log2([n for (_, n) in shared])
            y = np.log2([m for (m, _) in shared])
            z = np.log10([times[mn] / ref_times[mn] for mn in shared])

            # Span the actual data range so the colorbar carries real signal.
            zmin, zmax = float(np.min(z)), float(np.max(z))
            if zmax - zmin < 1e-9:  # nearly flat: pad so the levels are valid
                pad = max(abs(zmax) * 0.1, 1e-3)
                zmin, zmax = zmin - pad, zmax + pad
            levels = np.linspace(zmin, zmax, 15)

            if zmin < 0.0 < zmax:
                # Straddles parity: diverging, anchored at 0 (white == on par).
                cmap = "RdBu_r"
                norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=zmin, vmax=zmax)
            else:
                # Entirely one side: sequential over the data range, no wasted
                # half. All slower -> Reds (darker == slower); all faster -> Blues.
                cmap = "Reds" if zmin >= 0.0 else "Blues_r"
                norm = None

            fig, ax = plt.subplots(figsize=(7, 6))
            contour = ax.tricontourf(x, y, z, levels=levels, cmap=cmap, norm=norm)
            ax.tricontour(x, y, z, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
            ax.plot(x, y, "k.", markersize=3, alpha=0.5)

            bar = fig.colorbar(contour, ax=ax)
            bar.set_label(f"log10( {algo} / {ref_algo} )")

            _log2_ticks(ax.set_xticks, ax.set_xticklabels, x)
            _log2_ticks(ax.set_yticks, ax.set_yticklabels, y)
            ax.set_xlabel("N (columns)")
            ax.set_ylabel("M (rows)")
            ax.set_title(
                f"{algo} vs {ref_algo} <{dtype}> -- runtime ratio "
                "(red = slower than reference)"
            )

            path = os.path.join(outdir, f"contour_{algo}_vs_{ref_algo}_{dtype}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("wrote", path)


def _log2_ticks(set_ticks, set_labels, values):
    """Place ticks at each integer power of two spanned by ``values``."""
    lo, hi = int(np.floor(values.min())), int(np.ceil(values.max()))
    ticks = list(range(lo, hi + 1))
    set_ticks(ticks)
    set_labels([str(2**t) for t in ticks])


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
        _draw_series(ax, sizes, times, algo)

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
            _draw_series(ax, sizes, times, algo)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("M (rows)")
        ax.set_ylabel("runtime [ms]")
        ax.set_title(f"Skinny QR (N/M = {ratio * 100:g}%): runtime by algorithm")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
        ax.legend()

        path = os.path.join(outdir, f"skinny_comparison_{label}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


def plot_constant_width(records, outdir):
    """Runtime vs M at fixed N, one figure per N."""
    for N in SUPER_SKINNY_NS:
        by_algo = defaultdict(list)
        for rec in records:
            if rec["N"] == N:
                by_algo[rec["algo"]].append(rec)
        if not by_algo:
            print(f"no data for N = {N}; skipping")
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for algo in _algos(records):
            if algo not in by_algo:
                continue
            sizes, times = _series(by_algo[algo])
            _draw_series(ax, sizes, times, algo)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("M (rows)")
        ax.set_ylabel("runtime [ms]")
        ax.set_title(f"Super-skinny QR (N = {N}): runtime by algorithm")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
        ax.legend()

        path = os.path.join(outdir, f"super_skinny_comparison_N{N}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


def plot_constant_elements(records, outdir):
    """Compare different skinnyness ratios for a fixed number of elements (M*N)."""
    # Group records by the total number of elements (M*N)
    by_elements = defaultdict(list)
    for rec in records:
        total_elements = rec["M"] * rec["N"]
        by_elements[total_elements].append(rec)

    # Only plot where at least 4 different aspect ratios are present
    for total_elements, recs in by_elements.items():
        aspect_ratios = {r["N"] / r["M"] for r in recs}
        if len(aspect_ratios) < 4:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for algo in _algos(records):
            algo_results = [
                (r["N"] / r["M"], r["time_ms"]) for r in recs if r["algo"] == algo
            ]
            if not algo_results:
                continue
            # Sort by aspect ratio for plotting
            algo_results.sort(key=lambda x: x[0])
            aspect_ratios, times = zip(*algo_results, strict=True)
            _draw_series(ax, aspect_ratios, times, algo)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("aspect ratio (N / M)")
        ax.set_ylabel("runtime [ms]")
        ax.set_title(
            f"QR with constant elements (M*N = {total_elements}): runtime by algorithm"
        )
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
        ax.legend()

        path = os.path.join(
            outdir, f"constant_elements_comparison_{total_elements}.png"
        )
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
    plot_contour_vs_reference(records, args.outdir)
    plot_square(records, args.outdir)
    plot_skinny(records, args.outdir)
    plot_constant_width(records, args.outdir)
    plot_constant_elements(records, args.outdir)


if __name__ == "__main__":
    main()
