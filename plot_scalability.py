#!/usr/bin/env python3

import csv
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = str(SCRIPT_DIR / "scalability_results.csv")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "scalability_plot.png")


def read_results(path):
    sizes, speedups, seq_times, cuda_times = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["speedup"]:
                continue
            sizes.append(int(row["size"]))
            speedups.append(float(row["speedup"]))
            seq_times.append(float(row["seq_time_avg"]))
            cuda_times.append(float(row["cuda_time_avg"]))
    return sizes, speedups, seq_times, cuda_times


def main():
    parser = argparse.ArgumentParser(description="Plot GPU scalability results")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Input CSV file (default: %(default)s)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output plot file (default: %(default)s)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print("ERROR: {} not found. Run run_scalability.py first.".format(args.input),
              file=sys.stderr)
        sys.exit(1)

    sizes, speedups, seq_times, cuda_times = read_results(args.input)

    if not sizes:
        print("ERROR: no valid data rows in CSV.", file=sys.stderr)
        sys.exit(1)

    total_cells = [s * s for s in sizes]

    base_cells = total_cells[0]
    base_speedup = speedups[0]
    ideal_speedups = [base_speedup * (c / base_cells) for c in total_cells]

    # ── Figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(sizes, speedups, "o-", color="#2563eb", linewidth=2,
            markersize=7, zorder=3, label="Achieved speedup")
    ax.plot(sizes, ideal_speedups, "--", color="#dc2626", linewidth=1.5,
            zorder=2, label="Ideal scaling")

    ax.set_xlabel("Problem size", fontsize=16)
    ax.set_ylabel("Speedup", fontsize=16)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: "{:g}".format(v)))
    ax.set_xticks(sizes)

    ax.tick_params(axis="both", which="major", labelsize=14)

    ax.legend(fontsize=14, loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    for s, sp in zip(sizes, speedups):
        ax.annotate("{:.1f}x".format(sp), (s, sp),
                    textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=11, color="#2563eb",
                    fontweight="bold")

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print("Plot saved to {}".format(args.output))


if __name__ == "__main__":
    main()
