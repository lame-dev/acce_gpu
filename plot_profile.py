#!/usr/bin/env python3
"""
Render a horizontal stacked bar chart showing execution time breakdown
per test scenario from profile_results.csv.
"""

import csv
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = str(SCRIPT_DIR / "profile_results.csv")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "profile_plot.png")

CATEGORIES = ["alloc_init", "rainfall", "spillage", "propagation", "readback"]
LABELS = ["Alloc + init", "Rainfall", "Spillage", "Propagation", "Readback"]
COLORS = ["#6366f1", "#2563eb", "#059669", "#d97706", "#dc2626"]


def read_results(path):
    scenarios = []
    data = {cat: [] for cat in CATEGORIES}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(row["scenario"])
            for cat in CATEGORIES:
                data[cat].append(float(row[cat]))
    return scenarios, data


def main():
    parser = argparse.ArgumentParser(description="Plot execution time breakdown")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Input CSV file (default: %(default)s)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output plot file (default: %(default)s)")
    parser.add_argument("-sec", "--scenario", type=str, default=None,
                        help="Show only this scenario (e.g. large_mountains)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print("ERROR: {} not found. Run run_profile.py first.".format(args.input),
              file=sys.stderr)
        sys.exit(1)

    scenarios, data = read_results(args.input)

    if not scenarios:
        print("ERROR: no data rows in CSV.", file=sys.stderr)
        sys.exit(1)

    if args.scenario:
        if args.scenario not in scenarios:
            print("ERROR: scenario '{}' not in CSV. Available: {}".format(
                args.scenario, ", ".join(scenarios)), file=sys.stderr)
            sys.exit(1)
        idx = scenarios.index(args.scenario)
        scenarios = [scenarios[idx]]
        for cat in CATEGORIES:
            data[cat] = [data[cat][idx]]

    n = len(scenarios)
    single = (n == 1)
    y_pos = np.arange(n)

    fig_height = max(3, 1.2 * n + 2) if not single else 3.5
    fig, ax = plt.subplots(figsize=(9, fig_height))

    lefts = np.zeros(n)
    bars = []
    for cat, label, color in zip(CATEGORIES, LABELS, COLORS):
        vals = np.array(data[cat])
        b = ax.barh(y_pos, vals, left=lefts, height=0.55,
                     color=color, edgecolor="white", linewidth=0.5, label=label)
        bars.append(b)
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenarios)
    ax.invert_yaxis()

    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("Scenario", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)

    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    if single:
        ax.set_ylim(n - 0.5, -1.5)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print("Plot saved to {}".format(args.output))


if __name__ == "__main__":
    main()
