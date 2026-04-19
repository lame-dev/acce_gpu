#!/usr/bin/env python3
"""
Profile benchmark: runs flood_cuda (compiled with -DPROFILE) across all test
input files, parses per-category CUDA event timings from stderr, and writes
results to a CSV file.

Usage:
    python3 run_profile.py
    python3 run_profile.py --runs 3 --no-prun
"""

import subprocess
import sys
import re
import csv
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FLOOD_CUDA = str(SCRIPT_DIR / "flood_cuda")
RESULTS_FILE = str(SCRIPT_DIR / "profile_results.csv")
TEST_DIR = SCRIPT_DIR / "test_files"

SCENARIOS = [
    "debug",
    "small_mountains",
    "medium_higher_dam",
    "medium_lower_dam",
    "custom_clouds",
    "large_mountains",
]

NUM_RUNS = 3

PROFILE_RE = re.compile(
    r"PROFILE:\s+"
    r"alloc_init=([\d.]+)\s+"
    r"rainfall=([\d.]+)\s+"
    r"spillage=([\d.]+)\s+"
    r"propagation=([\d.]+)\s+"
    r"readback=([\d.]+)"
)
CATEGORIES = ["alloc_init", "rainfall", "spillage", "propagation", "readback"]


def run_once(input_file, use_prun):
    args_text = input_file.read_text().strip()
    if use_prun:
        cmd = "prun -t 10:00 -np 1 -native '-C gpunode,TitanX' bash -c '{} {} 2>&1'".format(
            FLOOD_CUDA, args_text)
    else:
        cmd = "{} {} 2>&1".format(FLOOD_CUDA, args_text)

    try:
        result = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", file=sys.stderr)
        return None

    combined = result.stdout + "\n" + result.stderr
    m = PROFILE_RE.search(combined)
    if not m:
        print("  Could not parse PROFILE line", file=sys.stderr)
        print("  output: {}".format(combined[:500]), file=sys.stderr)
        return None

    return {cat: float(m.group(i + 1)) for i, cat in enumerate(CATEGORIES)}


def write_csv(path, results):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario"] + CATEGORIES + ["total"])
        for r in results:
            total = sum(r[cat] for cat in CATEGORIES)
            writer.writerow(
                [r["scenario"]] +
                ["{:.6f}".format(r[cat]) for cat in CATEGORIES] +
                ["{:.6f}".format(total)]
            )


def main():
    parser = argparse.ArgumentParser(description="Run flood_cuda profile benchmarks")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help="Number of repetitions per scenario (default: %(default)s)")
    parser.add_argument("--output", type=str, default=RESULTS_FILE,
                        help="Output CSV file (default: %(default)s)")
    parser.add_argument("--no-prun", action="store_true",
                        help="Run directly (when already on a GPU node)")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                        help="Scenarios to run (default: all)")
    args = parser.parse_args()

    if not Path(FLOOD_CUDA).exists():
        print("ERROR: {} not found. Run 'make clean && make profile' first.".format(FLOOD_CUDA),
              file=sys.stderr)
        sys.exit(1)

    results = []

    for scenario in args.scenarios:
        input_file = TEST_DIR / "{}.in".format(scenario)
        if not input_file.exists():
            print("WARNING: {} not found, skipping".format(input_file), file=sys.stderr)
            continue

        print("[PROFILE] {} ...".format(scenario), end="", flush=True)

        accum = {cat: [] for cat in CATEGORIES}
        for run in range(args.runs):
            timings = run_once(input_file, use_prun=not args.no_prun)
            if timings is not None:
                for cat in CATEGORIES:
                    accum[cat].append(timings[cat])
                total = sum(timings[cat] for cat in CATEGORIES)
                print(" {:.3f}s".format(total), end="", flush=True)
        print()

        if not accum[CATEGORIES[0]]:
            print("  -> No valid runs, skipping")
            continue

        row = {"scenario": scenario}
        for cat in CATEGORIES:
            row[cat] = sum(accum[cat]) / len(accum[cat])
        results.append(row)

        write_csv(args.output, results)
        total = sum(row[cat] for cat in CATEGORIES)
        print("  -> Total: {:.4f}s  (alloc={:.4f} rain={:.4f} spill={:.4f} prop={:.4f} read={:.4f})".format(
            total, row["alloc_init"], row["rainfall"], row["spillage"],
            row["propagation"], row["readback"]))
        print()

    print("\nResults written to {}".format(args.output))
    print("Run 'python3 plot_profile.py' to generate the plot.")


if __name__ == "__main__":
    main()
