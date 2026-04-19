#!/usr/bin/env python3
"""
Scalability benchmark: runs flood_seq and flood_cuda across multiple problem
sizes, records runtimes, and writes results to a CSV file.

The sequential version runs directly on the head node.
The CUDA version is dispatched to a GPU node via prun.

Usage:
    python3 run_scalability.py
    python3 run_scalability.py --sizes 64 128 256 512 1024 --runs 3
"""

import subprocess
import sys
import re
import csv
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FLOOD_SEQ = str(SCRIPT_DIR / "flood_seq")
FLOOD_CUDA = str(SCRIPT_DIR / "flood_cuda")
RESULTS_FILE = str(SCRIPT_DIR / "scalability_results.csv")

# Fixed simulation parameters (mountain scenario)
SIM_ARGS = "M 0.0000001 200 10 30 70 45 240 40 3 80 32 35 83766"

SIZES = [64, 128, 256, 512, 768, 1024, 1536, 2048]
NUM_RUNS = 5


def run_experiment(executable, rows, cols, use_prun=False):
    """Run the flood simulation and return the measured runtime in seconds."""
    if use_prun:
        cmd = "prun -t 05:00 -np 1 -native '-C gpunode' {} {} {} {}".format(
            executable, rows, cols, SIM_ARGS)
    else:
        cmd = "{} {} {} {}".format(executable, rows, cols, SIM_ARGS)

    try:
        result = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: {} {}x{}".format(executable, rows, cols),
              file=sys.stderr)
        return None

    if result.returncode != 0:
        print("  ERROR ({} {}x{}): {}".format(executable, rows, cols,
              result.stderr.strip()), file=sys.stderr)
        return None

    match = re.search(r"Time:\s+([\d.]+)", result.stdout)
    if not match:
        print("  Could not parse time from: {}".format(executable),
              file=sys.stderr)
        print("  stdout: {}".format(result.stdout[:500]), file=sys.stderr)
        return None

    return float(match.group(1))


def write_csv(path, results):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "size", "total_cells",
            "seq_time_avg", "cuda_time_avg", "speedup",
            "seq_times_raw", "cuda_times_raw",
        ])
        for r in results:
            writer.writerow([
                r["size"],
                r["total_cells"],
                "{:.6f}".format(r["seq_time"]) if r["seq_time"] is not None else "",
                "{:.6f}".format(r["cuda_time"]) if r["cuda_time"] is not None else "",
                "{:.4f}".format(r["speedup"]) if r["speedup"] is not None else "",
                ";".join("{:.6f}".format(t) for t in r["seq_times_raw"]),
                ";".join("{:.6f}".format(t) for t in r["cuda_times_raw"]),
            ])


def main():
    parser = argparse.ArgumentParser(description="Run flood scalability benchmarks")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=SIZES,
        help="Square grid sizes to benchmark (default: %(default)s)")
    parser.add_argument(
        "--runs", type=int, default=NUM_RUNS,
        help="Number of repetitions per configuration (default: %(default)s)")
    parser.add_argument(
        "--output", type=str, default=RESULTS_FILE,
        help="Output CSV file path (default: %(default)s)")
    parser.add_argument(
        "--skip-seq", action="store_true",
        help="Skip sequential runs")
    parser.add_argument(
        "--skip-cuda", action="store_true",
        help="Skip CUDA runs")
    parser.add_argument(
        "--no-prun", action="store_true",
        help="Run CUDA binary directly (when already on a GPU node)")
    args = parser.parse_args()

    for binary in [FLOOD_SEQ, FLOOD_CUDA]:
        if not Path(binary).exists():
            print("ERROR: {} not found. Run 'make all' first.".format(binary),
                  file=sys.stderr)
            sys.exit(1)

    results = []

    for size in args.sizes:
        rows = cols = size
        total_cells = rows * cols

        seq_times = []
        cuda_times = []

        if not args.skip_seq:
            print("[SEQ ] {}x{} ({:>10,} cells) ...".format(rows, cols, total_cells),
                  end="", flush=True)
            for _ in range(args.runs):
                t = run_experiment(FLOOD_SEQ, rows, cols, use_prun=False)
                if t is not None:
                    seq_times.append(t)
                    print(" {:.3f}s".format(t), end="", flush=True)
            print()

        if not args.skip_cuda:
            use_prun = not args.no_prun
            print("[CUDA] {}x{} ({:>10,} cells) ...".format(rows, cols, total_cells),
                  end="", flush=True)
            for _ in range(args.runs):
                t = run_experiment(FLOOD_CUDA, rows, cols, use_prun=use_prun)
                if t is not None:
                    cuda_times.append(t)
                    print(" {:.3f}s".format(t), end="", flush=True)
            print()

        seq_avg = sum(seq_times) / len(seq_times) if seq_times else None
        cuda_avg = sum(cuda_times) / len(cuda_times) if cuda_times else None

        if seq_avg is not None and cuda_avg is not None and cuda_avg > 0:
            speedup = seq_avg / cuda_avg
        else:
            speedup = None

        results.append({
            "size": size,
            "total_cells": total_cells,
            "seq_time": seq_avg,
            "cuda_time": cuda_avg,
            "speedup": speedup,
            "seq_times_raw": seq_times,
            "cuda_times_raw": cuda_times,
        })

        write_csv(args.output, results)
        if speedup is not None:
            print("  -> Speedup: {:.2f}x".format(speedup))
        else:
            print("  -> Speedup: N/A")
        print()

    print("\nResults written to {}".format(args.output))
    print("Run 'python3 plot_scalability.py' to generate the plot.")


if __name__ == "__main__":
    main()
