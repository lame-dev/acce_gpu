#!/usr/bin/env python3

"""Average the sequential, CUDA, and nvprof evaluation results.

Expected layout:

results/<scenario>/01/sequential/result.out
results/<scenario>/01/cuda/result.out
...
results/<scenario>/05/sequential/result.out
results/<scenario>/05/cuda/result.out

The script can be pointed at a single scenario directory or at the results
root. When given the root, it analyzes every scenario folder found there.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable
import re
import argparse


NVPROF_ROW_RE = re.compile(
    r"^\s*(?P<time_pct>\S+)\s+(?P<time>\S+)\s+(?P<calls>\S+)\s+"
    r"(?P<avg>\S+)\s+(?P<min>\S+)\s+(?P<max>\S+)\s+(?P<name>.+?)\s*$"
)


def parse_result_file(path: Path) -> tuple[float, list[float]]:
    time_value = None
    result_values = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("Time:"):
                time_value = float(line.split(":", 1)[1].strip())
            elif line.startswith("Result:"):
                result_values = [float(value.strip()) for value in line.split(":", 1)[1].split(",")]

    if time_value is None or result_values is None:
        raise ValueError(f"{path} does not contain both a Time line and a Result line")

    return time_value, result_values


def parse_time_value(token: str) -> float:
    token = token.strip()
    if token.endswith("ms"):
        return float(token[:-2]) / 1000.0
    if token.endswith("us"):
        return float(token[:-2]) / 1_000_000.0
    if token.endswith("ns"):
        return float(token[:-2]) / 1_000_000_000.0
    if token.endswith("s"):
        return float(token[:-1])
    return float(token)


def parse_count_value(token: str) -> float:
    return float(token.replace(",", ""))


def parse_nvprof_eval(path: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    summary: dict[str, float] = {}
    rows: dict[str, dict[str, float]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            if not line:
                continue

            stripped = line.lstrip()
            if stripped.startswith("GPU activities:") or stripped.startswith("API calls:"):
                _, _, remainder = stripped.partition(":")
                match = NVPROF_ROW_RE.match(remainder)
                if match is None:
                    continue

                name = match.group("name")
                rows[name] = {
                    "time_pct": float(match.group("time_pct").rstrip("%")),
                    "time": parse_time_value(match.group("time")),
                    "calls": parse_count_value(match.group("calls")),
                    "avg": parse_time_value(match.group("avg")),
                    "min": parse_time_value(match.group("min")),
                    "max": parse_time_value(match.group("max")),
                }
            elif line.startswith("Time:"):
                summary["time"] = float(line.split(":", 1)[1].strip())

    return summary, rows


def average(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected)


def format_values(values: Iterable[float]) -> str:
    return ", ".join(f"{value:.6f}" for value in values)


def format_cell(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    print(render_row(headers))
    print(separator)
    for row in rows:
        print(render_row(row))


def analyze_impl(scenario_dir: Path, implementation: str) -> tuple[float, list[float]]:
    times: list[float] = []
    runs: list[list[float]] = []

    for index in range(1, 6):
        result_file = scenario_dir / f"{index:02d}" / implementation / "result.out"
        if not result_file.exists():
            raise FileNotFoundError(f"Missing result file: {result_file}")

        time_value, result_values = parse_result_file(result_file)
        times.append(time_value)
        runs.append(result_values)

    averaged_results = [average(values) for values in zip(*runs)]
    return average(times), averaged_results


def analyze_nvprof(scenario_dir: Path) -> tuple[float, dict[str, dict[str, float]]]:
    times: list[float] = []
    per_run_rows: list[dict[str, dict[str, float]]] = []

    for index in range(1, 6):
        eval_file = scenario_dir / f"{index:02d}" / "nvprof_eval.txt"
        if not eval_file.exists():
            raise FileNotFoundError(f"Missing nvprof file: {eval_file}")

        summary, rows = parse_nvprof_eval(eval_file)
        if "time" in summary:
            times.append(summary["time"])
        per_run_rows.append(rows)

    aggregated: dict[str, dict[str, float]] = {}
    row_names = sorted({name for rows in per_run_rows for name in rows})

    for name in row_names:
        collected = [rows[name] for rows in per_run_rows if name in rows]
        aggregated[name] = {
            "time_pct": average(row["time_pct"] for row in collected),
            "time": average(row["time"] for row in collected),
            "calls": average(row["calls"] for row in collected),
            "avg": average(row["avg"] for row in collected),
            "min": average(row["min"] for row in collected),
            "max": average(row["max"] for row in collected),
        }

    return average(times) if times else 0.0, aggregated


def resolve_scenario_dir(argument: str) -> Path:
    candidate = Path(argument)
    if candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / argument
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not find scenario directory: {argument}")


def find_scenarios(results_root: Path) -> list[Path]:
    return sorted(
        [path for path in results_root.iterdir() if path.is_dir() and path.name != "__pycache__"],
        key=lambda path: path.name,
    )


def print_scenario_report(scenario_dir: Path) -> dict[str, object]:
    sequential_time, sequential_results = analyze_impl(scenario_dir, "sequential")
    cuda_time, cuda_results = analyze_impl(scenario_dir, "cuda")
    nvprof_time, nvprof_rows = analyze_nvprof(scenario_dir)

    return {
        "scenario": scenario_dir.name,
        "sequential_time": sequential_time,
        "cuda_time": cuda_time,
        "speedup": sequential_time / cuda_time if cuda_time > 0 else 0.0,
        "sequential_results": sequential_results,
        "cuda_results": cuda_results,
        "nvprof_time": nvprof_time,
        "nvprof_rows": nvprof_rows,
    }


def print_summary_table(reports: list[dict[str, object]]) -> None:
    print("Summary")
    rows = []
    for report in reports:
        rows.append(
            [
                str(report["scenario"]),
                f"{report['sequential_time']:.6f}",
                f"{report['cuda_time']:.6f}",
                f"{report['speedup']:.2f}x",
            ]
        )

    print_table(["Scenario", "Seq time", "CUDA time", "Speedup"], rows)
    print()


def get_top_nvprof_entry(nvprof_rows: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    return max(nvprof_rows.items(), key=lambda item: item[1]["time"])


def print_nvprof_summary_table(reports: list[dict[str, object]]) -> None:
    print("nvprof summary")
    rows = []
    for report in reports:
        nvprof_rows = report["nvprof_rows"]
        if nvprof_rows:
            top_name, top_metrics = get_top_nvprof_entry(nvprof_rows)
            rows.append(
                [
                    str(report["scenario"]),
                    top_name,
                    f"{top_metrics['time']:.6f}",
                    f"{top_metrics['calls']:.0f}",
                    str(len(nvprof_rows)),
                ]
            )
        else:
            rows.append([str(report["scenario"]), "-", "-", "-", "-"])

    print_table(["Scenario", "Top entry", "Top time", "Calls", "Rows"], rows)
    print()


def print_run_table(report: dict[str, object]) -> None:
    print(f"Scenario: {report['scenario']}")
    print_table(
        ["Metric", "Sequential", "CUDA"],
        [
            ["Average time", f"{report['sequential_time']:.6f}", f"{report['cuda_time']:.6f}"],
            ["Iterations", format_cell(report['sequential_results'][0]), format_cell(report['cuda_results'][0])],
            ["Max flow iteration", format_cell(report['sequential_results'][1]), format_cell(report['cuda_results'][1])],
            ["Max flow amount", f"{report['sequential_results'][2]:.6f}", f"{report['cuda_results'][2]:.6f}"],
            ["Highest water level", f"{report['sequential_results'][3]:.6f}", f"{report['cuda_results'][3]:.6f}"],
            ["Total rain", f"{report['sequential_results'][4]:.6f}", f"{report['cuda_results'][4]:.6f}"],
            ["Final water", f"{report['sequential_results'][5]:.6f}", f"{report['cuda_results'][5]:.6f}"],
            ["Final outflow", f"{report['sequential_results'][6]:.6f}", f"{report['cuda_results'][6]:.6f}"],
        ],
    )
    print()


def print_nvprof_table(report: dict[str, object]) -> None:
    nvprof_rows = report["nvprof_rows"]
    if not nvprof_rows:
        return

    print("nvprof")
    rows = []
    for name, metrics in nvprof_rows.items():
        rows.append(
            [
                name,
                f"{metrics['time_pct']:.2f}%",
                f"{metrics['time']:.6f}",
                f"{metrics['calls']:.0f}",
                f"{metrics['avg']:.6f}",
                f"{metrics['min']:.6f}",
                f"{metrics['max']:.6f}",
            ]
        )

    print_table(["Name", "Time %", "Time", "Calls", "Avg", "Min", "Max"], rows)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze averaged scenario and nvprof results.")
    parser.add_argument("path", nargs="?", help="Scenario directory or results root. Defaults to results/.")
    parser.add_argument("--verbose", action="store_true", help="Show per-scenario detailed tables.")
    args = parser.parse_args()

    target = resolve_scenario_dir(args.path) if args.path else Path(__file__).resolve().parent

    scenario_dirs = [target] if (target / "01").is_dir() else find_scenarios(target)
    if not scenario_dirs:
        print(f"No scenario directories found under {target}")
        return 1

    reports = [print_scenario_report(scenario_dir) for scenario_dir in scenario_dirs]
    print_summary_table(reports)
    print_nvprof_summary_table(reports)

    if args.verbose:
        for report in reports:
            print_run_table(report)
            print_nvprof_table(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())