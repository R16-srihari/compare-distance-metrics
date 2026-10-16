"""Timing experiment protocol 2: repeat runs per metric per K.

This script builds the Beta product GMM once per K (untimed) and then
performs multiple timed calls (default 5) to each metric's distance function
(GMM with itself). It records the timing for each run and writes two CSV files:
one with raw per-run data and one with summary statistics (mean, std, min, median).
"""

from __future__ import annotations

import sys
from typing import Sequence

import numpy as np

from src.airm_gmm import gmm_airm
from src.beta_gmm import build_beta_product_gmm
from src.gmm_types import GMM
from src.hellinger_gmm import gmm_hellinger
from src.timing_utils import summarize, thread_env_report, timed, write_csv


def run_repeats(
    k_list: Sequence[int],
    metrics: Sequence[str],
    repeats: int = 5,
    chunk: int = 64,
    solver: str = "pot",
) -> tuple[list[dict], list[dict]]:
    """Run the repeat timing experiment.

    Parameters
    ----------
    k_list : sequence of int
        List of K values to test.
    metrics : sequence of str
        List of metric names to run ('hellinger', 'airm').
    repeats : int, optional
        Number of repetitions per (K, metric) (default: 5).
    chunk : int, optional
        Chunk size for Hellinger cost matrix (default: 64).
    solver : str, optional
        Solver for Hellinger transport LP (default: 'pot').

    Returns
    -------
    tuple[list[dict], list[dict]]
        (raw_rows, summary_rows) where raw_rows contains one dict per run and
        summary_rows contains the aggregated statistics.
    """
    raw_results = []
    for k in k_list:
        # Build GMM once (untimed)
        gmm = build_beta_product_gmm(k=k)
        gmm.validate()  # Ensure it's valid

        for metric in metrics:
            for run_idx in range(1, repeats + 1):
                # Time the distance computation (GMM with itself)
                if metric == "hellinger":
                    func = lambda: gmm_hellinger(gmm, gmm, chunk=chunk, solver=solver)
                elif metric == "airm":
                    func = lambda: gmm_airm(gmm, gmm, method="auto")
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                result, elapsed = timed(func)
                # Extract the relevant fields from the result
                if metric == "hellinger":
                    stage_times = result.stage_times
                    distance = result.distance
                else:  # airm
                    stage_times = result.stage_times
                    distance = result.distance

                raw_results.append(
                    {
                        "k": k,
                        "metric": metric,
                        "run_index": run_idx,
                        "total_s": elapsed,
                        "stage_times": stage_times,
                        "distance": distance,
                    }
                )

    # Compute summary statistics
    summary_results = summarize(raw_results)
    return raw_results, summary_results


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for benchmark_repeat_runs.py.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark repeat runs per metric per K"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000],
        help="List of K values (default: 100 200 500 1000)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["hellinger", "airm"],
        help="Metrics to benchmark (default: hellinger airm)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repetitions per (K, metric) (default: 5)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="Chunk size for Hellinger cost matrix (default: 64)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["pot", "scipy"],
        default="pot",
        help="Solver for Hellinger transport LP (default: pot)",
    )
    parser.add_argument(
        "--out-raw",
        type=str,
        default="results/benchmark_repeat.csv",
        help="Output CSV file for raw runs (default: results/benchmark_repeat.csv)",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="results/benchmark_repeat_summary.csv",
        help="Output CSV file for summary (default: results/benchmark_repeat_summary.csv)",
    )
    args = parser.parse_args(argv)

    # Print thread environment report
    print("Thread environment:")
    for var, val in thread_env_report().items():
        print(f"  {var}: {val}")
    print()

    # Run the benchmark
    raw_results, summary_results = run_repeats(
        k_list=args.k,
        metrics=args.metrics,
        repeats=args.repeats,
        chunk=args.chunk,
        solver=args.solver,
    )

    # Print raw results as a table (first few rows)
    from src.timing_utils import format_table

    # Prepare rows for display (we'll show total_s and distance)
    display_rows = []
    for r in raw_results[: min(10, len(raw_results))]:  # show first 10 rows
        display_rows.append(
            {
                "K": r["k"],
                "Metric": r["metric"],
                "Run": r["run_index"],
                "Total_s": f"{r['total_s']:.10e}",
                "Distance": f"{r['distance']:.10e}",
            }
        )
    columns = ["K", "Metric", "Run", "Total_s", "Distance"]
    print("First 10 raw runs:")
    print(format_table(display_rows, columns))
    print()

    # Print summary results as a table
    display_rows = []
    for r in summary_results:
        display_rows.append(
            {
                "K": r["k"],
                "Metric": r["metric"],
                "Mean_s": f"{r['mean_s']:.10e}",
                "Std_s": f"{r['std_s']:.10e}",
                "Min_s": f"{r['min_s']:.10e}",
                "Median_s": f"{r['median_s']:.10e}",
                "Max_s": f"{r['max_s']:.10e}",
                "Distance": f"{r['distance']:.10e}",
            }
        )
    columns = ["K", "Metric", "Mean_s", "Std_s", "Min_s", "Median_s", "Max_s", "Distance"]
    print("Summary statistics:")
    print(format_table(display_rows, columns))
    print()

    # Write raw CSV with only k, metric, run_index, total_s, distance
    import csv
    from pathlib import Path
    raw_path = Path(args.out_raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "metric", "run_index", "total_s", "distance"])
        writer.writeheader()
        for row in raw_results:
            writer.writerow({
                "k": row["k"],
                "metric": row["metric"],
                "run_index": row.get("run_index", 1),
                "total_s": f"{float(row['total_s']):.10e}",
                "distance": f"{float(row['distance']):.10e}",
            })
    # Write summary CSV with mean, min, max, distance
    summary_path = Path(args.out_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "metric", "mean_s", "min_s", "max_s", "distance"])
        writer.writeheader()
        for row in summary_results:
            writer.writerow({
                "k": row["k"],
                "metric": row["metric"],
                "mean_s": f"{float(row['mean_s']):.10e}",
                "min_s": f"{float(row['min_s']):.10e}",
                "max_s": f"{float(row['max_s']):.10e}",
                "distance": f"{float(row['distance']):.10e}",
            })
    print(f"Raw results written to {args.out_raw}")
    print(f"Summary results written to {args.out_summary}")

    # Print ratio of AIRM to Hellinger mean times for each K
    print("\nRatio (AIRM mean / Hellinger mean) per K:")
    for k in args.k:
        # Extract mean times for this K
        hellinger_mean = None
        airm_mean = None
        for row in summary_results:
            if row["k"] == k:
                if row["metric"] == "hellinger":
                    hellinger_mean = row["mean_s"]
                elif row["metric"] == "airm":
                    airm_mean = row["mean_s"]
        if hellinger_mean is not None and airm_mean is not None and hellinger_mean > 0:
            ratio = airm_mean / hellinger_mean
            print(f"  K={k}: {ratio:.4f}")
        else:
            print(f"  K={k}: insufficient data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
