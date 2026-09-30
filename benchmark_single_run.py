"""Timing experiment protocol 1: single timed run per metric per K.

This script builds the Beta product GMM once per K (untimed) and then
performs exactly one timed call to each metric's distance function
(GMM with itself). It records the timing and writes a CSV file.
"""

from __future__ import annotations

import sys
from typing import Sequence

import numpy as np

from src.airm_gmm import gmm_airm
from src.beta_gmm import build_beta_product_gmm
from src.gmm_types import GMM
from src.hellinger_gmm import gmm_hellinger
from src.timing_utils import thread_env_report, timed, write_csv


def run_single(
    k_list: Sequence[int],
    metrics: Sequence[str],
    chunk: int = 64,
    solver: str = "pot",
) -> list[dict]:
    """Run the single timing experiment.

    Parameters
    ----------
    k_list : sequence of int
        List of K values to test.
    metrics : sequence of str
        List of metric names to run ('hellinger', 'airm').
    chunk : int, optional
        Chunk size for Hellinger cost matrix (default: 64).
    solver : str, optional
        Solver for Hellinger transport LP (default: 'pot').

    Returns
    -------
    list of dict
        Each dict contains the results for one (K, metric) pair.
    """
    results = []
    for k in k_list:
        # Build GMM once (untimed)
        gmm = build_beta_product_gmm(k=k)
        gmm.validate()  # Ensure it's valid

        for metric in metrics:
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

            results.append(
                {
                    "k": k,
                    "metric": metric,
                    "total_s": elapsed,
                    "stage_times": stage_times,
                    "distance": distance,
                }
            )
    return results


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for benchmark_single_run.py.

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
        description="Benchmark single run per metric per K"
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
        "--out",
        type=str,
        default="results/benchmark_single.csv",
        help="Output CSV file (default: results/benchmark_single.csv)",
    )
    args = parser.parse_args(argv)

    # Print thread environment report
    print("Thread environment:")
    for var, val in thread_env_report().items():
        print(f"  {var}: {val}")
    print()

    # Run the benchmark
    results = run_single(
        k_list=args.k,
        metrics=args.metrics,
        chunk=args.chunk,
        solver=args.solver,
    )

    # Print results as a table
    from src.timing_utils import format_table

    # Prepare rows for display (we'll show total_s and distance)
    display_rows = []
    for r in results:
        display_rows.append(
            {
                "K": r["k"],
                "Metric": r["metric"],
                "Total_s": f"{r['total_s']:.10e}",
                "Distance": f"{r['distance']:.10e}",
            }
        )
    columns = ["K", "Metric", "Total_s", "Distance"]
    print(format_table(display_rows, columns))
    print()

    # Write CSV with only k, metric, total_s, distance
    import csv
    from pathlib import Path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "metric", "total_s", "distance"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "k": r["k"],
                "metric": r["metric"],
                "total_s": f"{float(r['total_s']):.10e}",
                "distance": f"{float(r['distance']):.10e}",
            })
    print(f"Results written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
