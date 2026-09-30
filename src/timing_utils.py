"""Timing utilities for the benchmark.

This module provides helper functions for timing, CSV writing, summarizing,
and formatting tables.
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from pathlib import Path
from typing import Any


def timed(func, *args, **kwargs) -> tuple[Any, float]:
    """Wrap a call between two time.perf_counter() reads.

    Parameters
    ----------
    func : callable
        The function to time.
    *args, **kwargs
        Arguments to pass to func.

    Returns
    -------
    tuple[Any, float]
        (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def write_csv(
    path: str | Path,
    rows: list[dict],
    stage_order: list[str],
) -> None:
    """Write a list of dictionaries to a CSV file.

    The CSV schema is fixed to have columns:
        k, metric, total_s, stage_1_name, stage_1_s, stage_2_name, stage_2_s, distance
    plus run_index for repeat runs.

    Parameters
    ----------
    path : str or Path
        Output CSV file path.
    rows : list of dict
        Each dict must contain at least:
            k, metric, run_index (optional), total_s, stage_times (dict), distance.
    stage_order : list of str
        Ordered list of stage names (e.g., ['cost_matrix', 'transport_lp']).
        Only the first two stages are used; if there are more, they are ignored.
        If there are fewer than two, missing stages are filled with empty strings and 0.0.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine if we have run_index (i.e., repeat runs)
    has_run_index = any("run_index" in row for row in rows)

    fieldnames = [
        "k",
        "metric",
    ]
    if has_run_index:
        fieldnames.append("run_index")
    fieldnames.extend([
        "total_s",
        "stage_1_name",
        "stage_1_s",
        "stage_2_name",
        "stage_2_s",
        "distance",
    ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {
                "k": row["k"],
                "metric": row["metric"],
            }
            if has_run_index:
                out_row["run_index"] = row.get("run_index", 1)
            # Format floats in scientific notation with 10 decimal places
            out_row["total_s"] = f"{float(row['total_s']):.10e}"
            out_row["distance"] = f"{float(row['distance']):.10e}"

            # Flatten stage_times
            stage_times = row.get("stage_times", {})
            # Take the first two stages in stage_order
            stage_names = stage_order[:2] + [""] * max(0, 2 - len(stage_order))
            stage_values = [stage_times.get(name, 0.0) for name in stage_order[:2]]
            # Pad to two stages
            while len(stage_names) < 2:
                stage_names.append("")
                stage_values.append(0.0)
            out_row["stage_1_name"] = stage_names[0]
            out_row["stage_1_s"] = f"{float(stage_values[0]):.10e}"
            out_row["stage_2_name"] = stage_names[1]
            out_row["stage_2_s"] = f"{float(stage_values[1]):.10e}"

            writer.writerow(out_row)


def summarize(rows: list[dict]) -> list[dict]:
    """Group rows by (k, metric) and compute summary statistics.

    Parameters
    ----------
    rows : list of dict
        Each dict must contain:
            k, metric, total_s, distance (and optionally run_index).

    Returns
    -------
    list of dict
        Each dict contains:
            k, metric, mean_s, std_s, min_s, median_s, max_s, distance
        where distance is the value from the first row in the group (assumed
        to be the same across runs for the same GMM).
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for row in rows:
        key = (row["k"], row["metric"])
        groups[key].append(row)

    summary_rows = []
    for (k, metric), group in groups.items():
        total_times = [r["total_s"] for r in group]
        # Distance should be the same across runs; take the first
        distance = group[0]["distance"]
        mean_s = statistics.mean(total_times)
        if len(total_times) > 1:
            std_s = statistics.stdev(total_times)
        else:
            std_s = 0.0
        min_s = min(total_times)
        median_s = statistics.median(total_times)
        max_s = max(total_times)
        summary_rows.append(
            {
                "k": k,
                "metric": metric,
                "mean_s": mean_s,
                "std_s": std_s,
                "min_s": min_s,
                "median_s": median_s,
                "max_s": max_s,
                "distance": distance,
            }
        )
    return summary_rows


def format_table(rows: list[dict], columns: list[str]) -> str:
    """Format a list of dictionaries as a fixed-width table.

    Parameters
    ----------
    rows : list of dict
        Each dict represents a row; keys must match the column names.
    columns : list of str
        Column names in the order they should appear.

    Returns
    -------
    str
        The formatted table as a string.
    """
    if not rows:
        return ""
    # Determine the width for each column
    col_widths = {}
    for col in columns:
        # Header width
        width = len(col)
        # Check each row's value for this column
        for row in rows:
            val = row.get(col, "")
            # Convert to string for width calculation
            str_val = str(val)
            if len(str_val) > width:
                width = len(str_val)
        col_widths[col] = width

    # Build the separator
    sep = " + ".join("-" * (col_widths[col] + 2) for col in columns)
    # Header
    header = " | ".join(
        f" {col:<{col_widths[col]}} " for col in columns
    )
    lines = [sep, header, sep]
    # Rows
    for row in rows:
        line = " | ".join(
            f" {str(row.get(col, '')):<{col_widths[col]}} " for col in columns
        )
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def thread_env_report() -> dict[str, str]:
    """Read thread-related environment variables.

    Returns
    -------
    dict[str, str]
        Mapping of variable name to its value (or "Not set").
    """
    vars_to_check = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    report = {}
    for var in vars_to_check:
        report[var] = os.environ.get(var, "Not set")
    return report