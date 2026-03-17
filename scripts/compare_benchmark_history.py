"""Compare two benchmark runs and detect latency/memory regressions.

Usage
-----
    python scripts/compare_benchmark_history.py artifacts/benchmark_history.json

By default, compares the second-to-last run against the last run in the JSON
history and prints absolute/percent deltas for each optimization variant.

Exit codes:
    0: Comparison completed and thresholds passed (or threshold checks disabled)
    2: Threshold regression detected when --fail-on-regression is set
    3: Invalid input/history format
"""

from __future__ import annotations

import argparse
import json
from math import inf
from pathlib import Path
from typing import Any


def _load_runs(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        raise ValueError(f"History file not found: {history_path}")

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("History JSON must contain a top-level object")

    if isinstance(payload.get("history"), list):
        return [run for run in payload["history"] if isinstance(run, dict)]

    if isinstance(payload.get("rows"), list):
        # Supports single-run export files from visualize_benchmark.py
        return [payload]

    raise ValueError("No runs found. Expected either top-level 'history' or 'rows'.")


def _select_run(runs: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not runs:
        raise ValueError("No benchmark runs available in file")

    try:
        return runs[index]
    except IndexError as exc:
        raise ValueError(
            f"Run index {index} is out of range for {len(runs)} available runs"
        ) from exc


def _rows_by_variant(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = run.get("rows", [])
    if not isinstance(rows, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("variant")
        if isinstance(name, str):
            out[name] = row
    return out


def _pct_delta(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else inf
    return ((candidate - baseline) / baseline) * 100.0


def compare_runs(
    baseline_run: dict[str, Any], candidate_run: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = _rows_by_variant(baseline_run)
    cand_rows = _rows_by_variant(candidate_run)

    if not base_rows or not cand_rows:
        raise ValueError("Both runs must contain non-empty 'rows' with variant metrics")

    ordered_variants = [
        row.get("variant")
        for row in baseline_run.get("rows", [])
        if isinstance(row, dict) and isinstance(row.get("variant"), str)
    ]
    variants = [v for v in ordered_variants if v in cand_rows]

    if not variants:
        raise ValueError("No common variants found between baseline and candidate runs")

    metrics = ["mean_ms", "p95_ms", "p99_ms", "peak_gpu_mb"]
    rows: list[dict[str, Any]] = []

    worst_p95_pct = float("-inf")
    worst_peak_pct = float("-inf")

    for variant in variants:
        b = base_rows[variant]
        c = cand_rows[variant]
        row: dict[str, Any] = {"variant": variant}

        for metric in metrics:
            b_val = float(b.get(metric, 0.0))
            c_val = float(c.get(metric, 0.0))
            d_val = c_val - b_val
            d_pct = _pct_delta(b_val, c_val)

            row[f"baseline_{metric}"] = round(b_val, 3)
            row[f"candidate_{metric}"] = round(c_val, 3)
            row[f"delta_{metric}"] = round(d_val, 3)
            row[f"delta_{metric}_pct"] = round(d_pct, 3) if d_pct != inf else inf

            if metric == "p95_ms":
                worst_p95_pct = max(worst_p95_pct, d_pct)
            if metric == "peak_gpu_mb":
                worst_peak_pct = max(worst_peak_pct, d_pct)

        rows.append(row)

    summary = {
        "common_variant_count": len(rows),
        "worst_p95_regression_pct": worst_p95_pct,
        "worst_peak_mem_regression_pct": worst_peak_pct,
        "baseline_run": baseline_run.get("run", {}),
        "candidate_run": candidate_run.get("run", {}),
    }

    return rows, summary


def print_comparison(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("Benchmark regression comparison")
    print("  Columns show candidate - baseline deltas")
    print(
        "\n"
        "  "
        f"{'Variant':<30} {'dP95(ms)':>10} {'dP95(%)':>10} {'dMem(MB)':>10} {'dMem(%)':>10}"
    )
    print("  " + "-" * 78)

    for row in rows:
        print(
            "  "
            f"{row['variant']:<30} "
            f"{row['delta_p95_ms']:>10.3f} "
            f"{row['delta_p95_ms_pct']:>10.3f} "
            f"{row['delta_peak_gpu_mb']:>10.3f} "
            f"{row['delta_peak_gpu_mb_pct']:>10.3f}"
        )

    print("\nWorst regressions")
    print(f"  p95 latency: {summary['worst_p95_regression_pct']:.3f}%")
    print(f"  peak memory: {summary['worst_peak_mem_regression_pct']:.3f}%")


def _severity_from_thresholds(
    *,
    exceeds_p95: bool,
    exceeds_mem: bool,
    worst_p95_pct: float,
    worst_peak_pct: float,
    max_p95_pct: float,
    max_peak_pct: float,
) -> str:
    if exceeds_p95 or exceeds_mem:
        if (max_p95_pct > 0 and worst_p95_pct >= max_p95_pct * 2) or (
            max_peak_pct > 0 and worst_peak_pct >= max_peak_pct * 2
        ):
            return "high"
        return "medium"

    if worst_p95_pct > 0 or worst_peak_pct > 0:
        return "low"

    return "none"


def _severity_label(severity: str) -> str:
    mapping = {
        "high": "BLOCKER",
        "medium": "REGRESSION",
        "low": "WATCH",
        "none": "CLEAN",
    }
    return mapping.get(severity, "UNKNOWN")


def _severity_rank(severity: str) -> int:
    rank = {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }
    return rank.get(severity, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark runs from history JSON")
    parser.add_argument(
        "history_json",
        type=Path,
        nargs="?",
        default=Path("artifacts/benchmark_history.json"),
        help="Path to benchmark history JSON",
    )
    parser.add_argument(
        "--baseline-index",
        type=int,
        default=-2,
        help="Index of baseline run (default: -2)",
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=-1,
        help="Index of candidate run (default: -1)",
    )
    parser.add_argument(
        "--max-p95-regression-pct",
        type=float,
        default=5.0,
        help="Maximum allowed p95 latency regression percentage",
    )
    parser.add_argument(
        "--max-peak-mem-regression-pct",
        type=float,
        default=10.0,
        help="Maximum allowed peak memory regression percentage",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 2 if thresholds are exceeded",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write comparison payload as JSON",
    )
    args = parser.parse_args()

    try:
        runs = _load_runs(args.history_json)
        baseline = _select_run(runs, args.baseline_index)
        candidate = _select_run(runs, args.candidate_index)
        rows, summary = compare_runs(baseline, candidate)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(3) from exc

    print_comparison(rows, summary)

    exceeds_p95 = summary["worst_p95_regression_pct"] > args.max_p95_regression_pct
    exceeds_mem = summary["worst_peak_mem_regression_pct"] > args.max_peak_mem_regression_pct
    passed = not exceeds_p95 and not exceeds_mem
    severity = _severity_from_thresholds(
        exceeds_p95=exceeds_p95,
        exceeds_mem=exceeds_mem,
        worst_p95_pct=summary["worst_p95_regression_pct"],
        worst_peak_pct=summary["worst_peak_mem_regression_pct"],
        max_p95_pct=args.max_p95_regression_pct,
        max_peak_pct=args.max_peak_mem_regression_pct,
    )
    severity_label = _severity_label(severity)

    print("\nThreshold check")
    print(
        f"  p95 <= {args.max_p95_regression_pct:.3f}% : " f"{'PASS' if not exceeds_p95 else 'FAIL'}"
    )
    print(
        f"  peak_mem <= {args.max_peak_mem_regression_pct:.3f}% : "
        f"{'PASS' if not exceeds_mem else 'FAIL'}"
    )
    print(f"  severity: {severity.upper()}")
    print(f"  triage: {severity_label}")
    print("\nMachine-readable status")
    print(f"BENCHMARK_PASSED={'true' if passed else 'false'}")
    print(f"BENCHMARK_SEVERITY={severity}")
    print(f"BENCHMARK_TRIAGE={severity_label}")
    print(f"BENCHMARK_SEVERITY_RANK={_severity_rank(severity)}")
    print(f"BENCHMARK_P95_EXCEEDED={'true' if exceeds_p95 else 'false'}")
    print(f"BENCHMARK_PEAK_MEM_EXCEEDED={'true' if exceeds_mem else 'false'}")

    payload = {
        "rows": rows,
        "summary": {
            **summary,
            "max_p95_regression_pct": args.max_p95_regression_pct,
            "max_peak_mem_regression_pct": args.max_peak_mem_regression_pct,
            "passed": passed,
            "severity": severity,
            "severity_label": severity_label,
            "severity_rank": _severity_rank(severity),
            "threshold_exceeded": {
                "p95": exceeds_p95,
                "peak_mem": exceeds_mem,
            },
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Comparison JSON saved to: {args.output_json}")

    if args.fail_on_regression and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
