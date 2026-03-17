from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_benchmark_history.py"


def _write_history(path: Path) -> None:
    payload = {
        "history": [
            {
                "run": {"timestamp_utc": "2026-03-17T10:00:00Z", "model_id": "baseline"},
                "rows": [
                    {
                        "variant": "Baseline (FP16)",
                        "mean_ms": 120.0,
                        "p50_ms": 110.0,
                        "p95_ms": 150.0,
                        "p99_ms": 165.0,
                        "peak_gpu_mb": 2100.0,
                    },
                    {
                        "variant": "torch.compile (FP16)",
                        "mean_ms": 95.0,
                        "p50_ms": 90.0,
                        "p95_ms": 120.0,
                        "p99_ms": 130.0,
                        "peak_gpu_mb": 2200.0,
                    },
                ],
            },
            {
                "run": {"timestamp_utc": "2026-03-17T11:00:00Z", "model_id": "candidate"},
                "rows": [
                    {
                        "variant": "Baseline (FP16)",
                        "mean_ms": 123.0,
                        "p50_ms": 112.0,
                        "p95_ms": 156.0,
                        "p99_ms": 170.0,
                        "peak_gpu_mb": 2120.0,
                    },
                    {
                        "variant": "torch.compile (FP16)",
                        "mean_ms": 93.0,
                        "p50_ms": 88.0,
                        "p95_ms": 118.0,
                        "p99_ms": 126.0,
                        "peak_gpu_mb": 2210.0,
                    },
                ],
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_benchmark_history_cli_pass(tmp_path: Path) -> None:
    history = tmp_path / "benchmark_history.json"
    output_json = tmp_path / "comparison.json"
    _write_history(history)

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(history),
            "--max-p95-regression-pct",
            "10",
            "--max-peak-mem-regression-pct",
            "10",
            "--output-json",
            str(output_json),
            "--fail-on-regression",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Threshold check" in proc.stdout
    assert "severity:" in proc.stdout.lower()
    assert "BENCHMARK_PASSED=true" in proc.stdout
    assert "BENCHMARK_TRIAGE=" in proc.stdout
    assert output_json.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    assert summary.get("passed") is True
    assert summary.get("severity") in {"none", "low"}
    assert summary.get("severity_label") in {"CLEAN", "WATCH"}
    assert isinstance(summary.get("severity_rank"), int)


def test_compare_benchmark_history_cli_fail_on_regression(tmp_path: Path) -> None:
    history = tmp_path / "benchmark_history.json"
    _write_history(history)

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(history),
            "--max-p95-regression-pct",
            "1",
            "--max-peak-mem-regression-pct",
            "0.1",
            "--fail-on-regression",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "FAIL" in proc.stdout
    assert "severity:" in proc.stdout.lower()
    assert "triage:" in proc.stdout.lower()
    assert "BENCHMARK_PASSED=false" in proc.stdout
    assert "BENCHMARK_P95_EXCEEDED=true" in proc.stdout


def test_compare_benchmark_history_cli_invalid_input(tmp_path: Path) -> None:
    history = tmp_path / "missing.json"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(history)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 3
    assert "Error:" in proc.stdout
