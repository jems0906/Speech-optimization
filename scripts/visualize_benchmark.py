"""Visualize before-vs-after optimization latency and memory benchmarks.

Usage
-----
    python scripts/visualize_benchmark.py <audio_path> [options]

The script runs ``profile_latency`` and ``measure_gpu_memory`` for each
optimization variant, then produces two sub-plots:

    1. Bar chart of mean / p95 / p99 latency in milliseconds.
    2. Bar chart of peak GPU memory in MiB (zeros on CPU-only machines).

The output is saved to ``artifacts/benchmark_comparison.png`` (or a custom
path via ``--output``).

Requires: matplotlib
    pip install matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src._optional_imports import optional_import
from src.models.asr_pipeline import ASRConfig, ASRPipeline
from src.optim.inference_optimizer import (
    OptimizationConfig,
    optimize_asr_pipeline,
)
from src.profiling.benchmark import LatencyStats, MemoryStats, measure_gpu_memory, profile_latency

plt = optional_import("matplotlib.pyplot")
mticker = optional_import("matplotlib.ticker")
if plt is None or mticker is None:
    raise SystemExit(
        "visualize_benchmark.py requires matplotlib. Install with: pip install matplotlib"
    )

torchaudio = optional_import("torchaudio")
if torchaudio is None:
    raise SystemExit(
        "visualize_benchmark.py requires torchaudio. Install with: pip install '.[asr]'"
    )

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

_VARIANTS: list[dict[str, Any]] = [
    {
        "label": "Baseline\n(FP16)",
        "dtype": "float16",
        "enable_dynamic_int8": False,
        "enable_torch_compile": False,
        "pruning_amount": 0.0,
    },
    {
        "label": "Dynamic\nINT8",
        "dtype": "float32",
        "enable_dynamic_int8": True,
        "enable_torch_compile": False,
        "pruning_amount": 0.0,
    },
    {
        "label": "torch.compile\n(FP16)",
        "dtype": "float16",
        "enable_dynamic_int8": False,
        "enable_torch_compile": True,
        "pruning_amount": 0.0,
    },
    {
        "label": "Pruned 20%\n+ FP16",
        "dtype": "float16",
        "enable_dynamic_int8": False,
        "enable_torch_compile": False,
        "pruning_amount": 0.2,
    },
]


def _build_pipeline(model_id: str, device: str, variant: dict[str, Any]) -> ASRPipeline:
    asr = ASRPipeline(
        ASRConfig(
            backend="transformers",
            model_id=model_id,
            device=device,
            dtype=variant["dtype"],
        )
    )
    asr = optimize_asr_pipeline(
        asr,
        OptimizationConfig(
            enable_dynamic_int8=variant["enable_dynamic_int8"],
            enable_torch_compile=variant["enable_torch_compile"],
            enable_pruning=variant["pruning_amount"] > 0.0,
            pruning_amount=variant["pruning_amount"],
        ),
    )
    return asr


def run_benchmarks(
    waveform: Any,
    sample_rate: int,
    model_id: str,
    device: str,
    iterations: int,
    warmup: int,
) -> list[tuple[str, LatencyStats, MemoryStats]]:
    results: list[tuple[str, LatencyStats, MemoryStats]] = []

    for variant in _VARIANTS:
        print(f"  Benchmarking: {variant['label'].replace(chr(10), ' ')} …", flush=True)
        pipeline = _build_pipeline(model_id, device, variant)

        def fn() -> None:
            pipeline.transcribe(waveform=waveform, sample_rate=sample_rate)

        latency = profile_latency(fn, iterations=iterations, warmup=warmup)
        memory = measure_gpu_memory(fn)
        results.append((variant["label"], latency, memory))

    return results


def _serialize_results(
    results: list[tuple[str, LatencyStats, MemoryStats]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, lat, mem in results:
        rows.append(
            {
                "variant": label.replace("\n", " "),
                "mean_ms": round(lat.mean_ms, 3),
                "p50_ms": round(lat.p50_ms, 3),
                "p95_ms": round(lat.p95_ms, 3),
                "p99_ms": round(lat.p99_ms, 3),
                "peak_gpu_mb": round(mem.peak_mb, 3),
                "allocated_gpu_mb": round(mem.allocated_mb, 3),
                "device": mem.device,
            }
        )
    return rows


def export_results(
    results: list[tuple[str, LatencyStats, MemoryStats]],
    *,
    run_metadata: dict[str, Any],
    summary: dict[str, Any],
    json_path: Path | None,
    csv_path: Path | None,
    append_json_history: bool,
) -> None:
    rows = _serialize_results(results)

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        run_payload = {
            "run": run_metadata,
            "summary": summary,
            "rows": rows,
        }
        if append_json_history:
            if json_path.exists():
                existing = json.loads(json_path.read_text(encoding="utf-8"))
                history = existing.get("history", []) if isinstance(existing, dict) else []
            else:
                history = []
            history.append(run_payload)
            payload = {"history": history}
        else:
            payload = run_payload

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Metrics JSON saved to: {json_path}")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_rows: list[dict[str, Any]] = []
        for row in rows:
            csv_rows.append({**run_metadata, **row})

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Metrics CSV saved to: {csv_path}")


def summarize_results(
    results: list[tuple[str, LatencyStats, MemoryStats]], latency_target_ms: float
) -> dict[str, Any]:
    if not results:
        return {
            "variant_count": 0,
            "all_variants_meet_target": False,
            "latency_target_ms": latency_target_ms,
        }

    best_label, best_lat, _ = min(results, key=lambda x: x[1].p95_ms)
    baseline_label, baseline_lat, _ = results[0]
    speedup = baseline_lat.p95_ms / best_lat.p95_ms if best_lat.p95_ms > 0 else 0.0
    all_passed = all(lat.p95_ms <= latency_target_ms for _, lat, _ in results)

    return {
        "variant_count": len(results),
        "latency_target_ms": latency_target_ms,
        "all_variants_meet_target": all_passed,
        "baseline_variant": baseline_label.replace("\n", " "),
        "baseline_p95_ms": round(baseline_lat.p95_ms, 3),
        "best_variant": best_label.replace("\n", " "),
        "best_p95_ms": round(best_lat.p95_ms, 3),
        "best_speedup_vs_baseline": round(speedup, 4),
    }


def print_target_summary(
    results: list[tuple[str, LatencyStats, MemoryStats]], latency_target_ms: float
) -> bool:
    if not results:
        return False

    summary = summarize_results(results, latency_target_ms=latency_target_ms)
    print("\nLatency target summary (p95 <= target):")
    for label, lat, _ in results:
        passed = lat.p95_ms <= latency_target_ms
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label.replace(chr(10), ' ')}: p95={lat.p95_ms:.1f} ms")

    print(
        "Best p95 variant: "
        f"{summary['best_variant']} ({summary['best_p95_ms']:.1f} ms, "
        f"speedup vs baseline={summary['best_speedup_vs_baseline']:.2f}x "
        f"from {summary['baseline_variant']})"
    )
    return bool(summary["all_variants_meet_target"])


def plot_results(
    results: list[tuple[str, LatencyStats, MemoryStats]],
    output: Path,
    latency_target_ms: float = 200.0,
) -> None:
    labels = [r[0] for r in results]
    means = [r[1].mean_ms for r in results]
    p95s = [r[1].p95_ms for r in results]
    p99s = [r[1].p99_ms for r in results]
    peaks = [r[2].peak_mb for r in results]
    device_name = results[0][2].device or "cpu"

    x = range(len(labels))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ASR Inference: Before vs After Optimization", fontsize=14, fontweight="bold")

    # --- Latency chart ---
    bars_mean = ax1.bar([xi - width for xi in x], means, width, label="Mean", color="#4C72B0")
    ax1.bar(x, p95s, width, label="p95", color="#DD8452")
    ax1.bar([xi + width for xi in x], p99s, width, label="p99", color="#55A868")

    ax1.axhline(
        latency_target_ms,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"{latency_target_ms:.0f} ms target",
    )
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Transcription Latency")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax1.bar_label(bars_mean, fmt="%.1f", fontsize=7, padding=2)

    # --- Memory chart ---
    bar_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    mem_bars = ax2.bar(list(x), peaks, color=bar_colors[: len(labels)], alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Peak GPU Memory (MiB)")
    ax2.set_title(f"GPU Memory Usage\n({device_name})")
    ax2.bar_label(mem_bars, fmt="%.1f", fontsize=8, padding=2)
    if all(p == 0.0 for p in peaks):
        ax2.text(
            0.5,
            0.5,
            "GPU not available\n(all zeros)",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and visualize ASR optimization variants"
    )
    parser.add_argument("audio_path", type=Path, help="Path to WAV/FLAC audio clip")
    parser.add_argument("--model-id", type=str, default="openai/whisper-small")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmark_comparison.png"),
        help="Output path for the comparison chart PNG",
    )
    parser.add_argument(
        "--latency-target-ms",
        type=float,
        default=200.0,
        help="Reference line drawn on the latency chart (default 200 ms)",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("artifacts/benchmark_metrics.json"),
        help="Output path for machine-readable metrics JSON",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("artifacts/benchmark_metrics.csv"),
        help="Output path for machine-readable metrics CSV",
    )
    parser.add_argument(
        "--fail-if-target-missed",
        action="store_true",
        help="Exit with status 2 when any variant misses the p95 latency target",
    )
    parser.add_argument(
        "--append-json-history",
        action="store_true",
        help="Append run payload to a JSON history file instead of overwriting",
    )
    args = parser.parse_args()

    print(f"Loading audio: {args.audio_path}")
    waveform, sample_rate = torchaudio.load(args.audio_path)
    print(f"  shape={tuple(waveform.shape)}, sample_rate={sample_rate}")

    print("\nRunning benchmarks …")
    results = run_benchmarks(
        waveform=waveform,
        sample_rate=sample_rate,
        model_id=args.model_id,
        device=args.device,
        iterations=args.iterations,
        warmup=args.warmup,
    )

    print("\nResults summary:")
    print(f"  {'Variant':<30} {'mean_ms':>8} {'p95_ms':>8} {'p99_ms':>8} {'peak_MB':>9}")
    print("  " + "-" * 68)
    for label, lat, mem in results:
        flat_label = label.replace("\n", " ")
        print(
            f"  {flat_label:<30} {lat.mean_ms:>8.1f} {lat.p95_ms:>8.1f} "
            f"{lat.p99_ms:>8.1f} {mem.peak_mb:>9.1f}"
        )

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audio_path": str(args.audio_path),
        "model_id": args.model_id,
        "device": args.device,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "sample_rate": int(sample_rate),
        "waveform_shape": list(waveform.shape),
    }
    summary = summarize_results(results, latency_target_ms=args.latency_target_ms)
    all_passed = print_target_summary(results, latency_target_ms=args.latency_target_ms)
    export_results(
        results,
        run_metadata=run_metadata,
        summary=summary,
        json_path=args.metrics_json,
        csv_path=args.metrics_csv,
        append_json_history=args.append_json_history,
    )
    plot_results(results, args.output, latency_target_ms=args.latency_target_ms)

    if args.fail_if_target_missed and not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
