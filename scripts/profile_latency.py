from __future__ import annotations

import argparse
from pathlib import Path

from src._optional_imports import optional_import
from src.models.asr_pipeline import ASRConfig, ASRPipeline
from src.optim.inference_optimizer import OptimizationConfig, optimize_asr_pipeline
from src.profiling.benchmark import profile_latency, profile_with_torch_profiler

torchaudio = optional_import("torchaudio")
if torchaudio is None:
    raise SystemExit("profile_latency.py requires torchaudio. Install with: pip install '.[asr]'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ASR latency for an audio file")
    parser.add_argument("audio_path", type=Path, help="Path to WAV/FLAC audio file")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--model-id", type=str, default="openai/whisper-small")
    parser.add_argument(
        "--model-family",
        type=str,
        default="auto",
        choices=["auto", "whisper", "wav2vec2"],
        help="ASR model family. Use auto to infer from the loaded Hugging Face model.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"]
    )
    parser.add_argument(
        "--dynamic-int8", action="store_true", help="Apply dynamic INT8 quantization (CPU)"
    )
    parser.add_argument(
        "--torch-compile", action="store_true", help="Apply torch.compile (reduce-overhead mode)"
    )
    parser.add_argument(
        "--pruning-amount",
        type=float,
        default=0.0,
        help="Apply L1 magnitude pruning with this amount in [0, 1). Set 0 to disable.",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Run torch.profiler for operator-level breakdown",
    )
    args = parser.parse_args()

    waveform, sample_rate = torchaudio.load(args.audio_path)

    pipeline = ASRPipeline(
        ASRConfig(
            backend="transformers",
            model_id=args.model_id,
            model_family=args.model_family,
            device=args.device,
            dtype=args.dtype,
        )
    )

    pipeline = optimize_asr_pipeline(
        pipeline,
        OptimizationConfig(
            enable_dynamic_int8=args.dynamic_int8,
            enable_torch_compile=args.torch_compile,
            enable_pruning=args.pruning_amount > 0.0,
            pruning_amount=args.pruning_amount,
        ),
    )

    def transcribe_fn() -> str:
        return pipeline.transcribe(waveform=waveform, sample_rate=sample_rate)

    stats = profile_latency(
        transcribe_fn,
        iterations=args.iterations,
        warmup=args.warmup,
    )

    print(f"count={stats.count}")
    print(f"mean_ms={stats.mean_ms:.2f}")
    print(f"p50_ms={stats.p50_ms:.2f}")
    print(f"p95_ms={stats.p95_ms:.2f}")
    print(f"p99_ms={stats.p99_ms:.2f}")

    if args.torch_profiler:
        print("\n--- PyTorch Profiler (operator-level) ---")
        use_cuda = args.device == "cuda"
        table = profile_with_torch_profiler(transcribe_fn, iterations=5, use_cuda=use_cuda)
        print(table)


if __name__ == "__main__":
    main()
