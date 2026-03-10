"""Script to profile model performance."""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import ModelLoader
from src.profiling.profiler import benchmark_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Profile model performance."""
    parser = argparse.ArgumentParser(description="Profile ASR model performance")
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-base",
        help="Model name to profile",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )

    args = parser.parse_args()

    logger.info(f"Profiling model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"FP16: {args.fp16}")

    # Load model
    dtype = torch.float16 if args.fp16 else torch.float32
    model, processor = ModelLoader.load_whisper(
        model_name=args.model,
        device=args.device,
        torch_dtype=dtype,
    )

    # Create dummy input (5 seconds of audio)
    dummy_input = processor(
        torch.randn(1, 16000 * 5).squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(args.device)

    if args.fp16:
        dummy_input = dummy_input.half()

    # Benchmark
    logger.info("Starting benchmark...")
    stats = benchmark_model(
        model=model,
        input_tensor=dummy_input,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        device=args.device,
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("PROFILING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mean inference time: {stats['mean_ms']:.2f} ms")
    logger.info(f"Min inference time: {stats['min_ms']:.2f} ms")
    logger.info(f"Max inference time: {stats['max_ms']:.2f} ms")
    logger.info(f"Total iterations: {stats['count']}")
    logger.info(f"Total time: {stats['total_s']:.2f} s")

    if args.device == "cuda":
        logger.info(f"\nGPU Memory Allocated: {stats['allocated_gb']:.2f} GB")
        logger.info(f"GPU Memory Reserved: {stats['reserved_gb']:.2f} GB")
        logger.info(f"GPU Max Allocated: {stats['max_allocated_gb']:.2f} GB")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
