"""Comprehensive benchmarking script."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import ModelLoader
from src.optim.quantization import ModelOptimizer
from src.profiling.profiler import GPUProfiler, PerformanceProfiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_configuration(
    model_name: str,
    use_fp16: bool,
    use_quantization: bool,
    device: str,
    iterations: int,
):
    """Benchmark a specific configuration."""
    logger.info(
        f"Benchmarking: {model_name}, FP16={use_fp16}, Quant={use_quantization}"
    )

    # Load model
    dtype = torch.float16 if use_fp16 else torch.float32
    model, processor = ModelLoader.load_whisper(
        model_name=model_name, device=device, torch_dtype=dtype
    )

    # Apply optimizations
    if use_quantization and not use_fp16:
        model = ModelOptimizer.quantize_dynamic_int8(model)

    # Create dummy input
    dummy_input = processor(
        torch.randn(1, 16000 * 5).squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)

    if use_fp16:
        dummy_input = dummy_input.half()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.generate(dummy_input)

    # Benchmark
    profiler = PerformanceProfiler(device=device)
    GPUProfiler.reset_peak_memory_stats()

    for _ in range(iterations):
        with profiler.profile("inference"):
            with torch.no_grad():
                _ = model.generate(dummy_input)

    stats = profiler.get_stats("inference")
    gpu_info = GPUProfiler.get_gpu_memory_info()

    return {
        "model": model_name,
        "fp16": use_fp16,
        "quantization": use_quantization,
        "mean_ms": stats["mean_ms"],
        "min_ms": stats["min_ms"],
        "max_ms": stats["max_ms"],
        "gpu_memory_gb": gpu_info.get("max_allocated_gb", 0),
    }


def main():
    """Run comprehensive benchmarks."""
    parser = argparse.ArgumentParser(description="Comprehensive benchmark suite")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["whisper-tiny", "whisper-base"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Benchmark iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    results = []

    # Test configurations
    configs = [
        {"use_fp16": False, "use_quantization": False},  # Baseline
        {"use_fp16": True, "use_quantization": False},  # FP16
        {"use_fp16": False, "use_quantization": True},  # INT8
    ]

    for model_name in args.models:
        for config in configs:
            try:
                result = benchmark_configuration(
                    model_name=model_name,
                    device=args.device,
                    iterations=args.iterations,
                    **config,
                )
                results.append(result)

                # Log result
                logger.info(f"Result: {result['mean_ms']:.2f}ms")

                # Clean up
                torch.cuda.empty_cache()
                time.sleep(2)

            except Exception as e:
                logger.error(f"Benchmark failed for {model_name} with {config}: {e}")

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Model':<20} {'Config':<15} {'Mean (ms)':<12} {'GPU Mem (GB)':<12}")
    logger.info("-" * 80)

    for result in results:
        config = f"{'FP16' if result['fp16'] else 'FP32'}"
        if result["quantization"]:
            config += "+INT8"

        logger.info(
            f"{result['model']:<20} {config:<15} {result['mean_ms']:<12.2f} "
            f"{result['gpu_memory_gb']:<12.2f}"
        )

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
