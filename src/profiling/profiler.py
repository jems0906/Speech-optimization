"""Performance profiling utilities."""

import time
from contextlib import contextmanager
from typing import Dict, Optional

import torch


class PerformanceProfiler:
    """Profile model inference performance."""

    def __init__(self, device: str = "cuda"):
        """Initialize profiler."""
        self.device = device
        self.measurements = {}

    @contextmanager
    def profile(self, name: str):
        """Context manager for timing operations."""
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        yield
        
        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(elapsed)

    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a profiled operation."""
        if name not in self.measurements:
            return None

        measurements = self.measurements[name]
        return {
            "mean_ms": sum(measurements) / len(measurements) * 1000,
            "min_ms": min(measurements) * 1000,
            "max_ms": max(measurements) * 1000,
            "count": len(measurements),
            "total_s": sum(measurements),
        }

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all measurements."""
        return {name: self.get_stats(name) for name in self.measurements}

    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()


class GPUProfiler:
    """GPU memory and utilization profiler."""

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        }

    @staticmethod
    def reset_peak_memory_stats():
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark model inference."""
    model.eval()
    profiler = PerformanceProfiler(device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor)

    # Benchmark
    with torch.no_grad():
        for _ in range(num_iterations):
            with profiler.profile("inference"):
                _ = model(input_tensor)

    stats = profiler.get_stats("inference")
    gpu_info = GPUProfiler.get_gpu_memory_info()

    return {
        **stats,
        **gpu_info,
    }
