"""Tests for profiling utilities."""

import pytest
import torch

from src.profiling.profiler import GPUProfiler, PerformanceProfiler


def test_performance_profiler():
    """Test performance profiler."""
    profiler = PerformanceProfiler(device="cpu")

    with profiler.profile("test_op"):
        torch.randn(100, 100)

    stats = profiler.get_stats("test_op")
    assert stats is not None
    assert "mean_ms" in stats
    assert "count" in stats
    assert stats["count"] == 1


def test_profiler_multiple_measurements():
    """Test multiple measurements."""
    profiler = PerformanceProfiler(device="cpu")

    for _ in range(5):
        with profiler.profile("test_op"):
            torch.randn(100, 100)

    stats = profiler.get_stats("test_op")
    assert stats["count"] == 5


def test_profiler_reset():
    """Test profiler reset."""
    profiler = PerformanceProfiler(device="cpu")

    with profiler.profile("test_op"):
        torch.randn(100, 100)

    profiler.reset()
    assert profiler.get_stats("test_op") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_gpu_memory_info():
    """Test GPU memory info."""
    info = GPUProfiler.get_gpu_memory_info()

    assert "allocated_gb" in info
    assert "reserved_gb" in info
    assert "max_allocated_gb" in info
