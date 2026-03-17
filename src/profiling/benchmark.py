from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Callable

from src._optional_imports import optional_import

torch = optional_import("torch")


@dataclass(slots=True)
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass(slots=True)
class MemoryStats:
    allocated_mb: float
    peak_mb: float
    device: str


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * ratio))))
    return sorted_values[index]


def profile_latency(
    fn: Callable[[], object], iterations: int = 20, warmup: int = 5
) -> LatencyStats:
    for _ in range(max(0, warmup)):
        fn()

    samples: list[float] = []
    for _ in range(max(1, iterations)):
        start = perf_counter()
        fn()
        samples.append((perf_counter() - start) * 1000)

    sorted_samples = sorted(samples)
    return LatencyStats(
        count=len(sorted_samples),
        mean_ms=mean(sorted_samples),
        p50_ms=_percentile(sorted_samples, 0.50),
        p95_ms=_percentile(sorted_samples, 0.95),
        p99_ms=_percentile(sorted_samples, 0.99),
    )


def profile_with_torch_profiler(
    fn: Callable[[], object],
    iterations: int = 10,
    *,
    use_cuda: bool = False,
    row_limit: int = 20,
) -> str:
    """Run fn under torch.profiler and return a per-operator key-averages table.

    The returned string is suitable for printing or logging. It sorts by
    cpu_time_total and includes CUDA time when use_cuda=True and a GPU is
    available.

    Raises RuntimeError when torch is not installed.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch Profiler requires torch. Install with: pip install '.[asr]'")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for _ in range(max(1, iterations)):
            fn()

    return prof.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit)


def measure_gpu_memory(fn: Callable[[], object], device: int = 0) -> MemoryStats:
    """Measure GPU memory allocated before and peak during fn().

    Resets the per-device peak counter before running fn so the recorded peak
    reflects only the call itself.  Returns zeros on CPU-only machines or when
    torch is not installed rather than raising, so callers can always print the
    result unconditionally.

    Args:
        fn:     Callable to profile (called exactly once).
        device: CUDA device index (default 0).

    Returns:
        MemoryStats with allocated_mb (post-call resident), peak_mb (watermark
        during the call), and device name string.
    """
    if torch is None or not torch.cuda.is_available():
        return MemoryStats(allocated_mb=0.0, peak_mb=0.0, device="cpu")

    torch.cuda.reset_peak_memory_stats(device)  # pragma: no cover
    torch.cuda.synchronize(device)  # pragma: no cover

    fn()  # pragma: no cover

    torch.cuda.synchronize(device)  # pragma: no cover
    allocated = torch.cuda.memory_allocated(device) / (1024**2)  # pragma: no cover
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)  # pragma: no cover
    device_name = torch.cuda.get_device_name(device)  # pragma: no cover
    return MemoryStats(  # pragma: no cover
        allocated_mb=round(allocated, 2), peak_mb=round(peak, 2), device=device_name
    )
