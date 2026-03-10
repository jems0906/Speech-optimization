"""Profiling configuration."""

# Benchmark settings
BENCHMARK_CONFIG = {
    "warmup_iterations": 10,
    "test_iterations": 100,
    "audio_durations": [3, 5, 10, 30],  # seconds
    "sample_rate": 16000,
}

# Profiling output settings
PROFILING_OUTPUT = {
    "print_summary": True,
    "save_results": True,
    "results_dir": "profiling_results",
}

# GPU profiling tools
GPU_PROFILING_TOOLS = {
    "nsight_systems": {
        "enabled": False,
        "command": "nsys profile --trace=cuda,nvtx",
    },
    "nsight_compute": {
        "enabled": False,
        "command": "ncu --target-processes all",
    },
    "pytorch_profiler": {
        "enabled": True,
        "activities": ["cpu", "cuda"],
        "record_shapes": True,
    },
}
