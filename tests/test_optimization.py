"""Tests for optimization utilities."""

import torch
import torch.nn as nn

from src.optim.quantization import ModelOptimizer


def test_convert_to_half_precision():
    """Test FP16 conversion."""
    model = nn.Linear(10, 5)
    half_model = ModelOptimizer.convert_to_half_precision(model)

    assert next(half_model.parameters()).dtype == torch.float16


def test_estimate_speedup():
    """Test speedup calculation."""
    stats = ModelOptimizer.estimate_speedup(original_time=1.0, optimized_time=0.5)

    assert stats["speedup_factor"] == 2.0
    assert stats["reduction_percentage"] == 50.0
    assert stats["original_time_ms"] == 1000.0
    assert stats["optimized_time_ms"] == 500.0


def test_estimate_speedup_zero_time():
    """Test speedup with zero optimized time."""
    stats = ModelOptimizer.estimate_speedup(original_time=1.0, optimized_time=0.0)
    assert stats["speedup_factor"] == 0.0
