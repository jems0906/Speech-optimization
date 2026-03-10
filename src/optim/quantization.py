"""Model optimization: quantization and pruning."""

from typing import Optional

import torch
import torch.nn as nn
from torch.quantization import get_default_qconfig, prepare, quantize_dynamic


class ModelOptimizer:
    """Optimize models for inference."""

    @staticmethod
    def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization."""
        quantized_model = quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8, inplace=False
        )
        return quantized_model

    @staticmethod
    def convert_to_half_precision(model: nn.Module) -> nn.Module:
        """Convert model to FP16."""
        return model.half()

    @staticmethod
    def apply_torch_compile(
        model: nn.Module, mode: str = "reduce-overhead", backend: str = "inductor"
    ) -> nn.Module:
        """Apply torch.compile for graph optimization (PyTorch 2.0+)."""
        if hasattr(torch, "compile"):
            return torch.compile(model, mode=mode, backend=backend)
        return model

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        opset_version: int = 14,
        dynamic_axes: Optional[dict] = None,
    ):
        """Export model to ONNX format."""
        model.eval()

        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    @staticmethod
    def estimate_speedup(
        original_time: float, optimized_time: float
    ) -> dict:
        """Calculate optimization speedup metrics."""
        speedup = original_time / optimized_time if optimized_time > 0 else 0
        reduction_pct = ((original_time - optimized_time) / original_time) * 100

        return {
            "speedup_factor": speedup,
            "reduction_percentage": reduction_pct,
            "original_time_ms": original_time * 1000,
            "optimized_time_ms": optimized_time * 1000,
        }
