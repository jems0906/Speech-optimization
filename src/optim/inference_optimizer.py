from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src._optional_imports import optional_import
from src.models.asr_pipeline import ASRPipeline, build_export_metadata

torch = optional_import("torch")


@dataclass(slots=True)
class OptimizationConfig:
    enable_torch_compile: bool = False
    enable_dynamic_int8: bool = False
    enable_pruning: bool = False
    pruning_amount: float = 0.2


def optimize_asr_pipeline(asr_pipeline: ASRPipeline, config: OptimizationConfig) -> ASRPipeline:
    if asr_pipeline.config.backend == "mock":
        return asr_pipeline

    if (
        not config.enable_dynamic_int8
        and not config.enable_torch_compile
        and not config.enable_pruning
    ):
        return asr_pipeline

    if torch is None:
        raise RuntimeError(
            "Optimization requires torch. Install optional dependencies with: pip install .[asr]"
        )

    model = asr_pipeline.model

    if config.enable_pruning:
        model = apply_magnitude_pruning(model, amount=config.pruning_amount)

    if config.enable_dynamic_int8 and asr_pipeline.config.device == "cpu":
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

    if config.enable_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    asr_pipeline.model = model
    return asr_pipeline


def export_model_to_onnx(
    model: Any,
    output_path: Path,
    opset: int = 17,
    *,
    model_family: str = "auto",
    model_id: str = "openai/whisper-small",
) -> None:
    if torch is None:
        raise RuntimeError(
            "ONNX export requires torch and onnx. Install optional dependencies with: "
            "pip install .[asr,optimization]"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy_input, input_names = build_export_metadata(
        model,
        model_family=model_family,
        model_id=model_id,
        torch_module=torch,
    )

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["logits"],
    )


def compile_with_tensorrt(
    model: Any,
    input_shapes: list[tuple[int, ...]],
    *,
    dtype: str = "float16",
) -> Any:
    """Compile model with Torch-TensorRT for maximum GPU throughput.

    Falls back to the original model if torch_tensorrt is not installed or
    compilation fails, so callers are not forced to guard against its absence.

    Args:
        model: An nn.Module already on a CUDA device.
        input_shapes: List of concrete (batch, *dims) tuples, one per positional
            model input. Example for Whisper encoder: [(1, 80, 3000)].
        dtype: "float16" (recommended on Turing+) or "float32".
    """
    if torch is None:
        raise RuntimeError("TensorRT compilation requires torch.")

    try:
        import torch_tensorrt  # type: ignore[import]
    except ImportError:  # pragma: no cover
        return model

    precision_map: dict[str, Any] = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    trt_dtype = precision_map.get(dtype, torch.float32)
    trt_inputs = [
        torch_tensorrt.Input(shape=list(shape), dtype=trt_dtype) for shape in input_shapes
    ]

    try:
        return torch_tensorrt.compile(
            model,
            inputs=trt_inputs,
            enabled_precisions={trt_dtype},
        )
    except Exception:  # pragma: no cover
        return model


def apply_bitsandbytes_int8(model: Any) -> Any:
    """Replace eligible Linear layers with bitsandbytes 8-bit quantized versions.

    Targets GPU inference where VRAM is the bottleneck. The model must already
    be on a CUDA device. Falls back to the original model when bitsandbytes is
    not installed or the platform does not support it (non-Linux).
    """
    if torch is None:
        raise RuntimeError("bitsandbytes quantization requires torch.")

    try:
        import bitsandbytes as bnb  # type: ignore[import]
    except ImportError:  # pragma: no cover
        return model

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model if not parent_name else _get_submodule(model, parent_name)
        replacement = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            has_fp16_weights=False,
        )
        replacement.weight = module.weight
        if module.bias is not None:
            replacement.bias = module.bias
        setattr(parent, attr, replacement)

    return model


def _get_submodule(model: Any, path: str) -> Any:
    parts = path.split(".")
    m = model
    for part in parts:
        m = getattr(m, part)
    return m


def convert_to_fp16(model: Any) -> Any:
    """Cast all model parameters and buffers to FP16 (half precision).

    Reduces VRAM usage by roughly half versus FP32. The model must already be
    on a CUDA device; FP16 on CPU is unsupported by most operators.

    Raises RuntimeError when torch is not installed.
    """
    if torch is None:
        raise RuntimeError("FP16 conversion requires torch.")
    return model.half()


def apply_magnitude_pruning(model: Any, amount: float = 0.2) -> Any:
    """Apply unstructured L1 magnitude pruning to all Linear layers.

    Zeroes the ``amount`` fraction of weights with the smallest absolute
    values in every ``nn.Linear`` layer and makes the sparsity permanent by
    removing the pruning re-parametrisation immediately after.

    Args:
        model:  An ``nn.Module`` (already moved to the desired device).
        amount: Fraction of weights to prune per layer, in [0, 1).
                Default 0.2 removes the 20 % smallest-magnitude weights.

    Returns:
        The same model object with pruned weights (in-place operation).

    Raises:
        RuntimeError: When torch is not installed.
        ValueError:   When amount is not in [0, 1).
    """
    if torch is None:
        raise RuntimeError("Magnitude pruning requires torch. Install with: pip install '.[asr]'")
    if not (0.0 <= amount < 1.0):
        raise ValueError(f"amount must be in [0, 1), got {amount}")

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.utils.prune.l1_unstructured(module, name="weight", amount=amount)
            torch.nn.utils.prune.remove(module, "weight")

    return model
