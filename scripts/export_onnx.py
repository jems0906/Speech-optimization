from __future__ import annotations

import argparse
from pathlib import Path

from src.models.asr_pipeline import ASRConfig, ASRPipeline
from src.optim.inference_optimizer import export_model_to_onnx


def _build_tensorrt_engine(onnx_path: Path, engine_path: Path, *, fp16: bool = True) -> None:
    """Compile an ONNX model into a serialized TensorRT engine.

    Requires the ``tensorrt`` package from the NVIDIA pip index.
    FP16 is enabled automatically when the GPU supports it and fp16=True.
    """
    try:
        import tensorrt as trt  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit(
            "TensorRT engine build requires tensorrt. "
            "Install via NVIDIA pip index: pip install tensorrt"
        ) from exc

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse errors: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ASR model to ONNX (and optionally TensorRT)"
    )
    parser.add_argument("--model-id", type=str, default="openai/whisper-small")
    parser.add_argument(
        "--model-family",
        type=str,
        default="auto",
        choices=["auto", "whisper", "wav2vec2"],
        help="ASR model family. Use auto to infer from the loaded Hugging Face model.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/asr_model.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--tensorrt-engine",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Also compile a TensorRT engine and write it to PATH (.engine). "
            "Requires tensorrt and a CUDA GPU."
        ),
    )
    args = parser.parse_args()

    pipeline = ASRPipeline(
        ASRConfig(
            backend="transformers",
            model_id=args.model_id,
            model_family=args.model_family,
            device="cpu",
            dtype="float32",
        )
    )

    export_model_to_onnx(
        pipeline.model,
        args.output,
        opset=args.opset,
        model_family=pipeline.model_family,
        model_id=args.model_id,
    )
    print(f"Exported ONNX model to: {args.output}")

    if args.tensorrt_engine:
        _build_tensorrt_engine(args.output, args.tensorrt_engine)
        print(f"Built TensorRT engine: {args.tensorrt_engine}")


if __name__ == "__main__":
    main()
