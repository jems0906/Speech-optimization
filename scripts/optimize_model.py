"""Script to optimize model for inference."""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader import ModelLoader
from src.optim.quantization import ModelOptimizer
from src.optim.tensorrt_utils import TensorRTConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Optimize model for inference."""
    parser = argparse.ArgumentParser(description="Optimize ASR model")
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-base",
        help="Model name to optimize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/optimized",
        help="Output directory for optimized models",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Export to ONNX",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Convert to TensorRT",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {args.model}")

    # Load model
    dtype = torch.float16 if args.fp16 else torch.float32
    model, processor = ModelLoader.load_whisper(
        model_name=args.model,
        device="cuda",
        torch_dtype=dtype,
    )

    model_name_safe = args.model.replace("/", "_")

    # Export to ONNX
    if args.onnx:
        logger.info("Exporting to ONNX...")
        onnx_path = output_dir / f"{model_name_safe}.onnx"

        # Create dummy input
        dummy_input = processor(
            torch.randn(1, 16000 * 5).squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to("cuda")

        if args.fp16:
            dummy_input = dummy_input.half()

        ModelOptimizer.export_to_onnx(
            model=model,
            dummy_input=dummy_input,
            output_path=str(onnx_path),
        )

        logger.info(f"ONNX model saved to: {onnx_path}")

    # Convert to TensorRT
    if args.tensorrt:
        if not args.onnx:
            logger.error("TensorRT conversion requires ONNX export first")
            return

        logger.info("Converting to TensorRT...")
        engine_path = output_dir / f"{model_name_safe}.engine"

        converter = TensorRTConverter(fp16_mode=args.fp16)
        success = converter.convert_from_onnx(
            onnx_path=str(onnx_path),
            engine_path=str(engine_path),
        )

        if success:
            logger.info(f"TensorRT engine saved to: {engine_path}")
        else:
            logger.error("TensorRT conversion failed")

    logger.info("Optimization complete!")


if __name__ == "__main__":
    main()
