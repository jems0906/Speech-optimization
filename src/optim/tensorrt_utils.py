"""TensorRT conversion utilities."""

import logging

import torch

logger = logging.getLogger(__name__)


class TensorRTConverter:
    """Convert PyTorch models to TensorRT."""

    def __init__(
        self,
        workspace_size: int = 1 << 30,  # 1GB
        fp16_mode: bool = True,
        int8_mode: bool = False,
    ):
        """Initialize TensorRT converter."""
        self.workspace_size = workspace_size
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode

    def convert_from_onnx(
        self,
        onnx_path: str,
        engine_path: str,
        min_batch: int = 1,
        opt_batch: int = 8,
        max_batch: int = 16,
    ) -> bool:
        """Convert ONNX model to TensorRT engine."""
        try:
            import tensorrt as trt

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # Parse ONNX
            with open(onnx_path, "rb") as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("Failed to parse ONNX")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False

            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.workspace_size
            )

            if self.fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            if self.int8_mode and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)

            # Build engine
            logger.info("Building TensorRT engine...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                logger.error("Failed to build TensorRT engine")
                return False

            # Save engine
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            logger.info(f"TensorRT engine saved to {engine_path}")
            return True

        except ImportError as e:
            logger.error(f"TensorRT not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}", exc_info=True)
            return False

    def convert_torch_to_tensorrt(
        self,
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        engine_path: str,
    ) -> bool:
        """Convert PyTorch model directly to TensorRT using Torch-TensorRT."""
        try:
            import torch_tensorrt

            # Compile with Torch-TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[dummy_input],
                enabled_precisions=(
                    {torch.float16} if self.fp16_mode else {torch.float32}
                ),
                workspace_size=self.workspace_size,
            )

            # Save
            torch.jit.save(trt_model, engine_path)
            logger.info(f"Torch-TensorRT model saved to {engine_path}")
            return True

        except ImportError as e:
            logger.error(f"Torch-TensorRT not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Torch-TensorRT conversion failed: {e}", exc_info=True)
            return False
