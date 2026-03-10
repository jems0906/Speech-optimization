"""Model loader for ASR models (Whisper, Wav2Vec2)."""

from typing import Tuple

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


class ModelLoader:
    """Load and configure ASR models."""

    SUPPORTED_MODELS = {
        "whisper-tiny": "openai/whisper-tiny",
        "whisper-base": "openai/whisper-base",
        "whisper-small": "openai/whisper-small",
        "wav2vec2-base": "facebook/wav2vec2-base-960h",
        "wav2vec2-large": "facebook/wav2vec2-large-960h",
    }

    @staticmethod
    def load_whisper(
        model_name: str = "whisper-base",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
        """Load Whisper model and processor."""
        model_id = ModelLoader.SUPPORTED_MODELS.get(model_name, model_name)

        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )

        model = model.to(device)
        model.eval()

        return model, processor

    @staticmethod
    def load_wav2vec2(
        model_name: str = "wav2vec2-base", device: str = "cuda"
    ) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
        """Load Wav2Vec2 model and processor."""
        model_id = ModelLoader.SUPPORTED_MODELS.get(model_name, model_name)

        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)

        model = model.to(device)
        model.eval()

        return model, processor

    @staticmethod
    def get_model_info(model: nn.Module) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),  # FP32
        }
