"""Tests for model loading."""

import pytest
import torch

from src.models.loader import ModelLoader


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_model_loader_whisper():
    """Test Whisper model loading."""
    model, processor = ModelLoader.load_whisper(
        model_name="whisper-tiny", device="cuda", torch_dtype=torch.float16
    )

    assert model is not None
    assert processor is not None
    assert next(model.parameters()).is_cuda


def test_model_info():
    """Test model info extraction."""
    model = torch.nn.Linear(10, 5)
    info = ModelLoader.get_model_info(model)

    assert "total_parameters" in info
    assert "trainable_parameters" in info
    assert "model_size_mb" in info
    assert info["total_parameters"] == 55  # 10*5 + 5 bias


def test_supported_models():
    """Test supported models list."""
    assert "whisper-tiny" in ModelLoader.SUPPORTED_MODELS
    assert "whisper-base" in ModelLoader.SUPPORTED_MODELS
    assert "wav2vec2-base" in ModelLoader.SUPPORTED_MODELS
