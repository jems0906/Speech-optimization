"""Test fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    """Get compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dummy_audio():
    """Generate dummy audio tensor."""
    return torch.randn(1, 16000)  # 1 second of audio at 16kHz


@pytest.fixture
def sample_rate():
    """Sample rate fixture."""
    return 16000
