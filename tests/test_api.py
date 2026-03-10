"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

# Note: These are placeholder tests
# Actual tests would require mocking the model loading


@pytest.fixture
def client():
    """Create test client."""
    from src.serving.main import app

    return TestClient(app)


def test_health_endpoint_structure():
    """Test health endpoint response structure."""
    # This is a placeholder - actual test would need model mocked
    pass


def test_transcription_request_validation():
    """Test request validation."""
    from src.serving.schemas import TranscriptionRequest

    request = TranscriptionRequest(language="en", task="transcribe")
    assert request.language == "en"
    assert request.task == "transcribe"


def test_transcription_response_structure():
    """Test response structure."""
    from src.serving.schemas import TranscriptionResponse

    response = TranscriptionResponse(
        text="test", processing_time_ms=100.0, model_name="whisper-base"
    )
    assert response.text == "test"
    assert response.processing_time_ms == 100.0
