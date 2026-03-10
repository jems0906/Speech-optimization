"""Request and response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Request for transcription."""

    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es')")
    task: str = Field("transcribe", description="Task type: 'transcribe' or 'translate'")


class TranscriptionResponse(BaseModel):
    """Response from transcription."""

    text: str
    processing_time_ms: float
    model_name: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    gpu_memory_allocated_gb: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    total_parameters: int
    model_size_mb: float
    device: str
    optimization_enabled: bool
