"""FastAPI application for ASR serving."""

import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.models.inference import ASRInference
from src.models.loader import ModelLoader
from src.optim.quantization import ModelOptimizer
from src.profiling.profiler import GPUProfiler, PerformanceProfiler
from src.serving.config import settings
from src.serving.schemas import (
    HealthResponse,
    ModelInfoResponse,
    TranscriptionRequest,
    TranscriptionResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Speech-to-Text Optimization API",
    description="Real-time ASR with PyTorch and TensorRT optimization",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (thread-safe for read-heavy workloads)
import threading

asr_engine: Optional[ASRInference] = None
profiler: Optional[PerformanceProfiler] = None
_startup_lock = threading.Lock()


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global asr_engine, profiler

    with _startup_lock:
        if asr_engine is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model: {settings.model_name}")
        logger.info(f"Device: {settings.device}")

        try:
            # Load model
            if settings.model_type == "whisper":
                model, processor = ModelLoader.load_whisper(
                    model_name=settings.model_name,
                    device=settings.device,
                    torch_dtype=torch.float16 if settings.use_fp16 else torch.float32,
                )
            else:
                model, processor = ModelLoader.load_wav2vec2(
                    model_name=settings.model_name, device=settings.device
                )

            # Apply optimizations
            if settings.use_quantization:
                logger.info("Applying INT8 quantization")
                model = ModelOptimizer.quantize_dynamic_int8(model)

            # Initialize inference engine
            asr_engine = ASRInference(
                model=model,
                processor=processor,
                device=settings.device,
                sample_rate=settings.sample_rate,
            )

            # Warmup
            logger.info("Warming up model...")
            asr_engine.warmup(duration=3.0)

            # Initialize profiler
            if settings.enable_profiling:
                profiler = PerformanceProfiler(device=settings.device)

            logger.info("Model loaded and ready")

            # Log GPU info
            if torch.cuda.is_available():
                gpu_info = GPUProfiler.get_gpu_memory_info()
                logger.info(f"GPU Memory: {gpu_info}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global asr_engine, profiler
    
    logger.info("Shutting down...")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    asr_engine = None
    profiler = None
    
    logger.info("Shutdown complete")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_memory = None

    if gpu_available:
        gpu_info = GPUProfiler.get_gpu_memory_info()
        gpu_memory = gpu_info.get("allocated_gb")

    return HealthResponse(
        status="healthy" if asr_engine is not None else "not_ready",
        model_loaded=asr_engine is not None,
        device=settings.device,
        gpu_available=gpu_available,
        gpu_memory_allocated_gb=gpu_memory,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if asr_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = ModelLoader.get_model_info(asr_engine.model)

    return ModelInfoResponse(
        model_name=settings.model_name,
        total_parameters=info["total_parameters"],
        model_size_mb=info["model_size_mb"],
        device=settings.device,
        optimization_enabled=settings.use_quantization or settings.use_tensorrt,
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    task: str = "transcribe",
):
    """Transcribe audio file."""
    if asr_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if audio.content_type and not audio.content_type.startswith(("audio/", "application/octet-stream")):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {audio.content_type}. Expected audio file."
        )
    
    audio_buffer = None
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        audio_buffer = BytesIO(audio_bytes)

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_buffer)

        # Resample if needed
        if sample_rate != settings.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, settings.sample_rate)
            waveform = resampler(waveform)

        # Check duration
        duration = waveform.shape[1] / settings.sample_rate
        if duration > settings.max_audio_length_seconds:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long: {duration:.2f}s (max: {settings.max_audio_length_seconds}s)",
            )

        # Transcribe
        start_time = time.perf_counter()
        text = asr_engine.transcribe(waveform, language=language, task=task)
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000

        # Log profiling info
        if profiler:
            logger.info(f"Transcription time: {processing_time_ms:.2f}ms")

        return TranscriptionResponse(
            text=text,
            processing_time_ms=processing_time_ms,
            model_name=settings.model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Cleanup resources
        if audio_buffer is not None:
            audio_buffer.close()


@app.get("/profiling/stats")
async def get_profiling_stats():
    """Get profiling statistics."""
    if not settings.enable_profiling or profiler is None:
        raise HTTPException(status_code=404, detail="Profiling not enabled")

    return JSONResponse(content=profiler.summary())


def main():
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "src.serving.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
