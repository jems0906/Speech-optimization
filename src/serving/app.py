from __future__ import annotations

import io
import logging
import os
import time
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from src._optional_imports import optional_import
from src.models.asr_pipeline import ASRConfig, ASRPipeline, normalize_model_family
from src.optim.inference_optimizer import OptimizationConfig, optimize_asr_pipeline

torch = optional_import("torch")
torchaudio = optional_import("torchaudio")


@dataclass(slots=True)
class RuntimeSettings:
    asr_backend: str
    asr_model_id: str
    asr_model_family: str
    device: str
    dtype: str
    chunk_length_s: float
    stride_length_s: float
    enable_torch_compile: bool
    enable_dynamic_int8: bool
    enable_pruning: bool
    pruning_amount: float


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_yaml_defaults(path: Path) -> dict:
    """Load asr/optimization keys from a YAML config file.

    Returns an empty dict when the file does not exist so callers can always
    fall through to hard-coded defaults without raising.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    asr = data.get("asr", {})
    optim = data.get("optimization", {})
    return {
        "asr_backend": str(asr.get("backend", "transformers")),
        "asr_model_id": str(asr.get("model_id", "openai/whisper-small")),
        "asr_model_family": normalize_model_family(str(asr.get("model_family", "auto"))),
        "device": str(asr.get("device", "cpu")),
        "dtype": str(asr.get("dtype", "float16")),
        "chunk_length_s": float(asr.get("chunk_length_s", 15.0)),
        "stride_length_s": float(asr.get("stride_length_s", 5.0)),
        "enable_torch_compile": bool(optim.get("enable_torch_compile", False)),
        "enable_dynamic_int8": bool(optim.get("enable_dynamic_int8", False)),
        "enable_pruning": bool(optim.get("enable_pruning", False)),
        "pruning_amount": float(optim.get("pruning_amount", 0.2)),
    }


def load_runtime_settings() -> RuntimeSettings:
    config_path = Path(os.getenv("CONFIG_PATH", "configs/default.yaml"))
    yd = _load_yaml_defaults(config_path)
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return RuntimeSettings(
        asr_backend=os.getenv("ASR_BACKEND", yd.get("asr_backend", "transformers")),
        asr_model_id=os.getenv("ASR_MODEL_ID", yd.get("asr_model_id", "openai/whisper-small")),
        asr_model_family=normalize_model_family(
            os.getenv("ASR_MODEL_FAMILY", yd.get("asr_model_family", "auto"))
        ),
        device=os.getenv("ASR_DEVICE", yd.get("device", default_device)),
        dtype=os.getenv("ASR_DTYPE", yd.get("dtype", "float16")),
        chunk_length_s=float(os.getenv("ASR_CHUNK_LENGTH_S", yd.get("chunk_length_s", 15.0))),
        stride_length_s=float(os.getenv("ASR_STRIDE_LENGTH_S", yd.get("stride_length_s", 5.0))),
        enable_torch_compile=_env_bool(
            "ENABLE_TORCH_COMPILE", yd.get("enable_torch_compile", False)
        ),
        enable_dynamic_int8=_env_bool("ENABLE_DYNAMIC_INT8", yd.get("enable_dynamic_int8", False)),
        enable_pruning=_env_bool("ENABLE_PRUNING", yd.get("enable_pruning", False)),
        pruning_amount=float(os.getenv("PRUNING_AMOUNT", yd.get("pruning_amount", 0.2))),
    )


class HealthResponse(BaseModel):
    status: str
    backend: str
    device: str


class TranscribeResponse(BaseModel):
    text: str
    latency_ms: float
    duration_s: float
    sample_rate: int


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech RT Optimization", version="0.1.0")


@lru_cache(maxsize=1)
def get_asr_service() -> ASRPipeline:
    settings = load_runtime_settings()
    asr_service = ASRPipeline(
        ASRConfig(
            backend=settings.asr_backend,
            model_id=settings.asr_model_id,
            model_family=settings.asr_model_family,
            device=settings.device,
            dtype=settings.dtype,
            chunk_length_s=settings.chunk_length_s,
            stride_length_s=settings.stride_length_s,
        )
    )
    return optimize_asr_pipeline(
        asr_service,
        OptimizationConfig(
            enable_torch_compile=settings.enable_torch_compile,
            enable_dynamic_int8=settings.enable_dynamic_int8,
            enable_pruning=settings.enable_pruning,
            pruning_amount=settings.pruning_amount,
        ),
    )


def _decode_wav_fallback(data: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            raw = wav.readframes(frame_count)
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode WAV audio: {exc}") from exc

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported WAV bit depth: {sample_width * 8} bits",
        )

    if channels > 1:
        audio = audio.reshape(-1, channels).T
    else:
        audio = audio.reshape(1, -1)

    return audio, sample_rate


async def _decode_audio(file: UploadFile) -> tuple[object, int]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Received empty audio file")

    if torchaudio is not None:
        try:
            waveform, sample_rate = torchaudio.load(io.BytesIO(data))
            return waveform, sample_rate
        except Exception:
            pass

    # Fallback decoder keeps tests and basic WAV inference usable without torchaudio.
    waveform_np, sample_rate = _decode_wav_fallback(data)
    if torch is not None:
        return torch.from_numpy(waveform_np), sample_rate

    return waveform_np, sample_rate


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = load_runtime_settings()
    return HealthResponse(status="ok", backend=settings.asr_backend, device=settings.device)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio_file: UploadFile = File(...),
    language: str | None = Form(default=None),
    asr_service: ASRPipeline = Depends(get_asr_service),
) -> TranscribeResponse:
    start = time.perf_counter()
    waveform, sample_rate = await _decode_audio(audio_file)

    text = asr_service.transcribe(waveform=waveform, sample_rate=sample_rate, language=language)
    latency_ms = (time.perf_counter() - start) * 1000
    duration_s = float(waveform.shape[-1] / sample_rate) if sample_rate > 0 else 0.0

    logger.info(
        "POST /transcribe language=%s duration_s=%.3f latency_ms=%.2f",
        language or "auto",
        round(duration_s, 3),
        round(latency_ms, 2),
    )
    if latency_ms > 200:
        logger.warning(
            "POST /transcribe latency %.2f ms exceeds 200 ms real-time target", latency_ms
        )
    return TranscribeResponse(
        text=text,
        latency_ms=round(latency_ms, 2),
        duration_s=round(duration_s, 3),
        sample_rate=sample_rate,
    )


@app.websocket("/ws/transcribe")
async def ws_transcribe(
    websocket: WebSocket,
    asr_service: ASRPipeline = Depends(get_asr_service),
) -> None:
    """Real-time transcription over WebSocket.

    Protocol (client -> server):
    - Text frame "LANG:<code>": set ISO language code before audio (optional).
    - Binary frames:           raw mono float32 PCM at 16 kHz.
    - Text frame "END":        signal end of stream; triggers transcription.

    Server responds with a single JSON frame:
    {"text": str, "latency_ms": float, "duration_s": float, "sample_rate": int}
    """
    await websocket.accept()
    chunks: list[bytes] = []
    language: str | None = None

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes"):
                chunks.append(msg["bytes"])
            elif msg.get("text") == "END":
                break
            elif msg.get("text", "").startswith("LANG:"):
                language = msg["text"][5:].strip() or None
    except WebSocketDisconnect:
        return

    if not chunks:
        await websocket.send_json({"error": "No audio received"})
        await websocket.close()
        return

    raw = b"".join(chunks)
    waveform_np = np.frombuffer(raw, dtype=np.float32).copy()
    waveform: object = torch.from_numpy(waveform_np) if torch is not None else waveform_np
    sample_rate = 16000

    try:
        start = time.perf_counter()
        result = asr_service.transcribe(
            waveform=waveform, sample_rate=sample_rate, language=language
        )
        latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        logger.exception("WebSocket transcription error")
        await websocket.send_json({"error": str(exc)})
        await websocket.close()
        return

    duration_s = len(waveform_np) / sample_rate
    logger.info(
        "WS /ws/transcribe language=%s duration_s=%.3f latency_ms=%.2f",
        language or "auto",
        round(duration_s, 3),
        round(latency_ms, 2),
    )
    if latency_ms > 200:
        logger.warning(
            "WS /ws/transcribe latency %.2f ms exceeds 200 ms real-time target", latency_ms
        )
    await websocket.send_json(
        {
            "text": result,
            "latency_ms": round(latency_ms, 2),
            "duration_s": round(duration_s, 3),
            "sample_rate": sample_rate,
        }
    )
    await websocket.close()
