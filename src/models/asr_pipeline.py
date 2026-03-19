from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src._optional_imports import optional_attr_import, optional_import

torch = optional_import("torch")
torchaudio = optional_import("torchaudio")
hf_pipeline = optional_attr_import("transformers", "pipeline")

TARGET_SAMPLE_RATE = 16000


SUPPORTED_MODEL_FAMILIES = {"auto", "whisper", "wav2vec2"}


def _default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(slots=True)
class ASRConfig:
    backend: str = "transformers"
    model_id: str = "openai/whisper-small"
    model_family: str = "auto"
    device: str = _default_device()
    dtype: str = "float16"
    chunk_length_s: float = 15.0
    stride_length_s: float = 5.0


def normalize_model_family(value: str | None) -> str:
    family = (value or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "whisper": "whisper",
        "wav2vec": "wav2vec2",
        "wav2vec2": "wav2vec2",
    }
    normalized = aliases.get(family)
    if normalized is None:
        raise ValueError(
            f"Unsupported model_family '{value}'. Expected one of: {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )
    return normalized


def infer_model_family(model_id: str) -> str:
    normalized = model_id.strip().lower()
    if "wav2vec" in normalized or "hubert" in normalized or "ctc" in normalized:
        return "wav2vec2"
    return "whisper"


def resolve_model_family(configured_family: str | None, *, model_id: str, model: Any | None = None) -> str:
    family = normalize_model_family(configured_family)
    if family != "auto":
        return family

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if isinstance(model_type, str) and model_type:
        try:
            return normalize_model_family(model_type)
        except ValueError:
            pass

    return infer_model_family(model_id)


def build_export_metadata(
    model: Any,
    *,
    model_family: str | None,
    model_id: str,
    torch_module: Any | None = None,
) -> tuple[Any, list[str]]:
    active_torch = torch if torch_module is None else torch_module
    if active_torch is None:
        raise RuntimeError(
            "ONNX export requires torch and onnx. Install optional dependencies with: "
            "pip install .[asr,optimization]"
        )

    resolved_family = resolve_model_family(model_family, model_id=model_id, model=model)
    if resolved_family == "wav2vec2":
        return active_torch.randn(1, TARGET_SAMPLE_RATE, dtype=active_torch.float32), [
            "input_values"
        ]

    return active_torch.randn(1, 80, 3000, dtype=active_torch.float32), ["input_features"]


class ASRPipeline:
    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self._pipe: Any | None = None
        self.model_family = normalize_model_family(self.config.model_family)
        if self.config.backend != "mock":
            self._pipe = self._build_pipeline()
            self.model_family = resolve_model_family(
                self.config.model_family,
                model_id=self.config.model_id,
                model=self._pipe.model,
            )

    def _resolved_model_family(self) -> str:
        if self._pipe is None:
            return normalize_model_family(self.model_family)
        return resolve_model_family(
            self.model_family,
            model_id=self.config.model_id,
            model=getattr(self._pipe, "model", None),
        )

    @property
    def model(self) -> Any:
        if self._pipe is None:
            raise RuntimeError("Mock backend does not expose a model")
        return self._pipe.model

    @model.setter
    def model(self, value: Any) -> None:
        if self._pipe is None:
            raise RuntimeError("Mock backend does not expose a model")
        self._pipe.model = value

    def _build_pipeline(self) -> Any:
        if hf_pipeline is None or torch is None:
            raise RuntimeError(
                "Transformers ASR backend requires torch and transformers. "
                "Install optional dependencies with: pip install .[asr]"
            )

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.dtype.lower(), torch.float16)
        device = 0 if self.config.device == "cuda" and torch.cuda.is_available() else -1

        return hf_pipeline(
            task="automatic-speech-recognition",
            model=self.config.model_id,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=self.config.chunk_length_s,
            stride_length_s=(self.config.stride_length_s, self.config.stride_length_s),
        )

    def transcribe(self, waveform: Any, sample_rate: int, language: str | None = None) -> str:
        if self.config.backend == "mock":
            return "mock transcription"

        if torch is None or torchaudio is None:
            raise RuntimeError(
                "Audio transcription requires torch and torchaudio. "
                "Install optional dependencies with: pip install .[asr]"
            )

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        waveform = waveform.detach().cpu().to(torch.float32)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

        if self._pipe is None:
            raise RuntimeError("ASR pipeline was not initialized")

        payload = {"array": waveform.numpy(), "sampling_rate": sample_rate}
        generate_kwargs: dict[str, str] = {}
        if language and self._resolved_model_family() == "whisper":
            generate_kwargs["language"] = language

        if generate_kwargs:
            output = self._pipe(payload, generate_kwargs=generate_kwargs)
        else:
            output = self._pipe(payload)

        if isinstance(output, dict) and "text" in output:
            return str(output["text"]).strip()
        return str(output).strip()
