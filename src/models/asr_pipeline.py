from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src._optional_imports import optional_attr_import, optional_import

torch = optional_import("torch")
torchaudio = optional_import("torchaudio")
hf_pipeline = optional_attr_import("transformers", "pipeline")

TARGET_SAMPLE_RATE = 16000


def _default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass(slots=True)
class ASRConfig:
    backend: str = "transformers"
    model_id: str = "openai/whisper-small"
    device: str = _default_device()
    dtype: str = "float16"
    chunk_length_s: float = 15.0
    stride_length_s: float = 5.0


class ASRPipeline:
    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self._pipe: Any | None = None
        if self.config.backend != "mock":
            self._pipe = self._build_pipeline()

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
        if language:
            generate_kwargs["language"] = language

        if generate_kwargs:
            output = self._pipe(payload, generate_kwargs=generate_kwargs)
        else:
            output = self._pipe(payload)

        if isinstance(output, dict) and "text" in output:
            return str(output["text"]).strip()
        return str(output).strip()
