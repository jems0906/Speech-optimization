"""ASR inference wrapper."""

from typing import Optional, Union

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class ASRInference:
    """Real-time ASR inference handler."""

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        device: str = "cuda",
        sample_rate: int = 16000,
    ):
        """Initialize inference handler."""
        self.model = model
        self.processor = processor
        self.device = device
        self.sample_rate = sample_rate

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[torch.Tensor, str],
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> str:
        """Transcribe audio to text."""
        if isinstance(audio, str):
            audio, sr = torchaudio.load(audio)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)

        # Ensure mono channel
        if len(audio.shape) > 1 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Handle both tensor and already numpy arrays
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().numpy()
        else:
            audio = audio.squeeze() if hasattr(audio, 'squeeze') else audio

        # Process audio
        input_features = self.processor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features

        input_features = input_features.to(self.device)

        # Set language if provided
        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task=task
            )

        # Generate
        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription

    def warmup(self, duration: float = 5.0):
        """Warmup model with dummy input."""
        # Create dummy audio in correct shape (channels, samples)
        dummy_audio = torch.randn(1, int(self.sample_rate * duration))
        # warmup with actual audio tensor, not string path
        try:
            _ = self.transcribe(dummy_audio)
        except Exception as e:
            # Warmup failures are non-critical
            import logging
            logging.getLogger(__name__).warning(f"Warmup failed: {e}")
