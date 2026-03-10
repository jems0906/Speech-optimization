"""Configuration management."""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Model settings
    model_name: str = "whisper-base"
    model_type: str = "whisper"
    device: str = "cuda"
    use_fp16: bool = True

    # Optimization settings
    use_quantization: bool = False
    use_tensorrt: bool = False
    tensorrt_engine_path: Optional[str] = None

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    # Audio settings
    sample_rate: int = 16000
    max_audio_length_seconds: float = 30.0

    # Profiling
    enable_profiling: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.max_audio_length_seconds <= 0:
            raise ValueError(
                f"max_audio_length_seconds must be positive, got "
                f"{self.max_audio_length_seconds}"
            )
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be between 1-65535, got {self.port}")
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")
        if self.use_quantization and self.use_fp16:
            import logging

            logging.getLogger(__name__).warning(
                "Using both quantization and FP16 may not provide optimal performance"
            )


settings = Settings()
