"""Model optimization configuration."""

# Whisper model configurations
WHISPER_CONFIGS = {
    "tiny": {
        "model_id": "openai/whisper-tiny",
        "num_params": "39M",
        "relative_speed": 32,
    },
    "base": {
        "model_id": "openai/whisper-base",
        "num_params": "74M",
        "relative_speed": 16,
    },
    "small": {
        "model_id": "openai/whisper-small",
        "num_params": "244M",
        "relative_speed": 6,
    },
}

# Wav2Vec2 model configurations
WAV2VEC2_CONFIGS = {
    "base": {
        "model_id": "facebook/wav2vec2-base-960h",
        "num_params": "95M",
    },
    "large": {
        "model_id": "facebook/wav2vec2-large-960h",
        "num_params": "317M",
    },
}

# Optimization settings
OPTIMIZATION_PRESETS = {
    "low_latency": {
        "use_fp16": True,
        "use_quantization": False,
        "use_tensorrt": True,
        "batch_size": 1,
    },
    "balanced": {
        "use_fp16": True,
        "use_quantization": True,
        "use_tensorrt": False,
        "batch_size": 4,
    },
    "high_throughput": {
        "use_fp16": False,
        "use_quantization": False,
        "use_tensorrt": False,
        "batch_size": 16,
    },
}
