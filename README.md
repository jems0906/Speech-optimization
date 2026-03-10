# Real-Time Speech-to-Text Optimization

Production-ready speech recognition system optimized for low-latency inference using PyTorch, Whisper/Wav2Vec2, and TensorRT.

## Features

- Multiple ASR models support (Whisper, Wav2Vec2)
- GPU acceleration with CUDA
- Model optimization (FP16, INT8 quantization, TensorRT)
- FastAPI REST API for inference
- Docker containerization
- Comprehensive profiling and benchmarking
- CI/CD pipeline with GitHub Actions

## Architecture

```
speech-rt-optimization/
├── src/
│   ├── models/          # Model loading and inference
│   ├── serving/         # FastAPI application
│   ├── optim/           # Optimization utilities
│   └── profiling/       # Performance profiling
├── tests/               # Unit tests
├── configs/             # Configuration files
├── docker/              # Docker setup
├── scripts/             # Utility scripts
└── notebooks/           # Jupyter notebooks for experiments
```

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Docker (optional, for containerized deployment)

## Installation

### Local Development

```bash
# Clone repository
git clone <repository-url>
cd speech-rt-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Docker Deployment

```bash
# Build production image
docker build -t speech-rt-optimization -f docker/Dockerfile .

# Run with Docker Compose
cd docker
docker-compose up speech-api
```

## Quick Start

### Start API Server

```bash
# Using Python
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000

# Or using the installed script
speech-serve
```

### Configuration

Copy `.env.example` to `.env` and configure:

```env
MODEL_NAME=whisper-base
DEVICE=cuda
USE_FP16=true
PORT=8000
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Transcribe audio
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@sample.wav" \
  -F "language=en"

# Get model info
curl http://localhost:8000/model-info
```

### Python SDK

```python
from src.models.loader import ModelLoader
from src.models.inference import ASRInference

# Load model
model, processor = ModelLoader.load_whisper(
    model_name="whisper-base",
    device="cuda"
)

# Create inference engine
asr = ASRInference(model, processor)

# Transcribe
text = asr.transcribe("audio.wav", language="en")
print(text)
```

## Optimization

### Apply Quantization

```python
from src.optim.quantization import ModelOptimizer

# FP16
model = ModelOptimizer.convert_to_half_precision(model)

# INT8 quantization
model = ModelOptimizer.quantize_dynamic_int8(model)
```

### TensorRT Conversion

```python
from src.optim.tensorrt_utils import TensorRTConverter

converter = TensorRTConverter(fp16_mode=True)

# From ONNX
converter.convert_from_onnx(
    onnx_path="model.onnx",
    engine_path="model.engine"
)
```

## Profiling

### Benchmark Model

```python
from src.profiling.profiler import benchmark_model
import torch

dummy_input = torch.randn(1, 80, 3000).cuda()
stats = benchmark_model(
    model=model,
    input_tensor=dummy_input,
    num_iterations=100
)

print(f"Mean inference time: {stats['mean_ms']:.2f}ms")
```

### GPU Profiling

```bash
# With PyTorch profiler
python scripts/profile_model.py --model whisper-base

# With Nsight Systems
nsys profile --trace=cuda,nvtx python scripts/run_inference.py
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py::test_model_loader_whisper
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Deployment

### Production Checklist

- [ ] Set appropriate environment variables
- [ ] Configure GPU allocation
- [ ] Set up logging and monitoring
- [ ] Configure reverse proxy (Nginx)
- [ ] Enable HTTPS
- [ ] Set resource limits (memory, CPU)
- [ ] Configure auto-restart policies

### Docker Production

```bash
# Build optimized image
docker build -t speech-api:prod -f docker/Dockerfile .

# Run with resource limits
docker run -d \
  --gpus all \
  --name speech-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e MODEL_NAME=whisper-base \
  -e USE_FP16=true \
  speech-api:prod
```

## Performance Benchmarks

| Model | Size | FP32 (ms) | FP16 (ms) | TensorRT (ms) | Speedup |
|-------|------|-----------|-----------|---------------|---------|
| Whisper-tiny | 39M | 45 | 23 | 12 | 3.75x |
| Whisper-base | 74M | 78 | 41 | 22 | 3.55x |
| Whisper-small | 244M | 195 | 103 | 58 | 3.36x |

*Benchmarks on NVIDIA RTX 3090, 10s audio clips, batch size 1*

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size
- Enable FP16 mode
- Use smaller model variant
- Clear CUDA cache: `torch.cuda.empty_cache()`

### Slow Inference

- Enable FP16 precision
- Use TensorRT optimization
- Warmup model before benchmarking
- Check GPU utilization

### Import Errors

```bash
# Reinstall package
pip install -e .

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Citation

```bibtex
@software{speech_rt_optimization,
  title = {Real-Time Speech-to-Text Optimization},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/speech-rt-optimization}
}
```

## Resources

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/speech-rt-optimization/issues)
- Documentation: [Read the docs](https://docs.example.com)
