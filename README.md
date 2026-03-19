# Speech RT Optimization

Real-time speech-to-text service optimized for low latency and production readiness.

## Stack

- Python 3.10+
- Optional C++17 for custom ops or other performance-critical extensions
- PyTorch + Torchaudio
- Hugging Face Transformers ASR models (Whisper and Wav2Vec2 families)
- ONNX + TensorRT or Torch-TensorRT for runtime acceleration
- PyTorch quantization and bitsandbytes for INT8 / FP16 optimization paths
- FastAPI + Uvicorn/Gunicorn
- Docker + optional Nginx reverse proxy
- CUDA toolkit + cuDNN for GPU deployment
- PyTorch Profiler plus optional Nsight Systems / Nsight Compute for GPU profiling
- pytest, coverage, pre-commit, GitHub Actions

## Repository Layout

```text
speech-rt-optimization/
  src/
    models/
    serving/
    optim/
    profiling/
  tests/
  notebooks/
  configs/
  docker/
  scripts/
  requirements.txt
  pyproject.toml
  README.md
```

## Quick Start

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
pip install .[asr,optimization,dev]
```

2. Optional: install a build toolchain if you plan to add custom C++17 extensions.

```bash
# Windows: install Visual Studio Build Tools with C++ workload
# Linux: install gcc/g++ with C++17 support
```

3. Run the API.

```bash
# Development
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Production (single-worker GPU)
gunicorn src.serving.app:app -c gunicorn.conf.py
```

4. Test health endpoint.

```bash
curl http://localhost:8000/health
```

5. Send transcription request.

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio_file=@sample.wav" \
  -F "language=en"
```

6. Stream transcription over WebSocket.

```python
import asyncio
import json
import numpy as np
import websockets

async def transcribe_stream(audio_np: np.ndarray, sample_rate: int = 16000) -> dict:
    uri = "ws://localhost:8000/ws/transcribe"
    async with websockets.connect(uri) as ws:
        await ws.send("LANG:en")
        await ws.send(audio_np.astype(np.float32).tobytes())
        await ws.send("END")
        return json.loads(await ws.recv())

asyncio.run(transcribe_stream(np.zeros(16000, dtype=np.float32)))
```

The WebSocket protocol:
- `LANG:<code>` text frame: set language (optional, before audio).
- Binary frames: raw mono float32 PCM at 16 kHz.
- `END` text frame: signals end of audio; server replies with JSON and closes.

## Runtime Configuration

Environment variables:

- `ASR_BACKEND` (`transformers` or `mock`)
- `CONFIG_PATH` (path to a YAML config file; defaults to `configs/default.yaml`; env vars override YAML values)
- `ASR_MODEL_ID` (default `openai/whisper-small`)
- `ASR_MODEL_FAMILY` (`auto`, `whisper`, or `wav2vec2`)
- `ASR_DEVICE` (`cuda` or `cpu`)
- `ASR_DTYPE` (`float16`, `float32`, `bfloat16`)
- `ASR_CHUNK_LENGTH_S`
- `ASR_STRIDE_LENGTH_S`
- `ENABLE_TORCH_COMPILE`
- `ENABLE_DYNAMIC_INT8`
- `ENABLE_PRUNING`
- `PRUNING_AMOUNT` (float in `[0, 1)`, default `0.2`)

## Optimization Workflow

**Step 1 — Baseline**

```bash
python scripts/profile_latency.py audio.wav --device cpu --dtype float32 --model-family whisper
```

**Step 2 — torch.compile**

```bash
python scripts/profile_latency.py audio.wav --device cuda --torch-compile --model-family whisper
```

**Step 3 — Dynamic INT8 quantization (CPU)**

```bash
python scripts/profile_latency.py audio.wav --device cpu --dynamic-int8 --model-family whisper
```

**Step 4 — Magnitude pruning (sparsity)**

```bash
python scripts/profile_latency.py audio.wav --device cuda --pruning-amount 0.2 --model-family whisper
```

**Step 5 — bitsandbytes INT8 (GPU VRAM reduction)**

```python
from src.optim import apply_bitsandbytes_int8
model = apply_bitsandbytes_int8(pipeline.model)
```

**Step 6 — FP16 (half-precision, 2× memory, faster on Tensor Cores)**

```python
from src.optim import convert_to_fp16
model = convert_to_fp16(pipeline.model)
```

**Step 7 — Torch-TensorRT (maximum GPU throughput)**

```python
from src.optim import compile_with_tensorrt
# input_shapes matches the encoder: (batch, n_mels, frames)
model = compile_with_tensorrt(pipeline.model, input_shapes=[(1, 80, 3000)], dtype="float16")
```

**Step 8 — ONNX export (optionally build a TensorRT engine in one step)**

```bash
python scripts/export_onnx.py --model-id openai/whisper-small --model-family whisper --output artifacts/whisper.onnx

# Also serialize a TensorRT FP16 engine alongside the ONNX file:
python scripts/export_onnx.py --model-id openai/whisper-small --model-family whisper --output artifacts/whisper.onnx \
  --tensorrt-engine artifacts/whisper.trt

# Wav2Vec2 example:
python scripts/export_onnx.py --model-id facebook/wav2vec2-base-960h --model-family wav2vec2 --output artifacts/wav2vec2.onnx
```

**Step 9 — PyTorch Profiler (operator-level breakdown)**

```bash
python scripts/profile_latency.py audio.wav --device cuda --torch-profiler --model-family whisper
```

**Step 10 — Nsight Systems / Nsight Compute (GPU profiling)**

```bash
# Timeline/system view
python scripts/profile_nsight.py audio.wav --tool nsys --model-family whisper --output artifacts/nsys_speech

# Kernel-level inspection
python scripts/profile_nsight.py audio.wav --tool ncu --model-family whisper --output artifacts/ncu_speech
```

**Step 11 — Visualize before/after optimization**

```bash
python scripts/visualize_benchmark.py audio.wav --device cuda \
  --metrics-json artifacts/benchmark_metrics.json \
  --metrics-csv artifacts/benchmark_metrics.csv
```

To keep a longitudinal benchmark history, append each run into the JSON file:

```bash
python scripts/visualize_benchmark.py audio.wav --device cuda \
  --metrics-json artifacts/benchmark_history.json \
  --append-json-history
```

Compare two runs (baseline vs candidate) and gate regressions:

```bash
python scripts/compare_benchmark_history.py artifacts/benchmark_history.json \
  --max-p95-regression-pct 5 \
  --max-peak-mem-regression-pct 10 \
  --fail-on-regression
```

Automated GitHub workflow is available for this gate:
- Workflow: `.github/workflows/benchmark-regression.yml`
- Manual trigger: `workflow_dispatch` with configurable thresholds and run indices
- PR trigger: runs when benchmark history or comparator logic changes
- Artifact: uploads `artifacts/benchmark_comparison.json` and markdown summary
- PR feedback: posts/updates a sticky benchmark delta comment on pull requests
- PR header: the sticky summary header is prefixed with the triage label for faster scanning
- Triage labels: `BLOCKER` (high), `REGRESSION` (medium), `WATCH` (low), `CLEAN` (none)
- CLI status lines: emits `BENCHMARK_PASSED=...`, `BENCHMARK_SEVERITY=...`, `BENCHMARK_TRIAGE=...` for simple CI parsing
- Workflow outputs: exposes parsed status as job outputs (`benchmark_passed`, `benchmark_severity`, `benchmark_triage`, `benchmark_severity_rank`)
- Regression notice: a follow-up workflow job runs automatically for medium severity (`benchmark_severity_rank == 2`)
- Blocker escalation: a separate workflow job runs for high severity (`benchmark_severity_rank >= 3`) and uploads a blocker summary artifact

For CI gating against your latency objective, fail the command when any variant
misses the p95 target:

```bash
python scripts/visualize_benchmark.py audio.wav --device cuda \
  --latency-target-ms 200 \
  --fail-if-target-missed
```

**Step 12 — Deploy**

```bash
docker build -f docker/Dockerfile -t speech-rt-optimization . && docker run --gpus all -p 8000:8000 speech-rt-optimization
```

The production container entrypoint uses Gunicorn with a single Uvicorn worker via `gunicorn.conf.py`.

## Tests and Quality

```bash
pytest
coverage run -m pytest && coverage report -m
pre-commit run --all-files
```

## Packaging and Release

Build distribution artifacts locally:

```bash
python -m pip install -e .[dev]
python -m build
```

This creates:
- `dist/*.whl`
- `dist/*.tar.gz`

Tag-based releases are automated via GitHub Actions:
- Workflow: `.github/workflows/release.yml`
- Trigger: push a SemVer tag like `v0.1.1`
- Output: GitHub Release with `dist/*` artifacts and `SHA256SUMS.txt`

Example release flow:

```bash
git tag v0.1.1
git push origin v0.1.1
```

Use `CHANGELOG.md` to document release notes before tagging.

PyPI publishing is also automated:
- Workflow: `.github/workflows/publish-pypi.yml`
- Trigger: GitHub Release `published` (or manual workflow dispatch)
- Auth: Trusted Publishing via GitHub OIDC (`id-token: write`)
- Post-publish check: smoke-installs released version from PyPI (with retry for index propagation)

TestPyPI pre-release publishing is also available:
- Workflow: `.github/workflows/publish-testpypi.yml`
- Trigger: GitHub Release `prereleased` (or manual workflow dispatch)
- Auth: Trusted Publishing via GitHub OIDC (`id-token: write`)
- Post-publish check: smoke-installs prerelease from TestPyPI (with retry for index propagation)

PyPI trusted publishing setup (one-time):
1. Create your project on PyPI (same normalized name as `pyproject.toml`).
2. On PyPI, add a trusted publisher for this repository and workflow:
  - Owner/repo: your GitHub repository
  - Workflow file: `publish-pypi.yml`
  - Environment: `pypi`
3. In GitHub repository settings, create environment `pypi` and add any required approval rules.

TestPyPI trusted publishing setup (one-time):
1. Create your project on TestPyPI.
2. On TestPyPI, add a trusted publisher for this repository and workflow:
  - Owner/repo: your GitHub repository
  - Workflow file: `publish-testpypi.yml`
  - Environment: `testpypi`
3. In GitHub repository settings, create environment `testpypi` and add any required approval rules.

After setup, publishing a GitHub Release will build and publish `dist/*` to PyPI automatically.

Recommended release pattern:
1. Publish a prerelease first to validate installability on TestPyPI.
2. Publish a final release to push to production PyPI.

## Release Checklist

Use this checklist for each versioned release.

Preflight:
1. Update `CHANGELOG.md` for the target version.
2. Ensure `pyproject.toml` has the intended `project.version`.
3. Run local quality gates:

```bash
python -m black --check .
python -m isort --check-only .
python -m flake8 .
python -m coverage run -m pytest tests/ -q && python -m coverage report -m
```

Prerelease (TestPyPI):
1. Create and push a prerelease tag (example `v0.1.2-rc.1`).
2. Create a GitHub Release and mark it as prerelease.
3. Verify workflow `.github/workflows/publish-testpypi.yml` succeeds, including smoke-install.

Final release (PyPI):
1. Create and push final tag (example `v0.1.2`).
2. Publish GitHub Release (non-prerelease).
3. Verify workflow `.github/workflows/publish-pypi.yml` succeeds, including smoke-install.

Post-release checks:
1. Confirm package install in a clean environment.
2. Confirm `README.md` and `CHANGELOG.md` are consistent with released features.
3. Start next cycle by adding notes under `[Unreleased]` in `CHANGELOG.md`.

## Docker

Build and run the API alone:

```bash
docker build -f docker/Dockerfile -t speech-rt-optimization .
docker run --gpus all -p 8000:8000 speech-rt-optimization
```

Run with Nginx reverse proxy (prod-like):

```bash
docker compose -f docker/docker-compose.yml up --build
```

The compose stack exposes port 80 via Nginx and routes to the API container on port 8000.

## Notes

- `ASR_BACKEND=mock` is useful for API tests without loading model weights.
- `requirements.txt` contains the shared baseline packages; install `.[asr,optimization,dev]` to match the full project stack from this checklist.
- For production GPU serving, use a host with NVIDIA drivers, CUDA, and cuDNN support.
- Nsight CLI tools (`nsys`, `ncu`) must be installed separately from the NVIDIA CUDA toolkit if you want GPU timeline or kernel analysis.
