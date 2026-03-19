import asyncio
import importlib
import math
import os
import struct
import sys
import tempfile
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml
from fastapi import HTTPException, WebSocketDisconnect
from fastapi.testclient import TestClient

from src._optional_imports import optional_attr_import, optional_import
from src.models.asr_pipeline import (
    ASRConfig,
    ASRPipeline,
    build_export_metadata,
    infer_model_family,
    normalize_model_family,
    resolve_model_family,
)
from src.optim.inference_optimizer import (
    OptimizationConfig,
    apply_bitsandbytes_int8,
    apply_magnitude_pruning,
    compile_with_tensorrt,
    convert_to_fp16,
    optimize_asr_pipeline,
)
from src.profiling.benchmark import (
    LatencyStats,
    MemoryStats,
    measure_gpu_memory,
    profile_latency,
    profile_with_torch_profiler,
)
from src.serving.app import (
    _decode_wav_fallback,
    _load_yaml_defaults,
    app,
    get_asr_service,
    ws_transcribe,
)


class FakeASR:
    def transcribe(self, waveform, sample_rate, language=None):
        return "test transcription"


def _build_wav_bytes(duration_s: float = 0.15, sample_rate: int = 16000) -> bytes:
    frame_count = int(duration_s * sample_rate)
    amplitude = 12000
    freq = 440.0

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(frame_count):
            value = int(amplitude * math.sin((2 * math.pi * freq * i) / sample_rate))
            wav.writeframes(struct.pack("<h", value))

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["backend"]
    assert payload["device"]


def test_transcribe_endpoint_with_override() -> None:
    app.dependency_overrides[get_asr_service] = lambda: FakeASR()
    client = TestClient(app)

    wav_bytes = _build_wav_bytes()
    response = client.post(
        "/transcribe",
        files={"audio_file": ("sample.wav", wav_bytes, "audio/wav")},
        data={"language": "en"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "test transcription"
    assert payload["sample_rate"] == 16000
    assert payload["duration_s"] > 0
    assert payload["latency_ms"] >= 0

    app.dependency_overrides.clear()


def test_transcribe_empty_file_returns_400() -> None:
    app.dependency_overrides[get_asr_service] = lambda: FakeASR()
    client = TestClient(app)

    response = client.post(
        "/transcribe",
        files={"audio_file": ("empty.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    app.dependency_overrides.clear()


def test_decode_wav_fallback_without_torch() -> None:
    serving_module = sys.modules["src.serving.app"]

    wav_bytes = _build_wav_bytes()
    original_torch = serving_module.torch
    original_torchaudio = serving_module.torchaudio

    try:
        serving_module.torch = None  # type: ignore[assignment]
        serving_module.torchaudio = None  # type: ignore[assignment]

        # Create a mock UploadFile
        class MockFile:
            async def read(self):
                return wav_bytes

        # Call _decode_audio which exercises line 195
        waveform, sample_rate = asyncio.run(serving_module._decode_audio(MockFile()))
        assert isinstance(waveform, np.ndarray)
        assert sample_rate == 16000
        assert waveform.shape[0] > 0
    finally:
        serving_module.torch = original_torch
        serving_module.torchaudio = original_torchaudio


# ---------------------------------------------------------------------------
# WebSocket endpoint tests
# ---------------------------------------------------------------------------


def test_ws_transcribe_endpoint() -> None:
    app.dependency_overrides[get_asr_service] = lambda: FakeASR()
    client = TestClient(app)

    samples = np.zeros(1600, dtype=np.float32)  # 0.1 s of silence at 16 kHz

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text("LANG:en")
        ws.send_bytes(samples.tobytes())
        ws.send_text("END")
        data = ws.receive_json()

    assert data["text"] == "test transcription"
    assert data["sample_rate"] == 16000
    assert data["duration_s"] > 0
    assert data["latency_ms"] >= 0
    app.dependency_overrides.clear()


def test_ws_transcribe_empty_sends_error() -> None:
    app.dependency_overrides[get_asr_service] = lambda: FakeASR()
    client = TestClient(app)

    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text("END")
        data = ws.receive_json()

    assert "error" in data
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Profiling tests
# ---------------------------------------------------------------------------


def test_profile_latency_stats() -> None:
    counter: dict[str, int] = {"n": 0}

    def noop() -> None:
        counter["n"] += 1

    stats = profile_latency(noop, iterations=10, warmup=2)

    assert isinstance(stats, LatencyStats)
    assert stats.count == 10
    assert stats.mean_ms >= 0.0
    assert stats.p99_ms >= stats.p95_ms >= stats.p50_ms >= 0.0
    assert counter["n"] == 12  # 2 warmup + 10 measured


def test_profile_with_torch_profiler_without_torch_raises() -> None:
    import src.profiling.benchmark as bm

    original = bm.torch
    try:
        bm.torch = None  # type: ignore[assignment]
        try:
            profile_with_torch_profiler(lambda: None)
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "torch" in str(exc).lower()
    finally:
        bm.torch = original


def test_optional_import_helpers() -> None:
    assert optional_import("json") is not None
    assert optional_import("definitely_missing_module_for_test") is None
    assert callable(optional_attr_import("json", "loads"))
    assert optional_attr_import("definitely_missing_module_for_test", "loads") is None


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------


def test_optimize_asr_pipeline_mock_passthrough() -> None:
    pipeline = ASRPipeline(ASRConfig(backend="mock"))
    config = OptimizationConfig(enable_torch_compile=True, enable_dynamic_int8=True)

    result = optimize_asr_pipeline(pipeline, config)

    assert result is pipeline


def test_compile_with_tensorrt_without_torch_raises() -> None:
    import src.optim.inference_optimizer as oi

    original = oi.torch
    try:
        oi.torch = None  # type: ignore[assignment]
        try:
            compile_with_tensorrt(object(), input_shapes=[(1, 80, 3000)])
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "torch" in str(exc).lower()
    finally:
        oi.torch = original


def test_apply_bitsandbytes_int8_without_torch_raises() -> None:
    import src.optim.inference_optimizer as oi

    original = oi.torch
    try:
        oi.torch = None  # type: ignore[assignment]
        try:
            apply_bitsandbytes_int8(object())
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "torch" in str(exc).lower()
    finally:
        oi.torch = original


def test_convert_to_fp16_without_torch_raises() -> None:
    import src.optim.inference_optimizer as oi

    original = oi.torch
    try:
        oi.torch = None  # type: ignore[assignment]
        try:
            convert_to_fp16(object())
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "torch" in str(exc).lower()
    finally:
        oi.torch = original


# ---------------------------------------------------------------------------
# Config / YAML tests
# ---------------------------------------------------------------------------


def test_yaml_defaults_loaded_from_file() -> None:
    config = {
        "asr": {
            "backend": "mock",
            "model_id": "openai/whisper-tiny",
            "device": "cpu",
            "dtype": "float32",
            "chunk_length_s": 10,
            "stride_length_s": 3,
        },
        "optimization": {
            "enable_torch_compile": False,
            "enable_dynamic_int8": True,
            "enable_pruning": True,
            "pruning_amount": 0.35,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        tmp_path = Path(f.name)

    try:
        defaults = _load_yaml_defaults(tmp_path)
        assert defaults["asr_backend"] == "mock"
        assert defaults["asr_model_id"] == "openai/whisper-tiny"
        assert defaults["dtype"] == "float32"
        assert defaults["chunk_length_s"] == 10.0
        assert defaults["enable_dynamic_int8"] is True
        assert defaults["enable_torch_compile"] is False
        assert defaults["enable_pruning"] is True
        assert defaults["pruning_amount"] == 0.35
    finally:
        tmp_path.unlink(missing_ok=True)


def test_yaml_defaults_missing_file_returns_empty() -> None:
    defaults = _load_yaml_defaults(Path("/nonexistent/path/config.yaml"))
    assert defaults == {}


# ---------------------------------------------------------------------------
# Additional coverage tests (model + optimizer internals)
# ---------------------------------------------------------------------------


def test_asr_pipeline_mock_backend_transcribe_and_model_errors() -> None:
    pipeline = ASRPipeline(ASRConfig(backend="mock"))

    text = pipeline.transcribe(np.zeros(16, dtype=np.float32), sample_rate=16000)
    assert text == "mock transcription"

    try:
        _ = pipeline.model
        assert False, "Expected RuntimeError for mock backend model access"
    except RuntimeError as exc:
        assert "mock backend" in str(exc).lower()

    try:
        pipeline.model = object()
        assert False, "Expected RuntimeError for mock backend model assignment"
    except RuntimeError as exc:
        assert "mock backend" in str(exc).lower()


def test_asr_pipeline_requires_torch_and_transformers_for_real_backend() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    original_hf_pipeline = ap.hf_pipeline
    try:
        ap.torch = None  # type: ignore[assignment]
        ap.hf_pipeline = None  # type: ignore[assignment]

        try:
            ASRPipeline(ASRConfig(backend="transformers"))
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            msg = str(exc).lower()
            assert "torch" in msg and "transformers" in msg
    finally:
        ap.torch = original_torch
        ap.hf_pipeline = original_hf_pipeline


def test_optimize_asr_pipeline_nonmock_without_flags_passthrough() -> None:
    class DummyConfig:
        backend = "transformers"
        device = "cpu"

    class DummyPipeline:
        def __init__(self) -> None:
            self.config = DummyConfig()
            self.model = object()

    pipeline = DummyPipeline()
    config = OptimizationConfig(enable_torch_compile=False, enable_dynamic_int8=False)

    result = optimize_asr_pipeline(pipeline, config)
    assert result is pipeline


def test_optimize_asr_pipeline_pruning_branch_calls_helper() -> None:
    import src.optim.inference_optimizer as oi

    class DummyConfig:
        backend = "transformers"
        device = "cpu"

    class DummyPipeline:
        def __init__(self) -> None:
            self.config = DummyConfig()
            self.model = "initial"

    original_torch = oi.torch
    original_apply = oi.apply_magnitude_pruning
    try:
        oi.torch = object()  # type: ignore[assignment]

        calls: dict[str, object] = {}

        def fake_apply(model, amount):
            calls["model"] = model
            calls["amount"] = amount
            return "pruned-model"

        oi.apply_magnitude_pruning = fake_apply  # type: ignore[assignment]

        p = DummyPipeline()
        cfg = OptimizationConfig(
            enable_torch_compile=False,
            enable_dynamic_int8=False,
            enable_pruning=True,
            pruning_amount=0.3,
        )
        out = optimize_asr_pipeline(p, cfg)

        assert out is p
        assert p.model == "pruned-model"
        assert calls["model"] == "initial"
        assert calls["amount"] == 0.3
    finally:
        oi.torch = original_torch
        oi.apply_magnitude_pruning = original_apply


def test_export_model_to_onnx_calls_torch_export() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch
    calls: dict[str, object] = {}

    class FakeOnnx:
        @staticmethod
        def export(model, dummy_input, output_path, **kwargs):
            calls["model"] = model
            calls["dummy_input"] = dummy_input
            calls["output_path"] = output_path
            calls["kwargs"] = kwargs

    class FakeTorch:
        float32 = "float32"
        onnx = FakeOnnx

        @staticmethod
        def randn(*shape, dtype=None):
            calls["randn_shape"] = shape
            calls["randn_dtype"] = dtype
            return {"shape": shape, "dtype": dtype}

    class FakeModel:
        def cpu(self):
            calls["cpu_called"] = True
            return self

        def eval(self):
            calls["eval_called"] = True
            return self

    try:
        oi.torch = FakeTorch  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "nested" / "model.onnx"
            oi.export_model_to_onnx(FakeModel(), output, opset=18)

            assert output.parent.exists()
            assert calls["cpu_called"] is True
            assert calls["eval_called"] is True
            assert calls["randn_shape"] == (1, 80, 3000)
            assert calls["output_path"] == output
            assert isinstance(calls["kwargs"], dict)
    finally:
        oi.torch = original_torch


def test_export_model_to_onnx_wav2vec2_uses_input_values_shape() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch
    calls: dict[str, object] = {}

    class FakeOnnx:
        @staticmethod
        def export(model, dummy_input, output_path, **kwargs):
            calls["dummy_input"] = dummy_input
            calls["output_path"] = output_path
            calls["kwargs"] = kwargs

    class FakeTorch:
        float32 = "float32"
        onnx = FakeOnnx

        @staticmethod
        def randn(*shape, dtype=None):
            calls["randn_shape"] = shape
            return {"shape": shape, "dtype": dtype}

    class FakeConfig:
        model_type = "wav2vec2"

    class FakeModel:
        config = FakeConfig()

        def cpu(self):
            return self

        def eval(self):
            return self

    try:
        oi.torch = FakeTorch  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "wav2vec2.onnx"
            oi.export_model_to_onnx(
                FakeModel(),
                output,
                model_family="auto",
                model_id="facebook/wav2vec2-base-960h",
            )

        assert calls["randn_shape"] == (1, 16000)
        assert calls["kwargs"]["input_names"] == ["input_values"]
    finally:
        oi.torch = original_torch


def test_compile_with_tensorrt_returns_original_when_dependency_missing() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

    model = object()
    try:
        oi.torch = FakeTorch  # type: ignore[assignment]
        result = compile_with_tensorrt(model, input_shapes=[(1, 80, 3000)], dtype="float16")
        assert result is model
    finally:
        oi.torch = original_torch


def test_get_submodule_traverses_nested_attributes() -> None:
    import src.optim.inference_optimizer as oi

    class Leaf:
        pass

    class Mid:
        def __init__(self) -> None:
            self.leaf = Leaf()

    class Root:
        def __init__(self) -> None:
            self.mid = Mid()

    root = Root()
    assert oi._get_submodule(root, "mid.leaf") is root.mid.leaf


def test_model_family_helpers_cover_alias_and_inference_paths() -> None:
    class FakeConfig:
        model_type = "wav2vec2"

    class FakeModel:
        config = FakeConfig()

    assert normalize_model_family("wav2vec") == "wav2vec2"
    assert infer_model_family("facebook/wav2vec2-base-960h") == "wav2vec2"
    assert infer_model_family("openai/whisper-small") == "whisper"
    assert resolve_model_family("auto", model_id="foo/bar", model=FakeModel()) == "wav2vec2"

    try:
        normalize_model_family("unknown")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "unsupported model_family" in str(exc).lower()


def test_build_export_metadata_uses_whisper_shape_by_default() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    calls: dict[str, object] = {}

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def randn(*shape, dtype=None):
            calls["shape"] = shape
            calls["dtype"] = dtype
            return {"shape": shape, "dtype": dtype}

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        dummy_input, input_names = build_export_metadata(
            object(),
            model_family="auto",
            model_id="openai/whisper-small",
        )

        assert dummy_input == {"shape": (1, 80, 3000), "dtype": "float32"}
        assert input_names == ["input_features"]
    finally:
        ap.torch = original_torch


def test_convert_to_fp16_calls_half_when_torch_present() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch

    class FakeModel:
        def __init__(self) -> None:
            self.called = False

        def half(self):
            self.called = True
            return "fp16-model"

    try:
        oi.torch = object()  # type: ignore[assignment]
        model = FakeModel()
        result = convert_to_fp16(model)
        assert result == "fp16-model"
        assert model.called is True
    finally:
        oi.torch = original_torch


def test_load_runtime_settings_env_overrides_yaml_defaults() -> None:
    app_mod = importlib.import_module("src.serving.app")

    old_env = {
        "ASR_BACKEND": os.getenv("ASR_BACKEND"),
        "ASR_MODEL_ID": os.getenv("ASR_MODEL_ID"),
        "ASR_MODEL_FAMILY": os.getenv("ASR_MODEL_FAMILY"),
        "ASR_DEVICE": os.getenv("ASR_DEVICE"),
        "ASR_DTYPE": os.getenv("ASR_DTYPE"),
        "ASR_CHUNK_LENGTH_S": os.getenv("ASR_CHUNK_LENGTH_S"),
        "ASR_STRIDE_LENGTH_S": os.getenv("ASR_STRIDE_LENGTH_S"),
        "ENABLE_TORCH_COMPILE": os.getenv("ENABLE_TORCH_COMPILE"),
        "ENABLE_DYNAMIC_INT8": os.getenv("ENABLE_DYNAMIC_INT8"),
        "ENABLE_PRUNING": os.getenv("ENABLE_PRUNING"),
        "PRUNING_AMOUNT": os.getenv("PRUNING_AMOUNT"),
    }
    original_loader = app_mod._load_yaml_defaults

    try:
        app_mod._load_yaml_defaults = lambda _: {
            "asr_backend": "transformers",
            "asr_model_id": "openai/whisper-small",
            "asr_model_family": "auto",
            "device": "cpu",
            "dtype": "float16",
            "chunk_length_s": 12.0,
            "stride_length_s": 4.0,
            "enable_torch_compile": False,
            "enable_dynamic_int8": False,
            "enable_pruning": False,
            "pruning_amount": 0.2,
        }

        os.environ["ASR_BACKEND"] = "mock"
        os.environ["ASR_MODEL_ID"] = "openai/whisper-tiny"
        os.environ["ASR_MODEL_FAMILY"] = "wav2vec2"
        os.environ["ASR_DEVICE"] = "cpu"
        os.environ["ASR_DTYPE"] = "float32"
        os.environ["ASR_CHUNK_LENGTH_S"] = "9"
        os.environ["ASR_STRIDE_LENGTH_S"] = "2"
        os.environ["ENABLE_TORCH_COMPILE"] = "yes"
        os.environ["ENABLE_DYNAMIC_INT8"] = "1"
        os.environ["ENABLE_PRUNING"] = "true"
        os.environ["PRUNING_AMOUNT"] = "0.4"

        settings = app_mod.load_runtime_settings()
        assert settings.asr_backend == "mock"
        assert settings.asr_model_id == "openai/whisper-tiny"
        assert settings.asr_model_family == "wav2vec2"
        assert settings.dtype == "float32"
        assert settings.chunk_length_s == 9.0
        assert settings.stride_length_s == 2.0
        assert settings.enable_torch_compile is True
        assert settings.enable_dynamic_int8 is True
        assert settings.enable_pruning is True
        assert settings.pruning_amount == 0.4
    finally:
        app_mod._load_yaml_defaults = original_loader
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_get_asr_service_builds_pipeline_and_applies_optimizer() -> None:
    app_mod = importlib.import_module("src.serving.app")

    captured: dict[str, object] = {}
    original_asr_pipeline = app_mod.ASRPipeline
    original_optimize = app_mod.optimize_asr_pipeline
    original_settings = app_mod.load_runtime_settings

    class DummyASR:
        def __init__(self, config):
            self.config = config

    try:
        app_mod.get_asr_service.cache_clear()
        app_mod.load_runtime_settings = lambda: app_mod.RuntimeSettings(
            asr_backend="mock",
            asr_model_id="id",
            asr_model_family="whisper",
            device="cpu",
            dtype="float32",
            chunk_length_s=8.0,
            stride_length_s=2.0,
            enable_torch_compile=True,
            enable_dynamic_int8=True,
            enable_pruning=True,
            pruning_amount=0.25,
        )
        app_mod.ASRPipeline = DummyASR

        def fake_optimize(service, cfg):
            captured["service"] = service
            captured["cfg"] = cfg
            return "optimized"

        app_mod.optimize_asr_pipeline = fake_optimize

        result = app_mod.get_asr_service()
        assert result == "optimized"
        assert isinstance(captured["service"], DummyASR)
        assert captured["service"].config.backend == "mock"
        assert captured["service"].config.model_family == "whisper"
        assert captured["cfg"].enable_torch_compile is True
        assert captured["cfg"].enable_dynamic_int8 is True
        assert captured["cfg"].enable_pruning is True
        assert captured["cfg"].pruning_amount == 0.25
    finally:
        app_mod.ASRPipeline = original_asr_pipeline
        app_mod.optimize_asr_pipeline = original_optimize
        app_mod.load_runtime_settings = original_settings
        app_mod.get_asr_service.cache_clear()


def test_decode_wav_fallback_uint8_and_invalid_inputs() -> None:
    # 8-bit PCM branch
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(16000)
            wav.writeframes(bytes([0, 127, 255]))
        data = buffer.getvalue()

    audio, sample_rate = _decode_wav_fallback(data)
    assert sample_rate == 16000
    assert audio.shape == (1, 3)

    # Invalid WAV payload -> decode error branch
    app_mod = importlib.import_module("src.serving.app")

    try:
        app_mod._decode_wav_fallback(b"not-a-wav")
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "decode wav" in exc.detail.lower()


def test_decode_wav_fallback_stereo_reshapes_to_channel_first() -> None:
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack("<hhhhhh", 1000, -1000, 2000, -2000, 3000, -3000))
        data = buffer.getvalue()

    audio, sample_rate = _decode_wav_fallback(data)
    assert sample_rate == 16000
    assert audio.shape == (2, 3)


def test_decode_wav_fallback_unsupported_bit_depth_raises() -> None:
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(3)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00\x00")
        data = buffer.getvalue()

    try:
        _decode_wav_fallback(data)
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "unsupported wav bit depth" in exc.detail.lower()


def test_decode_audio_torchaudio_success_and_fallback_to_torch_numpy() -> None:
    app_mod = importlib.import_module("src.serving.app")

    original_torchaudio = app_mod.torchaudio
    original_torch = app_mod.torch
    original_fallback = app_mod._decode_wav_fallback

    class DummyUpload:
        def __init__(self, data: bytes):
            self._data = data

        async def read(self):
            return self._data

    class FakeTorchaudioSuccess:
        @staticmethod
        def load(_):
            return "waveform", 12345

    class FakeTorchaudioFail:
        @staticmethod
        def load(_):
            raise RuntimeError("boom")

    class FakeTorch:
        @staticmethod
        def from_numpy(arr):
            return ("tensor", arr.shape)

    try:
        app_mod.torchaudio = FakeTorchaudioSuccess
        waveform, sr = asyncio.run(app_mod._decode_audio(DummyUpload(b"abc")))
        assert waveform == "waveform"
        assert sr == 12345

        app_mod.torchaudio = FakeTorchaudioFail
        app_mod._decode_wav_fallback = lambda _: (np.zeros((1, 4), dtype=np.float32), 16000)
        app_mod.torch = FakeTorch
        waveform, sr = asyncio.run(app_mod._decode_audio(DummyUpload(b"def")))
        assert waveform == ("tensor", (1, 4))
        assert sr == 16000
    finally:
        app_mod.torchaudio = original_torchaudio
        app_mod.torch = original_torch
        app_mod._decode_wav_fallback = original_fallback


def test_ws_transcribe_when_asr_raises_returns_error_json() -> None:
    class FailingASR:
        def transcribe(self, waveform, sample_rate, language=None):
            raise RuntimeError("forced failure")

    app.dependency_overrides[get_asr_service] = lambda: FailingASR()
    client = TestClient(app)

    samples = np.zeros(800, dtype=np.float32)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_bytes(samples.tobytes())
        ws.send_text("END")
        data = ws.receive_json()

    assert "error" in data
    assert "forced failure" in data["error"]
    app.dependency_overrides.clear()


def test_ws_transcribe_disconnect_returns_cleanly() -> None:
    class DisconnectingWebSocket:
        def __init__(self) -> None:
            self.accepted = False

        async def accept(self) -> None:
            self.accepted = True

        async def receive(self):
            raise WebSocketDisconnect(code=1000)

    ws = DisconnectingWebSocket()
    asyncio.run(ws_transcribe(websocket=ws, asr_service=FakeASR()))
    assert ws.accepted is True


# ---------------------------------------------------------------------------
# Additional deep coverage (ASR + optimizer internals)
# ---------------------------------------------------------------------------


def test_default_device_prefers_cuda_when_available() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        assert ap._default_device() == "cuda"
    finally:
        ap.torch = original_torch


def test_asr_build_pipeline_uses_dtype_and_cuda_device_index() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    original_hf_pipeline = ap.hf_pipeline
    captured: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakePipe:
        def __init__(self) -> None:
            self.model = object()

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"
        bfloat16 = "bfloat16"
        cuda = FakeCuda

    def fake_hf_pipeline(**kwargs):
        captured.update(kwargs)
        return FakePipe()

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        ap.hf_pipeline = fake_hf_pipeline  # type: ignore[assignment]

        ASRPipeline(
            ASRConfig(
                backend="transformers",
                model_id="fake/model",
                device="cuda",
                dtype="bfloat16",
                chunk_length_s=11.0,
                stride_length_s=3.0,
            )
        )

        assert captured["task"] == "automatic-speech-recognition"
        assert captured["model"] == "fake/model"
        assert captured["device"] == 0
        assert captured["torch_dtype"] == "bfloat16"
        assert captured["chunk_length_s"] == 11.0
        assert captured["stride_length_s"] == (3.0, 3.0)
    finally:
        ap.torch = original_torch
        ap.hf_pipeline = original_hf_pipeline


def test_asr_transcribe_full_path_with_resample_and_language() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    original_torchaudio = ap.torchaudio
    calls: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=np.float32)

        @property
        def ndim(self) -> int:
            return self.arr.ndim

        def mean(self, dim: int):
            return FakeTensor(self.arr.mean(axis=dim))

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, _dtype):
            return self

        def numpy(self):
            return self.arr

    class FakeTorch:
        Tensor = FakeTensor
        float32 = "float32"

        @staticmethod
        def as_tensor(value):
            calls["as_tensor"] = True
            return FakeTensor(value)

    class FakeFunctional:
        @staticmethod
        def resample(tensor, in_sr, out_sr):
            calls["resample"] = (in_sr, out_sr)
            return FakeTensor(tensor.arr)

    class FakeTorchaudio:
        functional = FakeFunctional

    class FakePipe:
        def __init__(self) -> None:
            self.model = object()

        def __call__(self, payload, generate_kwargs=None):
            calls["payload"] = payload
            calls["generate_kwargs"] = generate_kwargs
            return {"text": "  hello world  "}

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        ap.torchaudio = FakeTorchaudio  # type: ignore[assignment]

        pipeline = ASRPipeline(ASRConfig(backend="mock"))
        pipeline.config.backend = "transformers"
        pipeline._pipe = FakePipe()

        result = pipeline.transcribe([[0.1, 0.2], [0.3, 0.4]], sample_rate=8000, language="en")

        assert result == "hello world"
        assert calls["as_tensor"] is True
        assert calls["resample"] == (8000, 16000)
        assert calls["payload"]["sampling_rate"] == 16000
        assert calls["generate_kwargs"] == {"language": "en"}
    finally:
        ap.torch = original_torch
        ap.torchaudio = original_torchaudio


def test_asr_transcribe_no_language_and_string_output() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    original_torchaudio = ap.torchaudio

    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=np.float32)

        @property
        def ndim(self) -> int:
            return self.arr.ndim

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, _dtype):
            return self

        def numpy(self):
            return self.arr

    class FakeTorch:
        Tensor = FakeTensor
        float32 = "float32"

    class FakePipe:
        def __init__(self) -> None:
            self.model = object()

        def __call__(self, payload, generate_kwargs=None):
            assert payload["sampling_rate"] == 16000
            assert generate_kwargs is None
            return "  text-output  "

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        ap.torchaudio = object()  # type: ignore[assignment]

        pipeline = ASRPipeline(ASRConfig(backend="mock"))
        pipeline.config.backend = "transformers"
        pipeline._pipe = FakePipe()

        result = pipeline.transcribe(FakeTensor([1.0, 2.0]), sample_rate=16000, language=None)
        assert result == "text-output"
    finally:
        ap.torch = original_torch
        ap.torchaudio = original_torchaudio


def test_asr_transcribe_ignores_language_for_wav2vec2() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    original_torchaudio = ap.torchaudio
    calls: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=np.float32)

        @property
        def ndim(self) -> int:
            return self.arr.ndim

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, _dtype):
            return self

        def numpy(self):
            return self.arr

    class FakeTorch:
        Tensor = FakeTensor
        float32 = "float32"

    class FakeConfig:
        model_type = "wav2vec2"

    class FakePipe:
        def __init__(self) -> None:
            self.model = type("Model", (), {"config": FakeConfig()})()

        def __call__(self, payload, generate_kwargs=None):
            calls["payload"] = payload
            calls["generate_kwargs"] = generate_kwargs
            return {"text": "wav2vec2-output"}

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        ap.torchaudio = object()  # type: ignore[assignment]

        pipeline = ASRPipeline(ASRConfig(backend="mock", model_family="auto"))
        pipeline.config.backend = "transformers"
        pipeline.config.model_id = "facebook/wav2vec2-base-960h"
        pipeline._pipe = FakePipe()

        result = pipeline.transcribe(FakeTensor([1.0, 2.0]), sample_rate=16000, language="en")

        assert result == "wav2vec2-output"
        assert calls["payload"]["sampling_rate"] == 16000
        assert calls["generate_kwargs"] is None
    finally:
        ap.torch = original_torch
        ap.torchaudio = original_torchaudio


def test_asr_transcribe_raises_when_pipeline_missing() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    original_torchaudio = ap.torchaudio

    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=np.float32)

        @property
        def ndim(self) -> int:
            return self.arr.ndim

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, _dtype):
            return self

        def numpy(self):
            return self.arr

    class FakeTorch:
        Tensor = FakeTensor
        float32 = "float32"

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        ap.torchaudio = object()  # type: ignore[assignment]

        pipeline = ASRPipeline(ASRConfig(backend="mock"))
        pipeline.config.backend = "transformers"
        pipeline._pipe = None

        try:
            pipeline.transcribe(FakeTensor([1.0, 2.0]), sample_rate=16000)
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "not initialized" in str(exc).lower()
    finally:
        ap.torch = original_torch
        ap.torchaudio = original_torchaudio


def test_optimize_asr_pipeline_raises_without_torch_on_real_backend() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch

    class DummyConfig:
        backend = "transformers"
        device = "cpu"

    class DummyPipeline:
        def __init__(self) -> None:
            self.config = DummyConfig()
            self.model = object()

    try:
        oi.torch = None  # type: ignore[assignment]
        try:
            optimize_asr_pipeline(
                DummyPipeline(),
                OptimizationConfig(enable_torch_compile=True, enable_dynamic_int8=False),
            )
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "requires torch" in str(exc).lower()
    finally:
        oi.torch = original_torch


def test_optimize_asr_pipeline_applies_dynamic_int8_and_compile() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch
    calls: dict[str, object] = {}

    class Linear:
        pass

    class Quantization:
        @staticmethod
        def quantize_dynamic(model, layers, dtype):
            calls["quant"] = (model, layers, dtype)
            return "quantized-model"

    class NN:
        pass

    NN.Linear = Linear

    class FakeTorch:
        quantization = Quantization
        nn = NN
        qint8 = "qint8"

        @staticmethod
        def compile(model, mode, fullgraph):
            calls["compile"] = (model, mode, fullgraph)
            return "compiled-model"

    class DummyConfig:
        backend = "transformers"
        device = "cpu"

    class DummyPipeline:
        def __init__(self) -> None:
            self.config = DummyConfig()
            self.model = "base-model"

    try:
        oi.torch = FakeTorch  # type: ignore[assignment]
        pipe = DummyPipeline()
        result = optimize_asr_pipeline(
            pipe,
            OptimizationConfig(enable_torch_compile=True, enable_dynamic_int8=True),
        )

        assert result is pipe
        assert pipe.model == "compiled-model"
        assert calls["quant"][0] == "base-model"
        assert calls["compile"] == ("quantized-model", "reduce-overhead", False)
    finally:
        oi.torch = original_torch


def test_compile_with_tensorrt_success_path() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch
    original_trt = sys.modules.get("torch_tensorrt")
    calls: dict[str, object] = {}

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

    class FakeTorchTensorRT:
        @staticmethod
        def Input(shape, dtype):
            return {"shape": shape, "dtype": dtype}

        @staticmethod
        def compile(model, inputs, enabled_precisions):
            calls["compile"] = (model, inputs, enabled_precisions)
            return "trt-model"

    try:
        oi.torch = FakeTorch  # type: ignore[assignment]
        sys.modules["torch_tensorrt"] = FakeTorchTensorRT

        result = compile_with_tensorrt("base", input_shapes=[(1, 80, 3000)], dtype="float16")
        assert result == "trt-model"
        assert calls["compile"][0] == "base"
        assert calls["compile"][2] == {"float16"}
    finally:
        oi.torch = original_torch
        if original_trt is None:
            sys.modules.pop("torch_tensorrt", None)
        else:
            sys.modules["torch_tensorrt"] = original_trt


def test_apply_bitsandbytes_int8_replaces_linear_layers() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch
    original_bnb = sys.modules.get("bitsandbytes")

    class FakeLinear:
        def __init__(self, in_features, out_features, bias=True):
            self.in_features = in_features
            self.out_features = out_features
            self.bias = "bias" if bias else None
            self.weight = "weight"

    class FakeLinear8bitLt:
        def __init__(self, in_features, out_features, bias, has_fp16_weights):
            self.in_features = in_features
            self.out_features = out_features
            self.bias = None
            self.weight = None
            self.has_fp16_weights = has_fp16_weights

    class FakeNN:
        Linear = FakeLinear

    class FakeTorch:
        nn = FakeNN

    class FakeBnbNN:
        Linear8bitLt = FakeLinear8bitLt

    class FakeBnb:
        nn = FakeBnbNN

    class Model:
        def __init__(self) -> None:
            self.proj = FakeLinear(4, 2, bias=True)

        def named_modules(self):
            return [("", self), ("proj", self.proj)]

    try:
        oi.torch = FakeTorch  # type: ignore[assignment]
        sys.modules["bitsandbytes"] = FakeBnb

        model = Model()
        out = apply_bitsandbytes_int8(model)

        assert out is model
        assert isinstance(model.proj, FakeLinear8bitLt)
        assert model.proj.weight == "weight"
        assert model.proj.bias == "bias"
    finally:
        oi.torch = original_torch
        if original_bnb is None:
            sys.modules.pop("bitsandbytes", None)
        else:
            sys.modules["bitsandbytes"] = original_bnb


# ---------------------------------------------------------------------------
# Final coverage closeout (profiling + remaining model/optimizer branches)
# ---------------------------------------------------------------------------


def test_percentile_empty_list_returns_zero() -> None:
    import src.profiling.benchmark as bm

    assert bm._percentile([], 0.5) == 0.0


def test_profile_with_torch_profiler_cpu_and_cuda_activity_paths() -> None:
    import src.profiling.benchmark as bm

    original_torch = bm.torch
    calls: dict[str, object] = {}

    class FakeProfilerActivity:
        CPU = "CPU"
        CUDA = "CUDA"

    class FakeProfileCtx:
        def __init__(self, *, activities, record_shapes, with_flops):
            calls["activities"] = activities
            calls["record_shapes"] = record_shapes
            calls["with_flops"] = with_flops

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def key_averages(self):
            return self

        def table(self, *, sort_by, row_limit):
            calls["table"] = (sort_by, row_limit)
            return "profiler-table"

    class FakeProfiler:
        ProfilerActivity = FakeProfilerActivity

        @staticmethod
        def profile(**kwargs):
            return FakeProfileCtx(**kwargs)

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        profiler = FakeProfiler
        cuda = FakeCuda

    try:
        bm.torch = FakeTorch  # type: ignore[assignment]
        out = profile_with_torch_profiler(lambda: None, iterations=2, use_cuda=True, row_limit=7)

        assert out == "profiler-table"
        assert calls["activities"] == ["CPU", "CUDA"]
        assert calls["record_shapes"] is True
        assert calls["with_flops"] is True
        assert calls["table"] == ("cpu_time_total", 7)
    finally:
        bm.torch = original_torch


def test_asr_model_property_get_and_set_when_pipe_initialized() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    original_hf_pipeline = ap.hf_pipeline

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakePipe:
        def __init__(self) -> None:
            self.model = "initial-model"

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"
        bfloat16 = "bfloat16"
        cuda = FakeCuda

    def fake_hf_pipeline(**_kwargs):
        return FakePipe()

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        ap.hf_pipeline = fake_hf_pipeline  # type: ignore[assignment]

        pipeline = ASRPipeline(ASRConfig(backend="transformers", device="cpu"))
        assert pipeline.model == "initial-model"

        pipeline.model = "updated-model"
        assert pipeline.model == "updated-model"
    finally:
        ap.torch = original_torch
        ap.hf_pipeline = original_hf_pipeline


def test_asr_transcribe_raises_when_torchaudio_missing() -> None:
    import src.models.asr_pipeline as ap

    original_torch = ap.torch
    original_torchaudio = ap.torchaudio

    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=np.float32)

        @property
        def ndim(self) -> int:
            return self.arr.ndim

    class FakeTorch:
        Tensor = FakeTensor

        @staticmethod
        def as_tensor(value):
            return FakeTensor(value)

    try:
        ap.torch = FakeTorch  # type: ignore[assignment]
        ap.torchaudio = None  # type: ignore[assignment]

        pipeline = ASRPipeline(ASRConfig(backend="mock"))
        pipeline.config.backend = "transformers"
        pipeline._pipe = object()

        try:
            pipeline.transcribe([1.0, 2.0], sample_rate=16000)
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "torchaudio" in str(exc).lower()
    finally:
        ap.torch = original_torch
        ap.torchaudio = original_torchaudio


def test_export_model_to_onnx_without_torch_raises() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch
    try:
        oi.torch = None  # type: ignore[assignment]
        try:
            oi.export_model_to_onnx(object(), Path("tmp.onnx"))
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "onnx export requires torch" in str(exc).lower()
    finally:
        oi.torch = original_torch


# ---------------------------------------------------------------------------
# measure_gpu_memory
# ---------------------------------------------------------------------------


def test_measure_gpu_memory_returns_zeros_on_cpu_only() -> None:
    """On a machine without CUDA (or without torch), all fields are zero."""
    import src.profiling.benchmark as bm

    original_torch = bm.torch
    try:
        # Simulate CUDA unavailable by patching cuda_is_available inside the module
        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return False

        bm.torch = FakeTorch  # type: ignore[assignment]
        result = measure_gpu_memory(lambda: None)
        assert result.allocated_mb == 0.0
        assert result.peak_mb == 0.0
        assert result.device == "cpu"
    finally:
        bm.torch = original_torch


def test_measure_gpu_memory_no_torch_returns_zeros() -> None:
    import src.profiling.benchmark as bm

    original_torch = bm.torch
    try:
        bm.torch = None  # type: ignore[assignment]
        result = measure_gpu_memory(lambda: None)
        assert isinstance(result, MemoryStats)
        assert result.peak_mb == 0.0
        assert result.device == "cpu"
    finally:
        bm.torch = original_torch


# ---------------------------------------------------------------------------
# apply_magnitude_pruning
# ---------------------------------------------------------------------------


def test_apply_magnitude_pruning_calls_prune_and_remove() -> None:
    """Pruning iterates modules, calls l1_unstructured + remove on each Linear."""
    pruned_modules: list[object] = []
    removed_modules: list[object] = []

    class FakePrune:
        @staticmethod
        def l1_unstructured(module: object, name: str, amount: float) -> None:
            pruned_modules.append(module)

        @staticmethod
        def remove(module: object, name: str) -> None:
            removed_modules.append(module)

    class FakeLinear:
        pass

    linear_instance = FakeLinear()

    class FakeNN:
        Linear = FakeLinear

        class utils:
            prune = FakePrune()

    class FakeModel:
        def modules(self) -> list[object]:
            return [linear_instance]

    class FakeTorch:
        nn = FakeNN()

        @staticmethod
        def is_available() -> bool:
            return False

    import src.optim.inference_optimizer as oi

    original_torch = oi.torch
    try:
        oi.torch = FakeTorch  # type: ignore[assignment]
        result = apply_magnitude_pruning(FakeModel(), amount=0.2)
        assert isinstance(result, FakeModel)
        assert len(pruned_modules) == 1
        assert len(removed_modules) == 1
    finally:
        oi.torch = original_torch


def test_apply_magnitude_pruning_invalid_amount_raises() -> None:
    import src.optim.inference_optimizer as oi

    class FakeTorch:
        class nn:
            class Linear:
                pass

    original_torch = oi.torch
    try:
        oi.torch = FakeTorch  # type: ignore[assignment]
        try:
            apply_magnitude_pruning(object(), amount=1.5)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "amount" in str(exc)
    finally:
        oi.torch = original_torch


def test_apply_magnitude_pruning_without_torch_raises() -> None:
    import src.optim.inference_optimizer as oi

    original_torch = oi.torch
    try:
        oi.torch = None  # type: ignore[assignment]
        try:
            apply_magnitude_pruning(object())
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "pruning requires torch" in str(exc).lower()
    finally:
        oi.torch = original_torch


# ---------------------------------------------------------------------------
# >200 ms latency warning  POST /transcribe
# ---------------------------------------------------------------------------


def test_transcribe_endpoint_logs_warning_when_slow() -> None:
    """When latency exceeds 200 ms the app logs a WARNING."""
    import logging as _logging
    import time

    app_mod = importlib.import_module("src.serving.app")

    app_mod.get_asr_service.cache_clear()  # type: ignore[attr-defined]

    slow_pipeline = ASRPipeline(ASRConfig(backend="mock"))
    original_transcribe = slow_pipeline.transcribe

    def slow_transcribe(*args: object, **kwargs: object) -> str:
        time.sleep(0.25)  # force > 200 ms
        return original_transcribe(*args, **kwargs)

    slow_pipeline.transcribe = slow_transcribe  # type: ignore[method-assign]

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        import struct as _struct

        samples = _struct.pack("<" + "h" * 160, *([0] * 160))
        wf.writeframes(samples)
    wav_bytes = buf.getvalue()

    log_records: list[str] = []

    class _Capture(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            log_records.append(_logging.getLevelName(record.levelno) + " " + record.getMessage())

    handler = _Capture()
    handler.setLevel(_logging.WARNING)
    svc_logger = _logging.getLogger("src.serving.app")
    svc_logger.addHandler(handler)

    client = TestClient(app)
    app.dependency_overrides[get_asr_service] = lambda: slow_pipeline
    try:
        resp = client.post(
            "/transcribe",
            files={"audio_file": ("audio.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        # When the mock pipeline plus sleep exceeds 200 ms, a warning must be logged
        if data["latency_ms"] > 200:
            assert any("200 ms" in r or "exceeds" in r for r in log_records)
    finally:
        svc_logger.removeHandler(handler)
        app.dependency_overrides.clear()
        app_mod.get_asr_service.cache_clear()  # type: ignore[attr-defined]


def test_ws_transcribe_logs_warning_when_slow() -> None:
    """WebSocket handler logs a WARNING when transcription exceeds 200 ms."""
    import logging as _logging
    import time

    app_mod = importlib.import_module("src.serving.app")
    app_mod.get_asr_service.cache_clear()  # type: ignore[attr-defined]

    slow_pipeline = ASRPipeline(ASRConfig(backend="mock"))
    original_transcribe = slow_pipeline.transcribe

    def slow_transcribe(*args: object, **kwargs: object) -> str:
        time.sleep(0.25)
        return original_transcribe(*args, **kwargs)

    slow_pipeline.transcribe = slow_transcribe  # type: ignore[method-assign]

    log_records: list[str] = []

    class _Capture(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            log_records.append(_logging.getLevelName(record.levelno) + " " + record.getMessage())

    handler = _Capture()
    handler.setLevel(_logging.WARNING)
    svc_logger = _logging.getLogger("src.serving.app")
    svc_logger.addHandler(handler)

    import struct as _struct

    raw_pcm = _struct.pack("<" + "f" * 160, *([0.0] * 160))

    client = TestClient(app)
    app.dependency_overrides[get_asr_service] = lambda: slow_pipeline
    try:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_bytes(raw_pcm)
            ws.send_text("END")
            data = ws.receive_json()
        assert "text" in data or "error" in data
        if data.get("latency_ms", 0) > 200:
            assert any("200 ms" in r or "exceeds" in r for r in log_records)
    finally:
        svc_logger.removeHandler(handler)
        app.dependency_overrides.clear()
        app_mod.get_asr_service.cache_clear()  # type: ignore[attr-defined]
