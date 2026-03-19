from __future__ import annotations

from pathlib import Path


class _RunResult:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_require_tool_missing_raises(monkeypatch) -> None:
    import scripts.profile_nsight as ns

    monkeypatch.setattr(ns.shutil, "which", lambda _name: None)

    try:
        ns._require_tool("nsys")
        assert False, "Expected SystemExit"
    except SystemExit as exc:
        assert "required tool" in str(exc).lower()
        assert "nsys" in str(exc).lower()


def test_main_builds_nsys_command(monkeypatch, tmp_path: Path) -> None:
    import scripts.profile_nsight as ns

    captured: dict[str, object] = {}

    def fake_require_tool(name: str) -> str:
        assert name == "nsys"
        return "/usr/bin/nsys"

    def fake_run(command, check=False):
        captured["command"] = command
        captured["check"] = check
        return _RunResult(0)

    monkeypatch.setattr(ns, "_require_tool", fake_require_tool)
    monkeypatch.setattr(ns.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ns.sys,
        "argv",
        [
            "profile_nsight.py",
            "audio.wav",
            "--tool",
            "nsys",
            "--output",
            str(tmp_path / "profiles" / "timeline"),
            "--model-family",
            "whisper",
            "--torch-compile",
            "--dynamic-int8",
            "--pruning-amount",
            "0.3",
        ],
    )

    try:
        ns.main()
        assert False, "Expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 0

    command = captured["command"]
    assert command[0] == "/usr/bin/nsys"
    assert "profile" in command
    assert "--trace=cuda,nvtx,osrt" in command
    assert any(str(item).startswith("--capture-range=") for item in command)
    assert any(str(item).startswith("--output=") for item in command)
    assert "--model-family" in command
    assert "whisper" in command
    assert "--torch-compile" in command
    assert "--dynamic-int8" in command
    assert "--pruning-amount" in command
    assert "0.3" in command


def test_main_builds_ncu_command(monkeypatch, tmp_path: Path) -> None:
    import scripts.profile_nsight as ns

    captured: dict[str, object] = {}

    def fake_require_tool(name: str) -> str:
        assert name == "ncu"
        return "/usr/bin/ncu"

    def fake_run(command, check=False):
        captured["command"] = command
        captured["check"] = check
        return _RunResult(7)

    monkeypatch.setattr(ns, "_require_tool", fake_require_tool)
    monkeypatch.setattr(ns.subprocess, "run", fake_run)
    monkeypatch.setattr(
        ns.sys,
        "argv",
        [
            "profile_nsight.py",
            "audio.wav",
            "--tool",
            "ncu",
            "--output",
            str(tmp_path / "profiles" / "kernel"),
            "--model-family",
            "wav2vec2",
        ],
    )

    try:
        ns.main()
        assert False, "Expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 7

    command = captured["command"]
    assert command[0] == "/usr/bin/ncu"
    assert "--set" in command
    assert "full" in command
    assert "--target-processes" in command
    assert "all" in command
    assert "--export" in command
    assert "--model-family" in command
    assert "wav2vec2" in command
    assert not any(str(item).startswith("--capture-range=") for item in command)
