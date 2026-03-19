from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _require_tool(tool_name: str) -> str:
    resolved = shutil.which(tool_name)
    if resolved is None:
        raise SystemExit(
            f"Required tool '{tool_name}' was not found on PATH. Install the NVIDIA Nsight CLI first."
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile an ASR latency run with Nsight Systems or Nsight Compute"
    )
    parser.add_argument("audio_path", type=Path, help="Path to the input audio file")
    parser.add_argument(
        "--tool",
        choices=["nsys", "ncu"],
        default="nsys",
        help="Profiler to launch: nsys for timeline/system profiling, ncu for kernel analysis.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/nsight_profile"))
    parser.add_argument("--model-id", type=str, default="openai/whisper-small")
    parser.add_argument(
        "--model-family",
        type=str,
        default="auto",
        choices=["auto", "whisper", "wav2vec2"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"]
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--dynamic-int8", action="store_true")
    parser.add_argument("--pruning-amount", type=float, default=0.0)
    parser.add_argument(
        "--capture-range",
        type=str,
        default="cudaProfilerApi",
        help="Nsight Systems capture range. Ignored by ncu.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).with_name("profile_latency.py")
    base_command = [
        sys.executable,
        str(script_path),
        str(args.audio_path),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--model-id",
        args.model_id,
        "--model-family",
        args.model_family,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]

    if args.torch_compile:
        base_command.append("--torch-compile")
    if args.dynamic_int8:
        base_command.append("--dynamic-int8")
    if args.pruning_amount > 0.0:
        base_command.extend(["--pruning-amount", str(args.pruning_amount)])

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.tool == "nsys":
        profiler = _require_tool("nsys")
        command = [
            profiler,
            "profile",
            "--trace=cuda,nvtx,osrt",
            f"--capture-range={args.capture_range}",
            f"--output={args.output}",
            *base_command,
        ]
    else:
        profiler = _require_tool("ncu")
        command = [
            profiler,
            "--set",
            "full",
            "--target-processes",
            "all",
            "--export",
            str(args.output),
            *base_command,
        ]

    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
