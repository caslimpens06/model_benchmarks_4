from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from benchmark import BenchmarkRunner
from models import create_models


VIDEO_PATH = Path("videos/benchmark_video.mp4")


def record_reachy_video(output_path: Path, duration: float = 10.0, fps: float = 30.0):
    try:
        from reachy_mini import ReachyMini
    except ImportError as exc:
        raise RuntimeError(
            "reachy_mini is not installed. Run the setup script first."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_frames = int(round(duration * fps))

    print("Connecting to Reachy Mini simulator...")
    print("Make sure this is already running:")
    print("  reachy-mini-daemon --sim --scene minimal")
    print()

    mini = ReachyMini(media_backend="webrtc")
    writer = None
    written = 0

    print("=" * 90)
    print("RECORDING FIXED BENCHMARK VIDEO")
    print("=" * 90)
    print(f"Duration : {duration:.1f}s")
    print(f"FPS      : {fps:.1f}")
    print(f"Frames   : {target_frames}")
    print(f"Output   : {output_path.resolve()}")
    print()
    print("Perform the same movement(s) you want to use for the benchmark.")
    print("Recording...")

    try:
        while written < target_frames:
            frame = mini.media.get_frame()
            if frame is None:
                continue

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create video file: {output_path}")

            writer.write(frame)
            written += 1
            print(
                f"\rFrames: {written:4d}/{target_frames} ({written / target_frames * 100:5.1f}%)",
                end="",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.release()
        try:
            mini.disconnect()
        except Exception:
            pass

    print()
    if written != target_frames:
        raise RuntimeError(
            f"Recording ended early: {written}/{target_frames} frames captured."
        )

    print(f"Saved video: {output_path.resolve()}")
    print(f"Frames    : {written}")
    print(f"Duration  : {written / fps:.2f}s")
    print(f"Resolution: {w}x{h}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple, reproducible Reachy Mini AI model benchmark."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record a new 10-second Reachy Mini video before benchmarking.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the benchmark on the fixed video.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Benchmark video path.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch models run on CPU or CUDA. BlazePose remains CPU-only in this project.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup input frames per model.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names: blazepose,movinet,tsm,vitpose or all.",
    )
    return parser.parse_args()


def select_models(spec: str):
    models = create_models()
    if spec.lower() == "all":
        return models

    aliases = {
        "blazepose": "BlazePose",
        "movinet": "MoViNet-A0",
        "tsm": "TSM-MobileNetV2",
        "vitpose": "ViTPose",
    }
    wanted = {aliases[item.strip().lower()] for item in spec.split(",") if item.strip().lower() in aliases}
    if not wanted:
        raise SystemExit("No valid model names supplied. Use --models all or a comma-separated list.")
    return [model for model in models if model.name in wanted]


def main():
    args = parse_args()

    if not args.record and not args.benchmark:
        print("Nothing to do. Use --record, --benchmark, or both.")
        return 0

    if args.record:
        record_reachy_video(args.video)

    if args.benchmark:
        if not args.video.exists():
            raise SystemExit(
                f"Benchmark video not found: {args.video}\n"
                "Run: python main.py --record"
            )
        models = select_models(args.models)
        print()
        print("Models:")
        for model in models:
            print(f"  - {model.name} [{model.family}]")
        print()
        runner = BenchmarkRunner(models)
        runner.benchmark(args.video, requested_device=args.device, warmup_frames=args.warmup)

    return 0


if __name__ == "__main__":
    sys.exit(main())
