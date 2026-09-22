from __future__ import annotations

import csv
import gc
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import psutil

from models.base import ModelAdapter, ModelOutput


RESULTS_DIR = Path("results")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            p,
        )
    )


def fmt(
    value: float | None,
    digits: int = 2,
) -> str:
    if value is None:
        return "-"

    return f"{value:.{digits}f}"


@dataclass
class Result:
    model: str
    family: str
    task: str
    input_type: str
    implementation: str
    device_requested: str
    device_used: str

    status: str = "OK"
    error: str = ""

    frames_seen: int = 0
    successful_inferences: int = 0
    errors: int = 0

    load_ms: float = 0.0

    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    output_fps: float = 0.0
    total_inference_s: float = 0.0

    avg_process_cpu_percent: float = 0.0

    ram_after_load_mb: float = 0.0
    peak_ram_delta_mb: float = 0.0

    vram_after_load_mb: float | None = None
    peak_vram_allocated_mb: float | None = None

    parameters: int | None = None
    model_size_mb: float | None = None

    video_width: int = 0
    video_height: int = 0
    video_fps: float = 0.0
    video_frames: int = 0
    video_duration_s: float = 0.0


class VideoInfo:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        frames: int,
    ):
        self.width = width
        self.height = height
        self.fps = (
            fps
            if fps > 0
            else 30.0
        )
        self.frames = frames
        self.duration_s = (
            frames / self.fps
            if self.fps
            else 0.0
        )


class BenchmarkRunner:

    def __init__(
        self,
        models: Iterable[ModelAdapter],
    ):
        self.models = list(models)
        self.results: list[Result] = []
        self.process = psutil.Process(
            os.getpid()
        )

    # ---------------------------------------------------------
    # Video
    # ---------------------------------------------------------

    @staticmethod
    def inspect_video(
        path: str | Path,
    ) -> VideoInfo:

        cap = cv2.VideoCapture(
            str(path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap.release()

        if (
            width <= 0
            or height <= 0
            or frames <= 0
        ):
            raise RuntimeError(
                "Video metadata is invalid "
                "or video is empty."
            )

        return VideoInfo(
            width,
            height,
            fps,
            frames,
        )

    # ---------------------------------------------------------
    # CUDA helpers
    # ---------------------------------------------------------

    def _torch_state(self):
        """
        Returns:
            (torch_module, allocated_vram_mb)
            or None when CUDA is unavailable.
        """

        try:
            import torch

        except ImportError:
            return None

        if not torch.cuda.is_available():
            return None

        torch.cuda.synchronize()

        current = (
            torch.cuda.memory_allocated()
            / (1024 * 1024)
        )

        return (
            torch,
            current,
        )

    # ---------------------------------------------------------
    # Model metadata helpers
    # ---------------------------------------------------------

    @staticmethod
    def _model_always_cpu(
        model: ModelAdapter,
    ) -> bool:

        return bool(
            getattr(
                model,
                "always_cpu",
                False,
            )
        )

    @staticmethod
    def _requested_device_for_model(
        model: ModelAdapter,
        requested_device: str,
    ) -> str:

        if BenchmarkRunner._model_always_cpu(
            model
        ):
            return "cpu"

        return requested_device

    @staticmethod
    def _get_model_device(
        model: ModelAdapter,
    ) -> str:
        """
        Safely retrieve the actual device from the
        model adapter.

        Supported adapter patterns:
          - device_used()
          - _device_used
          - device

        Falls back to CPU.
        """

        method = getattr(
            model,
            "device_used",
            None,
        )

        if callable(method):

            try:

                value = method()

                if value:
                    return str(value)

            except Exception:
                pass

        value = getattr(
            model,
            "_device_used",
            None,
        )

        if value:
            return str(value)

        value = getattr(
            model,
            "device",
            None,
        )

        if value is not None:
            return str(value)

        return "cpu"

    # ---------------------------------------------------------
    # Main benchmark
    # ---------------------------------------------------------

    def benchmark(
        self,
        video_path: str | Path,
        requested_device: str = "cpu",
        warmup_frames: int = 10,
    ):

        requested_device = (
            requested_device.lower()
        )

        if requested_device not in {
            "cpu",
            "cuda",
        }:

            raise ValueError(
                "Device must be 'cpu' or 'cuda'."
            )

        if requested_device == "cuda":

            torch_state = (
                self._torch_state()
            )

            if torch_state is None:

                raise RuntimeError(
                    "CUDA was requested, but "
                    "PyTorch CUDA is not available."
                )

        video_info = (
            self.inspect_video(
                video_path
            )
        )

        self.results = []

        print(
            "=" * 110
        )

        print(
            "REACHY MINI MODEL BENCHMARK"
        )

        print(
            "=" * 110
        )

        print(
            f"Video       : "
            f"{Path(video_path).resolve()}"
        )

        print(
            f"Resolution  : "
            f"{video_info.width}x"
            f"{video_info.height}"
        )

        print(
            f"FPS         : "
            f"{video_info.fps:.2f}"
        )

        print(
            f"Frames      : "
            f"{video_info.frames}"
        )

        print(
            f"Duration    : "
            f"{video_info.duration_s:.2f}s"
        )

        print(
            f"Device mode : "
            f"{requested_device}"
        )

        print(
            f"Warmup      : "
            f"{warmup_frames} input frame(s)"
        )

        print(
            "Timing      : decode is excluded; "
            "only model pipeline calls are timed"
        )

        print()

        for model in self.models:

            result = (
                self._benchmark_one(
                    model,
                    video_path,
                    video_info,
                    requested_device,
                    warmup_frames,
                )
            )

            self.results.append(
                result
            )

            gc.collect()

        self._print_summary()

        return self._save_results(
            video_path,
            video_info,
            requested_device,
        )

    # ---------------------------------------------------------
    # Single model benchmark
    # ---------------------------------------------------------

    def _benchmark_one(
        self,
        model: ModelAdapter,
        video_path,
        info: VideoInfo,
        requested_device: str,
        warmup_frames: int,
    ) -> Result:

        initial_device = (
            self._requested_device_for_model(
                model,
                requested_device,
            )
        )

        result = Result(
            model=model.name,
            family=getattr(
                model,
                "family",
                "Unknown",
            ),
            task=getattr(
                model,
                "task",
                "Unknown",
            ),
            input_type=getattr(
                model,
                "input_type",
                "Unknown",
            ),
            implementation=getattr(
                model,
                "implementation",
                "",
            ),
            device_requested=requested_device,
            device_used=(
                "CPU"
                if initial_device == "cpu"
                else "cuda"
            ),
            video_width=info.width,
            video_height=info.height,
            video_fps=info.fps,
            video_frames=info.frames,
            video_duration_s=(
                info.duration_s
            ),
        )

        print(
            "-" * 110
        )

        print(
            f"MODEL: {model.name}"
        )

        print(
            "-" * 110
        )

        try:

            # -------------------------------------------------
            # Load
            # -------------------------------------------------

            before_load_ram = (
                self.process
                .memory_info()
                .rss
                / (1024 * 1024)
            )

            load_start = (
                time.perf_counter()
            )

            # Give every adapter a safe initial
            # device value. This is only used when
            # adapters expose _device_used.
            try:

                model._device_used = (
                    initial_device
                )

            except Exception:
                pass

            model.load(
                requested_device
            )

            model.synchronize()

            result.load_ms = (
                time.perf_counter()
                - load_start
            ) * 1000.0

            result.device_used = (
                self._get_model_device(
                    model
                )
            )

            after_load_ram = (
                self.process
                .memory_info()
                .rss
                / (1024 * 1024)
            )

            result.ram_after_load_mb = (
                after_load_ram
            )

            # -------------------------------------------------
            # Model metadata
            # -------------------------------------------------

            try:

                result.parameters = (
                    model.parameter_count()
                )

            except Exception:

                result.parameters = None

            try:

                result.model_size_mb = (
                    model.model_size_mb()
                )

            except Exception:

                result.model_size_mb = None

            # -------------------------------------------------
            # GPU baseline
            # -------------------------------------------------

            torch_state = (
                self._torch_state()
            )

            if (
                torch_state is not None
                and result.device_used.lower()
                == "cuda"
            ):

                torch, current = (
                    torch_state
                )

                result.vram_after_load_mb = (
                    current
                )

                torch.cuda.reset_peak_memory_stats()

            print(
                f"Load time       : "
                f"{result.load_ms:.2f} ms"
            )

            print(
                f"Device used     : "
                f"{result.device_used}"
            )

            if (
                platform.system() == "Windows"
                and self._model_always_cpu(model)
            ):
                print(
                    "Device note     : "
                    "CPU-only on Windows"
                )

            if result.parameters:

                print(
                    f"Parameters      : "
                    f"{result.parameters:,}"
                )

            else:

                print(
                    "Parameters      : -"
                )

            if (
                result.model_size_mb
                is not None
            ):

                print(
                    f"Model memory    : "
                    f"{fmt(result.model_size_mb)} MB"
                )

            else:

                print(
                    "Model memory    : -"
                )

            # -------------------------------------------------
            # Warmup
            # -------------------------------------------------

            self._warmup(
                model,
                video_path,
                info,
                warmup_frames,
            )

            # IMPORTANT:
            #
            # Reset after warmup.
            #
            # This is especially important for:
            #   - MediaPipe VIDEO mode
            #   - TSM
            #   - other temporal models
            #
            # It makes frame 0 of the measured run
            # start from a clean state.
            model.reset()

            # Reset GPU peak statistics after warmup.
            torch_state = (
                self._torch_state()
            )

            if (
                torch_state is not None
                and result.device_used.lower()
                == "cuda"
            ):

                torch_state[
                    0
                ].cuda.reset_peak_memory_stats()

            # -------------------------------------------------
            # Timed inference
            # -------------------------------------------------

            latencies: list[float] = []
            cpu_samples: list[float] = []

            peak_ram_delta = 0.0

            errors = 0
            frames_seen = 0

            cap = cv2.VideoCapture(
                str(video_path)
            )

            if not cap.isOpened():

                raise RuntimeError(
                    f"Could not open video: "
                    f"{video_path}"
                )

            print(
                "Running...",
                end=" ",
                flush=True,
            )

            while True:

                ok, frame = cap.read()

                if not ok:
                    break

                frame_index = (
                    frames_seen
                )

                frames_seen += 1

                try:

                    # Synchronize before the timed
                    # section so pending GPU work is not
                    # included from previous operations.
                    model.synchronize()

                    t0 = (
                        time.perf_counter()
                    )

                    output: ModelOutput = (
                        model.process_frame(
                            frame,
                            frame_index,
                            info.fps,
                        )
                    )

                    # Synchronize before stopping the timer
                    # so GPU work is included in elapsed time.
                    model.synchronize()

                    elapsed = (
                        time.perf_counter()
                        - t0
                    )

                    # Adapter returned no result.
                    if output is None:
                        continue

                    # Temporal models may intentionally return
                    # count_as_inference=False while they fill
                    # their frame buffer.
                    if not getattr(
                        output,
                        "count_as_inference",
                        True,
                    ):

                        continue

                    # Backwards-compatible behavior:
                    # adapters that return ModelOutput(None)
                    # signal that no actual inference output
                    # was produced.
                    if getattr(
                        output,
                        "value",
                        None,
                    ) is None:

                        continue

                    latencies.append(
                        elapsed * 1000.0
                    )

                    rss = (
                        self.process
                        .memory_info()
                        .rss
                        / (1024 * 1024)
                    )

                    peak_ram_delta = max(
                        peak_ram_delta,
                        rss - after_load_ram,
                    )

                    # CPU measurement is deliberately
                    # performed outside the timed section.
                    if (
                        len(latencies) % 20
                        == 0
                    ):

                        cpu_samples.append(
                            self.process
                            .cpu_percent(
                                interval=0.05
                            )
                        )

                except Exception as exc:

                    errors += 1

                    if errors <= 3:

                        print(
                            "\n"
                            f"Frame "
                            f"{frame_index} "
                            f"error: "
                            f"{type(exc).__name__}: "
                            f"{exc}",
                            flush=True,
                        )

            cap.release()

            # -------------------------------------------------
            # Basic result information
            # -------------------------------------------------

            result.frames_seen = (
                frames_seen
            )

            result.successful_inferences = (
                len(latencies)
            )

            result.errors = errors

            result.peak_ram_delta_mb = (
                peak_ram_delta
            )

            result.avg_process_cpu_percent = (
                statistics.mean(
                    cpu_samples
                )
                if cpu_samples
                else 0.0
            )

            # -------------------------------------------------
            # Latency statistics
            # -------------------------------------------------

            if latencies:

                result.avg_ms = (
                    statistics.mean(
                        latencies
                    )
                )

                result.p50_ms = percentile(
                    latencies,
                    50,
                )

                result.p95_ms = percentile(
                    latencies,
                    95,
                )

                result.p99_ms = percentile(
                    latencies,
                    99,
                )

                result.min_ms = min(
                    latencies
                )

                result.max_ms = max(
                    latencies
                )

                result.total_inference_s = (
                    sum(latencies)
                    / 1000.0
                )

                result.output_fps = (
                    len(latencies)
                    / result.total_inference_s
                    if result.total_inference_s
                    else 0.0
                )

            # -------------------------------------------------
            # Detect invalid benchmark result
            # -------------------------------------------------

            # A model that saw frames but produced zero
            # measured inferences is not a valid result.
            #
            # This prevents something like ViTPose having
            # 300 errors but still being reported as OK.
            if (
                frames_seen > 0
                and len(latencies) == 0
            ):

                result.status = "FAILED"

                if not result.error:

                    if errors > 0:

                        result.error = (
                            "All benchmark "
                            "inference calls failed."
                        )

                    else:

                        result.error = (
                            "No inference output "
                            "was produced."
                        )

            # -------------------------------------------------
            # Peak GPU memory
            # -------------------------------------------------

            if (
                result.device_used.lower()
                == "cuda"
            ):

                try:

                    import torch

                    torch.cuda.synchronize()

                    result.peak_vram_allocated_mb = (
                        torch.cuda
                        .max_memory_allocated()
                        / (1024 * 1024)
                    )

                except Exception:
                    pass

            # -------------------------------------------------
            # Print result
            # -------------------------------------------------

            print()

            print(
                f"Successful calls: "
                f"{result.successful_inferences}"
            )

            print(
                f"Errors          : "
                f"{result.errors}"
            )

            print(
                f"Average latency : "
                f"{result.avg_ms:.2f} ms"
            )

            print(
                f"P50             : "
                f"{result.p50_ms:.2f} ms"
            )

            print(
                f"P95             : "
                f"{result.p95_ms:.2f} ms"
            )

            print(
                f"P99             : "
                f"{result.p99_ms:.2f} ms"
            )

            print(
                f"Output FPS      : "
                f"{result.output_fps:.2f}"
            )

            print(
                f"Peak RAM delta  : "
                f"{result.peak_ram_delta_mb:.1f} MB"
            )

            if (
                result.peak_vram_allocated_mb
                is not None
            ):

                print(
                    f"Peak VRAM alloc : "
                    f"{result.peak_vram_allocated_mb:.1f} MB"
                )

            if result.status != "OK":

                print(
                    f"Status          : "
                    f"{result.status}"
                )

                print(
                    f"Reason          : "
                    f"{result.error}"
                )

        except Exception as exc:

            result.status = "FAILED"

            result.error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f"FAILED: "
                f"{result.error}"
            )

        finally:

            try:

                model.close()

            except Exception as exc:

                print(
                    "Close warning: "
                    f"{exc}"
                )

            gc.collect()

        return result

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------

    def _warmup(
        self,
        model,
        video_path,
        info,
        warmup_frames,
    ):

        if warmup_frames <= 0:
            return

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
            )

        count = 0

        while count < warmup_frames:

            ok, frame = cap.read()

            if not ok:
                break

            try:

                model.process_frame(
                    frame,
                    count,
                    info.fps,
                )

            except Exception:

                # Warm-up failures are intentionally
                # ignored. The actual benchmark starts
                # after model.reset().
                pass

            count += 1

        cap.release()

        model.synchronize()

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    def _print_summary(self):

        print()

        print(
            "=" * 150
        )

        print(
            "FINAL RESULTS"
        )

        print(
            "=" * 150
        )

        print(
            f"{'Model':<22} "
            f"{'Status':<9} "
            f"{'Device':<7} "
            f"{'Avg ms':>9} "
            f"{'P95':>9} "
            f"{'FPS':>9} "
            f"{'RAM Δ':>10} "
            f"{'Params':>14}"
        )

        print(
            "-" * 150
        )

        for result in self.results:

            params = (
                f"{result.parameters:,}"
                if result.parameters
                else "-"
            )

            print(
                f"{result.model:<22} "
                f"{result.status:<9} "
                f"{result.device_used:<7} "
                f"{result.avg_ms:>9.2f} "
                f"{result.p95_ms:>9.2f} "
                f"{result.output_fps:>9.2f} "
                f"{result.peak_ram_delta_mb:>9.1f}M "
                f"{params:>14}"
            )

            if (
                result.status != "OK"
                and result.error
            ):

                print(
                    f"  -> "
                    f"{result.error}"
                )
        if platform.system() == "Windows":
            print()
            print(
                "Note: BlazePose is CPU-only on Windows "
                "in this benchmark (MediaPipe Tasks)."
    )
        print(
            "=" * 150
        )

    # ---------------------------------------------------------
    # Save CSV + JSON
    # ---------------------------------------------------------

    def _save_results(
        self,
        video_path,
        info,
        device,
    ):

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        stamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        base = (
            RESULTS_DIR
            / f"benchmark_{device}_{stamp}"
        )

        csv_path = (
            base.with_suffix(".csv")
        )

        json_path = (
            base.with_suffix(".json")
        )

        rows = [
            asdict(result)
            for result in self.results
        ]

        if rows:

            fieldnames = list(
                rows[0].keys()
            )

        else:

            fieldnames = list(
                asdict(
                    Result(
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    )
                ).keys()
            )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "created": (
                datetime.now()
                .isoformat()
            ),
            "system": {
                "platform": (
                    platform.platform()
                ),
                "python": (
                    platform.python_version()
                ),
                "processor": (
                    platform.processor()
                ),
                "cpu_count": os.cpu_count(),
            },
            "benchmark": {
                "video": str(
                    Path(
                        video_path
                    ).resolve()
                ),
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "frames": info.frames,
                "duration_s": (
                    info.duration_s
                ),
                "device_mode": device,
            },
            "results": rows,
        }

        json_path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()

        print(
            f"CSV : "
            f"{csv_path.resolve()}"
        )

        print(
            f"JSON: "
            f"{json_path.resolve()}"
        )

        return (
            csv_path,
            json_path,
        )


__all__ = [
    "BenchmarkRunner",
]