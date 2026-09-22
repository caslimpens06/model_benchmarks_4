from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

from .base import ModelAdapter, ModelOutput


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

MODEL_PATH = Path("weights/pose_landmarker_full.task")


class BlazePose(ModelAdapter):
    name = "BlazePose"
    family = "pose / skeleton"
    task = "2D human pose"
    input_type = "RGB frame"
    temporal = True
    implementation = "MediaPipe Pose Landmarker"
    license = "Apache 2.0"

    # MediaPipe Tasks in this benchmark is CPU-only.
    always_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.pose = None

    def _ensure_model(self) -> None:
        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if MODEL_PATH.exists():
            return

        print("Downloading BlazePose model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH,
        )

    def _create_pose(self) -> None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = (
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(
                        MODEL_PATH
                    )
                ),
                running_mode=(
                    vision.RunningMode.VIDEO
                ),
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.pose = (
            vision.PoseLandmarker
            .create_from_options(options)
        )

    def load(self, device: str) -> None:
        del device

        self.device = "cpu"

        self._ensure_model()

        self._create_pose()

    def reset(self) -> None:
        """
        Reset MediaPipe VIDEO state.

        MediaPipe VIDEO mode requires timestamps to increase
        monotonically. The benchmark performs a warm-up pass
        and then starts the real benchmark again at frame 0.

        Recreating the landmarker resets its internal timestamp.
        """

        if self.pose is not None:
            self.pose.close()
            self.pose = None

        self._create_pose()

    def process_frame(
        self,
        frame,
        frame_index: int,
        fps: float,
    ) -> ModelOutput:

        if self.pose is None:
            raise RuntimeError(
                "BlazePose is not loaded."
            )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = int(
            round(
                frame_index
                * 1000.0
                / fps
            )
        )

        result = (
            self.pose.detect_for_video(
                image,
                timestamp_ms,
            )
        )

        return ModelOutput(result)

    def parameter_count(self) -> int | None:
        # MediaPipe does not expose the parameter count
        # reliably through this Python API.
        return None

    def model_size_mb(self) -> float | None:
        try:
            return (
                MODEL_PATH.stat().st_size
                / (1024 * 1024)
            )
        except OSError:
            return None

    def synchronize(self) -> None:
        # CPU-only.
        pass

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()
            self.pose = None