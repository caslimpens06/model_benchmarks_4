from __future__ import annotations

import gc

import numpy as np

from .base import ModelAdapter, ModelOutput


MODEL_ID = (
    "usyd-community/vitpose-base-simple"
)


class ViTPose(ModelAdapter):
    name = "ViTPose"
    family = "pose / skeleton"
    task = "2D human pose"
    input_type = "RGB frame"
    temporal = False
    implementation = (
        "Hugging Face Transformers ViTPose"
    )
    license = (
        "See model card / checkpoint terms"
    )

    def __init__(self) -> None:
        super().__init__()

        self.model = None
        self.processor = None
        self.torch = None
        self.device = None

    def load(
        self,
        device: str,
    ) -> None:

        try:
            import torch

            from transformers import (
                AutoProcessor,
                VitPoseForPoseEstimation,
            )

        except ImportError as exc:

            raise RuntimeError(
                "ViTPose requires PyTorch, "
                "Transformers and Pillow."
            ) from exc

        if (
            device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "GPU mode requested, but "
                "PyTorch CUDA is not available."
            )

        self.torch = torch

        self.device = torch.device(
            device
        )

        # --------------------------------------------------
        # Work around the current Transformers ViTPose
        # image-processing bug where `inv` is referenced
        # without being defined.
        # --------------------------------------------------

        import transformers.models.vitpose.image_processing_vitpose as vitpose_processor

        if not hasattr(
            vitpose_processor,
            "inv",
        ):
            vitpose_processor.inv = np.linalg.inv

        # --------------------------------------------------
        # Load processor and model
        # --------------------------------------------------

        self.processor = (
            AutoProcessor.from_pretrained(
                MODEL_ID
            )
        )

        self.model = (
            VitPoseForPoseEstimation
            .from_pretrained(
                MODEL_ID
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def process_frame(
        self,
        frame,
        frame_index: int,
        fps: float,
    ) -> ModelOutput:

        del frame_index
        del fps

        if (
            self.model is None
            or self.processor is None
        ):
            raise RuntimeError(
                "ViTPose is not loaded."
            )

        height, width = frame.shape[:2]

        # OpenCV uses BGR.
        # ViTPose expects RGB.
        rgb = frame[:, :, ::-1].copy()

        # ViTPose is a top-down pose estimator.
        #
        # Normally a person detector provides the
        # bounding boxes. To keep this benchmark fair,
        # we intentionally do NOT add a detector.
        #
        # Every frame receives the same full-frame
        # person box.

        person_boxes = [
            [
                0.0,
                0.0,
                float(width),
                float(height),
            ]
        ]

        inputs = self.processor(
            rgb,
            boxes=[person_boxes],
            return_tensors="pt",
        )

        inputs = inputs.to(
            self.device
        )

        with self.torch.inference_mode():

            outputs = self.model(
                **inputs
            )

        pose_results = (
            self.processor
            .post_process_pose_estimation(
                outputs,
                boxes=[person_boxes],
            )
        )

        return ModelOutput(
            pose_results
        )

    def parameter_count(self) -> int | None:

        if self.model is None:
            return None

        return sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )

    def model_size_mb(self) -> float | None:

        if self.model is None:
            return None

        total_bytes = 0

        for tensor in list(
            self.model.parameters()
        ) + list(
            self.model.buffers()
        ):

            total_bytes += (
                tensor.numel()
                * tensor.element_size()
            )

        return (
            total_bytes
            / (1024 * 1024)
        )

    def synchronize(self) -> None:

        if (
            self.torch is not None
            and self.device is not None
            and self.device.type == "cuda"
        ):
            self.torch.cuda.synchronize()

    def close(self) -> None:

        self.model = None
        self.processor = None

        gc.collect()

        if (
            self.torch is not None
            and self.torch.cuda.is_available()
        ):

            self.torch.cuda.empty_cache()

        self.device = None