from __future__ import annotations

import gc
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from .base import ModelAdapter, ModelOutput


TSM_ROOT = Path(
    "external/temporal-shift-module"
)

CHECKPOINT = Path(
    "weights/"
    "TSM_kinetics_RGB_mobilenetv2_shift8_blockres_"
    "avg_segment8_e100_dense.pth"
)

CHECKPOINT_URL = (
    "https://hanlab18.mit.edu/projects/tsm/models/"
    "TSM_kinetics_RGB_mobilenetv2_shift8_blockres_"
    "avg_segment8_e100_dense.pth"
)

NUM_SEGMENTS = 8
NUM_CLASSES = 400


class TSMMobileNetV2(ModelAdapter):
    name = "TSM-MobileNetV2"
    family = "direct activity recognition"
    task = "video action classification"
    input_type = "8-frame RGB clip"
    temporal = True
    implementation = "Official TSM MobileNetV2"
    license = "MIT"

    def __init__(self):
        super().__init__()

        self.model = None
        self.device = None
        self.buffer = []

    # ---------------------------------------------------------
    # Official TSM repository
    # ---------------------------------------------------------

    def _ensure_official_repo(self):
        if (
            TSM_ROOT.exists()
            and (TSM_ROOT / "ops").exists()
        ):
            return

        TSM_ROOT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "Downloading official TSM repository..."
        )

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/"
                "mit-han-lab/"
                "temporal-shift-module.git",
                str(TSM_ROOT),
            ],
            check=True,
        )

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    def _ensure_checkpoint(self):
        CHECKPOINT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if CHECKPOINT.exists():
            return

        print(
            "Downloading official TSM "
            "MobileNetV2 Kinetics-400 checkpoint..."
        )

        urllib.request.urlretrieve(
            CHECKPOINT_URL,
            CHECKPOINT,
        )

    # ---------------------------------------------------------
    # Import official TSM
    # ---------------------------------------------------------

    def _import_official_tsm(self):
        import importlib

        self._ensure_official_repo()

        root = str(
            TSM_ROOT.resolve()
        )

        if root not in sys.path:
            sys.path.insert(
                0,
                root,
            )

        # Remove previously loaded TSM modules.
        # This makes sure we import the repository
        # that belongs to this benchmark.
        for module_name in list(
            sys.modules.keys()
        ):
            if (
                module_name == "ops"
                or module_name.startswith(
                    "ops."
                )
            ):
                del sys.modules[
                    module_name
                ]

        ops_models = importlib.import_module(
            "ops.models"
        )

        return ops_models.TSN

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    def load(
        self,
        device: str,
    ):
        if (
            device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "GPU mode requested, but "
                "PyTorch CUDA is not available."
            )

        self.device = torch.device(
            device
        )

        TSN = self._import_official_tsm()

        self._ensure_checkpoint()

        print(
            "Creating official TSM "
            "MobileNetV2 model..."
        )

        # Official TSM configuration for:
        #
        # TSM_kinetics_RGB_mobilenetv2_shift8_
        # blockres_avg_segment8_e100_dense.pth
        self.model = TSN(
            NUM_CLASSES,
            NUM_SEGMENTS,
            "RGB",
            base_model="mobilenetv2",
            consensus_type="avg",
            before_softmax=True,
            dropout=0.8,
            img_feature_dim=256,
            crop_num=1,
            partial_bn=True,
            print_spec=False,
            pretrain="imagenet",
            is_shift=True,
            shift_div=8,
            shift_place="blockres",
            fc_lr5=False,
            temporal_pool=False,
            non_local=False,
        )

        # -----------------------------------------------------
        # Load checkpoint
        # -----------------------------------------------------

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
        )

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and "state_dict" in checkpoint
        ):
            state_dict = checkpoint[
                "state_dict"
            ]
        else:
            state_dict = checkpoint

        # Remove "module." prefix if the checkpoint
        # was created with DataParallel.
        cleaned = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[
                    len("module.") :
                ]

            cleaned[key] = value

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # The downloaded official checkpoint contains:
        #
        #   base_model.classifier.weight
        #   base_model.classifier.bias
        #
        # The TSN model created with dropout=0.8 expects:
        #
        #   new_fc.weight
        #   new_fc.bias
        #
        # This is the same classifier rename used by the
        # official TSM test code.
        # -----------------------------------------------------

        if (
            "base_model.classifier.weight"
            in cleaned
        ):
            cleaned[
                "new_fc.weight"
            ] = cleaned.pop(
                "base_model.classifier.weight"
            )

        if (
            "base_model.classifier.bias"
            in cleaned
        ):
            cleaned[
                "new_fc.bias"
            ] = cleaned.pop(
                "base_model.classifier.bias"
            )

        state_dict = cleaned

        # Load all weights strictly.
        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.reset()

        print(
            "TSM-MobileNetV2 loaded."
        )

    # ---------------------------------------------------------
    # Reset temporal buffer
    # ---------------------------------------------------------

    def reset(self):
        self.buffer.clear()

    # ---------------------------------------------------------
    # Preprocess one frame
    # ---------------------------------------------------------

    @staticmethod
    def _preprocess_frame(
        image,
    ):
        # OpenCV uses BGR.
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # Resize to MobileNetV2 input size.
        rgb = cv2.resize(
            rgb,
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert uint8 -> float32 [0, 1].
        tensor = (
            rgb.astype(
                np.float32
            )
            / 255.0
        )

        # ImageNet normalization.
        mean = np.array(
            [
                0.485,
                0.456,
                0.406,
            ],
            dtype=np.float32,
        )

        std = np.array(
            [
                0.229,
                0.224,
                0.225,
            ],
            dtype=np.float32,
        )

        tensor = (
            tensor - mean
        ) / std

        # HWC -> CHW
        tensor = tensor.transpose(
            2,
            0,
            1,
        )

        return tensor

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    def process_frame(
        self,
        frame,
        frame_index: int,
        fps: float,
    ) -> ModelOutput:

        del frame_index
        del fps

        if self.model is None:
            raise RuntimeError(
                "TSM-MobileNetV2 "
                "is not loaded."
            )

        # Add frame to rolling temporal buffer.
        self.buffer.append(
            frame.copy()
        )

        # TSM needs 8 frames.
        if (
            len(self.buffer)
            < NUM_SEGMENTS
        ):
            return ModelOutput(
                None
            )

        # Keep exactly the latest 8 frames.
        if (
            len(self.buffer)
            > NUM_SEGMENTS
        ):
            self.buffer.pop(0)

        # -----------------------------------------------------
        # Preprocess all 8 frames
        # -----------------------------------------------------

        frames = [
            self._preprocess_frame(
                image
            )
            for image in self.buffer
        ]

        clip = np.stack(
            frames,
            axis=0,
        )

        # Shape before batch:
        #
        # [8, 3, 224, 224]
        #
        # Add batch dimension:
        #
        # [1, 8, 3, 224, 224]
        clip_tensor = (
            torch.from_numpy(
                clip
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # -----------------------------------------------------
        # Forward pass
        # -----------------------------------------------------

        with torch.inference_mode():
            output = self.model(
                clip_tensor
            )

        return ModelOutput(
            output
        )

    # ---------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------

    def parameter_count(
        self,
    ) -> int | None:

        if self.model is None:
            return None

        return sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )

    # ---------------------------------------------------------
    # Model memory
    # ---------------------------------------------------------

    def model_size_mb(
        self,
    ) -> float | None:

        if self.model is None:
            return None

        total_bytes = 0

        for parameter in (
            self.model.parameters()
        ):
            total_bytes += (
                parameter.numel()
                * parameter.element_size()
            )

        return (
            total_bytes
            / (1024 * 1024)
        )

    # ---------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------

    def synchronize(
        self,
    ):

        if (
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(
        self,
    ):

        self.model = None
        self.buffer.clear()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.device = None