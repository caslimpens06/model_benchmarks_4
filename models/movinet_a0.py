from __future__ import annotations

import gc

from .base import ModelAdapter, ModelOutput


class MoViNetA0(ModelAdapter):
    name = "MoViNet-A0"
    family = "direct activity recognition"
    task = "video action classification"
    input_type = "RGB frame stream"
    temporal = True
    implementation = "Atze00 MoViNet PyTorch port, pretrained Kinetics-600"
    license = "MIT for the PyTorch port; original model has its own terms"

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.torch = None

    def load(self, device: str) -> None:
        try:
            import torch
            from movinets import MoViNet
            from movinets.config import _C
        except ImportError as exc:
            raise RuntimeError(
                "MoViNet-A0 requires the PyTorch MoViNet package. "
                "Run the setup script first."
            ) from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU mode requested, but PyTorch CUDA is not available."
            )

        self.torch = torch
        self.device = torch.device(device)
        self.model = MoViNet(
            _C.MODEL.MoViNetA0,
            causal=True,
            pretrained=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.reset()

    def reset(self) -> None:
        if self.model is not None:
            clean = getattr(self.model, "clean_activation_buffers", None)
            if clean is not None:
                clean()

    def process_frame(self, frame, frame_index: int, fps: float) -> ModelOutput:
        if self.model is None:
            raise RuntimeError("MoViNet-A0 is not loaded")

        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (172, 172), interpolation=cv2.INTER_AREA)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        tensor = tensor.float().div_(255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(2).to(self.device)

        with self.torch.inference_mode():
            output = self.model(tensor)
        return ModelOutput(output)

    def parameter_count(self) -> int | None:
        if self.model is None:
            return None
        return sum(p.numel() for p in self.model.parameters())

    def model_size_mb(self) -> float | None:
        if self.model is None:
            return None
        total = 0
        for tensor in list(self.model.parameters()) + list(self.model.buffers()):
            total += tensor.numel() * tensor.element_size()
        return total / (1024 * 1024)

    def synchronize(self) -> None:
        if self.torch is not None and self.device == "cuda":
            self.torch.cuda.synchronize()

    def close(self) -> None:
        self.model = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
