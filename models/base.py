from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelOutput:
    value: Any = None


class ModelAdapter:
    """
    Common interface for every benchmark model.

    Every model inherits these defaults so that the
    benchmark runner can safely use the same interface.
    """

    name = "Unknown"
    family = "Unknown"
    task = "Unknown"
    input_type = "Unknown"
    temporal = False
    implementation = ""
    license = ""

    # True when a model must always run on CPU.
    always_cpu = False

    def __init__(self) -> None:
        self._device_used = "cpu"

    def load(self, device: str) -> None:
        raise NotImplementedError

    def process_frame(
        self,
        frame,
        frame_index: int,
        fps: float,
    ) -> ModelOutput:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def device_used(self) -> str:
        """
        Return the actual device used by this model.

        Model adapters may override this, but normally this
        default implementation is sufficient.
        """

        if self.always_cpu:
            return "CPU"

        return str(self._device_used)

    def parameter_count(self) -> int | None:
        return None

    def model_size_mb(self) -> float | None:
        return None

    def close(self) -> None:
        pass