from __future__ import annotations

import importlib
import platform

print("Python:", platform.python_version())
print("Platform:", platform.platform())

for name in ["cv2", "numpy", "psutil", "mediapipe", "torch", "transformers"]:
    try:
        module = importlib.import_module(name)
        print(f"{name:12}: OK", getattr(module, "__version__", ""))
    except Exception as exc:
        print(f"{name:12}: ERROR -> {exc}")

try:
    import torch
    print("torch CUDA :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch GPU  :", torch.cuda.get_device_name(0))
except Exception:
    pass
