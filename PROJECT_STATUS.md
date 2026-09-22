# Project status

Implemented in this repository:

- Reachy Mini / MuJoCo recording
- Fixed 10-second MP4 workflow
- CPU and CUDA benchmark modes
- BlazePose adapter
- ViTPose adapter
- MoViNet-A0 adapter
- TSM-MobileNetV2 adapter
- CSV + JSON export
- RAM / CPU / optional CUDA memory metrics
- Model metadata (parameters / memory where available)
- Simple CLI

The benchmark intentionally does not include WHAM, CameraHMR, FSPose, GraphEnet or SAM2. Those were excluded from this focused project because they do not belong to the same two task groups selected for the new research scope, or require different input/hardware assumptions.
