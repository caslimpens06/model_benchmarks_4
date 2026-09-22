# Benchmark methodology

## Research comparison

The benchmark contains four intentionally different approaches:

- **BlazePose** and **ViTPose**: pose / skeleton-based.
- **MoViNet-A0** and **TSM-MobileNetV2**: direct activity recognition from visual data.

The goal of the benchmark is to compare **computational performance**, not to rank the models by movement-recognition accuracy.

## Controlled input

A single 10-second video is recorded using the Reachy Mini simulator. That file is never changed between the CPU and GPU runs or between model runs.

The benchmark reopens the same MP4 for each model. Video decoding is performed before the timed model call and therefore is not included in the reported model latency.

## Model-specific input granularity

The phrase “same way” means the same **source video**, not an artificial requirement that every architecture must receive exactly the same tensor shape.

- BlazePose: one video frame at a time.
- ViTPose: one video frame at a time, with one identical full-frame person box.
- MoViNet-A0: one frame at a time in streaming/causal mode.
- TSM-MobileNetV2: the same video, converted to a sliding sequence of 8-frame RGB windows, matching the official 8-frame TSM setup.

## CPU and GPU

Two separate runs are performed:

- **CPU mode**: Torch models explicitly run on CPU.
- **CUDA mode**: Torch models explicitly run on NVIDIA CUDA when available.

BlazePose is reported as CPU-only in this project because the desktop MediaPipe Python task path used here is CPU-based.

## Metrics

Latency is reported as mean, P50, P95, P99, minimum and maximum. Throughput is reported as successful output calls per second. Memory measurements include process RAM growth and, for CUDA Torch models, allocated GPU memory.

## What not to conclude

A faster model is not automatically a better activity-recognition model. Pose models and direct activity models solve different intermediate problems, and pretrained action models are not necessarily trained for the team's physiotherapy classes.

A PC benchmark is also not a substitute for a benchmark on the Reachy Mini's onboard CM4. The PC run is a controlled comparative development benchmark; deployment performance on the robot should be measured on the target hardware when available.
