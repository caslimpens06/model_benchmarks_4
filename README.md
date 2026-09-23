# Reachy Mini Model Benchmark

A deliberately small benchmark project for comparing four AI approaches on the **same 10-second Reachy Mini video**:

| Model | Type | Main output |
|---|---|---|
| BlazePose | Pose / skeleton | 2D body landmarks |
| ViTPose | Pose / skeleton | 2D body keypoints |
| MoViNet-A0 | Direct activity recognition | Video action classification |
| TSM-MobileNetV2 | Direct activity recognition | Video action classification |

The project is intended to answer a **performance / computational-cost** question, not to claim that the models have the same task accuracy.

## Why these four models?

BlazePose and ViTPose are pose-estimation approaches: they estimate body keypoints which can be represented as a skeleton. MoViNet-A0 and TSM-MobileNetV2 instead operate on image/video information to classify activities, so they cover the second approach requested by the research scope: activity recognition directly from the visual stream.

## Benchmark principle

1. The Reachy Mini simulator records one fixed 10-second MP4.
2. That exact MP4 is reused for every model.
3. Video decoding is **not** part of the timed inference measurement.
4. Each model is loaded separately and warmed up before timing.
5. Frame/stream models process the same source frames; TSM uses the same sliding 8-frame windows from the same video.
6. CPU and CUDA are separate benchmark modes.
7. BlazePose is kept on CPU because this project uses the desktop MediaPipe Python task API. Torch-based models can use CPU or CUDA.
