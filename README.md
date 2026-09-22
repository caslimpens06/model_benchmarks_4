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

## Metrics

The benchmark records:

- model load time
- average latency
- P50 / P95 / P99 latency
- minimum / maximum latency
- output FPS
- peak process RAM increase
- process CPU usage samples
- parameter count when available
- model memory footprint when available
- CUDA allocated memory when CUDA is used
- errors / successful inference calls

This is intentionally enough to compare computational cost without turning the project into a large framework.

## Important interpretation

The direct-activity models are pretrained on general action-recognition datasets. Their predictions are **not** treated as an accuracy benchmark for the team's physiotherapy exercises. To evaluate activity recognition for the actual application, a separate labeled dataset and task-specific evaluation would be required.

ViTPose is a top-down pose estimator and normally uses an object detector first. For this performance benchmark, it receives one identical full-frame person box on every frame. This avoids introducing a second detector into the timing and keeps the benchmark focused on the pose model itself.

TSM-MobileNetV2 uses the official TSM architecture/checkpoint. Its official repository provides an 8-frame MobileNetV2 model for Kinetics-400. The benchmark uses the same video and a sliding 8-frame window for each output.

MoViNet-A0 uses an unofficial PyTorch port with pretrained Kinetics-600 weights, because the PyTorch implementation makes CPU/CUDA benchmarking straightforward in the same environment. The official MoViNet project itself is from TensorFlow Model Garden / TF Hub.

## Setup: CPU

Use Python 3.12. The Reachy Mini SDK documentation in the project research specifies Python 3.10-3.12.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_cpu.ps1
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\check_environment.py
```

## Setup: GPU

For a CUDA-capable NVIDIA development PC:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_gpu.ps1
```

Then verify:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
```

The setup script uses a documented PyTorch 2.6.0 + CUDA 12.6 wheel for Windows.

## Run the simulator

Start the Reachy Mini daemon in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
reachy-mini-daemon --sim --scene minimal
```

Use one daemon only. If port 8000 is already occupied, stop the existing daemon before starting another one.

## Record the fixed benchmark video

```powershell
.\.venv\Scripts\Activate.ps1
python main.py --record
```

The result is:

```text
videos/benchmark_video.mp4
```

Do the same movement sequence during this recording that you want to use in every comparison.

## CPU benchmark

```powershell
python main.py --benchmark --device cpu
```

## GPU benchmark

```powershell
python main.py --benchmark --device cuda
```

The same `benchmark_video.mp4` is used for both runs. If you want perfectly paired runs, record only once and do not recreate the video between the CPU and GPU tests.

## Test one model

```powershell
python main.py --benchmark --device cpu --models blazepose
python main.py --benchmark --device cpu --models movinet
python main.py --benchmark --device cpu --models tsm
python main.py --benchmark --device cpu --models vitpose
```

## Results

Results are saved in:

```text
results/benchmark_cpu_YYYYMMDD_HHMMSS.csv
results/benchmark_cpu_YYYYMMDD_HHMMSS.json
results/benchmark_cuda_YYYYMMDD_HHMMSS.csv
results/benchmark_cuda_YYYYMMDD_HHMMSS.json
```

## Research text

The four models are deliberately split into two categories. BlazePose and ViTPose estimate human pose/keypoints first, producing skeletal representations that can later be used for activity classification. MoViNet-A0 and TSM-MobileNetV2 perform direct video/action recognition without requiring a skeleton as the intermediate representation. Testing both categories on the same video makes it possible to compare not only different model sizes, but also the computational implications of these two different approaches.

## Sources

- MediaPipe Pose Landmarker: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker
- ViTPose: https://huggingface.co/docs/transformers/model_doc/vitpose
- MoViNet: https://www.tensorflow.org/hub/tutorials/movinet
- TSM: https://github.com/mit-han-lab/temporal-shift-module
- PyTorch installation: https://docs.pytorch.org/get-started/previous-versions/
