# Windows quick start

## 1. Prepare Python

Install Python 3.12 and Git.

## 2. CPU setup

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_cpu.ps1
```

## 3. GPU setup

Use this instead of the CPU setup when the PC has an NVIDIA CUDA GPU:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_gpu.ps1
```

Do not run the CPU setup and GPU setup into the same environment after switching modes. Delete `.venv` and recreate it when switching.

## 4. Start simulator

Terminal A:

```powershell
.\.venv\Scripts\Activate.ps1
reachy-mini-daemon --sim --scene minimal
```

Use exactly one daemon. A second daemon will usually fail because TCP port 8000 is already in use.

## 5. Record once

Terminal B:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py --record
```

This records exactly 300 frames at 30 FPS into `videos\benchmark_video.mp4`.

## 6. Benchmark the same file on CPU

```powershell
python main.py --benchmark --device cpu
```

## 7. Benchmark the same file on CUDA

```powershell
python main.py --benchmark --device cuda
```

The file in `videos\benchmark_video.mp4` is not recreated, so the CPU and CUDA measurements use the same video.

## 8. Compare runs

```powershell
python compare_results.py results\benchmark_cpu_<timestamp>.json results\benchmark_cuda_<timestamp>.json
```
