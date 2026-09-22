from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def main():
    parser = argparse.ArgumentParser(description="Compare CPU and CUDA benchmark JSON files.")
    parser.add_argument("cpu_json", type=Path)
    parser.add_argument("gpu_json", type=Path)
    args = parser.parse_args()

    cpu = {row["model"]: row for row in load(args.cpu_json)}
    gpu = {row["model"]: row for row in load(args.gpu_json)}
    names = list(dict.fromkeys([*cpu.keys(), *gpu.keys()]))

    print("Model                       CPU ms    CUDA ms   Speedup   CPU FPS   CUDA FPS")
    print("-" * 82)
    for name in names:
        c = cpu.get(name, {})
        g = gpu.get(name, {})
        c_ms = c.get("avg_ms", 0.0) or 0.0
        g_ms = g.get("avg_ms", 0.0) or 0.0
        speedup = c_ms / g_ms if c_ms > 0 and g_ms > 0 else 0.0
        print(
            f"{name:<27}"
            f"{c_ms:>9.2f} "
            f"{g_ms:>9.2f} "
            f"{speedup:>8.2f}x "
            f"{(c.get('output_fps', 0.0) or 0.0):>9.2f} "
            f"{(g.get('output_fps', 0.0) or 0.0):>9.2f}"
        )


if __name__ == "__main__":
    main()
