#!/usr/bin/env python3
"""Spot-check source video fps across datasets.

Determines whether AVSpeech/LRS2 need reprocessing after the HP-2 fps fix.
If >95% of videos are 25fps, reprocessing is not needed.

Usage:
    python scripts/check_dataset_fps.py --dataset avspeech --max_samples 200
    python scripts/check_dataset_fps.py --dataset lrs2 --max_samples 200
    python scripts/check_dataset_fps.py --dataset dfdc --max_samples 50
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2


def check_fps(data_dir: str, max_samples: int = 200):
    """Check source fps distribution for mp4 files in a directory."""
    root = Path(data_dir)
    if not root.exists():
        print(f"ERROR: Directory not found: {root}")
        return

    videos = sorted(root.rglob("*.mp4"))[:max_samples]
    if not videos:
        print(f"No .mp4 files found in {root}")
        return

    fps_counts = Counter()
    fps_values = []

    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_rounded = round(fps)
            fps_counts[fps_rounded] += 1
            fps_values.append(fps)
            cap.release()

    total = len(fps_values)
    print(f"\nFPS distribution ({total} videos sampled from {root.name}):")
    print("-" * 50)
    for fps, count in sorted(fps_counts.items()):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {fps:3d} fps: {count:5d} ({pct:5.1f}%) {bar}")

    n_25fps = fps_counts.get(25, 0)
    pct_25 = n_25fps / total * 100

    print(f"\nVerdict:")
    if pct_25 >= 95:
        print(f"  {pct_25:.1f}% are 25fps — NO reprocessing needed.")
        print(f"  The HP-2 fps fix produces identical output for 25fps sources.")
    else:
        print(f"  Only {pct_25:.1f}% are 25fps — REPROCESSING RECOMMENDED.")
        print(f"  {total - n_25fps} videos would produce different frame counts with the fix.")


def main():
    parser = argparse.ArgumentParser(description="Check source video fps distribution")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fakeavceleb", "avspeech", "lrs2", "dfdc"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    from src.utils.config import load_config
    config = load_config(args.config)

    dir_key = f"{args.dataset}_dir"
    data_dir = config["data"].get(dir_key)
    if not data_dir:
        print(f"No '{dir_key}' in config")
        sys.exit(1)

    check_fps(data_dir, args.max_samples)


if __name__ == "__main__":
    main()
