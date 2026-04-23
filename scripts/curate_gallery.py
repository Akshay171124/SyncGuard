#!/usr/bin/env python
"""Select 10 gallery clips from cascade predictions.

Reads outputs/logs/predictions_cascade_fakeavceleb.npz and picks
top-confidence correct predictions per category. sample_id from the eval
is the processed-directory path; we read metadata.json from that directory
to resolve the raw video path.

Writes demo_assets/gallery/manifest.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np


CATEGORY_QUOTAS = {
    "RV-RA": (2, 0),
    "FV-RA": (3, 1),
    "RV-FA": (3, 1),
    "FV-FA": (2, 1),
}


def pick_top_k(scores, labels, sample_ids, categories, category_name,
               k, ground_truth):
    """Pick k clips where model predicted correctly with highest confidence."""
    mask = (categories == category_name) & (labels == ground_truth)
    if ground_truth == 1:
        correct = mask & (scores > 0.5)
    else:
        correct = mask & (scores < 0.5)

    idxs = np.where(correct)[0]
    if len(idxs) == 0:
        return []

    conf = np.abs(scores[idxs] - 0.5)
    order = np.argsort(-conf)[:k]
    picked = idxs[order]
    return [(str(sample_ids[i]), float(scores[i])) for i in picked]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--repo_root", required=True,
                        help="SyncGuard repo root (for resolving processed paths)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    d = np.load(args.predictions)
    if "sample_ids" not in d.files:
        raise RuntimeError("predictions npz missing 'sample_ids'.")
    scores = d["max_scores"]
    labels = d["labels"]
    sample_ids = d["sample_ids"]
    categories = d["categories"]

    repo_root = Path(args.repo_root)

    manifest = {"clips": []}
    for cat, (count, gt) in CATEGORY_QUOTAS.items():
        picks = pick_top_k(scores, labels, sample_ids, categories,
                           cat, count, gt)
        print(f"[{cat}] found {len(picks)} candidates (wanted {count})")
        for i, (sid, score) in enumerate(picks, start=1):
            processed_dir = repo_root / sid
            meta_path = processed_dir / "metadata.json"
            if not meta_path.exists():
                print(f"  [skip] {sid}: no metadata.json at {meta_path}")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            manifest["clips"].append({
                "clip_id": f"{cat.lower().replace('-', '_')}_{i:02d}",
                "sample_id": sid,
                "category": cat,
                "ground_truth": "real" if gt == 0 else "fake",
                "hpc_confidence": score,
                "processed_dir": sid,
                "video_path": meta["video_path"],
                "audio_duration_s": meta.get("audio_duration_s", 0.0),
                "speaker_id": meta.get("speaker_id", ""),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest['clips'])} clips to {args.output}")


if __name__ == "__main__":
    main()
