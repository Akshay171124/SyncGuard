"""Gate G2: verify preprocessed sample integrity.

Checks one sample per dataset for the artefacts the training dataloader
expects: a 4-D mouth-crop array, an extracted audio track, and a speech mask.
"""
import json
import sys
from pathlib import Path

import numpy as np

DATASETS = ["fakeavceleb", "avspeech", "lrs2"]
failures = []

print("=== G2: preprocessed sample integrity ===")
for ds in DATASETS:
    root = Path(f"data/processed/{ds}")
    meta_path = next(root.rglob("metadata.json"), None)
    if meta_path is None:
        print(f"{ds:14s} NO SAMPLES FOUND")
        failures.append(f"{ds}: no samples")
        continue

    d = meta_path.parent
    meta = json.loads(meta_path.read_text())
    crops = np.load(d / "mouth_crops.npy")
    has_wav = (d / "audio.wav").exists()
    mask_path = d / "speech_mask.npy"
    smask = np.load(mask_path) if mask_path.exists() else None
    errors = [k for k in meta if k.startswith("error")]

    print(f"{ds:14s} sample={d.name}")
    print(f"               crops={crops.shape} dtype={crops.dtype}")
    print(f"               audio.wav={has_wav}  speech_mask="
          f"{'missing' if smask is None else smask.shape}")
    print(f"               frames={meta.get('num_frames')} "
          f"valid={meta.get('num_valid_frames')} "
          f"det_rate={meta.get('detection_rate')}")
    print(f"               errors={errors or 'none'}")

    if crops.ndim != 4:
        failures.append(f"{ds}: crops ndim {crops.ndim}, expected 4 (T,C,H,W)")
    if not has_wav:
        failures.append(f"{ds}: audio.wav missing")
    if smask is None:
        failures.append(f"{ds}: speech_mask.npy missing")
    if errors:
        failures.append(f"{ds}: sample carries {errors}")

print()
if failures:
    print("G2 FAIL")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("G2 PASS")
