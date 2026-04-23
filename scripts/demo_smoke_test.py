#!/usr/bin/env python
"""End-to-end smoke test: run DemoInference on every gallery clip.

Exits 0 if every clip's verdict matches its ground truth; 1 otherwise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.demo.inference import DemoInference
from src.demo.gallery import load_gallery_lazy
from src.demo.explain import generate_explanation


CONFIG = "configs/default.yaml"
CHECKPOINT = "demo_assets/checkpoints/finetune_best.pt"
AUDIO_CHECKPOINT = "demo_assets/checkpoints/audio_clf_best.pt"
GALLERY = "demo_assets/gallery"


def main():
    print("Loading model...")
    infer = DemoInference(CONFIG, CHECKPOINT,
                          audio_checkpoint_path=AUDIO_CHECKPOINT)
    print(f"  device={infer.device}")

    rows = []
    for clip in load_gallery_lazy(GALLERY):
        result = infer.analyze(
            mouth_crops=clip.mouth_crops,
            audio_waveform=clip.audio_waveform,
            ear_features=clip.ear_features,
            clip_duration_s=clip.duration_s,
        )
        match = "OK" if result.verdict == clip.ground_truth else "MISMATCH"
        timings = result.timings
        rows.append((clip.clip_id, clip.category, clip.ground_truth,
                     result.verdict, result.confidence,
                     timings.get("forward", 0.0), match))
        print(f"{clip.clip_id:15s} [{clip.category:20s}] "
              f"gt={clip.ground_truth:4s} pred={result.verdict:4s} "
              f"conf={result.confidence:.2f} "
              f"fwd={timings.get('forward', 0.0):.2f}s  {match}")
        print(f"   {generate_explanation(result)}")

    mismatches = [r for r in rows if r[-1] == "MISMATCH"]
    print(f"\nResult: {len(rows) - len(mismatches)}/{len(rows)} match ground truth")
    if mismatches:
        print("Mismatches:")
        for r in mismatches:
            print(f"  {r[0]} ({r[1]}): expected {r[2]}, got {r[3]} @ {r[4]:.2f}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
