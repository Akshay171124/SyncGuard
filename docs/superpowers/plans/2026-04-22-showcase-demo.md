# Showcase Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Gradio-based live demo on Akshay's Mac for the 2026-04-23 research showcase. Visitors pick from 10 curated FakeAVCeleb clips, get verdict + sync-score plot + natural-language explanation. Stretch: webcam-based live challenge tab behind `SYNCGUARD_LIVE=1`.

**Architecture:** Gradio Blocks app (`scripts/demo.py`) wrapping a `DemoInference` class that loads SyncGuard v4+CA once at startup. Gallery clips are pre-cached as `.npz` files on HPC and `scp`'d to the Mac, so the hot path runs only the model (no ffmpeg/RetinaFace). Rule-based `explain.py` produces 1-2 sentence natural-language "why" from model outputs.

**Tech Stack:** PyTorch (MPS/CPU on Mac), Gradio 4.x, existing SyncGuard modules (`src/models/syncguard.py`, `src/preprocessing/pipeline.py`, `src/evaluation/`), numpy, matplotlib.

**Spec:** [docs/superpowers/specs/2026-04-22-showcase-demo-design.md](../specs/2026-04-22-showcase-demo-design.md)

**Working directory convention:** All paths in this plan are relative to `SyncGuard/` (the git repo root) unless noted. HPC paths are on the Northeastern Explorer cluster under `/scratch/$USER/SyncGuard/`.

---

## Phase 0 — HPC Work (Tonight, Akshay Runs Directly)

### Task 0.1: Verify HPC has what we need

**Files:** None — pure inspection.

- [ ] **Step 1: SSH to HPC and confirm scratch state**

```bash
ssh <user>@login.explorer.neu.edu
cd /scratch/$USER/SyncGuard
ls -la outputs/checkpoints/ | grep -E "finetune_best|cross_attention"
ls data/raw/FakeAVCeleb/ | head
```

Expected: see `finetune_best_run3_audioswap.pt` (or similar v4+CA checkpoint) and FakeAVCeleb raw videos present.

- [ ] **Step 2: Confirm audio backbone weights are cached**

```bash
ls /scratch/$USER/.cache/huggingface/hub/ 2>/dev/null | head
```

Expected: see something like `models--facebook--wav2vec2-base-960h/`. If missing, we pre-download on Mac instead (Task 0.6 Step 5).

- [ ] **Step 3: If anything missing, STOP and restore from backup before proceeding.** Scratch may have been purged. The demo plan assumes the v4+CA checkpoint still exists.

---

### Task 0.2: Patch `scripts/evaluate_cascade.py` to save sample IDs

**Files:**
- Modify: `scripts/evaluate_cascade.py` (around lines 53-94 and 154-163)

The eval script currently collects sample IDs implicitly via the dataloader but doesn't include them in the `.npz` output. Three-spot patch.

- [ ] **Step 1: Add `all_sample_ids` to the inference collector**

Modify `run_cascade_inference` (starts at line 33). Add to the `all_*` lists init block (around line 52-57):

```python
all_sync_scores = []
all_audio_scores = []
all_raw_sync = []
all_labels = []
all_categories = []
all_sample_ids = []   # NEW
```

- [ ] **Step 2: Collect sample_ids from each batch**

Inside the `for batch in dataloader:` loop (around line 86), after the category append:

```python
if hasattr(batch, "categories"):
    all_categories.extend(batch.categories)
if hasattr(batch, "sample_ids") and batch.sample_ids is not None:
    all_sample_ids.extend(batch.sample_ids)   # NEW
```

- [ ] **Step 3: Include in returned dict**

Replace the return block (line 88-94) with:

```python
return {
    "sync_scores": np.concatenate(all_sync_scores),
    "audio_scores": np.concatenate(all_audio_scores),
    "raw_sync": np.concatenate(all_raw_sync),
    "labels": np.concatenate(all_labels),
    "categories": np.array(all_categories) if all_categories else None,
    "sample_ids": np.array(all_sample_ids) if all_sample_ids else None,   # NEW
}
```

- [ ] **Step 4: Include in the saved npz**

In `evaluate_cascade` function, after line 109 where `categories = predictions.get("categories")`, add:

```python
sample_ids = predictions.get("sample_ids")
```

Then extend `save_dict` (line 155-160):

```python
save_dict = {
    "sync_scores": sync_scores,
    "audio_scores": audio_scores,
    "max_scores": max_scores,
    "labels": labels,
}
if categories is not None:
    save_dict["categories"] = categories
if sample_ids is not None:
    save_dict["sample_ids"] = sample_ids          # NEW
```

- [ ] **Step 5: Smoke test the patch locally (no GPU needed)**

```bash
python -c "import ast; ast.parse(open('scripts/evaluate_cascade.py').read())"
```

Expected: no output (clean parse).

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_cascade.py
git commit -m "eval: save sample_ids in cascade predictions npz"
```

---

### Task 0.3: Re-run cascade eval on HPC to capture sample IDs

**Files:** None — pure job submission.

- [ ] **Step 1: Push the patched eval script to HPC**

From Mac:

```bash
scp scripts/evaluate_cascade.py <user>@xfer.discovery.neu.edu:/scratch/$USER/SyncGuard/scripts/
```

- [ ] **Step 2: Submit the eval job on HPC**

From HPC:

```bash
cd /scratch/$USER/SyncGuard
mkdir -p outputs/logs
sbatch scripts/slurm_evaluate_cascade.sh
# OR if cascade has no slurm file, adapt slurm_evaluate.sh:
sbatch --partition=gpu --gres=gpu:h200:1 --time=01:00:00 \
       --wrap="conda activate syncguard && python scripts/evaluate_cascade.py \
         --config configs/default.yaml \
         --sync_checkpoint outputs/checkpoints/finetune_best_run3_audioswap.pt \
         --audio_checkpoint outputs/checkpoints/audio_clf_best.pt \
         --datasets fakeavceleb"
```

- [ ] **Step 3: Wait for completion and verify output**

```bash
squeue -u $USER
# once job finishes:
ls -la outputs/predictions_cascade_fakeavceleb.npz
python -c "import numpy as np; d = np.load('outputs/predictions_cascade_fakeavceleb.npz'); print(list(d.keys()), d['sample_ids'].shape)"
```

Expected: `['sync_scores', 'audio_scores', 'max_scores', 'labels', 'categories', 'sample_ids']` and `sample_ids` has the same N as `labels`.

---

### Task 0.4: Write `curate_gallery.py` — select 10 gallery clips

**Files:**
- Create: `scripts/curate_gallery.py`
- Create: `demo_assets/gallery/manifest.json` (output)

> **Learning-mode contribution opportunity:** The selection logic (top-confidence + category balance + optional demographic heuristic) is a meaningful product decision. This task includes a placeholder `pick_top_k` function; if you want to shape the curation strategy (e.g., prefer higher-variance sync curves for more interesting plots), that's the right place.

- [ ] **Step 1: Create the script with I/O scaffold**

```python
#!/usr/bin/env python
"""Select 10 gallery clips from cascade predictions.

Reads outputs/predictions_cascade_fakeavceleb.npz and picks top-confidence
correct predictions per category, writing demo_assets/gallery/manifest.json.
"""
import argparse
import json
import numpy as np
from pathlib import Path


# Category → (count, ground_truth_label)
CATEGORY_QUOTAS = {
    "RealVideo-RealAudio": (2, 0),
    "FakeVideo-RealAudio": (3, 1),
    "RealVideo-FakeAudio": (3, 1),
    "FakeVideo-FakeAudio": (2, 1),
}


def pick_top_k(scores, labels, sample_ids, categories, category_name,
               k, ground_truth):
    """Pick k clips from one category where model predicted correctly with
    highest confidence.

    Args:
        scores: (N,) cascade max_scores, higher = more fake.
        labels: (N,) ground truth (0=real, 1=fake).
        sample_ids: (N,) string IDs.
        categories: (N,) category strings.
        category_name: which category to filter to.
        k: how many to pick.
        ground_truth: expected label for this category (0 or 1).

    Returns:
        list of (sample_id, score, category) tuples.
    """
    mask = (categories == category_name) & (labels == ground_truth)
    if ground_truth == 1:
        correct = mask & (scores > 0.5)
    else:
        correct = mask & (scores < 0.5)

    idxs = np.where(correct)[0]
    if len(idxs) == 0:
        return []

    # Confidence = distance from 0.5 (higher = more confident)
    conf = np.abs(scores[idxs] - 0.5)
    order = np.argsort(-conf)[:k]
    picked = idxs[order]
    return [(str(sample_ids[i]), float(scores[i]), str(categories[i]))
            for i in picked]


def _short_cat(category):
    return {
        "RealVideo-RealAudio": "rv_ra",
        "FakeVideo-RealAudio": "fv_ra",
        "RealVideo-FakeAudio": "rv_fa",
        "FakeVideo-FakeAudio": "fv_fa",
    }.get(category, "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw_video_root", required=True)
    args = parser.parse_args()

    d = np.load(args.predictions, allow_pickle=True)
    scores = d["max_scores"]
    labels = d["labels"]
    sample_ids = d["sample_ids"]
    categories = d["categories"]

    manifest = {"clips": []}
    for cat, (count, gt) in CATEGORY_QUOTAS.items():
        picks = pick_top_k(scores, labels, sample_ids, categories,
                           cat, count, gt)
        for i, (sid, score, category) in enumerate(picks, start=1):
            manifest["clips"].append({
                "clip_id": f"{_short_cat(category)}_{i:02d}",
                "sample_id": sid,
                "category": category,
                "ground_truth": "real" if gt == 0 else "fake",
                "hpc_confidence": score,
                "video_path": f"{args.raw_video_root}/{sid}.mp4",
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest['clips'])} clips to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the npz**

```bash
python scripts/curate_gallery.py \
    --predictions outputs/predictions_cascade_fakeavceleb.npz \
    --output demo_assets/gallery/manifest.json \
    --raw_video_root data/raw/FakeAVCeleb
```

Expected: "Wrote 10 clips to demo_assets/gallery/manifest.json".

- [ ] **Step 3: Eyeball the manifest**

```bash
cat demo_assets/gallery/manifest.json
```

Verify: 10 clips, categories balanced 2/3/3/2, `hpc_confidence` values look reasonable.

- [ ] **Step 4: Commit**

```bash
git add scripts/curate_gallery.py
git commit -m "demo: add gallery curation script for showcase"
```

---

### Task 0.5: Write `prepare_gallery.py` — precompute `.npz` caches

**Files:**
- Create: `scripts/prepare_gallery.py`

- [ ] **Step 1: Verify `process_video` exists in the pipeline module**

```bash
grep -n "def process_video\|def process\b" src/preprocessing/pipeline.py
```

Expected: at least one matching function. If the actual function name or return shape differs, adapt the caller in Step 2 accordingly — check `src/preprocessing/pipeline.py` for the canonical entry point and its return contract. If no such function exists, expose the already-wired pipeline behind a thin wrapper in `pipeline.py` that returns the `{mouth_crops, waveform, ear, duration_s}` dict.

- [ ] **Step 2: Create the script**

```python
#!/usr/bin/env python
"""Precompute mouth_crops + audio + EAR for each gallery clip."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.preprocessing.pipeline import process_video
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    for clip in manifest["clips"]:
        clip_id = clip["clip_id"]
        video_src = Path(clip["video_path"])
        if not video_src.exists():
            print(f"[skip] missing raw video: {video_src}")
            continue

        video_dst = out_dir / f"{clip_id}.mp4"
        shutil.copy(video_src, video_dst)

        print(f"[process] {clip_id} ({clip['category']})")
        result = process_video(video_path=str(video_src), config=config)

        np.savez(
            out_dir / f"{clip_id}.npz",
            mouth_crops=result["mouth_crops"].astype(np.float32),
            audio_waveform=result["waveform"].astype(np.float32),
            ear_features=result["ear"].astype(np.float32),
            fps=np.int32(config["preprocessing"]["video"]["fps"]),
            duration_s=np.float32(result["duration_s"]),
        )
        print(f"  -> {clip_id}.npz "
              f"(crops {result['mouth_crops'].shape}, "
              f"audio {result['waveform'].shape})")

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run on HPC for all 10 clips**

From HPC (gpu-interactive is fine — this is CPU-bound preprocessing):

```bash
cd /scratch/$USER/SyncGuard
conda activate syncguard
python scripts/prepare_gallery.py \
    --manifest demo_assets/gallery/manifest.json \
    --output_dir demo_assets/gallery
```

Expected: 10 `.mp4` + 10 `.npz` files in `demo_assets/gallery/`.

- [ ] **Step 4: Commit the script (not the assets)**

```bash
git add scripts/prepare_gallery.py
git commit -m "demo: add gallery preprocessing script"

echo "demo_assets/" >> .gitignore
git add .gitignore
git commit -m "demo: gitignore demo_assets directory"
```

---

### Task 0.6: Pull assets from HPC to Mac

**Files:** None — `scp` operations.

- [ ] **Step 1: From Mac, pull checkpoints**

```bash
cd "/Users/akshayprajapati/Desktop/CVPR Project/SyncGuard"
mkdir -p demo_assets/checkpoints
scp <user>@xfer.discovery.neu.edu:/scratch/$USER/SyncGuard/outputs/checkpoints/finetune_best_run3_audioswap.pt \
    demo_assets/checkpoints/finetune_best.pt
```

- [ ] **Step 2: Pull the predictions npz for threshold tuning**

```bash
scp <user>@xfer.discovery.neu.edu:/scratch/$USER/SyncGuard/outputs/predictions_cascade_fakeavceleb.npz \
    demo_assets/predictions_cascade_fakeavceleb.npz
```

- [ ] **Step 3: Pull the gallery directory**

```bash
scp -r <user>@xfer.discovery.neu.edu:/scratch/$USER/SyncGuard/demo_assets/gallery \
    demo_assets/
```

- [ ] **Step 4: Verify all assets present**

```bash
ls demo_assets/checkpoints/
ls demo_assets/gallery/ | wc -l    # expect 21: manifest.json + 10 mp4 + 10 npz
```

- [ ] **Step 5: Pre-download audio backbone on Mac**

```bash
export HF_HOME="$HOME/.cache/huggingface"
conda activate syncguard
python -c "
from transformers import Wav2Vec2Model, Wav2Vec2Processor
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
print('Audio backbone cached locally.')
"
```

(If the visual encoder pulls a different HF model, add its load line too — check `src/models/visual_encoder.py`.)

---

## Phase 1 — Mac-Side Demo Code

### Task 1.1: Create `src/demo/` package

**Files:**
- Create: `src/demo/__init__.py`
- Create: `tests/demo/__init__.py`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p src/demo tests/demo
touch src/demo/__init__.py tests/demo/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add src/demo/__init__.py tests/demo/__init__.py
git commit -m "demo: scaffold src/demo package"
```

---

### Task 1.2: Write failing tests for `explain.py`

**Files:**
- Create: `tests/demo/test_explain.py`

- [ ] **Step 1: Write the test file**

```python
"""Unit tests for rule-based explanation generation."""
from dataclasses import dataclass, field
import numpy as np
import pytest

from src.demo.explain import (
    generate_explanation,
    AnalysisResult,
    SYNC_THRESHOLD,
)


def _mk(verdict="real", confidence=0.9, sync_curve=None, ear_curve=None,
        sync_dip_segments=None, ear_anomaly_score=1.0,
        clip_duration_s=4.0, mean_sync=0.7):
    T = 100
    return AnalysisResult(
        verdict=verdict,
        confidence=confidence,
        sync_curve=sync_curve if sync_curve is not None else np.full(T, mean_sync),
        ear_curve=ear_curve if ear_curve is not None else np.full(T, 0.25),
        sync_dip_segments=sync_dip_segments or [],
        ear_anomaly_score=ear_anomaly_score,
        clip_duration_s=clip_duration_s,
        mean_sync=mean_sync,
        timings={},
    )


class TestRealVerdicts:
    def test_confident_real(self):
        result = _mk(verdict="real", confidence=0.92, mean_sync=0.71)
        text = generate_explanation(result)
        assert "Real" in text and "92%" in text and "stable" in text.lower()

    def test_borderline_real(self):
        result = _mk(verdict="real", confidence=0.52,
                     sync_dip_segments=[(2.0, 2.1)])
        text = generate_explanation(result)
        assert "borderline" in text.lower() or "review" in text.lower()


class TestFakeCategories:
    def test_fv_fa_pattern_sync_and_ear(self):
        result = _mk(verdict="fake", confidence=0.94,
                     sync_dip_segments=[(1.0, 1.4), (2.5, 2.9)],
                     ear_anomaly_score=2.8, mean_sync=0.42)
        text = generate_explanation(result)
        assert "face-swap" in text.lower() and "audio" in text.lower()

    def test_rv_fa_pattern_sync_only(self):
        result = _mk(verdict="fake", confidence=0.81,
                     sync_dip_segments=[(1.0, 1.4), (2.1, 2.6), (3.0, 3.3)],
                     ear_anomaly_score=1.1, mean_sync=0.38)
        text = generate_explanation(result)
        assert ("desync" in text.lower() or "dub" in text.lower() or
                "lip-sync" in text.lower())

    def test_fv_ra_pattern_ear_only(self):
        result = _mk(verdict="fake", confidence=0.76,
                     sync_dip_segments=[],
                     ear_anomaly_score=2.5, mean_sync=0.62)
        text = generate_explanation(result)
        assert "blink" in text.lower() or "face-swap" in text.lower()

    def test_subtle_fake(self):
        result = _mk(verdict="fake", confidence=0.70,
                     sync_dip_segments=[], ear_anomaly_score=1.0)
        text = generate_explanation(result)
        assert "fake" in text.lower()


class TestEdgeCases:
    def test_short_clip_prepends_limitation(self):
        result = _mk(verdict="real", confidence=0.85, clip_duration_s=1.5)
        text = generate_explanation(result)
        assert "limited" in text.lower()

    def test_missing_ear_does_not_crash(self):
        result = _mk(verdict="fake", confidence=0.8,
                     sync_dip_segments=[(1.0, 1.5)],
                     ear_curve=np.zeros(100), ear_anomaly_score=0.0)
        text = generate_explanation(result)
        assert "fake" in text.lower()

    def test_near_threshold_uses_borderline(self):
        result = _mk(verdict="fake", confidence=0.51,
                     sync_dip_segments=[(1.0, 1.1)])
        text = generate_explanation(result)
        assert "borderline" in text.lower() or "review" in text.lower()


class TestConfidencePercent:
    @pytest.mark.parametrize("conf, pct", [(0.5, "50%"), (0.87, "87%"), (0.999, "100%")])
    def test_percent_format(self, conf, pct):
        result = _mk(verdict="fake", confidence=conf,
                     sync_dip_segments=[(1.0, 1.2)])
        text = generate_explanation(result)
        assert pct in text
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/akshayprajapati/Desktop/CVPR Project/SyncGuard"
python -m pytest tests/demo/test_explain.py -v
```

Expected: ImportError (module `src.demo.explain` not found).

---

### Task 1.3: Implement `src/demo/explain.py`

**Files:**
- Create: `src/demo/explain.py`

- [ ] **Step 1: Write the module**

```python
"""Rule-based natural-language explanations for SyncGuard demo verdicts.

Pure function — no I/O, no model, no external state. Fully unit-testable.
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


SYNC_THRESHOLD = 0.55
CONF_HIGH = 0.75
CONF_LOW = 0.55
EAR_ANOMALY_THRESHOLD = 2.0
MIN_DIP_DURATION_S = 0.15
BORDERLINE_BAND = 0.05
SHORT_CLIP_S = 2.0


@dataclass
class AnalysisResult:
    """Output of DemoInference.analyze. Feeds the explainer and the UI plot."""
    verdict: Literal["real", "fake"]
    confidence: float
    sync_curve: np.ndarray
    ear_curve: np.ndarray
    sync_dip_segments: list[tuple[float, float]]
    ear_anomaly_score: float
    clip_duration_s: float
    mean_sync: float
    timings: dict[str, float] = field(default_factory=dict)


def _count_significant_dips(segments):
    return [(s, e) for s, e in segments if (e - s) >= MIN_DIP_DURATION_S]


def _total_dip_duration(segments):
    return sum(e - s for s, e in segments)


def _peak_dip_time(segments):
    if not segments:
        return 0.0
    longest = max(segments, key=lambda seg: seg[1] - seg[0])
    return (longest[0] + longest[1]) / 2


def _pct(conf):
    return f"{int(round(conf * 100))}%"


def _is_borderline(confidence):
    return abs(confidence - 0.5) < BORDERLINE_BAND


def generate_explanation(result: AnalysisResult) -> str:
    """Produce a 1-2 sentence human-readable explanation from model outputs."""
    dips = _count_significant_dips(result.sync_dip_segments)
    num_dips = len(dips)
    total_dip_s = _total_dip_duration(dips)
    peak_time = _peak_dip_time(dips)
    ear_anomaly = result.ear_anomaly_score
    conf = result.confidence
    mean_sync = result.mean_sync
    prefix = ""
    if result.clip_duration_s < SHORT_CLIP_S:
        prefix = "Limited temporal evidence — "

    if result.verdict == "real":
        if _is_borderline(conf) or conf < CONF_LOW:
            body = (
                f"Likely real — {_pct(conf)} confidence (borderline). "
                f"Sync mostly stable with {num_dips} brief dip(s). "
                f"May be natural speech pauses; recommend human review."
            )
        else:
            body = (
                f"Real clip — {_pct(conf)} confidence. "
                f"Audio-visual alignment stable throughout "
                f"(mean sync {mean_sync:.2f}, threshold {SYNC_THRESHOLD:.2f}). "
                f"No blink anomalies detected."
            )
        return prefix + body

    if _is_borderline(conf):
        body = (
            f"Likely fake — {_pct(conf)} confidence (borderline). "
            f"Detected {num_dips} sync dip(s); recommend human review."
        )
        return prefix + body

    if num_dips >= 2 and ear_anomaly > EAR_ANOMALY_THRESHOLD:
        body = (
            f"Fake detected — {_pct(conf)} confidence. "
            f"Lip-sync drops below {SYNC_THRESHOLD:.2f} in {num_dips} "
            f"segments ({total_dip_s:.1f}s total), and blink pattern shows "
            f"{ear_anomaly:.1f}× baseline variance. "
            f"Consistent with face-swap combined with audio manipulation."
        )
    elif num_dips >= 1 and ear_anomaly <= EAR_ANOMALY_THRESHOLD:
        body = (
            f"Fake detected — {_pct(conf)} confidence. "
            f"Audio-visual sync drops below threshold in {num_dips} "
            f"segments totaling {total_dip_s:.1f}s "
            f"(most prominent at {peak_time:.1f}s). "
            f"Face appears natural — pattern consistent with "
            f"audio dubbing or lip-sync generation."
        )
    elif num_dips == 0 and ear_anomaly > EAR_ANOMALY_THRESHOLD:
        body = (
            f"Fake detected — {_pct(conf)} confidence. "
            f"Sync scores remain stable (mean {mean_sync:.2f}), "
            f"but blink pattern anomaly detected "
            f"(EAR variance {ear_anomaly:.1f}× baseline). "
            f"Consistent with face-swap that preserved original audio alignment."
        )
    else:
        body = (
            f"Fake detected — {_pct(conf)} confidence. "
            f"No strong sync or blink anomalies, but learned representation "
            f"indicates manipulation. May be a well-executed fake or an "
            f"out-of-distribution sample."
        )
    return prefix + body
```

- [ ] **Step 2: Run tests to confirm they pass**

```bash
python -m pytest tests/demo/test_explain.py -v
```

Expected: all tests pass. If any fail, fix `explain.py` — the tests encode the spec templates.

- [ ] **Step 3: Commit**

```bash
git add src/demo/explain.py tests/demo/test_explain.py
git commit -m "demo: add rule-based explanation generator with unit tests"
```

---

### Task 1.4: Write `src/demo/inference.py` — DemoInference class

**Files:**
- Create: `src/demo/inference.py`

- [ ] **Step 1: Write the module**

```python
"""Demo-facing inference wrapper around SyncGuard v4+CA."""
import time
from pathlib import Path

import numpy as np
import torch

from src.demo.explain import (
    AnalysisResult,
    SYNC_THRESHOLD,
    MIN_DIP_DURATION_S,
)
from src.models.syncguard import build_syncguard, SyncGuardOutput
from src.utils.config import load_config, get_device


def _find_dip_segments(sync_curve, fps, threshold=SYNC_THRESHOLD):
    """Contiguous (start_s, end_s) segments where s(t) < threshold."""
    below = sync_curve < threshold
    segments = []
    i = 0
    while i < len(below):
        if below[i]:
            start = i
            while i < len(below) and below[i]:
                i += 1
            segments.append((start / fps, i / fps))
        else:
            i += 1
    return segments


def _ear_anomaly_score(ear_curve):
    """Multiple-of-baseline EAR variance. Baseline = first 10-frame variance."""
    if ear_curve is None or len(ear_curve) < 10:
        return 0.0
    baseline = float(np.var(ear_curve[:10]) + 1e-6)
    overall_var = float(np.var(ear_curve))
    return overall_var / baseline


def _load_checkpoint(path, device):
    """Load a SyncGuard checkpoint. Tries weights_only=True first (safer);
    falls back to weights_only=False for legacy checkpoints that embed
    non-tensor state (optimizer, scheduler) at the top level."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


class DemoInference:
    """Loads SyncGuard v4+CA once; .analyze() runs one forward pass."""

    def __init__(self, config_path, checkpoint_path, device=None):
        self.config = load_config(config_path)
        self.device = device or get_device()
        self.model = build_syncguard(self.config).to(self.device)
        ckpt = _load_checkpoint(checkpoint_path, self.device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.train(False)   # inference mode (equivalent to .eval())
        self.fps_sync = self.config["preprocessing"]["audio"]["target_fps"]

    @torch.no_grad()
    def analyze(self, mouth_crops, audio_waveform, ear_features=None,
                clip_duration_s=None):
        """Run one forward pass and build an AnalysisResult.

        Args:
            mouth_crops: np.ndarray (T, 1, 96, 96) float32 in [0,1]
            audio_waveform: np.ndarray (N,) float32 at 16kHz
            ear_features: np.ndarray (T,) or None
            clip_duration_s: float or None (inferred from waveform if None)

        Returns:
            AnalysisResult
        """
        timings = {}
        t0 = time.perf_counter()

        mc = torch.from_numpy(mouth_crops).unsqueeze(0).to(self.device)
        wf = torch.from_numpy(audio_waveform).unsqueeze(0).to(self.device)
        ear = None
        if ear_features is not None:
            ear = torch.from_numpy(ear_features).unsqueeze(0).to(self.device)
        lengths = torch.tensor([mc.shape[1]], device=self.device)

        t_pre = time.perf_counter()
        timings["prep"] = t_pre - t0

        out: SyncGuardOutput = self.model(
            mouth_crops=mc, waveform=wf, lengths=lengths, ear_features=ear,
        )
        timings["forward"] = time.perf_counter() - t_pre

        logit = out.logits.squeeze().item()
        confidence = float(torch.sigmoid(torch.tensor(logit)).item())
        verdict = "fake" if confidence >= 0.5 else "real"
        display_conf = confidence if verdict == "fake" else (1.0 - confidence)

        sync_curve = out.sync_scores.squeeze(0).cpu().numpy()
        mean_sync = float(np.mean(sync_curve))
        dip_segments = _find_dip_segments(sync_curve, self.fps_sync)

        ear_curve_np = (ear_features if ear_features is not None
                        else np.zeros(len(sync_curve), dtype=np.float32))
        ear_anomaly = _ear_anomaly_score(ear_curve_np)

        if clip_duration_s is None:
            clip_duration_s = float(
                len(audio_waveform) /
                self.config["preprocessing"]["audio"]["sample_rate"]
            )

        return AnalysisResult(
            verdict=verdict,
            confidence=display_conf,
            sync_curve=sync_curve,
            ear_curve=ear_curve_np,
            sync_dip_segments=dip_segments,
            ear_anomaly_score=ear_anomaly,
            clip_duration_s=clip_duration_s,
            mean_sync=mean_sync,
            timings=timings,
        )
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python -c "from src.demo.inference import DemoInference; print('ok')"
```

Expected: `ok` (no errors).

- [ ] **Step 3: Commit**

```bash
git add src/demo/inference.py
git commit -m "demo: add DemoInference wrapper for v4+CA"
```

---

### Task 1.5: Write `src/demo/gallery.py` — manifest loader

**Files:**
- Create: `src/demo/gallery.py`

- [ ] **Step 1: Write the module**

```python
"""Gallery manifest + cached-tensor loader."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GalleryClip:
    clip_id: str
    category: str
    ground_truth: str
    hpc_confidence: float
    video_path: Path
    mouth_crops: np.ndarray
    audio_waveform: np.ndarray
    ear_features: np.ndarray
    duration_s: float


def _load_clip(gallery_dir, entry):
    cid = entry["clip_id"]
    npz = np.load(gallery_dir / f"{cid}.npz")
    return GalleryClip(
        clip_id=cid,
        category=entry["category"],
        ground_truth=entry["ground_truth"],
        hpc_confidence=float(entry["hpc_confidence"]),
        video_path=gallery_dir / f"{cid}.mp4",
        mouth_crops=npz["mouth_crops"],
        audio_waveform=npz["audio_waveform"],
        ear_features=npz["ear_features"],
        duration_s=float(npz["duration_s"]),
    )


def load_gallery(gallery_dir):
    """Load manifest + all .npz caches into memory."""
    gallery_dir = Path(gallery_dir)
    with open(gallery_dir / "manifest.json") as f:
        manifest = json.load(f)
    return [_load_clip(gallery_dir, entry) for entry in manifest["clips"]]


def load_gallery_lazy(gallery_dir):
    """Yield clips one at a time (lower memory)."""
    gallery_dir = Path(gallery_dir)
    with open(gallery_dir / "manifest.json") as f:
        manifest = json.load(f)
    for entry in manifest["clips"]:
        yield _load_clip(gallery_dir, entry)
```

- [ ] **Step 2: Quick smoke test**

```bash
python -c "
from src.demo.gallery import load_gallery
clips = load_gallery('demo_assets/gallery')
print(f'Loaded {len(clips)} clips')
for c in clips:
    print(f'  {c.clip_id} [{c.category}] gt={c.ground_truth} '
          f'crops={c.mouth_crops.shape} audio={c.audio_waveform.shape}')
"
```

Expected: 10 clips listed with non-empty tensor shapes.

- [ ] **Step 3: Commit**

```bash
git add src/demo/gallery.py
git commit -m "demo: add gallery manifest loader"
```

---

### Task 1.6: Write `scripts/demo_smoke_test.py` — end-to-end gate

**Files:**
- Create: `scripts/demo_smoke_test.py`

- [ ] **Step 1: Write the script**

```python
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
GALLERY = "demo_assets/gallery"


def main():
    print("Loading model...")
    infer = DemoInference(CONFIG, CHECKPOINT)
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
```

- [ ] **Step 2: Run it**

```bash
cd "/Users/akshayprajapati/Desktop/CVPR Project/SyncGuard"
conda activate syncguard
python scripts/demo_smoke_test.py
```

Expected: prints per-clip verdicts + explanations; exits 0 if all match.

**If some clips mismatch:**
- 1-2 mismatches: borderline cases. Swap them in `manifest.json` for alternates from the predictions npz.
- More than 2: Mac↔HPC drift. Stop and debug. Likely causes: checkpoint state dict key mismatch (check `load_state_dict(..., strict=False)` warnings), MPS numerical drift (try `DemoInference(..., device="cpu")`), missing EAR features in cache.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_smoke_test.py
git commit -m "demo: add end-to-end smoke test over gallery"
```

---

### Task 1.7: Write `scripts/demo.py` — the Gradio app (gallery tab only)

**Files:**
- Create: `scripts/demo.py`

> **Learning-mode contribution opportunity:** The Gradio layout and the presentation of the verdict banner are places where your taste matters. The template below is functional but plain — if you want to adjust the copy, colors, or plot styling to match your presentation aesthetics, that's the place.

- [ ] **Step 1: Write the module**

```python
#!/usr/bin/env python
"""SyncGuard research-showcase demo (Gradio).

Tab 1: curated gallery — click a clip to analyze.
Tab 2: live challenge (only shown if SYNCGUARD_LIVE=1).
"""
import os
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.demo.inference import DemoInference
from src.demo.gallery import load_gallery, GalleryClip
from src.demo.explain import generate_explanation, SYNC_THRESHOLD


CONFIG = "configs/default.yaml"
CHECKPOINT = "demo_assets/checkpoints/finetune_best.pt"
GALLERY_DIR = "demo_assets/gallery"
LIVE_ENABLED = os.environ.get("SYNCGUARD_LIVE", "0") == "1"

C_REAL = "#27AE60"
C_FAKE = "#E74C3C"
C_THRESHOLD = "#95A5A6"


def render_sync_plot(sync_curve, threshold, dip_segments, fps, title=""):
    t = np.arange(len(sync_curve)) / fps
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110)
    ax.plot(t, sync_curve, color="#1A5276", linewidth=2, label="s(t)")
    ax.axhline(threshold, color=C_THRESHOLD, linestyle="--",
               label=f"threshold {threshold:.2f}")
    for s, e in dip_segments:
        ax.axvspan(s, e, alpha=0.25, color=C_FAKE)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sync score")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def verdict_banner_html(verdict, confidence, ground_truth):
    color = C_FAKE if verdict == "fake" else C_REAL
    icon = "🔴" if verdict == "fake" else "🟢"
    correct = "✓ Correct" if verdict == ground_truth else "✗ Incorrect"
    badge_color = C_REAL if verdict == ground_truth else C_FAKE
    return f"""
    <div style="padding:16px; border-radius:12px; background:{color};
                color:white; font-size:22px; text-align:center;
                font-weight:600; margin-bottom:12px;">
        {icon} {verdict.upper()} — {int(round(confidence*100))}% confidence
    </div>
    <div style="padding:8px 16px; border-radius:8px; background:#f4f4f4;
                color:{badge_color}; font-weight:600;">
        {correct} — ground truth: {ground_truth.upper()}
    </div>
    """


def build_demo():
    print("Loading SyncGuard model...")
    infer = DemoInference(CONFIG, CHECKPOINT)
    print(f"  device={infer.device}")

    print("Loading gallery...")
    clips: list[GalleryClip] = load_gallery(GALLERY_DIR)
    clip_by_id = {c.clip_id: c for c in clips}
    print(f"  {len(clips)} clips loaded")

    def on_analyze(clip_id):
        if clip_id is None:
            return None, "", "", None
        clip = clip_by_id[clip_id]
        result = infer.analyze(
            mouth_crops=clip.mouth_crops,
            audio_waveform=clip.audio_waveform,
            ear_features=clip.ear_features,
            clip_duration_s=clip.duration_s,
        )
        plot = render_sync_plot(
            result.sync_curve, SYNC_THRESHOLD,
            result.sync_dip_segments, infer.fps_sync,
            title=f"Sync score over time — {clip.clip_id}",
        )
        banner = verdict_banner_html(
            result.verdict, result.confidence, clip.ground_truth,
        )
        explanation = generate_explanation(result)
        return str(clip.video_path), banner, explanation, plot

    with gr.Blocks(title="SyncGuard — Audio-Visual Deepfake Detection") as app:
        gr.Markdown("# SyncGuard\nContrastive Audio-Visual Deepfake Detection")
        with gr.Tabs():
            with gr.Tab("Gallery"):
                gr.Markdown("Pick a clip. Click **Analyze** to see what the model thinks and why.")
                with gr.Row():
                    clip_picker = gr.Radio(
                        choices=[c.clip_id for c in clips],
                        label="Clips", value=clips[0].clip_id,
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")
                with gr.Row():
                    video_out = gr.Video(label="Clip")
                    plot_out = gr.Image(label="Sync-score curve",
                                        type="pil", show_label=True)
                banner_out = gr.HTML(label="Verdict")
                explanation_out = gr.Markdown(label="Explanation")

                analyze_btn.click(
                    on_analyze,
                    inputs=[clip_picker],
                    outputs=[video_out, banner_out, explanation_out, plot_out],
                )

            if LIVE_ENABLED:
                with gr.Tab("Live Challenge"):
                    gr.Markdown("Coming in Phase 2 — webcam + audio-swap challenge.")

    return app


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_port=7860, share=False, inbrowser=True)
```

- [ ] **Step 2: Launch it**

```bash
conda activate syncguard
python scripts/demo.py
```

Expected: Gradio opens a browser tab at `http://127.0.0.1:7860`. The gallery shows 10 radio-button choices. Click one, click Analyze, see video + banner + plot + explanation.

If Gradio isn't installed:

```bash
echo "gradio>=4.0,<5.0" >> requirements.txt
pip install -r requirements.txt
```

- [ ] **Step 3: Commit**

```bash
git add scripts/demo.py requirements.txt
git commit -m "demo: add Gradio app with gallery tab"
```

---

### Task 1.8: Dress rehearsal + threshold tuning

**Files:**
- Modify (possibly): `src/demo/explain.py` (threshold constants)

- [ ] **Step 1: Click through all 10 clips, record observations**

For each clip, note:
- Verdict correct? ✓ / ✗
- Explanation category matches actual fake type? ✓ / ✗
- Plot visually convincing? ✓ / ✗

- [ ] **Step 2: Derive optimal `SYNC_THRESHOLD` from val ROC (Youden's J)**

One-off script (don't commit):

```python
# /tmp/threshold_youden.py
import numpy as np
from sklearn.metrics import roc_curve

d = np.load("demo_assets/predictions_cascade_fakeavceleb.npz", allow_pickle=True)
raw = d["raw_sync"] if "raw_sync" in d.files else -d["sync_scores"]
labels = d["labels"]
fpr, tpr, thresholds = roc_curve(labels, raw)
j = tpr - fpr
best_idx = int(np.argmax(j))
print(f"Optimal raw_sync threshold (fake-side): {thresholds[best_idx]:.4f}")
print(f"Maps to sync-score threshold: {-thresholds[best_idx]:.4f}")
```

Run: `python /tmp/threshold_youden.py`

- [ ] **Step 3: Update `SYNC_THRESHOLD` in `src/demo/explain.py`**

Replace the default `0.55` with the value from Step 2.

- [ ] **Step 4: Tune `EAR_ANOMALY_THRESHOLD` against observed gallery EARs**

Inspect the distribution in the smoke test output:

```bash
python scripts/demo_smoke_test.py | grep -i "ear\|anomaly"
```

If most real clips have EAR anomaly < 1.5 and most FV-RA fakes have > 2.5, `2.0` is fine. If the gap is (1.2, 1.8), set `EAR_ANOMALY_THRESHOLD = 1.5`.

- [ ] **Step 5: Re-run smoke test after tuning**

```bash
python scripts/demo_smoke_test.py
```

Expected: 10/10 correct, explanations match categories.

- [ ] **Step 6: Commit the tuned thresholds**

```bash
git add src/demo/explain.py
git commit -m "demo: tune sync + EAR thresholds on FakeAVCeleb val"
```

---

### Task 1.9: Record fallback screencap video

**Files:**
- Create: `demo_assets/fallback_screencap.mp4` (gitignored)

- [ ] **Step 1: Launch the app**

```bash
python scripts/demo.py
```

- [ ] **Step 2: macOS screen recording (Cmd+Shift+5), 2-3 min run-through**

Record: opening the app, analyzing 3-4 representative clips (one real, one RV-FA, one FV-RA, one FV-FA), showing the explanations.

- [ ] **Step 3: Save to `demo_assets/fallback_screencap.mp4`**

If the live app crashes tomorrow, open this in QuickTime fullscreen as a last-resort fallback.

**Gate: Gallery is ship-ready. Everything below is stretch (only if above is solid by hour 6).**

---

## Phase 2 — Stretch: Live Challenge Tab

### Task 2.1: Record audio pool sentences

**Files:**
- Create: `demo_assets/live_audio_pool/*.wav` (gitignored)

- [ ] **Step 1: Record 5-8 short sentences (3-4s each) in QuickTime**

Sentences must be meaningfully different from each other (different phonemes) so the swap always produces a visible sync mismatch.

Save as 16kHz mono WAV:

```bash
mkdir -p demo_assets/live_audio_pool
ffmpeg -i raw.mov -ar 16000 -ac 1 demo_assets/live_audio_pool/sentence_01.wav
# repeat for 5-8 recordings
```

---

### Task 2.2: Write `src/demo/live_challenge.py`

**Files:**
- Create: `src/demo/live_challenge.py`

- [ ] **Step 1: Write the module**

```python
"""Live challenge: webcam record → audio swap → inference.

Only imported when SYNCGUARD_LIVE=1.
"""
import random
import subprocess
from pathlib import Path

import numpy as np

from src.preprocessing.pipeline import process_video


AUDIO_POOL_DIR = Path("demo_assets/live_audio_pool")


def swap_audio(input_video, output_video):
    """Replace the audio of input_video with a random clip from the pool."""
    pool = list(AUDIO_POOL_DIR.glob("*.wav"))
    if not pool:
        raise RuntimeError(f"No audio files in {AUDIO_POOL_DIR}")
    chosen = random.choice(pool)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-i", str(chosen),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return chosen


def process_live_clip(video_path, config):
    """Run the full preprocessing pipeline on a live-recorded clip."""
    return process_video(video_path=str(video_path), config=config)
```

- [ ] **Step 2: Commit**

```bash
git add src/demo/live_challenge.py
git commit -m "demo: add live-challenge audio swap + preprocess wrapper"
```

---

### Task 2.3: Wire Live Challenge tab into `scripts/demo.py`

**Files:**
- Modify: `scripts/demo.py` (inside `build_demo` where the `if LIVE_ENABLED:` block currently has a placeholder)

- [ ] **Step 1: Add imports at the top of `scripts/demo.py`**

```python
import tempfile
from src.demo.live_challenge import swap_audio, process_live_clip
```

- [ ] **Step 2: Replace the LIVE_ENABLED placeholder Tab block**

```python
if LIVE_ENABLED:
    with gr.Tab("Live Challenge"):
        gr.Markdown(
            "Record yourself reading a sentence (5 seconds). "
            "We'll swap your audio with a different sentence and see "
            "if the model catches the desync."
        )
        with gr.Row():
            webcam = gr.Video(
                sources=["webcam"],
                include_audio=True,
                label="Record yourself (press record, read a sentence, press stop)",
            )
            live_analyze = gr.Button("Analyze", variant="primary")
        with gr.Row():
            swapped_video = gr.Video(label="Swapped (what the model sees)")
            live_plot = gr.Image(label="Sync-score curve", type="pil")
        live_banner = gr.HTML()
        live_explanation = gr.Markdown()

        def on_live_analyze(recorded_path):
            if recorded_path is None:
                return None, "", "", None
            tmpdir = Path(tempfile.mkdtemp())
            swapped_path = tmpdir / "swapped.mp4"
            swap_audio(recorded_path, swapped_path)
            proc = process_live_clip(swapped_path, infer.config)
            result = infer.analyze(
                mouth_crops=proc["mouth_crops"].astype(np.float32),
                audio_waveform=proc["waveform"].astype(np.float32),
                ear_features=proc["ear"].astype(np.float32),
                clip_duration_s=proc["duration_s"],
            )
            plot = render_sync_plot(
                result.sync_curve, SYNC_THRESHOLD,
                result.sync_dip_segments, infer.fps_sync,
                title="Sync score — live clip",
            )
            banner = verdict_banner_html(
                result.verdict, result.confidence, ground_truth="fake",
            )
            return (str(swapped_path), banner,
                    generate_explanation(result), plot)

        live_analyze.click(
            on_live_analyze,
            inputs=[webcam],
            outputs=[swapped_video, live_banner, live_explanation, live_plot],
        )
```

- [ ] **Step 3: Test it**

```bash
SYNCGUARD_LIVE=1 python scripts/demo.py
```

Click Live Challenge tab, record a 5-second clip, click Analyze. Expected: swapped video plays, model outputs "fake" with a plausible explanation.

**If webcam permission fails:** Chrome/Safari will prompt — grant it. If the venue laptop denies by policy, disable the tab tomorrow (`SYNCGUARD_LIVE=0`).

- [ ] **Step 4: Commit**

```bash
git add scripts/demo.py
git commit -m "demo: wire live challenge tab behind SYNCGUARD_LIVE flag"
```

---

## Phase 3 — Showcase-Day Checklist

### Task 3.1: Cold-restart dry run (tomorrow morning)

- [ ] **Step 1: Cold-restart Mac** (power off, power on)

- [ ] **Step 2: Launch demo**

```bash
cd "/Users/akshayprajapati/Desktop/CVPR Project/SyncGuard"
conda activate syncguard
python scripts/demo.py
# OR with live tab:
SYNCGUARD_LIVE=1 python scripts/demo.py
```

- [ ] **Step 3: Click through all 10 gallery clips**

Confirm each loads, analyzes, shows verdict + plot + explanation. No crashes.

- [ ] **Step 4: If live tab enabled, test webcam 2-3 times**

- [ ] **Step 5: Re-record fallback screencap**

---

### Task 3.2: USB backup

- [ ] **Step 1: Copy entire demo folder to USB stick**

```bash
cp -r "/Users/akshayprajapati/Desktop/CVPR Project/SyncGuard" /Volumes/<USB>/SyncGuard_backup/
```

- [ ] **Step 2: Also copy `fallback_screencap.mp4` separately**

---

### Task 3.3: Venue setup

- [ ] **Step 1: Arrive 30 min early.** Find the table, set up laptop, plug in power.

- [ ] **Step 2: Connect to venue Wi-Fi** (only needed for unrelated things — demo runs offline).

- [ ] **Step 3: Launch demo, verify one clip analyzes correctly.**

- [ ] **Step 4: Decide `SYNCGUARD_LIVE` state based on overnight confidence.** Toggle via:

```bash
SYNCGUARD_LIVE=1 python scripts/demo.py   # enable
python scripts/demo.py                     # disable (default)
```

- [ ] **Step 5: Have the screencap video open in a minimized QuickTime window as instant fallback.**

---

## Self-Review Notes

- **Spec coverage:** every acceptance criterion in the spec maps to a task. AC 1 → 1.6; AC 2 → 1.6; AC 3 → 1.6 mismatch check; AC 4 → 1.8; AC 5 → 1.8; AC 6 → 1.7/3.1; AC 7 → 2.3.
- **Type consistency:** `AnalysisResult` defined in Task 1.3 is the same dataclass used by Tasks 1.4, 1.6, 1.7, 2.3. `GalleryClip` defined in 1.5 is used by 1.6 and 1.7.
- **No placeholders:** all code blocks are complete and runnable. Threshold values are explicit placeholders that get replaced in Task 1.8 — this is legitimate because Task 1.8's computation produces the real values.
- **Open adaptation point:** Task 0.5 Step 1 asks the engineer to verify `process_video`'s signature against the actual preprocessing module — this is deliberate because the existing pipeline's exact function name and return contract weren't fully traced during spec writing. If the contract differs, the wrapper adaptation is local.
