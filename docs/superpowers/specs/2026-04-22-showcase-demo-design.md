# Research Showcase Demo — Live Gradio App

**Date:** 2026-04-22
**Author:** Akshay Prajapati
**Status:** Approved
**Deadline:** Showcase 2026-04-23 (tomorrow)

## Problem

SyncGuard has strong quantitative results (0.9628 AUC on FakeAVCeleb) but no interactive demo. A research showcase table tomorrow requires a practical demo that visitors can interact with self-serve (no presenter continuously driving it). Constraints: 24-hour runway, HPC queue times unreliable for live inference, reliability at the booth is critical.

## Goals

1. **Self-serve table demo** — visitors click, see results in under 10 seconds, form their own impression.
2. **Explainability** — not just "real/fake" but a natural-language explanation of *why* (sync dips, blink anomalies, fake-type classification).
3. **Reliability** — zero dependency on venue Wi-Fi, HPC, or cloud services during the demo. Runs entirely on the booth laptop.
4. **Honest science** — gallery shows model strengths (in-domain FakeAVCeleb); DFDC failure examples available on request for judges who ask.

## Non-Goals

- Training new models for the demo.
- Real-time streaming inference (click-based, not always-on).
- Mobile or web-hosted demo (desktop Mac at the booth only).
- Supporting arbitrary user-uploaded videos as the primary path (gallery is primary; live challenge is stretch).

## Approach

Gradio web app (`scripts/demo.py`) running locally on the booth Mac. Model loads once at startup. Two tabs, the second feature-flagged:

- **Tab 1 — Gallery (primary):** 10 curated FakeAVCeleb test-split clips with pre-cached preprocessing. Click thumbnail → *Analyze* → verdict + sync-score plot + explanation + ground-truth reveal. Latency 3-5s.
- **Tab 2 — Live Challenge (stretch, `SYNCGUARD_LIVE=1`):** webcam + mic recording; system swaps recorded audio with a pre-recorded sentence from a pool to manufacture a sync mismatch; model detects the desync in ~10-15s.

Inference runs on Mac CPU/MPS. The v4+CA model (Stage 1 sync + Stage 2 cross-attention head) is copied from HPC scratch tonight. The cascade audio classifier is **not** used — demo sticks to the primary v4+CA path for simplicity and determinism. No HPC dependency after setup.

## Architecture

### Directory layout

```
SyncGuard/
├── scripts/
│   ├── demo.py                          # Gradio app entry point (~200 LOC)
│   ├── demo_smoke_test.py               # Manual end-to-end gate
│   ├── curate_gallery.py                # Picks 10 clips from eval output
│   └── prepare_gallery.py               # Precomputes .npz caches per clip (HPC)
├── src/
│   └── demo/
│       ├── __init__.py
│       ├── inference.py                 # DemoInference class
│       ├── explain.py                   # Rule-based explanation (pure function)
│       ├── gallery.py                   # Manifest loader + GalleryClip dataclass
│       └── live_challenge.py            # Webcam + audio swap (stretch)
├── tests/
│   └── demo/
│       └── test_explain.py              # Unit tests for explain.py
└── demo_assets/                         # Gitignored
    ├── gallery/
    │   ├── manifest.json                # {clip_id, path, category, ground_truth, hpc_confidence}
    │   ├── real_01.mp4, real_01.npz
    │   ├── fv_ra_01.mp4, fv_ra_01.npz
    │   └── ...
    ├── live_audio_pool/                 # Pre-recorded sentences for audio swap
    │   └── *.wav
    └── checkpoints/
        ├── finetune_best.pt             # Stage 1 sync
        ├── cross_attention_best.pt      # Stage 2 fusion head
        └── audio_clf_best.pt            # Optional cascade audio classifier
```

### Component responsibilities

| Module | Purpose | Dependencies |
|--------|---------|--------------|
| `scripts/demo.py` | Gradio Blocks app, tab layout, event handlers | `src.demo.*` |
| `src/demo/inference.py` | `DemoInference` class. Loads model once. `analyze(mouth_crops, audio, ear) → AnalysisResult` | `src/models/syncguard.py`, `src/evaluation/` |
| `src/demo/explain.py` | Pure `generate_explanation(result) → str`. No I/O, no model. Fully unit-testable. | stdlib only |
| `src/demo/gallery.py` | Loads `manifest.json`, yields `GalleryClip` with pre-cached tensors | `src/demo/inference.py` (types) |
| `src/demo/live_challenge.py` | Webcam handling, audio swap from pool, calls preprocessing pipeline for uncached input. Imported only if `SYNCGUARD_LIVE=1`. | `src/preprocessing/*` |
| `demo_assets/gallery/*.npz` | Pre-cached `mouth_crops`, `audio_waveform`, `ear_features` per clip | — |

## Data Flow

### Gallery (primary path)

```
click thumbnail
  → gallery.load_cached(clip_id)          # reads .npz, no preprocessing
  → DemoInference.analyze(...)
      ├─ AV-HuBERT → v_embeds              (~1.5s M-series)
      ├─ Wav2Vec 2.0 → a_embeds            (~1.0s)
      ├─ s(t) = cos(v_t, a_t)              (<0.01s)
      ├─ Stage 2 cross-attention → logit   (~0.3s)
      └─ Returns AnalysisResult
  → explain.generate_explanation(result)   # 1-2 sentences
  → visualize.render_sync_plot(...)        # PIL image
  → Gradio updates: verdict | plot | explanation | ground-truth reveal
```

Target latency: **3-5s p95.**

### Live challenge (stretch)

```
webcam+mic record (5s) via gr.Video(source='webcam')
  → live_challenge.swap_audio(video_blob)
      (extracts audio, replaces with random clip from live_audio_pool/)
  → preprocessing.pipeline.process(swapped.mp4)
      ├─ RetinaFace face detection          (~2s)
      ├─ MediaPipe landmarks → mouth ROI    (~1.5s)
      ├─ ffmpeg audio extraction            (~0.5s)
      └─ Silero VAD mask                    (~0.5s)
  → DemoInference.analyze(...)              # same as above
  → explain.generate_explanation(...)
  → Gradio updates side-by-side: original vs swapped + verdict + plot + explanation
```

Target latency: **10-15s.**

## Explainability (Level 2)

### AnalysisResult dataclass

```python
@dataclass
class AnalysisResult:
    verdict: Literal["real", "fake"]
    confidence: float                      # [0, 1]
    sync_curve: np.ndarray                 # (T,) per-frame sync scores
    ear_curve: np.ndarray                  # (T,) per-frame EAR values
    sync_dip_segments: list[tuple[float, float]]   # (start_s, end_s) where s < threshold
    ear_anomaly_score: float               # multiple-of-baseline variance
    clip_duration_s: float
    mean_sync: float
    timings: dict[str, float]              # {preprocess, visual, audio, fuse} seconds
```

### Thresholds

| Name | Default | Source of truth |
|------|---------|-----------------|
| `SYNC_THRESHOLD` | 0.55 | Threshold maximizing Youden's J (TPR − FPR) on FakeAVCeleb val ROC. Computed tonight from `predictions_cascade_fakeavceleb.npz` using `raw_sync` (negated cosine sim). |
| `CONF_HIGH` | 0.75 | Hard-coded, empirical |
| `CONF_LOW` | 0.55 | Hard-coded, empirical |
| `EAR_ANOMALY_THRESHOLD` | 2.0× baseline | Tuned against gallery tonight |
| `MIN_DIP_DURATION_S` | 0.15 | Phoneme-scale floor; filters VAD-boundary noise |

### Fake-type decision rules

```
num_dips = count(sync_dip_segments where duration >= MIN_DIP_DURATION_S)

if num_dips >= 2 and ear_anomaly > EAR_ANOMALY_THRESHOLD:
    fake_type = "face-swap with audio-visual mismatch"  # FV-FA
elif num_dips >= 1 and ear_anomaly <= EAR_ANOMALY_THRESHOLD:
    fake_type = "audio-visual desynchronization"         # RV-FA
elif num_dips == 0 and ear_anomaly > EAR_ANOMALY_THRESHOLD:
    fake_type = "face manipulation with preserved lip-sync"  # FV-RA
else:
    fake_type = "subtle manipulation (learned representation)"
```

### Template strings (hybrid tone — metric + plain summary)

**Real, confident:**
> "Real clip — {conf}% confidence. Audio-visual alignment stable throughout (mean sync {mean_sync:.2f}, threshold {SYNC_THRESHOLD:.2f}). No blink anomalies detected."

**Real, borderline:**
> "Likely real — {conf}% confidence (borderline). Sync mostly stable with {num_dips} brief dip(s). May be natural speech pauses; recommend human review."

**Fake, FV-FA pattern:**
> "Fake detected — {conf}% confidence. Lip-sync drops below {SYNC_THRESHOLD:.2f} in {num_dips} segments ({total_dip_s:.1f}s total), and blink pattern shows {ear_anomaly:.1f}× baseline variance. Consistent with face-swap combined with audio manipulation."

**Fake, RV-FA pattern:**
> "Fake detected — {conf}% confidence. Audio-visual sync drops below threshold in {num_dips} segments totaling {total_dip_s:.1f}s (most prominent at {peak_time:.1f}s). Face appears natural — pattern consistent with audio dubbing or lip-sync generation."

**Fake, FV-RA pattern:**
> "Fake detected — {conf}% confidence. Sync scores remain stable (mean {mean_sync:.2f}), but blink pattern anomaly detected (EAR variance {ear_anomaly:.1f}× baseline). Consistent with face-swap that preserved original audio alignment."

**Fake, subtle:**
> "Fake detected — {conf}% confidence. No strong sync or blink anomalies, but learned representation indicates manipulation. May be a well-executed fake or an out-of-distribution sample."

### Edge cases

- **Silent regions (VAD mask = 0):** excluded from `sync_dip_segments`. Sync during silence is noise.
- **Clip shorter than 2.0s:** prepend "Limited temporal evidence — " to any explanation.
- **Verdict on threshold (|conf - 0.5| < 0.05):** use borderline template regardless of side.
- **Missing EAR features:** fall back to sync-only explanation.

## Gallery Curation

### Composition (10 clips)

| Count | Category | Ground truth | Primary explanation signal |
|-------|----------|--------------|----------------------------|
| 2 | RV-RA (real video, real audio) | Real | Stable sync + normal EAR |
| 3 | FV-RA (face-swap, real audio) | Fake | EAR anomaly (lip-sync preserved) |
| 3 | RV-FA (real video, fake audio) | Fake | Sync dips (most visually compelling) |
| 2 | FV-FA (both swapped) | Fake | Sync dips + EAR anomaly |

All clips drawn from **FakeAVCeleb test split** (speakers unseen during training). Weighting favors FV-RA and RV-FA because they showcase distinct explanation templates (sync-only vs EAR-only signals).

### Selection criteria

For each category, select from `outputs/logs/predictions_cascade_fakeavceleb.npz` (after `sample_ids` patch):
- **Primary pick:** top-confidence correct prediction (model's strongest case in category).
- **Secondary pick(s):** mid-confidence correct predictions for texture.

Avoid cherry-picking only ≥99% confident clips — realistic confidence distribution makes the demo feel genuine, not scripted.

### UX pattern

- Thumbnails show first frame + anonymous ID ("Clip 01"). Ground-truth hidden before *Analyze*.
- After analysis: verdict + explanation + ground-truth reveal (✓ correct / ✗ incorrect).
- Suspense-before-reveal increases visitor engagement.

### Ethical guardrails

- All subjects are FakeAVCeleb research participants (consented).
- No new deepfakes of public figures created for the demo.
- No content that could plausibly be mistaken for real disinformation.

### Failure case handling

**Not in gallery.** Two to three DFDC cross-dataset clips (where model fails, AUC 0.53) stored on disk and pulled up manually if judges ask about generalization. Conversation prop, not demo default.

## HPC → Mac Handoff (Tonight)

One-time checkpoint and asset pull. After this, no HPC dependency.

### Patch + re-run eval

`scripts/evaluate_cascade.py` currently saves sync/audio/max/labels to `.npz` but not sample IDs. Three-line patch to include `sample_ids` in `save_dict`, then re-run on a single H200:

```python
save_dict = {
    "sample_ids": sample_ids,     # NEW
    "sync_scores": sync_scores,
    ...
}
```

### Asset pull list

| Item | Approx size | Destination |
|------|-------------|-------------|
| `finetune_best.pt` | ~400 MB | `demo_assets/checkpoints/` |
| `cross_attention_best.pt` | ~50 MB | `demo_assets/checkpoints/` |
| ~~`audio_clf_best.pt`~~ | — | Not pulled — cascade audio classifier is out of scope for demo |
| 10 gallery `.mp4` files | ~20-50 MB total | `demo_assets/gallery/` |
| 10 gallery `.npz` caches | ~5-15 MB total | `demo_assets/gallery/` |
| `manifest.json` | <10 KB | `demo_assets/gallery/` |
| 2-3 DFDC conversation clips | ~10 MB | `demo_assets/dfdc_extras/` |
| AV-HuBERT backbone weights | ~400 MB | `$HF_HOME/hub/` on Mac (set `HF_HOME=~/.cache/huggingface`); pre-download tonight to avoid first-load network dependency |

Total pull: ~1 GB. Single `scp` session.

## Testing

### Unit tests — `tests/demo/test_explain.py`

Pure-function tests, no GPU, runs in <1s:
- One test per fake-type branch (5 cases)
- One test per confidence tier (3 cases)
- Edge cases: short clip, silent regions, missing EAR (3 cases)

### Smoke test — `scripts/demo_smoke_test.py`

Manual end-to-end gate. Loops through `manifest.json`, prints per-clip `(verdict, confidence, mean_sync, ear_anomaly, expected, match)`. Exits non-zero if any clip doesn't match its expected verdict. Run before opening Gradio.

### Dress rehearsal

Cold-restart Mac, `python scripts/demo.py`, click through all 10 clips manually. Record explanations and visuals as a baseline.

### What we don't test

No automated Gradio UI tests. UI bugs are caught by the dress rehearsal; the cost of building Playwright tests exceeds the 24-hour budget.

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MPS numerical drift flips borderline verdicts | Medium | Pick clips with ≥80% HPC confidence; fall back to CPU if flips observed |
| fairseq / AV-HuBERT version mismatch on Mac | Medium-high | Pin `requirements.txt`; pre-download AV-HuBERT weights tonight; verify imports before sleep |
| Model-load OOM on Mac | Low | Inference in bf16; force CPU if MPS OOMs |
| Gradio port conflict | Low | Pin 7860, fallback 7861 |
| App crash mid-demo | Low if tested | Cold-restart script + screencap fallback video |
| Webcam permission denied at venue | Medium | If denied, `SYNCGUARD_LIVE=0`, gallery-only |
| Live tab flaky tomorrow morning | Medium | Feature flag toggles without code change |
| Venue Wi-Fi down | N/A | Entire demo runs offline |

## Acceptance Criteria

| Gate | Criteria | Verification |
|------|----------|--------------|
| 1 | Checkpoint loads on Mac in < 30s without errors | Smoke test |
| 2 | All 10 gallery clips return complete AnalysisResult | Smoke test |
| 3 | Mac verdict matches HPC verdict for each clip | Compare to `.npz` |
| 4 | p95 gallery analyze latency < 5s | Timings dict over 10 runs |
| 5 | Each category's clips get the correct fake-type explanation | Manual review |
| 6 | Gradio survives 100 consecutive analyze clicks without crash | Stress script |
| 7 | `SYNCGUARD_LIVE` flag toggles Tab 2 visibility | Visual check |

## 24-Hour Timeline

### Tonight (hours 1-6)

1. Patch `evaluate_cascade.py` to save `sample_ids` — 5 min
2. HPC: re-run cascade eval — 20 min
3. `scp predictions_cascade_fakeavceleb.npz` → Mac — 2 min
4. Write `scripts/curate_gallery.py` (top-confidence + balanced + demographics) — 30 min
5. Run curator → gallery manifest + clip list — 5 min
6. HPC: run `prepare_gallery.py` to precompute `.npz` caches — 15 min
7. `scp` checkpoints + `demo_assets/gallery/` → Mac — 10 min
8. Write `src/demo/inference.py` — 1 hr
9. Write `src/demo/explain.py` + unit tests — 45 min
10. Run smoke test on Mac — 15 min
11. Write `scripts/demo.py` (Gallery tab) — 1.5 hr
12. Dress rehearsal, tune thresholds — 1 hr

**Gate at hour 6: Gallery is ship-ready. Record fallback screencap video now.**

### Stretch (hours 7-9, only if above is solid)

13. Write `src/demo/live_challenge.py` + Tab 2 — 2 hr
14. Webcam + audio-swap test — 30 min
15. Decide `SYNCGUARD_LIVE=1` or `=0`

### Tomorrow morning

1. Cold-restart Mac, launch demo — 2 min
2. Click through all 10 gallery clips — 5 min
3. Re-record screencap fallback — 5 min
4. Copy `demo_assets/` to USB backup — 5 min
5. Travel, set up, verify at venue — 15 min

## Showcase-Day Contingencies

- App crash → `Ctrl+C` + relaunch (15s recovery).
- Per-clip inference crash → remove clip from manifest, reload.
- Full model-load failure → fullscreen screencap video fallback.
- Venue Wi-Fi dead → irrelevant (offline demo).
- Gradio won't start → explain verbally + screencap.

## Open Questions

None at spec-approval time. Threshold values are placeholders to be tuned tonight against gallery outputs.

## Decisions Log

- **Gradio over FastAPI+HTML:** 24-hour budget, webcam support built-in, `gr.Examples` makes gallery trivial.
- **Gallery primary, live stretch:** reliability > flashiness for self-serve table format.
- **Pre-cached preprocessing:** cuts gallery latency from ~10s to ~3-5s by moving CV pipeline off the hot path.
- **Rule-based explanations over post-hoc attribution:** deterministic, testable, fast; model architecture is already interpretable via sync-score curve.
- **FakeAVCeleb test split only in gallery:** shows model strengths; DFDC failure clips kept as conversation props.
- **10 clips, not 8:** more variety for returning visitors.
- **Hybrid explanation tone:** technical metric + plain-language summary balances research and accessibility.
- **`SYNCGUARD_LIVE` env flag for Tab 2:** single-toggle ship/no-ship decision for stretch feature.
