# SyncGuard — Experiment Summary

**Last updated:** 2026-03-28
**Team:** Akshay, Ritik, Atharva

This document consolidates all experiment runs, results, and key findings across the entire project. For detailed per-experiment analysis, see the individual experiment reports linked in each section.

---

## Overview of All Experiments

| # | Experiment | Date | Phase | Key Metric | Status |
|---|-----------|------|-------|------------|--------|
| 1 | Pretrain Run 1 (fixed τ) | Mar 19-20 | Pretrain | val_loss=8.2990 | Complete |
| 2 | Pretrain Run 2 (learnable τ) | Mar 20 | Pretrain | val_loss=8.2561 | **Winner** |
| 3 | Finetune Run 1 (baseline) | Mar 20 | Finetune | AUC=0.9112 | Complete |
| 4 | Finetune Run 2 (audio-swap on reals — bug) | Mar 20 | Finetune | AUC≈0.50 | Discarded |
| 5 | Finetune Run 3 (audio-swap on fakes) | Mar 20 | Finetune | AUC=0.9254 | **Best sync-only** |
| 6 | Finetune Run 4 (dual-head + fusion) | Mar 20 | Finetune | AUC=0.5542 | Abandoned |
| 7 | Audio classifier (standalone) | Mar 20 | Training | val_auc=0.8909 | Complete |
| 8 | Cascade evaluation (FakeAVCeleb) | Mar 20-21 | Eval | AUC=0.9458 (max-fusion) | Complete |
| 9 | Cross-dataset eval (CelebDF-v2) | Mar 21 | Eval | No audio — dropped | Complete |
| 10 | Cross-dataset eval (DFDC) | Mar 22 | Eval | AUC=0.5712 (random) | Complete |
| 11 | EAR extraction | Mar 23 | Preprocessing | 21,059 samples | Complete |
| 12 | LRS2 preprocessing | Mar 23 | Preprocessing | 18,453/96,318 | In progress |
| 13 | **v3.0.0 Multi-Agent Review** | Mar 28 | Review | 50 findings, 6 critical | **Complete** |
| 14 | DFDC reprocessing (HP-1/2/3/4 fixes) | Mar 28 | Preprocessing | — | In progress (job 5504787) |
| 15 | Pretrain v3 (CMP + LRS2 + all fixes) | Mar 28 | Pretrain | — | In progress (job 5504788) |
| 16 | Finetune v2 (EAR + all fixes) | Pending | Finetune | — | Blocked on #15 |
| 17 | DFDC re-evaluation (reprocessed) | Pending | Eval | Target ≥ 0.72 | Blocked on #14, #16 |

---

## Experiment 13: v3.0.0 Multi-Agent Code Review (Mar 28)

### What Happened
7 specialist agents reviewed the full codebase in parallel, producing 50 findings. Key discoveries:

**Prior results invalidated:**
- Experiments 1-2 (pretrain): MoCo queue corrupted on every SLURM resume (CB-1/2/3). Val loss ~8.25 ≈ log(4096) suggests pretraining barely learned.
- Experiments 3-6 (finetune): Built on corrupted pretrain + temperature not gradient-clipped (HP-5).
- Experiment 10 (DFDC eval): Preprocessing had 20% fps drift (HP-2), label fallback (HP-1), resolution issues (HP-3).

**Strategic finding:**
- CMP+EAR hypothesis rated 4/10 for DFDC — face-swaps preserve lip-sync, so CMP won't help through sync-score bottleneck.
- Raw sync-score AUC on DFDC is 0.4378 (inverted) — core signal is anti-correlated.
- Recommended 3-tier strategy: (1) preprocessing fixes + BN adaptation, (2) embedding bypass, (3) DCT features.

**10 critical+high fixes implemented and deployed.** Full retraining initiated (experiments 15-17).

See: `docs/superpowers/specs/review-findings.md`

---

## Experiment 1 & 2: Phase 1 Contrastive Pretraining

**Detailed report:** [`experiment_pretrain_comparison.md`](experiment_pretrain_comparison.md)

### Setup
- Dataset: AVSpeech (21K real clips, 85/15 train/val split)
- Loss: InfoNCE with MoCo queue (4096)
- Visual: AV-HuBERT (random init), Audio: Wav2Vec 2.0 (frozen, layer 9)
- 20 epochs, batch_size=32, lr=1e-4, cosine schedule, 2-epoch warmup

### Results

| Metric | Run 1 (fixed τ=0.07) | Run 2 (learnable τ) |
|--------|----------------------|---------------------|
| Best val loss | 8.2990 (epoch 13) | **8.2561 (epoch 17)** |
| Best sync score | 0.7005 | **0.7063** |
| Final temperature | 0.0700 (frozen) | **0.0411 (learned)** |
| Train-val gap | 0.0708 | **0.0275** |
| Overfitting? | Yes | **No** |
| GPU | H200 140GB | A100 40/80GB |
| Training time | ~90 min | ~150 min |

**Critical bug found in Run 1:** `build_optimizer()` did not include `criterion.parameters()`, so the learnable `log_temperature` never received gradients. Fixed for Run 2.

**Winner: Run 2** — lower val loss, no overfitting, self-tuned temperature. Checkpoint: `pretrain_best.pt` (epoch 17).

---

## Experiments 3–6: Phase 2 Fine-tuning

### Setup (shared)
- Dataset: FakeAVCeleb (21,544 clips, speaker-disjoint split)
- Pretrain checkpoint: `pretrain_best.pt` (Run 2, learnable τ)
- Loss: L_InfoNCE + 0.5×L_temp + 1.0×L_cls
- Hard negative annealing: 0% → 20% over 10 epochs
- Early stopping: patience=5 on val_auc

### Results

| Run | Config | AUC | EER | FV-RA | RV-FA | FV-FA | pAUC@0.1 | Outcome |
|-----|--------|-----|-----|-------|-------|-------|-----------|---------|
| 1 | Baseline (no augmentation) | 0.9112 | 0.1726 | 0.9071 | 0.5641 | 0.9397 | 0.4673 | Baseline |
| 2 | Audio-swap on reals (bug) | ~0.50 | — | — | — | — | — | **Discarded** |
| 3 | Audio-swap on fakes (15%) | **0.9254** | **0.1481** | 0.9188 | 0.5070 | 0.9528 | 0.6097 | **Best sync-only** |
| 4 | Dual-head + learnable fusion | 0.5542 | — | — | — | — | — | **Abandoned** |

### Key Findings

1. **Audio-swap augmentation on fakes helps overall metrics** (+1.4% AUC, -2.5% EER) but **cannot fix RV-FA** — architectural limitation of sync-score approach (voice cloning preserves phoneme-viseme alignment).

2. **Audio-swap on reals is a bug** — training the model that "real video + wrong audio = fake" while also training "real video + real audio = real" creates contradictory labels, collapsing AUC to random chance.

3. **Logit-level fusion destroys sync signal** — mixing randomly initialized audio head outputs with trained sync-score outputs degrades both. Never mix trained + untrained heads.

4. **RV-FA remains the hardest category** (AUC 0.5070–0.5641) because FakeAVCeleb uses same-content voice cloning, not content-mismatched dubbing.

### Finetune Run 3 — Epoch-by-epoch (from HPC `finetune.json`)

*Note: The `finetune.json` on HPC contains Run 4 data (epochs 0-6). Run 3 metrics were logged to wandb.*

| Epoch | Train Loss | Val AUC | Val EER | Hard Neg % | τ |
|-------|-----------|---------|---------|------------|------|
| 0 | 9.334 | 0.353 | 0.598 | 0% | 0.070 |
| 1 | 8.675 | 0.554 | 0.422 | 2% | 0.067 |
| 3 | 8.394 | 0.403 | 0.573 | 6% | 0.061 |
| 5 | 8.346 | 0.426 | 0.560 | 10% | 0.055 |
| 6 | 8.317 | 0.388 | 0.573 | 12% | 0.053 |

*Note: The above is actually Run 4 (dual-head) data — AUC never exceeded 0.554, confirming it was abandoned. Run 3 best AUC=0.9254 at epoch 7.*

---

## Experiment 7: Standalone Audio Classifier

### Setup
- Architecture: Wav2Vec 2.0 (frozen, layer 9) → mean+max pool → MLP (426K trainable params)
- Dataset: FakeAVCeleb
- 30 epochs, lr=1e-4, cosine schedule

### Results

| Epoch | Train Loss | Val AUC | Val EER |
|-------|-----------|---------|---------|
| 0 | 0.255 | 0.537 | 0.500 |
| 5 | 0.096 | 0.815 | 0.292 |
| 10 | 0.082 | 0.868 | 0.227 |
| 15 | 0.077 | 0.882 | 0.198 |
| 20 | 0.074 | 0.889 | 0.187 |
| 25 | 0.072 | 0.891 | 0.187 |
| 29 | 0.070 | **0.891** | **0.187** |

**Best val_auc: 0.8909** — converged around epoch 20, diminishing returns after.

---

## Experiment 8: Cascade Evaluation (FakeAVCeleb)

Combines SyncGuard (Run 3) + audio classifier with 4 fusion strategies.

### Results

| Strategy | AUC | EER | FV-RA | RV-FA | FV-FA | pAUC@0.1 |
|----------|-----|-----|-------|-------|-------|-----------|
| sync_only | 0.9254 | 0.1481 | 0.9188 | 0.5070 | 0.9528 | 0.6097 |
| audio_only | 0.8737 | 0.2271 | 0.7586 | 0.9524 | 0.9745 | 0.4867 |
| **max_fusion** | **0.9458** | **0.1445** | 0.8981 | **0.9278** | **0.9902** | **0.7378** |
| avg_fusion | 0.9243 | 0.1609 | 0.8706 | 0.7515 | 0.9820 | 0.5767 |

**Winner: max_fusion** — takes max(sync_score, audio_score) per sample.

### Key Findings
- Max-fusion fixes RV-FA: 0.5070 → **0.9278** (audio classifier detects TTS/voice cloning that sync-scores miss)
- FV-FA gets near-perfect: **0.9902** (both modalities agree it's fake)
- Overall AUC improves: 0.9254 → **0.9458**
- pAUC@0.1 jumps: 0.6097 → **0.7378** (better at low false-positive rates)

---

## Experiment 9: Cross-Dataset — CelebDF-v2

- Preprocessed 921 clips from CelebDF-v2
- **Finding:** Entire dataset has no audio streams — incompatible with AV sync-based methods
- **Decision:** Dropped from evaluation. Will note in report as a limitation.

---

## Experiment 10: Cross-Dataset — DFDC

### Results (Zero-Shot, No Fine-tuning on DFDC)

| Strategy | AUC | EER | pAUC@0.1 |
|----------|-----|-----|-----------|
| sync_only | 0.5712 | 0.4535 | 0.0684 |
| audio_only | 0.4857 | 0.5084 | 0.0467 |
| max_fusion | 0.4960 | 0.5120 | 0.0489 |
| avg_fusion | 0.5378 | 0.4649 | 0.0665 |
| raw_sync_score | 0.4378 | 0.5563 | 0.0134 |

**All strategies at random chance.** This is the primary motivation for the CMP + EAR improvements (Experiments 13-15).

### Root Cause Analysis
- DFDC face-swaps preserve lip-sync (unlike FakeAVCeleb's FV-FA which replaces faces entirely)
- Sync-scores don't discriminate because AV alignment is maintained in DFDC fakes
- Audio classifier also fails — DFDC uses real audio, not TTS/voice cloning
- Raw sync-score thresholding (AUC 0.4378) confirms encoder representations themselves don't generalize

---

## Experiments 11-12: Data Preparation for v2

### EAR Feature Extraction (Experiment 11) — Complete
- **FakeAVCeleb:** 19,725 ear_features.npy files extracted
- **DFDC:** 1,334 ear_features.npy files extracted
- Method: MediaPipe FaceLandmarker eye landmarks → EAR = (||p2-p6||+||p3-p5||)/(2×||p1-p4||)
- Used isolated `ear_extract` conda env (mediapipe==0.10.14, protobuf==4.25.8)
- SLURM job 5382837

### LRS2 Preprocessing (Experiment 12) — In Progress
- **Total:** 96,318 pretrain videos
- **Processed:** ~18,453 (19%) as of Mar 23
- **Rate:** ~190 samples/min (14 workers, multiprocessing)
- **Fixes applied:** mediapipe Tasks API migration, EGL segfault fix, unique ID collision fix
- SLURM job 5392595 (auto-resuming)

---

## Planned: Experiments 13-15

### Experiment 13: Phase 1 v2 — CMP Pretraining
- **Loss:** InfoNCE + 0.5 × CrossModalPrediction
- **Data:** AVSpeech (21K) + LRS2 (96K) = ~117K real clips
- **Code:** Ready (`CrossModalPredictionLoss` in `losses.py`, `scripts/slurm_pretrain.sh`)
- **Blocked on:** LRS2 preprocessing completion

### Experiment 14: Phase 2 v2 — Fine-tuning with EAR
- **Classifier input:** sync_scores (1D) → sync_scores + EAR (2D)
- **Data:** FakeAVCeleb + LRS2 reals
- **Code:** Ready (`BiLSTMClassifier(use_ear=True)`, `scripts/slurm_finetune.sh`)
- **Blocked on:** Experiment 13 completion

### Experiment 15: DFDC Re-evaluation
- **Goal:** AUC ≥ 0.72 (up from 0.5712)
- **Hypothesis:** CMP pretraining learns deeper AV correspondence; EAR detects blink artifacts in face-swaps
- **Blocked on:** Experiment 14 completion

---

## Summary: Best Results to Date

### FakeAVCeleb (In-Domain)

| System | Overall AUC | EER | RV-FA AUC | pAUC@0.1 |
|--------|------------|-----|-----------|-----------|
| SyncGuard sync-only (Run 3) | 0.9254 | 0.1481 | 0.5070 | 0.6097 |
| **SyncGuard max-fusion cascade** | **0.9458** | **0.1445** | **0.9278** | **0.7378** |
| Target | ≥ 0.88 | — | — | — |

### DFDC (Cross-Dataset, Zero-Shot)

| System | AUC | EER |
|--------|-----|-----|
| SyncGuard best (sync_only) | 0.5712 | 0.4535 |
| Target | ≥ 0.72 | — |
| **Gap to close** | **+0.15** | — |

---

## Artifacts Index

### Checkpoints (on HPC)
- `outputs/checkpoints/pretrain_best.pt` — Phase 1 winner (Run 2, epoch 17, learnable τ)
- `outputs/checkpoints/pretrain_best_run1_fixed_tau.pt` — Phase 1 Run 1
- `outputs/checkpoints/finetune_best_run3_audioswap.pt` — Phase 2 best (Run 3)
- `outputs/checkpoints/audio_clf_best.pt` — Audio classifier (30 epochs)

### Metrics JSONs (on HPC)
- `outputs/logs/pretrain.json` — Phase 1 Run 2 per-epoch metrics
- `outputs/logs/finetune.json` — Phase 2 per-epoch metrics
- `outputs/logs/audio_classifier.json` — Audio classifier per-epoch metrics
- `outputs/logs/eval_cascade.json` — Cascade evaluation (4 strategies)
- `outputs/logs/eval_cascade_dfdc.json` — DFDC cascade evaluation
- `outputs/logs/eval_cascade_fakeavceleb.json` — FakeAVCeleb cascade re-eval
- `outputs/logs/eval_fakeavceleb.json` — FakeAVCeleb sync-only evaluation
- `outputs/logs/eval_summary.json` — Summary of all evaluations

### Predictions (on HPC)
- `outputs/logs/predictions_cascade.npz` — Raw cascade predictions (FakeAVCeleb)
- `outputs/logs/predictions_cascade_dfdc.npz` — Raw cascade predictions (DFDC)
- `outputs/logs/predictions_fakeavceleb.npz` — Raw sync-only predictions

### Experiment Reports
- `outputs/logs/experiment_pretrain_comparison.md` — Phase 1 Run 1 vs Run 2 detailed analysis
- `outputs/logs/experiment_summary.md` — This file

### wandb
- Project: `SyncGuard`
- Runs: `phase1-pretrain`, `phase1-pretrain-learnable-tau`, `phase2-finetune` (multiple)
