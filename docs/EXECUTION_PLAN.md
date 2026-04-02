# SyncGuard — Execution Plan

**Created:** March 11, 2026
**Final Deadline:** April 13, 2026 (24 days remaining)
**Team:** Akshay (Visual Encoder, Preprocessing, Integration Lead) | Ritik (Audio Encoder, Contrastive Pretraining) | Atharva (Temporal Classifier, Evaluation)

---

## Current State (Updated Apr 2, 2026)

### ⚠ Prior Results Invalidated (v3.0.0 Review)
On Mar 28, a 7-agent code review found 50 issues including 6 critical blockers. Key impact:
- **Phase 1 pretrain results invalidated** — MoCo queue bugs corrupted contrastive learning on every SLURM resume (CB-1/2/3)
- **Phase 2 finetune results invalidated** — built on corrupted Phase 1 + missing grad clipping on temperature (HP-5)
- **DFDC 0.5712 AUC invalidated** — preprocessing had 20% fps drift, label fallback bug, resolution issues (HP-1/2/3)
- **FakeAVCeleb 0.9458 AUC likely reproducible** — BCE loss was correct, preprocessing unaffected
- All 10 critical+high fixes implemented and deployed. Full retraining initiated.
- See `docs/superpowers/specs/review-findings.md` for complete findings.

### Done
- Preprocessing pipeline (RetinaFace + MediaPipe mouth-ROI, audio extraction, Silero-VAD, temporal alignment)
- Dataset loaders: FakeAVCeleb, CelebDF-v2, AVSpeech
- Config system (`configs/default.yaml`)
- CLI tooling: `preprocess_dataset.py`, `download_avspeech.py`, `train_pretrain.py`, `train_finetune.py`, `gpu_smoke_test.py`
- Model architectures: visual encoder (AV-HuBERT), audio encoder (Wav2Vec 2.0), classifier (Bi-LSTM), SyncGuard integration
- Training: losses (InfoNCE, temporal, BCE, combined), dataset + collation, pretrain loop, finetune loop
- Evaluation: metrics (AUC-ROC, EER, pAUC), evaluate runner, visualization (7 plot types)
- HPC environment: conda env, all deps, Wav2Vec cached, GPU smoke test passed
- Full FakeAVCeleb uploaded & preprocessed on HPC: 21,544 clips (all 4 categories)
- AVSpeech uploaded & preprocessed on HPC: 24,760 clips
- **Phase 1 contrastive pretraining** — completed, 2 runs compared (fixed τ vs learnable τ). Winner: Run 2 (learnable τ), best val loss 8.2561, checkpoint: `pretrain_best.pt` (epoch 17)
- **Phase 2 fine-tuning** — 4 experiment runs completed:
  - Run 1 (no augmentation): AUC=0.9112, EER=0.1726
  - Run 2 (audio-swap on reals — bug): AUC collapsed, discarded
  - Run 3 (audio-swap on fakes, 15%): AUC=0.9254, EER=0.1481 — **best sync-only model**
  - Run 4 (dual-head + fusion): AUC=0.5542, abandoned — logit fusion destroyed sync signal
- **Standalone audio classifier** — trained (Wav2Vec2 frozen → MLP, 426K params), val_auc=0.8909
- **Cascade evaluation** — max-fusion system evaluated:
  - Overall AUC: **0.9458**, EER: **0.1445**, pAUC@0.1: **0.7378**
  - RV-FA AUC: **0.9278** (fixed from 0.5070 sync-only)
  - FV-FA AUC: **0.9902**, FV-RA AUC: **0.8981**
- wandb integration in both training loops
- SLURM scripts for all phases (preprocess, pretrain, finetune, audio clf, cascade eval)

### Completed (Cross-Dataset)
- **CelebDF-v2:** Preprocessed 921 clips — no audio streams in entire dataset, incompatible with AV methods. Dropped.
- **DFDC Part 0:** Preprocessed 1,334 clips — all cascade strategies at random chance (best AUC 0.5712). Face-swaps preserve lip-sync, so sync-scores don't discriminate.
- **Raw sync-score thresholding:** AUC 0.4378 on DFDC — encoder representations don't generalize, not just classifier.

### Completed (Post-Review Experiments — Mar 28 to Apr 2)
- **DFDC reprocessing:** 1,343/1,343 with corrected pipeline (fps fix, label fix, resolution normalization)
- **Pretrain v3** (unfrozen Wav2Vec, 121K clips, CMP): Saturated by epoch 3, sync=1.0. Not ideal.
- **Pretrain v4** (frozen Wav2Vec, 121K clips, CMP): InfoNCE=8.06, sync=0.978. Better representations.
- **Finetune v2** (v3 pretrain, frozen FT): AUC=0.910, DFDC=0.458
- **CA Stage 1+2** (on v2 finetune): FAV=0.927, **DFDC=0.526** — best cross-dataset
- **v4+CA finetune** (v4 pretrain, CA during FT): **FAV=0.961**, DFDC=0.468 — best in-domain
- **Cross-attention + DCT:** FAV=0.923, DFDC=0.512 — DCT didn't help DFDC

### Current Best Results
| Dataset | Model | AUC | EER |
|---------|-------|-----|-----|
| FakeAVCeleb | v4+CA fused | **0.961** | **0.082** |
| DFDC | CA Stage 1+2 on v2 | **0.526** | 0.491 |

### Still Pending
- BN adaptation on DFDC (Tier 1, never tried)
- Threshold recalibration (Tier 1, never tried)
- 3x seed variance runs (paper requirement)
- Bootstrap CIs (paper requirement)
- Ablation table compilation
- Paper writing (Days 7-14 remaining)

### Completed (HPC Deployment — Mar 23)
- **EAR extraction:** FakeAVCeleb (19,725) + DFDC (1,334) — all ear_features.npy saved ✓
- **LRS2 transfer:** ~50 GB, 144K videos transferred to HPC ✓
- **Code deployed to HPC:** mediapipe Tasks API migration, EGL fix, LRS2 unique ID fix, multiprocessing ✓
- **CMP + EAR code verified:** All training code (CrossModalPredictionLoss, BiLSTM use_ear, dataset loading) already implemented and ready ✓
- **SLURM training scripts ready:** `slurm_pretrain.sh` (Phase 1 CMP, H200), `slurm_finetune.sh` (Phase 2 EAR, H200) ✓

### HPC Deployment Queue
1. ~~Transfer LRS2 tar (60 GB) to HPC, extract, preprocess with EAR~~ — ✅ Transfer done, preprocessing in progress
2. ~~Extract EAR for existing FakeAVCeleb (21,544) + DFDC (1,334)~~ — ✅ Complete
3. ~~Push updated code to HPC~~ — ✅ Complete (mediapipe migration, EGL fix, unique ID fix, multiprocessing)
4. **Phase 1 CMP pretraining** on AVSpeech + LRS2 (20 epochs, H200) — blocked on LRS2 preprocessing
5. Phase 2 fine-tuning with EAR + LRS2 reals on FakeAVCeleb (30 epochs, H200)
6. Re-evaluate on DFDC with new model

### Revised Timeline (16 days: Mar 28 — Apr 13)

| Phase | Days | Tasks | Owner |
|-------|------|-------|-------|
| 0: Bug fixes | 1-2 (Mar 28-29) | All critical fixes ✅, DFDC reprocess, verify | All |
| 1: Pretraining v3 | 3-6 (Mar 30–Apr 2) | CMP pretrain on H200, DFDC diagnostics, BN adaptation | Ritik, Atharva |
| 2: Finetuning v2 | 7-8 (Apr 3-4) | Finetune + Tier 2 embedding bypass if needed | Ritik, Akshay |
| 3: Evaluation | 9-11 (Apr 5-7) | Full eval, DFDC Tier 2/3, ablations | All |
| 4: Paper | 12-16 (Apr 8-13) | Writing, figures, poster, submission | All |

**Gate checks:**
- Day 5 (Apr 2): Tier 1 DFDC AUC after preprocessing fixes + BN adaptation
- Day 9 (Apr 5): Full results table — go/no-go on Tier 3
- Day 11 (Apr 7): All results final, begin paper

### Not Started
- Ablation experiments (visual encoder, Wav2Vec layer, classifier)
- Wav2Lip adversarial set
- Visualizations / plots (10 required types)
- Poster, report, video demo

---

## Phase 3A — Define & Setup (Mar 11 – Mar 15)

> Goal: All data downloaded, models implemented, ready to train on Day 1 of development.

### Week 1, Day 1–2 (Mar 11–12): Data Procurement

| Task | Owner | Details |
|------|-------|---------|
| Download FakeAVCeleb | Akshay | Primary dataset. 19,500 clips. Use obtained access credentials. Place in `data/raw/FakeAVCeleb/` |
| Download CelebDF-v2 | Atharva | 590 real + 5,639 fake. Place in `data/raw/CelebDF-v2/` |
| Download VoxCeleb2 subset | Ritik | Download a practical subset (~500 hrs) via yt-dlp. If infeasible, fall back to FakeAVCeleb real subset (~4K clips) with augmentation |
| Download DFDC sample | Atharva | From Kaggle. At minimum the test partition (~5K clips) for zero-shot eval |

**Fallback plan for pretraining data:** If VoxCeleb2/LRS2 downloads stall, use FakeAVCeleb real subset + AVSpeech clips. Document which option was used.

### Week 1, Day 2–3 (Mar 12–13): Model Implementation

All three members implement their assigned components in parallel.

#### Akshay — Visual Encoder (`src/models/visual_encoder.py`)
```
- Load AV-HuBERT visual frontend (pretrained lip-reading weights from fairseq)
- Projection head: Linear → ReLU → Linear → L2-normalize to R^256
- Input: (B, T, 1, 96, 96) grayscale mouth crops
- Output: (B, T, 256) frame-level visual embeddings
- freeze_pretrained configurable from default.yaml
```

#### Ritik — Audio Encoder (`src/models/audio_encoder.py`)
```
- Load Wav2Vec 2.0 Base (facebook/wav2vec2-base-960h) via transformers
- Extract hidden states from configurable layer (default: 9)
- Projection head: Linear → ReLU → Linear → L2-normalize to R^256
- Input: (B, waveform_samples) raw audio at 16kHz
- Output: (B, T, 256) frame-level audio embeddings at 49Hz
- freeze_pretrained configurable (default: True for audio)
```

#### Atharva — Temporal Classifier (`src/models/classifier.py`)
```
- Sync-score computation: s(t) = cos(v_t, a_t), output (B, T, 1)
- Bi-LSTM: input_size=1, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True
- Pooling: concatenate mean-pool + max-pool over time → 512-dim → Linear(512, 256) → ReLU → Linear(256, 1) → Sigmoid
- Input: (B, T) sync-scores
- Output: (B, 1) real/fake probability
```

#### Joint — SyncGuard Model (`src/models/syncguard.py`)
```
- Combines visual_encoder + audio_encoder + classifier
- Forward pass returns: sync_scores (B, T), logits (B, 1), v_embeds (B, T, 256), a_embeds (B, T, 256)
- Akshay integrates after individual modules are done (Mar 13)
```

### Week 1, Day 3–4 (Mar 13–14): Loss Functions & Data Pipeline

#### Ritik — Loss Functions (`src/training/losses.py`)
```
- InfoNCE loss (frame-level):
    L = -(1/T) * sum_t log[ exp(cos(v_t, a_t)/τ) / sum_j exp(cos(v_t, a_j)/τ) ]
  With MoCo memory bank (queue_size=4096) for negatives
- Temporal consistency loss (real-only):
    L_temp = sum_t ||(v_{t+1} - v_t) - (a_{t+1} - a_t)||^2
- Classification loss: BCE
- Combined loss: L_total = L_InfoNCE + γ * L_temp + δ * L_cls
- Learnable temperature τ (init=0.07, clamp=[0.01, 0.5])
```

#### Akshay — Training Dataset (`src/training/dataset.py`)
```
- PyTorch Dataset that loads preprocessed samples (mouth_crops.npy, audio.wav, speech_mask.npy)
- Speaker-disjoint train/val/test splits for FakeAVCeleb
- Hard negative mining: sample same-speaker different-time windows
- Annealing: hard_negative_ratio ramps 0% → 20% over first 10 epochs
- Collation with padding + attention masks for variable-length sequences
```

#### Atharva — Evaluation Framework (`src/evaluation/metrics.py`)
```
- AUC-ROC (sklearn)
- EER computation (scipy interpolation)
- pAUC at FPR < 0.1
- Per-category AUC for FakeAVCeleb (RV-RA, FV-RA, RV-FA, FV-FA)
- Results logging to JSON + console table
```

### Week 1, Day 4–5 (Mar 14–15): Preprocessing Run & Integration

| Task | Owner | Details |
|------|-------|---------|
| Preprocess FakeAVCeleb | Akshay | Run `scripts/preprocess_dataset.py` on HPC. Output: `data/processed/FakeAVCeleb/` |
| Preprocess VoxCeleb2 (or fallback) | Ritik | Same pipeline on pretraining data |
| Preprocess CelebDF-v2 | Atharva | For zero-shot eval |
| Integration test | Akshay | End-to-end: load preprocessed data → forward pass through SyncGuard → loss computation. Verify shapes, no NaN, GPU memory fits |
| Smoke test on 10 samples | All | One batch forward + backward, confirm gradients flow through all components |

**Checkpoint: By end of Mar 15** — all models implemented, data preprocessed, one successful training step completed on HPC.

---

## Phase 3B — Development: Contrastive Pretraining (Mar 16 – Mar 21)

> Goal: Train encoders to produce meaningful audio-visual embeddings on real data.

### Training Loop (`src/training/pretrain.py`) — Ritik (lead), Akshay (review)

| Parameter | Value |
|-----------|-------|
| Dataset | VoxCeleb2 subset (or fallback) — real clips only |
| Loss | InfoNCE only (no classification, no temporal consistency) |
| Epochs | 20 |
| Batch size | 32 |
| LR | 1e-4, cosine schedule, 2-epoch warmup |
| Memory bank | MoCo queue, size 4096 |
| Temperature | Learnable, init 0.07 |
| Checkpointing | Save every 5 epochs + best val loss |

**Daily monitoring (Ritik):**
- InfoNCE loss should decrease steadily
- Temperature τ should stabilize around 0.05–0.10
- Spot-check sync-scores: real clips should show s(t) > 0.5 on average after ~10 epochs

**Parallelized work during pretraining:**

| Task | Owner | Dates |
|------|-------|-------|
| Implement evaluation runner (`src/evaluation/evaluate.py`) | Atharva | Mar 16–18 |
| Implement sync-score visualization (`src/evaluation/visualize.py`) | Atharva | Mar 18–19 |
| Implement fine-tuning loop (`src/training/finetune.py`) | Akshay | Mar 16–18 |
| Preprocess DFDC test set | Atharva | Mar 16–17 |
| Generate Wav2Lip adversarial set (~500 clips) | Ritik (if GPU is free) or Akshay | Mar 19–21 |

### Evaluation Runner (`src/evaluation/evaluate.py`) — Atharva
```
- Load checkpoint + test dataset
- Run inference, collect predictions + labels
- Compute all metrics (AUC-ROC, EER, pAUC)
- Per-category breakdown for FakeAVCeleb
- Save results JSON to outputs/logs/
- Print summary table
```

### Visualization (`src/evaluation/visualize.py`) — Atharva
```
- Plot s(t) curves for individual clips (real vs fake side-by-side)
- Overlay phoneme boundaries (if MFA timestamps available)
- Aggregate sync-score distribution histograms (real vs fake)
- Save to outputs/visualizations/
```

### Fine-tuning Loop (`src/training/finetune.py`) — Akshay
```
- Load pretrained encoder checkpoint
- Combined loss: L_InfoNCE + 0.5 * L_temp + 1.0 * L_cls
- Hard negative mining with annealing
- Validation AUC-ROC as checkpoint criterion
- Early stopping (patience=5 epochs)
```

**Checkpoint: By end of Mar 21** — Phase 1 pretraining complete, fine-tuning loop ready, evaluation framework operational.

---

## Phase 3C — Development: Fine-tuning & Evaluation (Mar 22 – Mar 30)

> Goal: Trained SyncGuard model meeting target metrics, ablations complete.

### Fine-tuning (Mar 22 – Mar 26) — Akshay (lead), Ritik (support)

| Parameter | Value |
|-----------|-------|
| Dataset | FakeAVCeleb (speaker-disjoint split) |
| Loss | L_InfoNCE + 0.5 * L_temp + 1.0 * L_cls |
| Epochs | 30 |
| Batch size | 16 |
| LR | 5e-5, cosine schedule, 3-epoch warmup |
| Hard negatives | 0% → 20% annealed over first 10 epochs |
| Checkpointing | Save every 5 epochs + best val AUC-ROC |

**Daily monitoring (Akshay):**
- Track: train loss, val loss, val AUC-ROC, val EER
- AUC-ROC should climb above 0.80 by epoch 10, target 0.88+ by convergence
- Watch for OOM — reduce batch size to 8 if needed

### Evaluation Runs (Mar 26 – Mar 28) — Atharva (lead)

Run best checkpoint through all 4 test axes:

| Test | Dataset | Target | Notes |
|------|---------|--------|-------|
| In-domain | FakeAVCeleb test split | AUC >= 0.88 | Per-category breakdown |
| Cross-generator | CelebDF-v2 | Report AUC | Zero-shot, no training data from CelebDF |
| In-the-wild | DFDC test | AUC >= 0.72 | Zero-shot, stratify by compression |
| Adversarial | Wav2Lip-generated | Report AUC | Hardest case — sync-optimized fakes |

### Ablation Studies (Mar 26 – Mar 30) — Split across team

Run ablations in parallel on HPC (each is an independent training run):

| Ablation | Variants | Owner | Priority |
|----------|----------|-------|----------|
| Visual encoder | AV-HuBERT vs ResNet-18 vs SyncNet | Akshay | High |
| Wav2Vec layer | Layers 3, 5, 7, 9, 11 | Ritik | High |
| Classifier | Statistical baseline vs 1D-CNN vs Bi-LSTM | Atharva | Medium |
| Hard negatives | 0% vs 20% | Ritik | Medium |

**For ablations:**
- Each variant: fine-tune for 15 epochs (reduced) on FakeAVCeleb, evaluate on test split
- Log: AUC-ROC, EER for each variant
- Create comparison table + bar chart

### Visualizations (Mar 28 – Mar 30) — Atharva

- s(t) curves: 3 real + 3 fake clips from each test set (12 plots total)
- Sync-score distribution histograms (real vs fake)
- Ablation comparison bar charts
- ROC curves for each test axis
- Confusion matrices at EER threshold

**Checkpoint: By end of Mar 30** — all experiments complete, results tables filled, visualizations generated.

---

## Phase 4 — Poster & Phase 4 Submission (Mar 30 – Apr 4)

> Goal: IEEE-format poster ready for submission.

| Task | Owner | Dates |
|------|-------|-------|
| Poster layout design | Akshay | Mar 30–31 |
| Write poster content: Introduction + Method | Akshay | Mar 31 – Apr 1 |
| Write poster content: Results + Ablations | Ritik | Mar 31 – Apr 1 |
| Create poster figures (architecture diagram, tables, plots) | Atharva | Mar 30 – Apr 1 |
| Poster review + iteration | All | Apr 2–3 |
| Phase 4 submission | Akshay | Apr 4 |

**Poster sections:**
1. Problem & Motivation (2 sentences + threat landscape figure)
2. Architecture Diagram (two-stream → sync-score → Bi-LSTM)
3. Key Results Table (4 test axes, AUC-ROC, EER)
4. Ablation Summary (mini table or grouped bar chart)
5. Sync-Score Visualization (real vs fake s(t) curves — the money shot)
6. Conclusion + Future Work (2-3 bullet points)

---

## Phase 5 — Final Report & Presentation (Apr 4 – Apr 13)

> Goal: Complete IEEE report, video demo, public GitHub release.

### Report (Apr 4 – Apr 9) — All

IEEE conference format. Target: 6–8 pages.

| Section | Owner | Dates |
|---------|-------|-------|
| Abstract + Introduction | Akshay | Apr 4–5 |
| Related Work | Ritik | Apr 4–5 |
| Methodology (full architecture, losses, training) | Akshay + Ritik | Apr 5–6 |
| Experiments (setup, datasets, implementation details) | Atharva | Apr 5–6 |
| Results (4 test axes, ablations, visualizations) | Atharva + Ritik | Apr 6–7 |
| Discussion + Limitations + Conclusion | Akshay | Apr 7–8 |
| Full review + revision pass | All | Apr 8–9 |

### Video Demo (Apr 7 – Apr 10) — Akshay

- 3–5 minute walkthrough
- Show: problem motivation → architecture overview → live demo on sample clips → results summary
- Include sync-score visualization running in real-time on a clip

### Lightweight Demo Script (Apr 7 – Apr 9) — Akshay

```
scripts/demo.py — CLI tool
  Input: path to video file
  Output: Real/Fake prediction + confidence + s(t) plot saved as PNG
  Target: <10s for a 30-second clip on CPU
  Optimizations: ONNX export or torch.jit.trace if needed
```

### GitHub Release (Apr 10 – Apr 11) — Akshay

- Clean up repo (remove debug code, ensure .gitignore is tight)
- Upload pretrained weights to GitHub Releases or HuggingFace
- Update README with: installation, quickstart, reproduction instructions, results table
- Add license (MIT or Apache-2.0)

### Final Submission (Apr 12 – Apr 13) — Akshay

- Submit report PDF
- Submit poster PDF
- Submit video link
- Submit GitHub link
- Final check: all deliverables present and formatted

---

## Critical Path & Dependencies

```
Data Download ──→ Preprocessing ──→ Phase 1 Pretrain ──→ Phase 2 Fine-tune ──→ Evaluation ──→ Report
                                         │                      │                   │
                                    (6 days, GPU)         (5 days, GPU)        (3 days)
                                         │                      │                   │
                                    Models must be         Pretrain ckpt       Fine-tune ckpt
                                    implemented first      must exist          must exist
```

**Bottleneck:** GPU time on HPC. Pretraining (20 epochs) + fine-tuning (30 epochs) + ablations (~4-6 runs x 15 epochs) = significant compute. Submit jobs early, use job queuing.

**Risk mitigations:**
1. **Data download delays:** Fall back to FakeAVCeleb real subset for pretraining
2. **OOM on HPC:** Reduce batch size (32→16→8), use gradient accumulation
3. **AV-HuBERT loading issues:** Fall back to ResNet-18 visual encoder (simpler, still viable)
4. **Metrics below target:** Tune hard negative ratio, try unfreezing audio encoder, increase pretraining epochs
5. **Time crunch:** Prioritize main model + FakeAVCeleb eval. Ablations and DFDC/Wav2Lip eval are lower priority

---

## File Deliverables Checklist

### Code (`src/`)
- [x] `src/models/visual_encoder.py` — AV-HuBERT wrapper + projection *(completed Mar 14)*
- [x] `src/models/audio_encoder.py` — Wav2Vec 2.0 wrapper + projection *(completed Mar 14)*
- [x] `src/models/classifier.py` — Bi-LSTM temporal classifier *(completed Mar 14)*
- [x] `src/models/syncguard.py` — Full model integration *(completed Mar 14)*
- [x] `src/models/audio_classifier.py` — Standalone Wav2Vec2 audio classifier for cascade *(completed Mar 20)*
- [x] `src/models/__init__.py` — Module exports *(completed Mar 14)*
- [x] `src/training/losses.py` — InfoNCE, temporal consistency, BCE, combined *(completed Mar 14)*
- [x] `src/training/dataset.py` — Training dataset + hard negative mining *(completed Mar 14)*
- [x] `src/training/pretrain.py` — Phase 1 contrastive pretraining loop *(completed Mar 15)*
- [x] `src/training/finetune.py` — Phase 2 fine-tuning loop *(completed Mar 15)*
- [x] `src/training/__init__.py` — Module exports *(completed Mar 14)*
- [x] `src/evaluation/metrics.py` — AUC-ROC, EER, pAUC computation ✓ Mar 18
- [x] `src/evaluation/evaluate.py` — Evaluation runner ✓ Mar 18
- [x] `src/evaluation/visualize.py` — Sync-score plots, ROC curves, ablation charts ✓ Mar 18
- [x] `src/evaluation/__init__.py` — Module exports ✓ Mar 18

### Scripts
- [x] `scripts/train_pretrain.py` — CLI for Phase 1 pretraining *(completed Mar 15)*
- [x] `scripts/train_finetune.py` — CLI for Phase 2 fine-tuning *(completed Mar 15)*
- [x] `scripts/evaluate.py` — (functionality in src/evaluation/evaluate.py) ✓ Mar 18
- [x] `scripts/train_audio_classifier.py` — Standalone audio classifier training ✓ Mar 20
- [x] `scripts/evaluate_cascade.py` — Cascade evaluation (sync + audio, 4 fusion strategies) ✓ Mar 20
- [ ] `scripts/demo.py` — Lightweight demo (video → prediction + s(t) plot)
- [x] `scripts/gpu_smoke_test.py` — GPU smoke test ✓ Mar 18
- [x] `scripts/slurm_preprocess_avspeech.sh` — Auto-resubmitting SLURM job ✓ Mar 18
- [x] `scripts/slurm_train_audio_clf.sh` — SLURM script for audio classifier ✓ Mar 20
- [x] `scripts/slurm_evaluate_cascade.sh` — SLURM script for cascade eval ✓ Mar 20

### Models
- [x] `src/models/audio_classifier.py` — Standalone Wav2Vec2 audio deepfake classifier ✓ Mar 20

### Outputs (generated, on HPC — gitignored)
- [x] `outputs/checkpoints/pretrain_best.pt` — Phase 1 best checkpoint (epoch 17, learnable τ)
- [x] `outputs/checkpoints/finetune_best_run3_audioswap.pt` — Phase 2 best checkpoint (Run 3, audio-swap)
- [x] `outputs/checkpoints/audio_clf_best.pt` — Standalone audio classifier checkpoint
- [x] `outputs/logs/eval_cascade.json` — Cascade evaluation results (4 fusion strategies)
- [x] `outputs/logs/predictions_cascade.npz` — Raw cascade predictions
- [ ] `outputs/visualizations/` — All generated plots

### Deliverables
- [ ] `docs/poster.pdf` — IEEE format poster
- [ ] `docs/report.pdf` — IEEE conference format report (6–8 pages)
- [ ] Video demo (hosted externally, link in README)
- [ ] GitHub Release with pretrained weights

---

## Daily Standup Protocol

Each team member posts a 3-line update daily (Slack/Discord):
1. **Done:** What I completed today
2. **Doing:** What I'm working on next
3. **Blocked:** Any blockers (data, GPU, bugs)

**Weekly sync (Sunday evening):** Review progress against this plan, adjust timeline if needed.

---

## Quick Reference: Key Commands

```bash
# Activate environment
source ~/.zshrc && conda activate syncguard

# Preprocess a dataset
python scripts/preprocess_dataset.py --dataset fakeavceleb --config configs/default.yaml

# Phase 1: Contrastive pretraining
python scripts/train_pretrain.py --config configs/default.yaml

# Phase 2: Fine-tuning
python scripts/train_finetune.py --config configs/default.yaml --pretrain_ckpt outputs/checkpoints/pretrain_best.pt

# Evaluate
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/finetune_best.pt --test_set fakeavceleb

# Demo
python scripts/demo.py --video path/to/video.mp4 --checkpoint outputs/checkpoints/finetune_best.pt
```
