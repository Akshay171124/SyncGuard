# BASELINES.md — Expected Metrics, Sanity Checks, and Red Flags

**Project:** SyncGuard — Contrastive Audio-Visual Deepfake Detection

> This file tells you what "normal" looks like at every stage.
> If your numbers fall outside these ranges, something is probably broken.
> Check here before debugging code — it's usually a config or data issue.

---

## 1. Contrastive Pretraining — What Normal Looks Like

### 1.1 InfoNCE Loss Trajectory

| Epoch | Expected Loss Range | What's Happening |
|-------|-------------------|------------------|
| 1 | 7.0–9.0 | Random embeddings. Loss ≈ log(queue_size) = log(4096) ≈ 8.3 |
| 5 | 4.0–6.0 | Model learning coarse audio-visual correspondence |
| 10 | 2.0–4.0 | Fine-grained temporal alignment emerging |
| 15 | 1.5–3.0 | Approaching convergence |
| 20 | 1.0–2.5 | Converged. If still > 4.0, something is wrong |

**Red flags — pretraining is broken if:**
- Loss stays flat at ~8.3 for >3 epochs → embeddings aren't updating (check LR, frozen layers, or optimizer)
- Loss drops to <0.5 → model collapsed (all embeddings identical). Check L2 normalization and temperature
- Loss oscillates wildly between epochs → LR too high, reduce by 5-10x
- Loss is NaN → numerical instability in log-softmax. Check for zero-length sequences or division by zero in temperature

### 1.2 Temperature τ (Learnable)

| Phase | Expected τ Range | Interpretation |
|-------|-----------------|----------------|
| Init | 0.07 (fixed) | Starting value |
| Epoch 5 | 0.04–0.08 | May decrease slightly as embeddings sharpen |
| Epoch 20 | 0.03–0.10 | Stabilized. Should NOT be at clamp boundaries |

**Red flags:**
- τ hits lower clamp (0.01) → model wants sharper softmax than allowed. Embeddings may be well-separated already — this is okay but monitor for gradient vanishing
- τ hits upper clamp (0.5) → model can't learn meaningful similarities. Check that projection heads are training
- τ is NaN → gradient explosion through τ. Use gradient clipping

### 1.3 Sync-Score Distribution (During Pretraining)

After pretraining on real data only, measure average sync-score on a held-out set:

| Measurement | Expected Value | Interpretation |
|-------------|---------------|----------------|
| Mean s(t) on real clips | 0.5–0.8 | High — model learned that real AV pairs are synchronous |
| Std of s(t) on real clips | 0.10–0.25 | Some variation across phonemes and speakers is normal |
| Mean s(t) on shuffled pairs (neg control) | -0.1–0.3 | Low — random pairings should have low similarity |
| Gap (real - shuffled) | ≥ 0.3 | Must be clearly separable. If < 0.2, pretraining didn't work |

**Red flags:**
- Mean s(t) on real ≈ mean on shuffled → embeddings are uninformative, pretraining failed
- Mean s(t) on real > 0.95 → embeddings collapsed to a constant, not discriminative
- Mean s(t) on real < 0.3 → model hasn't learned correspondence. Try more epochs or unfreeze encoders
- All s(t) values are exactly 0.0 → cosine similarity computation bug (check L2 normalization)

---

## 2. Fine-tuning — What Normal Looks Like

### 2.1 Loss Components

| Metric | Expected Range (Start) | Expected Range (End, 30 epochs) | Notes |
|--------|----------------------|-------------------------------|-------|
| L_InfoNCE | 1.5–3.0 | 0.8–2.0 | Should decrease but not collapse |
| L_temp (γ=0.5) | 0.01–0.10 | 0.005–0.05 | Real-only. Small values are normal |
| L_cls (δ=1.0) | 0.6–0.7 | 0.1–0.4 | BCE starts near -log(0.5)=0.693, should drop |
| L_total | 2.0–4.0 | 1.0–2.5 | Weighted sum of above |

**Red flags:**
- L_cls stays at 0.693 → classifier not learning (random predictions). Check labels, gradients through Bi-LSTM
- L_cls drops to <0.01 → overfitting. Check dataset size, add dropout, reduce LR
- L_temp increases while L_cls decreases → gradient conflict. Try reducing γ
- L_InfoNCE increases during fine-tuning → contrastive alignment degrading. Fine-tuning LR may be too high for the encoders

### 2.2 Validation Metrics During Fine-tuning

| Epoch | Expected Val AUC-ROC | Expected Val EER |
|-------|---------------------|-----------------|
| 5 | 0.65–0.75 | 0.30–0.40 |
| 10 | 0.75–0.85 | 0.20–0.30 |
| 20 | 0.82–0.90 | 0.15–0.25 |
| 30 | 0.85–0.92 | 0.12–0.22 |

**Red flags:**
- AUC < 0.55 after 10 epochs → barely above random. Check data labels, speaker leakage, or broken sync-score
- AUC plateaus at ~0.70 → pretraining may be insufficient, or hard negatives aren't effective
- AUC drops after epoch 15-20 → overfitting. Enable early stopping (patience=5)
- Val AUC much higher than expected (>0.95) → speaker leakage in splits. Verify speaker-disjoint partitioning

---

## 3. Evaluation — Expected Performance

### 3.1 Primary Test Results

| Test Set | Target AUC-ROC | Realistic Range | Interpretation |
|----------|---------------|----------------|----------------|
| FakeAVCeleb (all categories) | ≥ 0.88 | 0.82–0.93 | In-domain, speaker-disjoint |
| CelebDF-v2 (zero-shot) | Report | 0.65–0.80 | Unseen generator — drop expected |
| DFDC (zero-shot) | ≥ 0.72 | 0.62–0.78 | Diverse, compressed — harder |
| Wav2Lip adversarial | Report | 0.55–0.72 | Sync-optimized fakes — hardest case |

### 3.2 FakeAVCeleb Per-Category Expected AUC

| Category | Expected AUC | Reasoning |
|----------|-------------|-----------|
| RV-RA (Real Video, Real Audio) | ~1.0 | Trivially real — model should never flag these |
| FV-RA (Fake Video, Real Audio) | 0.82–0.90 | Visual manipulation disrupts lip-sync with real audio |
| RV-FA (Real Video, Fake Audio) | 0.78–0.88 | Audio manipulation — real face doesn't match fake audio timing |
| FV-FA (Fake Video, Fake Audio) | 0.75–0.88 | Both modalities manipulated — may accidentally re-sync, hardest |

**Red flags:**
- RV-RA AUC < 0.95 → model is incorrectly flagging real clips. Check preprocessing
- FV-FA AUC > FV-RA AUC → surprising, but possible if double manipulation creates more artifacts
- Any category AUC < 0.60 → sync-score signal not working for this manipulation type
- All categories have identical AUC → model may be using a non-sync shortcut (visual artifact, not temporal coherence)

### 3.3 Sync-Score Characteristics

| Clip Type | Expected Mean s(t) | Expected Std s(t) |
|-----------|-------------------|-------------------|
| Real clips | 0.55–0.80 | 0.08–0.18 |
| FV-RA (face-swapped) | 0.25–0.55 | 0.15–0.30 |
| RV-FA (audio-swapped) | 0.30–0.55 | 0.12–0.25 |
| FV-FA (both swapped) | 0.20–0.50 | 0.15–0.30 |
| Wav2Lip generated | 0.40–0.65 | 0.10–0.20 |

**Key insight:** Wav2Lip fakes will have HIGHER sync-scores than other fakes because Wav2Lip explicitly optimizes for lip-sync. The model must rely on subtle temporal dynamics (jitter, transition smoothness) rather than gross desynchronization.

---

## 4. Ablation — Expected Relative Performance

### 4.1 Visual Encoder Ablation

| Encoder | Expected AUC (FakeAVCeleb) | Notes |
|---------|---------------------------|-------|
| **AV-HuBERT** (ours) | 0.85–0.93 | Best — pretrained on lip-reading, captures articulatory motion |
| ResNet-18 | 0.75–0.85 | Decent but misses temporal lip dynamics |
| SyncNet | 0.78–0.87 | Designed for lip-sync, competitive but lower capacity |

**If ResNet-18 matches AV-HuBERT:** The lip-reading pretraining isn't as critical as hypothesized — interesting finding worth documenting.

### 4.2 Wav2Vec Layer Ablation

| Layer | Expected AUC | Notes |
|-------|-------------|-------|
| Layer 3 | 0.78–0.85 | Low-level acoustic features — less semantic |
| Layer 5 | 0.80–0.87 | Mid-level — balanced |
| Layer 7 | 0.82–0.89 | Starting to capture phonemic content |
| **Layer 9** (default) | 0.85–0.92 | Best expected — strong phonemic + temporal |
| Layer 11 | 0.82–0.88 | High-level — may be too abstract for temporal matching |

**Expected pattern:** Inverted U-shape — middle-to-upper layers best, extremes worse. This matches Pasad et al. (2021) findings on Wav2Vec layer analysis.

### 4.3 Classifier Ablation

| Classifier | Expected AUC | Notes |
|------------|-------------|-------|
| Statistical (mean, std, skew, kurtosis) | 0.68–0.78 | Baseline — no temporal modeling |
| 1D-CNN | 0.78–0.87 | Captures local temporal patterns |
| **Bi-LSTM** (ours) | 0.85–0.92 | Best — models global temporal dependencies |

**If statistical baseline gets >0.80:** The sync-score signal is so strong that even simple statistics suffice — simplify the architecture.

---

## 5. Training Time & Resource Estimates

| Phase | Dataset Size | Expected Time (1× H200) | Expected Time (1× A100 40GB) |
|-------|-------------|------------------------|------------------------------|
| Pretraining (20 epochs) | ~4K clips (fallback) | 3–5 hours | 5–8 hours |
| Pretraining (20 epochs) | ~50K clips (VoxCeleb2 subset) | 15–25 hours | 25–40 hours |
| Fine-tuning (30 epochs) | ~15K clips (FakeAVCeleb) | 8–14 hours | 14–22 hours |
| Single ablation (15 epochs) | ~15K clips | 4–7 hours | 7–11 hours |
| All ablations (4 studies) | varies | 20–35 hours total | 35–55 hours total |
| Evaluation (all 4 test sets) | ~25K clips | 1–2 hours | 2–3 hours |

**Total estimated GPU-hours:** 50–80 hours. Reserve 2× buffer (100–160 hours) for debugging and reruns.

---

## 6. Pre/Post Experiment Checklists

### Pre-Experiment (copy into experiment log, check each item)

```
PRE-EXPERIMENT CHECKS:
[ ] Conda environment activated (syncguard)
[ ] GPU detected (run quick torch.cuda check)
[ ] Config file reviewed (correct dataset, hyperparameters, paths)
[ ] Dataset exists at expected path and has correct sample count
[ ] Preprocessed data exists (mouth_crops.npy, audio.wav per sample)
[ ] Train/val/test splits are speaker-disjoint (verify with split script)
[ ] Output directory exists and previous results are saved
[ ] Checkpoint to load exists (for fine-tuning/evaluation)
[ ] Previous experiment results are committed to git
[ ] GPU memory looks right (check with nvidia-smi before launching)
```

### Post-Experiment (check after every run)

```
POST-EXPERIMENT CHECKS:
[ ] Training completed without errors (check last log line)
[ ] Checkpoint saved (*.pt file exists in outputs/checkpoints/)
[ ] Metrics JSON written and has correct number of entries
[ ] Loss curve is monotonically decreasing (roughly)
[ ] Final metric is within expected range (see Sections 1-4)
[ ] Results are different from previous experiment (not accidentally same config)
[ ] Experiment log filled out (see template in CLAUDE.md)
[ ] Results committed to git
```

---

## 7. Quick-Reference: What Numbers Go in the Report

After experiments, fill in these blanks:

```
PRETRAINING:
"Contrastive pretraining on [dataset] ([N] clips) for [E] epochs achieved
InfoNCE loss of [X], with mean sync-score of [Y] on real clips and [Z] on
shuffled negative pairs (gap = [Y-Z])."

MAIN RESULT:
"SyncGuard achieves AUC-ROC of [A] on FakeAVCeleb (speaker-disjoint test),
[B] on CelebDF-v2 (zero-shot), [C] on DFDC (zero-shot), and [D] on
Wav2Lip adversarial test set."

PER-CATEGORY:
"On FakeAVCeleb, per-category AUC: FV-RA=[X], RV-FA=[Y], FV-FA=[Z],
demonstrating detection across all manipulation types."

ABLATION KEY FINDING:
"AV-HuBERT outperforms ResNet-18 by [X]% AUC and SyncNet by [Y]% AUC,
confirming that lip-reading pretraining provides stronger articulatory features.
Wav2Vec layer [N] achieves the best performance, consistent with prior findings
on phonemic feature distribution across layers."

INTERPRETABILITY:
"Sync-score s(t) curves show clear temporal dips at manipulated segments,
with mean s(t) of [A] for real clips vs [B] for fake clips (p < 0.001)."
```

Fill in brackets with ACTUAL numbers. Never estimate or round favorably.
