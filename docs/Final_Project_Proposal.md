# SyncGuard: Contrastive Audio-Visual Deepfake Detection via Temporal Phoneme-Face Coherence

**Project Type:** Research Project
**CV Problem Types:** Classification · Detection · Prediction
**Course:** CS 5330 Computer Vision & Pattern Recognition
**Institution:** Khoury College of Computer Sciences, Northeastern University
**Course Professor:** Akram Bayat
**Date:** March 9, 2026

---

## Team

| Name | Email | Role |
|------|-------|------|
| Akshay Prajapati | prajapati.aksh@northeastern.edu | Visual Encoder & Preprocessing Pipeline, Integration Lead |
| Ritik Mahyavanshi | mahyavanshi.r@northeastern.edu | Audio Encoder & Contrastive Pretraining |
| Atharva Dhumal | dhumal.a@northeastern.edu | Temporal Classifier & Evaluation Pipeline |

---

## 1. Project Overview

### 1.1 Introduction

Deepfake videos pose a growing threat to media integrity, political discourse, and personal privacy. Current detection methods rely on identifying visual artifacts (texture glitches, frequency anomalies, blending boundaries) that are specific to particular generative models. When a new generator is released or improved, these artifact-based detectors become obsolete — this is the **generalization bottleneck**.

SyncGuard addresses this by exploiting a fundamentally different signal: the **biomechanical coupling between speech audio and facial motion**. In natural human speech, each phoneme (plosives, fricatives, vowels) requires a specific articulatory facial configuration synchronized at the millisecond level. No current generative model has been trained to faithfully reproduce this physical coupling, as doing so would require solving articulatory motor control rather than image synthesis.

### 1.2 Problem Statement

Existing deepfake detectors fail to generalize across unseen generators because they learn generator-specific visual artifacts. There is no robust, generator-agnostic detection method that exploits the temporal audio-visual synchronization of speech — a physical constraint that deepfake generators do not explicitly model. The core question is: **can fine-grained temporal phoneme-face coherence serve as a reliable and generalizable deepfake detection signal?**

### 1.3 Objectives

1. Develop a contrastive audio-visual framework that produces a frame-level sync-score sequence s(t) ∈ [-1, 1] measuring phoneme-face temporal coherence.
2. Achieve AUC-ROC ≥ 0.88 on FakeAVCeleb across all four manipulation categories.
3. Achieve AUC-ROC ≥ 0.72 on DFDC in a zero-shot setting (no DFDC data used in training).
4. Demonstrate cross-generator generalization on CelebDF-v2.
5. Validate robustness against sync-optimized fakes (Wav2Lip-generated adversarial test set).
6. Produce interpretable sync-score visualizations showing temporal dips at manipulated segments.

### 1.4 CV Problem Types

- **Classification:** Binary real vs. fake clip-level prediction via Bi-LSTM over the sync-score sequence.
- **Detection:** Temporal localization of desynchronization windows — identifying the specific seconds within a clip that are manipulated (evaluated qualitatively via s(t) curve visualizations and semi-quantitatively on a small synthetic splice benchmark).
- **Prediction:** Frame-level sync-score forecasting as a continuous temporal signal, analogous to anomaly detection in time-series data.

### 1.5 Deliverables

1. Trained SyncGuard model with contrastive pretrained encoders and Bi-LSTM classifier.
2. Evaluation results on FakeAVCeleb (per-category), CelebDF-v2, DFDC (zero-shot), and Wav2Lip adversarial set.
3. Ablation studies: visual encoder selection, Wav2Vec layer selection, classifier comparison, hard negative mining impact.
4. Sync-score curve visualizations with phoneme-level analysis.
5. Lightweight demo: 30-second clip analyzed in <10s on CPU.
6. Project poster, written report, GitHub repository, and video demonstration.

---

## 2. Data Collection Plan

### 2.1 Dataset Summary

| Dataset | Role | Size | Labeled? | Access Status |
|---------|------|------|----------|---------------|
| **FakeAVCeleb** | Primary Training & Validation | 19,500 clips; 4 AV manipulation categories (RV/RA, FV/RA, RV/FA, FV/FA); multilingual (English, Korean) | Yes (4-category) | **Obtained** |
| **VoxCeleb2 / LRS2-BBC** | Contrastive Pretraining (real-only) | VoxCeleb2: 2,000+ hrs, 6,112 speakers; LRS2-BBC: 224 hrs | Natural | Pending (VoxCeleb2 via yt-dlp; LRS2 via Oxford request) |
| **CelebDF-v2** | Cross-generator Zero-shot Test | 590 real + 5,639 fake videos | Yes (binary) | **Obtained** |
| **DFDC** | In-the-wild Zero-shot Test | ~100,000 clips; diverse actors, backgrounds, compression | Yes (binary) | Kaggle (public) |
| **Wav2Lip self-generated** | Adversarial Test | ~500 clips generated from real data | Self-labeled | Self-generated (open-source Wav2Lip) |

### 2.2 Dataset Roles & Rationale

**Pretraining (VoxCeleb2 / LRS2-BBC):** Real talking-head videos only. Used to teach the model what genuine audio-visual synchrony looks like via InfoNCE contrastive loss. VoxCeleb2 preferred for its massive speaker diversity (6,112 speakers). LRS2-BBC as fallback if VoxCeleb2 download proves impractical. If neither is available, FakeAVCeleb's real subset (~4,000+ clips) serves as a minimal fallback with aggressive augmentation.

**Training (FakeAVCeleb):** 4-category taxonomy (RV/RA, FV/RA, RV/FA, FV/FA) uniquely enables per-modality ablation — testing whether the sync-score detects audio-only fakes, video-only fakes, and joint AV fakes separately. Split by speaker ID to prevent identity leakage.

**Evaluation — 4 axes:**
1. **In-domain:** FakeAVCeleb held-out test split (speaker-disjoint)
2. **Cross-generator:** CelebDF-v2 (unseen face-swap method)
3. **In-the-wild:** DFDC (compression, diverse lighting, non-frontal faces)
4. **Adversarial:** Wav2Lip-generated fakes (sync-optimized, hardest case)

### 2.3 Preprocessing Pipeline

```
Video → RetinaFace (face detection)
      → MediaPipe FaceMesh (468 landmarks)
      → Affine-aligned 96×96 mouth-ROI crop

Audio → ffmpeg extraction (16 kHz, mono, PCM)
      → Silero-VAD (speech/non-speech gating)
      → Wav2Vec 2.0 input preparation

Temporal Alignment:
      → Video features upsampled from 25 fps → 49 Hz via linear interpolation
        (preserves Wav2Vec native temporal resolution)
```

### 2.4 Ground Truth & Annotation

All datasets provide video-level binary or category labels. No frame-level annotations are required for the primary classification task. For temporal localization evaluation, a small synthetic splice benchmark (~100 clips) will be created by inserting 2-5 second face-swapped segments into real clips with known splice timestamps.

---

## 3. Methods

### 3.1 Architecture Overview

SyncGuard uses a **two-stream contrastive learning framework** with three stages:

```
┌─────────────────────────────────────────────────────────┐
│  Visual Stream          Audio Stream                     │
│  ┌──────────────┐      ┌──────────────┐                 │
│  │ AV-HuBERT    │      │ Wav2Vec 2.0  │                 │
│  │ Visual Front. │      │ Base         │                 │
│  └──────┬───────┘      └──────┬───────┘                 │
│         │ 96×96 crops         │ 16kHz waveform          │
│         ▼                     ▼                          │
│  ┌──────────────┐      ┌──────────────┐                 │
│  │ Projection   │      │ Projection   │                 │
│  │ → R^256, L2  │      │ → R^256, L2  │                 │
│  └──────┬───────┘      └──────┬───────┘                 │
│         │ v_t                 │ a_t                      │
│         └────────┬────────────┘                          │
│                  ▼                                        │
│         s(t) = cos(v_t, a_t)   ∈ [-1, 1]                │
│                  │                                        │
│                  ▼                                        │
│         ┌──────────────┐                                 │
│         │  Bi-LSTM     │                                 │
│         │  (hidden=128)│                                 │
│         └──────┬───────┘                                 │
│                ▼                                          │
│         Real / Fake (Binary)                             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

**Visual Encoder: AV-HuBERT Visual Frontend**
- Pretrained on lip-reading (LRS3), directly encodes articulatory motion
- Produces per-frame embeddings projected to R^256 on the unit hypersphere
- Operates on 96×96 mouth-ROI crops
- Selected over EfficientNet-B3 (identity-biased VGGFace2 pretraining, overparameterized for 96×96 input) — validated via ablation

**Audio Encoder: Wav2Vec 2.0 Base**
- Self-supervised speech model (`facebook/wav2vec2-base-960h`)
- Hidden states extracted at a layer selected via ablation (candidates: 3, 5, 7, 9, 11)
- Projected to R^256 and L2-normalized
- Outputs features at native 49 Hz

**Sync-Score Sequence:**

```
s(t) = cos(v_t, a_t) = (v_t · a_t) / (‖v_t‖ ‖a_t‖),    s(t) ∈ [-1, 1]
```

For genuine clips, s(t) is high and relatively smooth. For deepfakes, s(t) exhibits sharp temporal dips at phoneme boundaries where the manipulated face's motion diverges from the audio dynamics.

**Temporal Classifier: Bi-directional LSTM**
- Hidden size 128, 2 layers, dropout 0.3
- Mean and max pooling over hidden states → concatenated to 256-dim vector
- Linear(256 → 1) + Sigmoid for binary classification
- Learns to detect the pattern of sync dips characteristic of manipulation

### 3.3 Training Protocol

**Phase 1 — Contrastive Pretraining (real data only, ~20 epochs):**
- Dataset: VoxCeleb2 / LRS2-BBC (or FakeAVCeleb real subset as fallback)
- Loss: InfoNCE only
- MoCo-style memory bank (size 4,096) for large effective negative pool
- Learnable temperature τ initialized at 0.07, clamped to [0.01, 0.5]

**Phase 2 — Fine-tuning on FakeAVCeleb (~30 epochs):**
- Full composite loss: L_total = L_InfoNCE + γ·L_temp + δ·L_cls
- Hard negative mining: 20% temporal hard negatives (same speaker, different time window), annealed from 0% → 20% over first 10 epochs

### 3.4 Loss Functions

**Total Loss:**
```
L_total = L_InfoNCE + γ · L_temp + δ · L_cls
```
Recommended: γ = 0.5, δ = 1.0

**Temporal-Aware InfoNCE (frame-level):**
```
L_InfoNCE = -(1/T) Σ_t log[ exp(cos(v_t, a_t)/τ) / Σ_j exp(cos(v_t, a_j)/τ) ]
```
Operating at frame-level (not clip-level) makes this loss sensitive to partial, localized desynchronization typical of face-swapped deepfakes.

**Temporal Consistency Loss (real clips only):**
```
L_temp = Σ_t ‖(v_{t+1} - v_t) - (a_{t+1} - a_t)‖² · 1[real]
```
Penalizes divergence in the rate-of-change of visual and audio embeddings. In real speech, vocal and facial dynamics accelerate/decelerate together; in deepfakes, they decouple.

**Classification Loss:**
Standard binary cross-entropy on clip-level real/fake prediction.

### 3.5 Hard Negative Mining

20% of contrastive negatives are **temporal hard negatives** — same speaker, different temporal window from the same video. This prevents the model from learning speaker identity as a proxy for authenticity. The hard-negative ratio is annealed from 0% to 20% over the first 10 epochs to avoid early training instability.

### 3.6 Software & Tools

PyTorch, fairseq (AV-HuBERT & Wav2Vec 2.0 pretrained weights), RetinaFace, MediaPipe, Silero-VAD, Montreal Forced Aligner (phoneme timestamps), ffmpeg, OpenCV, scikit-learn, matplotlib.

---

## 4. Evaluation Plan

### 4.1 Metrics

| Metric | Purpose |
|--------|---------|
| **AUC-ROC** | Overall detection ranking |
| **EER** (Equal Error Rate) | Operating-point-free comparison |
| **pAUC** (FPR < 0.1) | Performance at low false-positive rates (forensic use case) |
| **Per-category AUC** | Which FakeAVCeleb manipulation types are detected best |

### 4.2 Quantitative Targets

- AUC-ROC ≥ 0.88 on FakeAVCeleb (all four categories)
- AUC-ROC ≥ 0.72 on DFDC (zero-shot, no DFDC training data)
- Beat SyncNet-based baselines on FakeAVCeleb by ≥ 2% AUC
- Cross-generator generalization on CelebDF-v2

### 4.3 Testing Plan

| Test | Dataset | Condition |
|------|---------|-----------|
| In-domain | FakeAVCeleb held-out split | Speaker-disjoint split |
| Cross-generator | CelebDF-v2 | Zero-shot (no CelebDF training data) |
| In-the-wild | DFDC | Zero-shot; stratified by compression level |
| Adversarial | Wav2Lip self-generated | Sync-optimized fakes (hardest case) |

### 4.4 Ablation Studies

| Ablation | Variants |
|----------|----------|
| Visual encoder | AV-HuBERT vs ResNet-18 vs SyncNet original |
| Wav2Vec layer | Layers 3, 5, 7, 9, 11 |
| Classifier | Statistical baseline vs 1D-CNN vs Bi-LSTM |
| Hard negative mining | On (20%) vs Off (0%) |

### 4.5 Qualitative Evaluation

- Sync-score s(t) curve visualizations: real clips (smooth, high) vs fake clips (sharp dips at phoneme boundaries)
- Phoneme category analysis: which phoneme types (plosives, fricatives, vowels) show strongest desynchronization signal
- Semi-quantitative temporal localization on synthetic splice benchmark

### 4.6 Handling Edge Cases

- **Non-speech segments:** Silero-VAD gates non-speech frames; classifier aggregates only over speech segments
- **Face quality:** RetinaFace confidence threshold (0.8) drops low-quality / non-frontal frames
- **DFDC compression:** Results reported stratified by compression level

---

## 5. Presentation Plan

| Deliverable | Description |
|-------------|-------------|
| **Poster** | SyncGuard pipeline overview, key AUC tables, per-category breakdowns, sync-score visualizations |
| **GitHub Repo** | Public repository with all source code, preprocessing scripts, training pipeline, pretrained weights, README with reproduction instructions |
| **Write-Up** | IEEE conference format report: problem formulation, method, experiments, ablations, results, limitations discussion |
| **Video** | 3-5 minute demo: end-to-end pipeline on sample clips, real-time sync-score visualizations, detection results comparison |

---

## 6. Tentative Schedule

| Milestone | Start | Finish | Phase |
|-----------|-------|--------|-------|
| Form Project Team | 01/06/2026 | 02/13/2026 | Phase 1 |
| Brainstorming & Phase 1 Submission | 02/13/2026 | 02/22/2026 | Phase 1 |
| Finalize Proposal & Phase 2 Submission | 02/22/2026 | 03/09/2026 | Phase 2 |
| Define Phase (Dataset Procurement, Preprocessing Pipeline, Environment Setup) | 03/10/2026 | 03/15/2026 | Phase 3 |
| Development Phase (Contrastive Pretraining, Fine-tuning, Model Training & Evaluation) | 03/16/2026 | 03/30/2026 | Phase 3 |
| Poster Design & Phase 4 Submission | 03/30/2026 | 04/04/2026 | Phase 4 |
| Final Report, Presentation & Phase 5 Submission | 04/04/2026 | 04/13/2026 | Phase 5 |

---

## 7. References

[1] S. Khalid, S. Tariq, J. Kim, S. S. Woo, "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset," NeurIPS Datasets and Benchmarks Track, 2021.

[2] J. S. Chung, A. Nagrani, A. Zisserman, "VoxCeleb2: Deep Speaker Recognition," INTERSPEECH, 2018.

[3] Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu, "Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics," IEEE/CVF CVPR, 2020.

[4] B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, C. C. Ferrer, "The DeepFake Detection Challenge (DFDC) Dataset," arXiv:2006.07397, 2020.

[5] A. Baevski, Y. Zhou, A. Mohamed, M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS, 2020.

[6] A. Shi, B. Hsu, W. Lakber, A. Mohamed, "Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction," ICLR, 2022.

[7] J. S. Chung, A. Zisserman, "Out of Time: Automated Lip Sync in the Wild," ACCV, 2016.

[8] Z. Feng, Y. Lu, J. Liang, J. Li et al., "AVoiD-DF: Audio-Visual Joint Learning for Deepfake Detection," IEEE TIFS, 2023.

[9] D. Guera, E. J. Delp, "Deepfake Video Detection Using Recurrent Neural Networks," IEEE WIFS, 2018.

[10] K. R. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, C. V. Jawahar, "A Lip Sync Expert Is All You Need for Speech to Lip Generation in the Wild," ACM MM, 2020.

[11] T. Afouras, J. S. Chung, A. Senior, O. Zisserman, "LRS3-TED: A Large-Scale Dataset for Visual Speech Recognition," arXiv:1809.00496, 2018.

[12] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Niessner, "FaceForensics++: Learning to Detect Manipulated Facial Images," IEEE/CVF ICCV, 2019.

[13] A. Pasad, J.-C. Chou, K. Livescu, "Layer-Wise Analysis of a Self-Supervised Speech Representation Model," IEEE ASRU, 2021.
