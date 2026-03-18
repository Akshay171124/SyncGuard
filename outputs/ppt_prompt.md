# Prompt for Claude to Generate SyncGuard Project Presentation

Create a visually polished, professional 7-slide presentation (Google Slides or PowerPoint style) for a Computer Vision course (CS 5330, Northeastern University). This is a project presentation — it should tell the story of our project, not read like a checklist. Make it feel like a top-tier research project presentation. Use clean, modern design with consistent colors, clear hierarchy, and minimal text per point. Favor diagrams and visual layouts over walls of text.

---

## PROJECT OVERVIEW

**Project Name:** SyncGuard
**Tagline:** Contrastive Audio-Visual Deepfake Detection via Temporal Phoneme-Face Coherence
**Course:** CS 5330 — Computer Vision, Northeastern University, Khoury College of Computer Sciences
**Team:** Akshay Prajapati, Ritik, Atharva
**Date:** March 2026
**GitHub:** github.com/Akshay171124/SyncGuard

---

## SLIDE 1 — Title Slide

Project name "SyncGuard" prominently displayed. Include the full tagline, course info, team names, and date. Dark professional theme.

---

## SLIDE 2 — The Problem & Our Approach

### The Problem
Deepfake videos are increasingly realistic. Current detection methods rely on **visual artifacts** (blending boundaries, texture glitches, flickering) — but these are **generator-specific**. A detector trained on FaceSwap fails on Wav2Lip. When tested on unseen generators, AUC drops from ~0.95 to ~0.65 (Rossler et al., 2019). Detectors are always playing catch-up.

Most detectors also **completely ignore audio**, even though deepfakes manipulate audio (voice cloning) or create mismatches between face and voice.

### Our Key Insight
Real human speech has **tight temporal coupling** between lip movements and audio — governed by biomechanics of speech production:
- Lip closure for /p/ precedes the acoustic burst by 10-30ms
- Lip rounding for /o/ co-occurs with specific acoustic formants
- When speech speeds up, both visual and audio dynamics accelerate together

**No deepfake generator perfectly replicates this coupling at the frame level (20-50ms resolution).** Face-swap methods use a different person's articulation. Voice-clone methods create audio that doesn't match the face's timing. Even Wav2Lip (specifically trained for lip-sync) only matches coarse lip shape, not full articulatory dynamics.

This is a **generator-agnostic signal** — it comes from physics of speech production, not from artifacts of any specific model.

### Design this slide as:
Left side: "The Problem" with 3-4 concise points about why current detection fails
Right side: "Our Approach" with the key insight about AV synchrony being generator-agnostic
Use a visual contrast between the two sides (e.g., red/fragile vs green/robust)

---

## SLIDE 3 — System Architecture (Pipeline Diagram)

### Create a clear architecture diagram showing the full pipeline:

```
Input Video → [Preprocessing] → Two parallel streams:

VISUAL STREAM:
  Video frames → RetinaFace face detection → MediaPipe FaceMesh (468 landmarks)
  → Mouth ROI crop (96×96 grayscale)
  → AV-HuBERT Visual Encoder (3D Conv frontend + ResNet-18 trunk)
  → Projection Head (Linear → ReLU → Linear → L2 Normalize)
  → v_t ∈ R^(B×T×256)   [frame-level visual embeddings]

AUDIO STREAM:
  Audio → FFmpeg extract → 16kHz mono → Silero-VAD speech mask
  → Wav2Vec 2.0 Audio Encoder (frozen backbone, layer 9 hidden states)
  → Projection Head (Linear → ReLU → Linear → L2 Normalize)
  → a_t ∈ R^(B×T×256)   [frame-level audio embeddings]

SYNC SCORE:
  s(t) = cos(v_t, a_t) for each frame t
  Real clips: s(t) ≈ 0.7–0.9 (synchronized)
  Fake clips: s(t) ≈ 0.1–0.4 (desynchronized)

CLASSIFIER:
  s(t) sequence → Bi-LSTM (2-layer, bidirectional, hidden=128)
  → Mean + Max pooling → MLP → Real/Fake prediction

LOSSES (bottom of diagram):
  Phase 1 (Pretraining): InfoNCE loss with MoCo memory bank (4096 negatives)
  Phase 2 (Fine-tuning): L = L_InfoNCE + 0.5·L_temporal + 1.0·L_classification
```

Make this a clean, color-coded pipeline diagram. Use different colors for:
- Preprocessing (green)
- Pretrained encoders (blue)
- Classifier (purple)
- Loss functions (yellow/orange)

Label the key dimensions at each stage. Show Phase 1 and Phase 2 training boundaries.

---

## SLIDE 4 — Technical Deep Dive

### Four component boxes side by side:

**Visual Encoder (AV-HuBERT):**
- Pretrained on lip-reading (LRS3, 433 hours)
- Learned to decode speech from mouth movements — encodes exactly the articulatory dynamics we need
- 3D Conv frontend (5×7×7) captures temporal mouth motion
- ResNet-18 trunk for spatial features
- Projection to 256-dim, L2-normalized
- Why AV-HuBERT: lip-reading pretraining teaches articulatory motion, not just appearance (Shi et al., 2022)

**Audio Encoder (Wav2Vec 2.0):**
- facebook/wav2vec2-base-960h (pretrained on 960h Librispeech)
- Backbone frozen (94M parameters saved from gradient computation)
- Layer 9 hidden states — peak phonemic content (Pasad et al., 2021)
- Layers 1-4: acoustics, Layers 5-8: emerging phonemics, Layer 9-10: peak phonemic, Layer 11-12: linguistic
- Projection to 256-dim, L2-normalized
- Native 49Hz output → visual upsampled from 25fps to match

**Training Strategy:**
- Phase 1 — Contrastive Pretraining (real data only):
  - InfoNCE loss: frame-level contrastive learning
  - MoCo memory bank: 4096 past audio embeddings as negatives (decoupled from batch size)
  - Learnable temperature τ (init=0.07, clamped [0.01, 0.5])
  - 20 epochs, cosine LR with warmup, batch_size=32
- Phase 2 — Fine-tuning (FakeAVCeleb):
  - Combined: L_nce + 0.5·L_temporal + 1.0·L_classification
  - Temporal consistency loss: L2 on first derivatives of embeddings — penalizes divergent rate-of-change between visual and audio streams. Applied ONLY to real clips (fakes are supposed to be desynchronized)
  - Hard negative mining: same-speaker different-time windows, annealed 0%→20% over 10 epochs (prevents speaker identity shortcuts)
  - Early stopping: patience=5 on validation AUC-ROC
  - 30 epochs, lr=5e-5

**Ablation Dimensions (all implemented, config-swappable):**
- Visual encoder: AV-HuBERT vs ResNet-18 (ImageNet) vs SyncNet → Does lip-reading pretraining matter?
- Wav2Vec layer: 3, 5, 7, 9, 11 → Which layer best encodes phonetics?
- Classifier: Bi-LSTM vs 1D-CNN vs Statistical (mean/std/skew/kurtosis) → Does temporal modeling matter?
- Hard negatives: 0% vs 20% → Does same-speaker mining help?

---

## SLIDE 5 — Data & Preprocessing

### Datasets (show as visual cards):

| Dataset | Role | Size | Details |
|---------|------|------|---------|
| **FakeAVCeleb** | Primary train/val/test | 19,500 clips | 4 manipulation categories (see below). Speaker-disjoint splits: 70/15/15 |
| **VoxCeleb2** | Pretraining (real only) | ~500 hrs subset | Diverse speakers for learning AV correspondence |
| **CelebDF-v2** | Zero-shot evaluation | 6,229 clips | Different face-swap generator — tests cross-generator generalization |
| **DFDC** | Zero-shot evaluation | ~5K test clips | Facebook DeepFake Detection Challenge — hardest in-the-wild benchmark |
| **Wav2Lip generated** | Adversarial test | ~500 clips | Self-generated sync-optimized fakes — hardest case for our approach |

### FakeAVCeleb 4 Categories (important — show this visually):
- **RV-RA:** Real Video + Real Audio → Genuine (label: real)
- **FV-RA:** Fake Video + Real Audio → Face swap only (label: fake)
- **RV-FA:** Real Video + Fake Audio → Voice clone only (label: fake)
- **FV-FA:** Fake Video + Fake Audio → Full deepfake (label: fake)

Hypothesis: FV-FA easiest to detect (both streams manipulated), RV-FA hardest (only audio is fake — very subtle desync)

### Preprocessing Pipeline:
1. Video → RetinaFace face detection (confidence > 0.8)
2. MediaPipe FaceMesh → 468 facial landmarks → mouth ROI extraction
3. Crop & resize to 96×96 grayscale, normalize to [0, 1]
4. Audio → FFmpeg extraction → resample to 16kHz mono
5. Silero-VAD → speech activity detection mask
6. Temporal alignment: 25fps visual ↔ 49Hz Wav2Vec native rate

Speaker-disjoint splits prevent identity leakage — same speaker never appears in train and test.

---

## SLIDE 6 — Implementation Progress & Early Results

### What's Built (show as a checklist with green checkmarks):
- ✅ Preprocessing pipeline — RetinaFace + MediaPipe + Silero-VAD + audio extraction
- ✅ Visual encoder — AV-HuBERT with 3D Conv + ResNet-18 + projection head (+ 2 ablation variants)
- ✅ Audio encoder — Wav2Vec 2.0 with frozen backbone, layer 9, projection head
- ✅ Temporal classifier — Bi-LSTM with masked pooling (+ 2 ablation variants)
- ✅ Full model integration — 107M total parameters, 13M trainable (Wav2Vec frozen)
- ✅ Loss functions — InfoNCE with MoCo queue, temporal consistency, BCE, combined weighted loss
- ✅ Training dataset — speaker-disjoint splits, hard negative mining, variable-length collation with padding masks
- ✅ Phase 1 pretraining loop — cosine LR, warmup, checkpointing, resume support
- ✅ Phase 2 fine-tuning loop — combined loss, hard negative annealing, early stopping
- ✅ CLI scripts — ready to launch on Northeastern HPC (H200 GPUs)
- ⏳ Evaluation framework — AUC-ROC, EER, pAUC (next)
- ⏳ HPC training runs — pending data transfer

### Verification Results:
- **Shape pipeline verified end-to-end:** (B,T,1,96,96) → (B,T,256) visual | (B,samples) → (B,T,256) audio | → (B,T) sync scores | → (B,1) logit
- **L2 normalization confirmed** on both embedding streams
- **Gradient flow verified** through all trainable projection heads and classifier
- **Training loop test (2 epochs, CPU):** Pretraining loss 5.02→4.87 (decreasing ✓), sync-score -0.03→0.18 (learning alignment ✓)
- **Critical bug found & fixed:** Wav2Vec 2.0 produces NaN on zero-padded waveforms in train mode due to group normalization — fixed by forcing frozen backbone to eval() mode. Caught through systematic CPU testing before HPC deployment.

---

## SLIDE 7 — Road Ahead + Thank You

### Timeline to Final Submission (April 13, 2026):

| Period | Phase | Key Activities |
|--------|-------|----------------|
| **This Week** | Data + Eval Setup | Transfer datasets to HPC, run preprocessing, build evaluation framework, GPU smoke test |
| **Mar 16–21** | Phase 1: Pretrain | Contrastive pretraining on real speech data, 20 epochs on H200. Target: sync-score > 0.5 for real clips |
| **Mar 22–28** | Phase 2: Fine-tune | Fine-tune on FakeAVCeleb with combined loss. Evaluate on CelebDF & DFDC |
| **Mar 29–Apr 5** | Ablations | 8 experiments across visual encoder, Wav2Vec layer, classifier, hard negatives |
| **Apr 6–13** | Deliverables | Paper, poster, video demo, final evaluation |

### Target Metrics:
- FakeAVCeleb AUC-ROC ≥ 0.88
- CelebDF-v2 (zero-shot) AUC ≥ 0.79
- DFDC (zero-shot) AUC ≥ 0.72
- Wav2Lip adversarial: report AUC (hardest case — research contribution regardless of number)

### End with: "Thank You — Questions?"
Include GitHub link: github.com/Akshay171124/SyncGuard

---

## DESIGN GUIDELINES

- **Color palette:** Dark blue (#1A5276) for headers, medium blue (#2E86C1) for accents, green (#27AE60) for positive/complete, red (#E74C3C) for problems/challenges, orange (#F39C12) for warnings/in-progress, purple (#8E44AD) for classifier/ablation
- **Font:** Clean sans-serif (Arial, Helvetica, or Inter)
- **Slide backgrounds:** Light gray (#F8F9FA) for content slides, dark blue for title and closing
- **No walls of text** — use bullet points, boxes, diagrams, and visual hierarchy
- **Architecture diagram on slide 3 is the centerpiece** — make it large and clear
- **Slide 2 should feel like a compelling pitch** — why this problem matters and why our approach is different
- **Slide 6 should feel impressive** — we have 10/12 components built before training even starts

## REFERENCES (include small on relevant slides):
- Khalid et al., 2021 — FakeAVCeleb dataset
- Pasad et al., 2021 — Wav2Vec layer analysis
- Shi et al., 2022 — AV-HuBERT
- He et al., 2020 — MoCo
- Rossler et al., 2019 — FaceForensics++ (generalization failure)
- Prajwal et al., 2020 — Wav2Lip
