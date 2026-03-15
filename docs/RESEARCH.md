# RESEARCH.md — Technical Foundations & Design Rationale

**Project:** SyncGuard — Contrastive Audio-Visual Deepfake Detection

> This document explains the WHY behind every design choice.
> Read it alongside CLAUDE.md (the WHAT) and BASELINES.md (the EXPECTED NUMBERS).

---

## 1. The Core Signal: Why Audio-Visual Synchrony?

### 1.1 The Biomechanical Argument

Human speech production involves precise coordination between the vocal tract (producing sound) and facial articulators (lips, jaw, tongue visible through lip shape). This coupling is governed by physics — the same neural motor commands drive both modalities simultaneously.

**Key properties of natural AV synchrony:**
- **Phoneme-specific:** Each phoneme (e.g., /p/, /f/, /a/) requires a specific articulatory configuration. Bilabial plosives (/p/, /b/, /m/) require lip closure. Labiodental fricatives (/f/, /v/) require lower lip contact with upper teeth.
- **Millisecond-precise:** Lip closure for /p/ must precede the acoustic burst by 10–30ms. This timing is consistent within a speaker and across speakers.
- **Rate-coupled:** When speech speeds up, both acoustic and visual dynamics accelerate together. The first derivative of visual motion tracks the first derivative of the audio envelope.

**Why deepfakes fail this test:**
- Face-swap methods (DeepFaceLab, FaceSwap) replace the face but preserve the original audio → the new face's lip movements come from a different speaker's articulation patterns
- Audio-swap methods (voice cloning, TTS) replace the audio but preserve the face → the face's lip movements don't match the new audio's phoneme timing
- Even Wav2Lip (explicitly trained for lip-sync) only matches coarse lip shape, not the full articulatory dynamics across all facial muscles

### 1.2 Why This Generalizes Across Generators

Visual artifact detectors learn generator-specific patterns:
- FaceForensics++ detectors learn compression artifacts specific to the training generators
- When tested on unseen generators, AUC drops from ~0.95 to ~0.65 (Rossler et al., 2019)

Audio-visual synchrony is **generator-agnostic** because:
- It doesn't depend on how the face was generated, only on whether it matches the audio
- Any manipulation that changes either modality will break the temporal coupling
- The signal comes from physics (articulatory motor control), not from artifacts of a specific generative model
- Even if a generator produces perfectly realistic frames, the temporal dynamics will be wrong unless the generator explicitly models articulatory biomechanics

### 1.3 Limitations of This Signal

Be honest about where this approach struggles:
- **Wav2Lip adversarial case:** Wav2Lip is specifically trained to match lip shape to audio, creating fakes that pass coarse sync tests. Our approach must detect subtle temporal dynamics (transition speed, jitter) rather than gross desynchronization.
- **Non-speech segments:** When there's no speech, there's no sync signal. The model must rely on the Bi-LSTM learning to handle these gaps via the speech mask.
- **Music videos / dubbed content:** Legitimate post-production dubbing creates real desynchronization. The model may flag these as fake. Scope: we focus on talking-head content only.

---

## 2. Architecture Design Decisions

### 2.1 Why AV-HuBERT for Visual Encoding?

**Alternatives considered:**

| Encoder | Pretraining Data | What It Learns | Pros | Cons |
|---------|-----------------|---------------|------|------|
| ResNet-18 (ImageNet) | Object recognition | General visual features | Simple, well-understood | No temporal lip knowledge |
| EfficientNet-B3 (VGGFace2) | Face recognition | Identity features | Good face representations | Identity-biased, not motion-aware |
| SyncNet (LRS2) | Lip-sync detection | AV correspondence | Designed for sync | Small capacity, older architecture |
| **AV-HuBERT (LRS3)** | Lip-reading | Articulatory motion | Directly models what we need | Complex loading (fairseq), heavier |

**Why AV-HuBERT wins:**
- Pretrained on lip-reading (LRS3, 433 hours) — it literally learned to decode speech from mouth movements
- Its internal representations encode articulatory features, not just appearance
- The visual frontend (3D-CNN + ResNet) is designed for temporal mouth-ROI processing
- Shi et al. (2022) showed that AV-HuBERT's visual features contain phonemic information that transfers across tasks

**The key insight:** We're not doing lip-reading, but lip-reading pretraining teaches exactly the temporal articulatory dynamics we need to detect manipulation.

### 2.2 Why Wav2Vec 2.0 for Audio Encoding?

**Why not raw spectrograms?**
- Spectrograms require learning acoustic-to-phonemic mapping from scratch
- Wav2Vec 2.0 already encodes phonemic content in its hidden states (Pasad et al., 2021)
- Self-supervised pretraining on 960 hours of Librispeech provides robust features

**Why layer 9 (default)?**

Pasad et al. (2021) showed that Wav2Vec 2.0 layers encode different information:
- **Layers 1-4:** Low-level acoustics (pitch, energy, speaker characteristics)
- **Layers 5-8:** Emerging phonemic content
- **Layers 9-10:** Peak phonemic/word-level information
- **Layers 11-12:** High-level linguistic structure (less temporal)

For sync-score computation, we need **phonemic temporal information** — the identity and timing of each phoneme. Layer 9 is the sweet spot: rich phonemic content with preserved temporal resolution. We validate this via ablation across layers 3, 5, 7, 9, 11.

**Why 49 Hz output rate?**
- Wav2Vec 2.0 processes 16kHz audio with a stride of ~326 samples → 16000/326 ≈ 49 features/second
- This is native to the model — no resampling needed on the audio side
- Visual features (25 fps) are upsampled to 49 Hz via linear interpolation to match
- 49 Hz provides sufficient temporal resolution for phoneme boundaries (phonemes last ~50-200ms)

### 2.3 Why Bi-LSTM for Classification?

The sync-score sequence s(t) is a 1D temporal signal. The classifier must learn patterns like:
- Sustained low s(t) = likely fake
- Sharp dips at irregular intervals = likely manipulated segments
- Smooth, high s(t) with natural phoneme-driven variation = likely real

**Why not simpler approaches?**
- **Statistical features (mean, std, skew, kurtosis):** Loses temporal structure. Can't detect localized dips.
- **1D-CNN:** Captures local temporal patterns (kernel size ~5-10 frames) but misses long-range dependencies
- **Transformer:** Overkill for a 1D signal of ~150-500 timesteps. Would need positional encoding and attention, adding complexity without clear benefit for this task

**Why Bi-LSTM works:**
- Bidirectional processing captures both past and future context for each timestep
- Hidden size 128 × 2 directions = 256-dim representation — sufficient for binary classification
- 2 layers with dropout 0.3 provides enough capacity without overfitting on ~15K training clips
- Mean + max pooling over time captures both average behavior and extreme events (dips)

### 2.4 Why Projection to R^256?

Both visual and audio embeddings are projected to 256 dimensions and L2-normalized:
- **L2 normalization** maps embeddings to the unit hypersphere. Cosine similarity then equals dot product, simplifying computation
- **256 dimensions** is the standard for contrastive learning (SimCLR, MoCo, CLIP). Enough capacity for rich representations, small enough for efficient memory bank storage
- **Shared dimensionality** is required for cosine similarity computation between modalities

---

## 3. Loss Function Design

### 3.1 Why InfoNCE (not Triplet Loss, not Contrastive Loss)?

**InfoNCE (Noise-Contrastive Estimation):**
```
L = -(1/T) Σ_t log[ exp(cos(v_t, a_t)/τ) / Σ_j exp(cos(v_t, a_j)/τ) ]
```

Compared to alternatives:

| Loss | Pros | Cons | Why not? |
|------|------|------|----------|
| Triplet loss | Simple, well-understood | Requires explicit triplet mining, margins are fragile | Hard to tune margin, slow convergence |
| NT-Xent (SimCLR-style) | Clean, no memory bank needed | Effective negatives limited to batch size | Batch size 32 = only 31 negatives, too few for fine-grained alignment |
| **InfoNCE + MoCo** | Large negative pool (4096), decoupled from batch size | Memory bank adds complexity | **Our choice** — best of both worlds |

**Why frame-level, not clip-level?**
- Clip-level InfoNCE: one similarity score per clip → can't detect partial manipulation
- Frame-level InfoNCE: one similarity score per frame → sensitive to localized desynchronization
- A 5-second clip with a 1-second face-swap has 80% real frames and 20% fake frames. Clip-level loss would average this away. Frame-level loss targets the exact problematic frames

### 3.2 Why MoCo-Style Memory Bank?

With batch size 32, each frame only sees 31 in-batch negatives. This is insufficient for learning fine-grained AV correspondence — many negatives will be trivially easy (different speakers, different content).

MoCo (He et al., 2020) maintains a FIFO queue of past audio embeddings (size 4096). For each visual embedding v_t, the denominator includes:
- The positive a_t (matched audio frame)
- All 4096 queue entries as negatives

This gives 4096 negatives per frame regardless of batch size. The queue is updated with current-batch embeddings after each step (no gradient through queue).

**Queue size 4096:** A balance between diversity (more negatives = harder task = better representations) and staleness (very old embeddings may be inconsistent with current model).

### 3.3 Temporal Consistency Loss — The Rate-of-Change Constraint

```
L_temp = Σ_t ||(v_{t+1} - v_t) - (a_{t+1} - a_t)||²  ·  1[real]
```

**Intuition:** In real speech, when the visual motion accelerates (e.g., lips opening quickly for a vowel), the audio dynamics should also accelerate (rising energy, changing spectral shape). The first derivatives should track each other.

**Why only on real clips?**
- For real clips: enforces that the model learns the natural coupling between visual and audio dynamics
- For fake clips: the dynamics ARE desynchronized — that's the signal. Penalizing the model for this would be counterproductive

**Why L2 norm (not L1, not cosine)?**
- L2 penalizes large deviations quadratically — a single badly mismatched frame contributes more to the loss than many small mismatches
- This is desirable: we want the model to strongly encode the temporal dynamics at phoneme boundaries where the rate changes are largest

### 3.4 The Combined Loss Balance

```
L_total = L_InfoNCE + γ · L_temp + δ · L_cls
```

**Default weights: γ = 0.5, δ = 1.0**

**Rationale:**
- **L_InfoNCE (weight 1.0):** Primary loss. Drives the core learning of AV correspondence
- **L_cls (weight 1.0, δ):** Equal weight to classification. The model must learn to make binary decisions, not just embeddings
- **L_temp (weight 0.5, γ):** Lower weight because temporal consistency is a supporting signal, not the primary objective. Too high → model focuses on smoothness over discriminability

**When to adjust:**
- If AUC plateaus but sync-scores look good → increase δ (more weight on classification)
- If sync-scores are noisy/uninformative → increase γ (enforce smoother temporal dynamics)
- If model overfits to FakeAVCeleb specific patterns → increase InfoNCE weight relative to L_cls

### 3.5 Hard Negative Mining — Same Speaker, Different Time

Standard negatives in InfoNCE are random audio frames from other clips. These are trivially different (different speakers, content, acoustics). The model can "cheat" by learning speaker identity rather than fine-grained sync.

**Hard negatives:** Sample audio frames from the SAME speaker but a different temporal window. The model must learn precise temporal alignment, not just "this voice matches this face."

**Annealing (0% → 20% over 10 epochs):**
- Starting with hard negatives immediately makes training unstable — the model hasn't yet learned basic correspondence
- Gradually introducing them forces the model to refine its representations after learning the basics
- 20% maximum keeps training tractable — too many hard negatives cause underfitting

---

## 4. Evaluation Design Decisions

### 4.1 Why Speaker-Disjoint Splits?

If the same speaker appears in both train and test, the model can learn to recognize speakers (identity-based detection) rather than sync patterns. This gives inflated test performance that doesn't generalize.

**Our split strategy:**
- Group all clips by speaker ID
- Assign entire speakers to train/val/test (e.g., 70/15/15 by speaker count)
- No speaker appears in more than one partition
- This is the gold standard for deepfake detection evaluation (Khalid et al., 2021)

### 4.2 Why Four Test Axes?

Each test axis measures a different generalization property:

| Axis | What It Tests | Why It Matters |
|------|-------------|---------------|
| **In-domain (FakeAVCeleb)** | Basic detection capability | Can the model detect fakes at all? |
| **Cross-generator (CelebDF-v2)** | Generator generalization | Does it work on unseen face-swap methods? |
| **In-the-wild (DFDC)** | Real-world robustness | Handles compression, lighting, non-frontal? |
| **Adversarial (Wav2Lip)** | Sync-optimized robustness | Can it detect fakes designed to pass sync tests? |

**The progression matters:** In-domain should be highest, adversarial lowest. If adversarial > in-domain, the results are suspicious.

### 4.3 Why AUC-ROC as Primary Metric?

- **Threshold-independent:** Doesn't require choosing a detection threshold, which would be arbitrary
- **Standard in forensics:** All major deepfake detection papers report AUC-ROC, enabling direct comparison
- **Complemented by EER:** Equal Error Rate gives a single operating point where FPR = FNR
- **pAUC at FPR < 0.1:** For forensic applications, false positives are costly. pAUC measures performance specifically in the low-FPR regime

---

## 5. What Could Invalidate Our Approach

### 5.1 Wav2Lip Produces Perfect Temporal Dynamics

If Wav2Lip (or a future model) generates lip movements with perfectly accurate articulatory dynamics — not just lip shape but transition timing, jaw motion, tongue influence — our sync-score would be high for fakes. The Bi-LSTM would need to rely on extremely subtle signals.

**Likelihood:** Low for current Wav2Lip, medium for future models. Wav2Lip optimizes for lip-sync detection (SyncNet loss), not articulatory biomechanics. There's a gap between "looks synchronized" and "is biomechanically consistent."

**Mitigation:** The Wav2Lip adversarial test set directly measures this. If AUC drops below 0.55 (near random), acknowledge the limitation and discuss how temporal resolution or additional signals (jaw dynamics, micro-expressions) could help.

### 5.2 AV-HuBERT Features Are Identity-Biased

If AV-HuBERT's visual features encode speaker identity strongly, the model might learn identity-based detection (recognizing known speakers) rather than sync-based detection.

**Diagnosis:** Check cross-generator generalization (CelebDF-v2, DFDC). If CelebDF-v2 AUC is much lower than FakeAVCeleb AUC (>15% gap), identity bias may be present.

**Mitigation:** Speaker-disjoint splits prevent this during training. If identity bias persists, add a gradient reversal layer on speaker identity prediction (adversarial training).

### 5.3 The Pretraining Data Is Insufficient

If we fall back to using FakeAVCeleb's real subset (~4K clips) instead of VoxCeleb2 (~50K+ clips), the contrastive pretraining may not learn robust enough AV correspondence.

**Diagnosis:** Check sync-score separation between real and shuffled pairs after pretraining. If gap < 0.2, pretraining is insufficient.

**Mitigation:** Increase pretraining epochs (20 → 40), use aggressive augmentation (time-stretch, pitch shift, crop jitter), or try semi-supervised pretraining with unlabeled real videos.

### 5.4 The Sync-Score Is Not Discriminative Enough

If the distribution of mean s(t) overlaps significantly between real and fake clips, the Bi-LSTM has little to work with.

**Diagnosis:** Plot histograms of mean s(t) for real vs fake after pretraining. If the distributions overlap by more than 50%, the embeddings need improvement.

**Mitigation:** Try unfreezing the pretrained encoders during fine-tuning (at lower LR), use a larger projection head, or add layer-wise features from multiple Wav2Vec layers.

---

## 6. Paper & Reference Map

Every claim in the project should be traceable to one of these:

| Claim | Source |
|-------|--------|
| AV sync is a deepfake detection signal | Chung & Zisserman, 2016 (SyncNet); AVoiD-DF, 2023 |
| FakeAVCeleb dataset and 4-category taxonomy | Khalid et al., 2021 |
| Wav2Vec 2.0 layers encode phonemic information | Pasad et al., 2021 (Layer-Wise Analysis) |
| AV-HuBERT learns articulatory visual features | Shi et al., 2022 |
| InfoNCE loss for contrastive learning | Oord et al., 2018 (CPC) |
| MoCo memory bank for large negative pools | He et al., 2020 |
| Speaker-disjoint evaluation for deepfake detection | Khalid et al., 2021; Rossler et al., 2019 |
| Generalization failure of visual artifact detectors | Rossler et al., 2019 (FaceForensics++) |
| Wav2Lip for adversarial lip-sync generation | Prajwal et al., 2020 |
| CelebDF-v2 for cross-generator testing | Li et al., 2020 |
| DFDC for in-the-wild testing | Dolhansky et al., 2020 |
| Hard negative mining improves contrastive learning | Robinson et al., 2021 |
| Temporal consistency in AV deepfake detection | AVoiD-DF (Feng et al., 2023) |
