# CLIP Backbone + SBI Augmentation for DFDC Cross-Dataset Generalization

**Date:** 2026-04-06
**Author:** Akshay Prajapati + Claude
**Status:** Approved

## Problem

SyncGuard achieves 96.3% AUC on FakeAVCeleb but only 52.6% on DFDC (zero-shot). All attempted interventions (cross-attention, DCT features, BN adaptation, threshold recalibration) failed to close the gap because:

1. **Training data limitation:** Model only sees FakeAVCeleb fakes, which have different artifacts than DFDC face-swaps
2. **Weak visual backbone:** AV-HuBERT (25M params, random init) lacks the general visual understanding to detect forgery artifacts across methods
3. **No blending boundary signal:** DFDC face-swaps produce blending artifacts our model never learns to detect

## Evidence

| Model | FakeAVCeleb AUC | DFDC AUC |
|-------|:-:|:-:|
| SyncGuard current (v4+CA) | 0.963 | 0.497 |
| SyncGuard best DFDC (CA Stage 1+2) | 0.927 | 0.526 |
| GenD (CLIP ViT-L/14, WACV 2026) | — | **0.871** |
| SBI baseline (CVPR 2022) | — | 0.724 |

## Solution: CLIP Visual Backbone + Self-Blended Image Augmentation

Two complementary changes that address different aspects of the DFDC failure:

### Part 1: CLIP ViT-L/14 Visual Backbone

Replace AV-HuBERT (25M params, random init) with CLIP ViT-L/14 (300M params, pretrained on 400M image-text pairs). Freeze entire model except LayerNorm parameters (~90K trainable).

**Why:** Foundation models encode universal visual features (texture, structure, identity, frequency patterns) that capture forgery artifacts across ALL face-swap methods. GenD proves CLIP + minimal tuning achieves 87.1% DFDC zero-shot.

**Architecture:**
```
CURRENT:
  Mouth Crops (96×96, gray) → AV-HuBERT (25M, random) → Proj(512→256) → v_t

NEW:
  Mouth Crops (224×224, RGB) → CLIP ViT-L/14 (300M, frozen except LN)
       → CLS token per frame → Proj(768→256) → v_t
```

**Key decisions:**
- Input: resize 96×96 → 224×224, grayscale → RGB (repeat channels)
- Tuning: LayerNorm only (0.03% of params) — prevents overfitting to source dataset
- Frame-level: each frame processed independently, temporal modeling via Bi-LSTM/cross-attention downstream
- Audio encoder: Wav2Vec 2.0 unchanged (frozen)

### Part 2: Self-Blended Image (SBI) Augmentation

Generate synthetic face-swap training data from real videos by blending face regions with transformed versions of themselves.

**Why:** SBI creates blending boundary artifacts that are common to ALL face-swap methods (including DFDC's). The model learns to detect blending itself, not algorithm-specific artifacts.

**How SBI works (per frame):**
1. Extract face landmarks mask (convex hull from existing MediaPipe landmarks)
2. Create target: apply random transforms to original face (color jitter ±10%, Gaussian blur σ=1-3, affine warp ±5%)
3. Blend source (original) and target using landmarks mask with feathered edges (Gaussian blur on mask, σ=5-15)
4. Optionally apply JPEG compression (quality 70-95)
5. Label as fake (label=1)

**Integration:**
- Applied in `SyncGuardDataset.__getitem__()` as data augmentation
- 30% of real samples → SBI fakes per epoch
- Audio unchanged (creates visual-only fakes — matching DFDC's attack type)

## Combined Architecture

```
Inputs:
  Mouth Crops (224×224, RGB) → CLIP ViT-L/14 (frozen, LN trainable)
  Raw Audio (16kHz) → Wav2Vec 2.0 (frozen)
        │                              │
        ▼                              ▼
  ProjectionHead (768→256)    ProjectionHead (768→256)
        │                              │
        ▼                              ▼
  v_t ∈ ℝ²⁵⁶                    a_t ∈ ℝ²⁵⁶
        │                              │
        ├──→ cos(v,a) + EAR → Bi-LSTM → sync_logits
        ├──→ Cross-Attention (V→A, A→V) → Pool → MLP → embed_logits
        └──→ Learnable Fusion: w·sync + (1-w)·embed → Real/Fake
```

Everything downstream of the projection heads remains identical to the current architecture.

## Training Strategy

**Single phase — no pretraining.** CLIP is already pretrained. Skip Phase 1 contrastive pretraining entirely. Go directly to supervised fine-tuning.

**Rationale:** GenD achieves 87.1% DFDC without AV pretraining. Adding contrastive pretraining risks the same saturation we saw with v3 (sync=1.0, CMP=0). If results are underwhelming, Phase 1 can be added as a follow-up.

**Training details:**

| Setting | Value |
|---------|-------|
| Visual encoder | CLIP ViT-L/14, frozen except LayerNorm (~90K trainable) |
| Audio encoder | Wav2Vec 2.0, frozen |
| Trainable | Visual LN + ProjectionHeads + Bi-LSTM + CrossAttention + EmbedClassifier + fusion |
| Total trainable | ~2.5M |
| Dataset | FakeAVCeleb + LRS2 reals + SBI synthetic fakes (~140K samples) |
| SBI ratio | 30% of real samples per epoch |
| Batch size | 16 (CLIP ViT-L/14 memory: ~8GB for batch=16 at 224×224) |
| Learning rate | 5e-5, cosine schedule, 3-epoch warmup |
| Epochs | 30 (early stopping patience=5 on val AUC) |
| Loss | BCE (classification) + lightweight InfoNCE (projection alignment) |
| Input | 224×224 RGB (resized from 96×96 grayscale) |

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/models/clip_visual_encoder.py` | CLIP ViT-L/14 wrapper: frame-level processing, LayerNorm-only tuning, CLS token extraction, ProjectionHead (768→256) |
| `src/augmentation/sbi.py` | Self-Blended Image augmentation: landmarks mask, random transforms, alpha blending with feathered edges |
| `configs/clip_sbi.yaml` | Config: CLIP backbone + SBI enabled + frozen LN-only |
| `scripts/train_clip_sbi.py` | Training script: single-phase, direct fine-tuning |
| `scripts/slurm_train_clip_sbi.sh` | SLURM job for HPC |

### Modified Files

| File | Change |
|------|--------|
| `src/models/syncguard.py` | Add `clip` option to `build_visual_encoder()` dispatch |
| `src/training/dataset.py` | Add SBI augmentation call in `__getitem__()` for real samples, handle 224×224 RGB resize |

### No Changes Needed

| File | Why |
|------|-----|
| `src/models/audio_encoder.py` | Wav2Vec unchanged |
| `src/models/cross_attention.py` | Operates on 256-dim embeddings regardless of visual backbone |
| `src/models/classifier.py` | Bi-LSTM input unchanged |
| `src/evaluation/evaluate.py` | Already supports FakeAVCeleb + DFDC |

## Config

```yaml
model:
  visual_encoder:
    name: "clip"
    model_id: "openai/clip-vit-large-patch14"
    embedding_dim: 256
    freeze_pretrained: true  # Freeze everything
    tune_layernorm: true     # Except LayerNorm
    input_size: 224
  audio_encoder:
    name: "wav2vec2"
    model_id: "facebook/wav2vec2-base-960h"
    layer: 9
    embedding_dim: 256
    freeze_pretrained: true
  classifier:
    name: "bilstm"
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    use_ear: true
  cross_attention:
    enabled: true
    num_heads: 2
    dropout: 0.1
    embed_classifier_hidden: 256
    fusion_init: 0.0
  dct_extractor:
    enabled: false  # Disabled — didn't help in previous experiments

augmentation:
  sbi:
    enabled: true
    ratio: 0.3           # 30% of real samples
    color_jitter: 0.1    # ±10%
    blur_sigma: [1.0, 3.0]
    warp_strength: 0.05  # ±5% affine
    mask_blur_sigma: [5, 15]
    jpeg_quality: [70, 95]

training:
  finetune:
    epochs: 30
    batch_size: 16
    lr: 5.0e-5
    weight_decay: 1.0e-4
    warmup_epochs: 3
    gamma: 0.5
    delta: 1.0
```

## Expected Outcomes

| Configuration | FakeAVCeleb AUC | DFDC AUC |
|---------------|:-:|:-:|
| Current best (v4+CA) | 0.963 | 0.497 |
| Current best DFDC (CA Stage 1+2) | 0.927 | 0.526 |
| CLIP + SBI (predicted) | 0.95-0.98 | **0.75-0.88** |

## Risk Mitigation

- **OOM with CLIP:** ViT-L/14 at 224×224 with batch=16 should fit in A100 40GB (~8GB model + ~6GB activations). If OOM, reduce batch to 8.
- **FakeAVCeleb regression:** SBI may slightly reduce in-domain AUC. If FAV drops below 0.93, reduce SBI ratio from 30% to 15%.
- **DFDC still fails:** If DFDC < 0.65 despite CLIP + SBI, add Phase 1 contrastive pretraining as follow-up experiment.
- **Slow training:** CLIP ViT-L/14 is slower per frame than AV-HuBERT. Expected ~2x slower per epoch. With batch=16, ~40 min/epoch on A100.

## Dependencies

```
# New pip dependency needed
pip install open_clip_torch  # or: transformers (already installed, has CLIPModel)
```

CLIP ViT-L/14 available via HuggingFace `transformers` (already in our env):
```python
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
```

## References

- GenD (WACV 2026): 87.1% DFDC with CLIP + LayerNorm tuning — https://arxiv.org/abs/2508.06248
- SBI (CVPR 2022): 72.4% DFDC with self-blended augmentation — https://arxiv.org/abs/2204.08376
- ForAda++ (CVPR 2025): 86.3% DFDC with CLIP + adapter — https://arxiv.org/abs/2411.19715
- CLIP (ICML 2021): Foundation model — https://arxiv.org/abs/2103.00020
