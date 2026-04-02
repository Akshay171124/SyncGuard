# Cross-Attention Embedding Bypass for DFDC Generalization

**Date:** 2026-03-29
**Author:** Akshay Prajapati + Claude
**Status:** Approved

## Problem

SyncGuard compresses rich 256-dim AV embeddings into a scalar cosine similarity `s(t) = cos(v_t, a_t)`. The Bi-LSTM classifier only sees this 1D time series. DFDC face-swaps preserve lip-sync, so the sync-score is uninformative (AUC 0.4579 — below random chance). The identity-mismatch information needed to detect face-swaps IS present in the raw embeddings but is discarded by the cosine reduction.

## Evidence

| Model | FakeAVCeleb AUC | DFDC AUC | Problem |
|-------|:-:|:-:|---|
| Sync-only (v2 finetune) | 0.9225 | 0.4579 | Sync signal inverted on DFDC |
| AVFF (CVPR 2024, reference) | — | 0.862 | Uses cross-modal embedding fusion |

## Solution: Parallel Cross-Attention Head

Add a cross-attention module that operates on the full AV embeddings as a parallel classification path alongside the existing sync-score Bi-LSTM. Fuse both paths with a learnable weight.

### Architecture

```
EXISTING PATH (preserved, unchanged):

  v_t (B,T,256) ──→ cos(v_t, a_t) ──→ Bi-LSTM ──→ sync_logits (B,1)
  a_t (B,T,256) ──┘

NEW PARALLEL PATH (cross-attention):

  v_t (B,T,256) ──→ CrossAttention(Q=v, K=a, V=a) ──→ v_attended (B,T,256)
  a_t (B,T,256) ──→ CrossAttention(Q=a, K=v, V=v) ──→ a_attended (B,T,256)
                           │                                    │
                           └──────────┬─────────────────────────┘
                                      │
                                Concatenate
                      [v_attended; a_attended] (B,T,512)
                                      │
                                Temporal Pool
                           (masked mean + max → 1024)
                                      │
                                MLP (1024 → 256 → 1)
                                      │
                                embed_logits (B,1)

FUSION:

  sync_logits ──┐
                ├──→ w * sync_logits + (1-w) * embed_logits ──→ final_logits
  embed_logits ─┘    (w = learnable sigmoid parameter, init 0.5)
```

### Cross-Attention Module Details

- **Type:** Multi-head cross-attention (nn.MultiheadAttention)
- **Heads:** 2
- **Embed dim:** 256 (matches encoder output)
- **Layers:** 1 (lightweight — we have limited data)
- **Dropout:** 0.1 on attention weights
- **Bidirectional:** Visual attends to audio AND audio attends to visual
- **Residual connection:** `v_attended = v_t + CrossAttn(Q=v, K=a, V=a)`
- **Layer norm:** After residual connection

### Temporal Pooling

- Masked mean pooling + masked max pooling concatenated (same as existing Bi-LSTM pooling)
- Uses `lengths` tensor to mask padded positions
- Output: (B, 1024) — 512 from mean, 512 from max

### Embed Classifier MLP

- Linear(1024, 256) → ReLU → Dropout(0.3) → Linear(256, 1)
- Single output logit (pre-sigmoid)

### Fusion

- Learnable parameter `fusion_weight` initialized to 0.0 (sigmoid → 0.5)
- `w = sigmoid(fusion_weight)`
- `final_logits = w * sync_logits + (1-w) * embed_logits`
- During Stage 1, only `embed_logits` is trained; fusion activates in Stage 2

### Parameter Count Estimate

| Component | Parameters |
|-----------|-----------|
| CrossAttention (V→A) | 256×256×3 + 256 = ~197K |
| CrossAttention (A→V) | ~197K |
| LayerNorm × 2 | 256×2×2 = ~1K |
| MLP (1024→256→1) | 1024×256 + 256×1 = ~262K |
| Fusion weight | 1 |
| **Total new params** | **~657K** |

Lightweight — less than 1% of the existing model. Low OOM risk.

## Training Strategy

### Stage 1: Train cross-attention head only (~4 hours on A100)

```
Frozen:    visual encoder, audio encoder, Bi-LSTM, sync path
Trainable: CrossAttentionModule + EmbedClassifier MLP
Dataset:   FakeAVCeleb + LRS2 reals (112K samples)
Loss:      BCE(embed_logits, labels)
Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
Scheduler: Cosine with 2-epoch warmup
Epochs:    20 (early stopping patience=5 on val AUC)
```

### Stage 2: End-to-end fusion fine-tuning (~2 hours)

```
Frozen:    visual encoder, audio encoder
Trainable: Bi-LSTM + CrossAttention + MLP + fusion weight
Dataset:   FakeAVCeleb + LRS2 reals
Loss:      BCE(final_logits, labels)
Optimizer: AdamW, lr=5e-5 (lower — preserving sync path)
Epochs:    10 (early stopping patience=3)
```

### Stage 3: Evaluation

```
Test sets:     FakeAVCeleb test split + DFDC (reprocessed)
Ablation:      sync-only, embed-only, fused
Metrics:       AUC-ROC, EER, pAUC@0.1, per-category breakdown
Bootstrap CIs: 5000 iterations
```

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/models/cross_attention.py` | `CrossAttentionModule` + `EmbedClassifier` classes |
| `scripts/train_cross_attention.py` | Stage 1 + Stage 2 training script |
| `scripts/slurm_train_cross_attention.sh` | SLURM job for HPC |

### Modified Files

| File | Change |
|------|--------|
| `src/models/syncguard.py` | Add `use_cross_attention` flag, parallel forward path, fusion |
| `configs/finetune_frozen.yaml` | Add `model.cross_attention` config section |
| `scripts/evaluate.py` | Already supports DFDC (just fixed) |

### No Changes Needed

| File | Why |
|------|-----|
| `src/training/losses.py` | BCE loss already exists, no new loss needed |
| `src/training/dataset.py` | Same FakeAVCeleb data, no changes |
| `src/models/visual_encoder.py` | Encoders frozen, unchanged |
| `src/models/audio_encoder.py` | Encoders frozen, unchanged |

## Config

```yaml
model:
  cross_attention:
    enabled: true
    num_heads: 2
    num_layers: 1
    dropout: 0.1
    embed_classifier_hidden: 256
    fusion_init: 0.0  # sigmoid(0) = 0.5
```

## Expected Outcomes

| Configuration | FakeAVCeleb AUC | DFDC AUC | Notes |
|---------------|:-:|:-:|---|
| Sync-only (current) | 0.9225 | 0.4579 | Baseline |
| Embed-only (Stage 1) | 0.85-0.90 | 0.55-0.70 | Cross-attention alone |
| Fused (Stage 2) | 0.92-0.94 | 0.60-0.75 | Best of both paths |

Conservative DFDC estimate: **0.60-0.75 AUC** (up from 0.46).

## Timeline

| Day | Task | Time |
|-----|------|------|
| Day 3 (Mar 30) | Implement `cross_attention.py`, modify `syncguard.py` | 3-4 hours |
| Day 3 (Mar 30) | Write training script, create SLURM job | 1-2 hours |
| Day 3-4 | Stage 1 training on HPC | ~4 hours |
| Day 4 | Stage 2 fusion training | ~2 hours |
| Day 4 | Full evaluation (FakeAVCeleb + DFDC + ablations) | ~1 hour |
| Day 4 | Results analysis, update docs | 1 hour |

Total: ~2 days from start to DFDC number.

## Risk Mitigation

- **FakeAVCeleb regression:** Sync path is frozen in Stage 1. Fusion weight starts at 0.5 and learns. If embed path hurts FakeAVCeleb, fusion weight will shift toward sync path.
- **DFDC still fails:** If cross-attention doesn't generalize to DFDC face-swaps, this is a valid negative result. We report the analysis: "Identity-mismatch signals from FakeAVCeleb face-swaps do not transfer to DFDC face-swap methods."
- **OOM:** Only 657K new parameters. Negligible memory impact.
- **Training instability:** Stage 1 freezes everything except the new head — minimal risk. Stage 2 uses low learning rate.

## Why Cross-Attention Over Simpler Alternatives

1. **vs MLP on `v-a` difference:** MLP is per-frame, can't learn temporal relationships. Cross-attention learns "which audio frames matter for this visual frame" — captures temporal identity inconsistencies.
2. **vs Concatenation to Bi-LSTM:** Changing Bi-LSTM input from 1D to 1024D breaks existing checkpoint, risks OOM, needs more training data.
3. **vs Full AVFF reimplementation:** AVFF replaces the entire pipeline. We preserve our working sync path and add cross-attention as a parallel module — faster, safer, and produces ablation tables for the paper.

## Paper Contribution

"We augment temporal sync-score classification with bidirectional cross-modal attention that operates directly on encoder embeddings. This parallel architecture preserves in-domain detection (AUC 0.92 on FakeAVCeleb) while enabling cross-dataset generalization to face-swap deepfakes that preserve lip-sync (DFDC)."
