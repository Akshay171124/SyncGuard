# Experiment: Phase 1 Contrastive Pretraining — Run 1 vs Run 2

**Date:** 2026-03-20
**Owner:** Akshay / Ritik

## Setup (shared)
- Phase: pretrain (contrastive only, no classifier)
- Dataset: AVSpeech (21K real clips, 85/15 train/val split)
- Config: `configs/default.yaml` — batch_size=32, lr=1e-4, epochs=20, warmup=2, moco_queue=4096
- Visual encoder: AV-HuBERT architecture (random init, no pretrained weights)
- Audio encoder: Wav2Vec 2.0 (frozen backbone, trainable projection, layer 9)
- Checkpoint loaded: from scratch (both runs)

### Run 1: Fixed τ=0.07
- **GPU:** H200 140GB (SLURM job 5238944)
- **Bug:** `build_optimizer()` did not include `criterion.parameters()` — τ was never updated
- **wandb run:** `phase1-pretrain`
- **Training time:** ~90 min (20 epochs × ~270s/epoch)

### Run 2: Learnable τ (bug fixed)
- **GPU:** A100 40GB → A100 80GB (SLURM jobs 5244509 → 5248320, auto-resubmit)
- **Fix:** Added `criterion.parameters()` to optimizer so `log_temperature` gets gradients
- **wandb run:** `phase1-pretrain-learnable-tau`
- **Training time:** ~150 min (20 epochs, A100 slower than H200)

## Results

| Metric | Run 1 (fixed τ) | Run 2 (learnable τ) |
|--------|-----------------|---------------------|
| Best val loss | 8.2990 (epoch 13) | **8.2561 (epoch 17)** |
| Final val loss | 8.3134 | **8.2565** |
| Best sync score | 0.7005 | **0.7063** |
| Final temperature | 0.0700 (frozen) | **0.0411 (learned)** |
| Train-val gap | 0.0708 | **0.0275** |
| Overfitting? | Yes (val loss ↑ for 6 epochs) | No (flat after epoch 17) |

### Epoch-by-epoch: Run 1 (fixed τ=0.07)

| Epoch | Train Loss | Val Loss | Sync Score | τ | LR |
|-------|-----------|----------|------------|------|--------|
| 0 | 8.3265 | 8.3246 | 0.0067 | 0.07 | 5.05e-5 |
| 5 | 8.3106 | 8.3079 | 0.3175 | 0.07 | 7.79e-5 |
| 10 | 8.2677 | 8.3013 | 0.6150 | 0.07 | 3.94e-5 |
| 13 | 8.2492 | 8.2990 | 0.6673 | 0.07 | 2.10e-5 |
| 15 | 8.2430 | 8.3073 | 0.6875 | 0.07 | 1.16e-5 |
| 19 | 8.2426 | 8.3134 | 0.7005 | 0.07 | 1.00e-6 |

### Epoch-by-epoch: Run 2 (learnable τ)

| Epoch | Train Loss | Val Loss | Sync Score | τ | LR |
|-------|-----------|----------|------------|------|--------|
| 0 | 8.3266 | 8.3246 | 0.0068 | 0.069 | 5.05e-5 |
| 5 | 8.3085 | 8.3053 | 0.2993 | 0.063 | 7.79e-5 |
| 10 | 8.2523 | 8.2714 | 0.5784 | 0.051 | 3.94e-5 |
| 15 | 8.2318 | 8.2578 | 0.6881 | 0.043 | 1.16e-5 |
| 17 | 8.2301 | 8.2561 | 0.6990 | 0.042 | 5.69e-6 |
| 19 | 8.2290 | 8.2565 | 0.7063 | 0.041 | 1.00e-6 |

## Observations

1. **Learnable τ eliminates overfitting.** Run 1's val loss peaked at epoch 13 and deteriorated for 6 straight epochs. Run 2 kept improving until epoch 17 and was essentially flat after — the train-val gap (0.028 vs 0.071) confirms much better generalization.

2. **τ learned to sharpen.** Temperature dropped from 0.07 → 0.041, making the contrastive softmax distribution sharper. This means the model learned to make more confident positive/negative distinctions rather than relying on a hand-tuned temperature.

3. **Sync scores comparable.** Both runs reached ~0.70 sync score, indicating the visual and audio embeddings are well-aligned regardless of τ. The slight edge for Run 2 (0.7063 vs 0.7005) comes from better optimization, not a fundamentally different representation.

4. **Random-init visual encoder works.** Despite no pretrained AV-HuBERT weights, the model learned meaningful audio-visual alignment from scratch on only 21K clips. This validates the contrastive pretraining approach.

5. **InfoNCE loss context.** Random chance = ln(4096) ≈ 8.318. Both runs reached ~8.26, a modest but meaningful reduction. With only 21K clips and random-init visual encoder, this is expected — the model is learning, just slowly.

## Decision

**Winner: Run 2 (learnable τ)**

Reasons:
- Lower val loss (8.2561 vs 8.2990) — better learned representations
- No overfitting — the checkpoint is more reliable for transfer
- Smaller train-val gap — better generalization to unseen data
- Temperature self-tuned — one fewer hyperparameter to worry about in Phase 2

**Next step:** Use `pretrain_best.pt` (Run 2, epoch 17) as the backbone for Phase 2 fine-tuning on FakeAVCeleb.

## Artifacts

- **Winner checkpoint:** `outputs/checkpoints/pretrain_best.pt` (Run 2, epoch 17)
- **Run 1 checkpoint:** `outputs/checkpoints/pretrain_best_run1_fixed_tau.pt` (epoch 13)
- **Metrics JSON:** `outputs/logs/pretrain.json` (Run 2 full history)
- **wandb:** Project `SyncGuard`, runs `phase1-pretrain` and `phase1-pretrain-learnable-tau`
- **This report:** `outputs/logs/experiment_pretrain_comparison.md`

---

## Planned: Phase 1 v2 — CMP Pretraining (Cross-Modal Prediction)

**Status:** Blocked on LRS2 preprocessing (~18,453/96,318 done as of Mar 23)

### Motivation
Phase 1 v1 (above) trained on AVSpeech only (21K clips) with InfoNCE loss. The resulting encoder generalizes well to FakeAVCeleb (AUC 0.9254 after fine-tuning) but fails on DFDC (AUC 0.5712 — random chance). DFDC face-swaps preserve lip-sync, so surface-level sync-scores don't discriminate.

**Hypothesis:** Cross-modal prediction (AVFF-style) forces encoders to learn *deep* AV correspondence beyond surface sync. Combined with expanded training data (AVSpeech 21K + LRS2 96K), this should produce representations that detect face-swap artifacts invisible to pure sync-score approaches.

### Changes from v1
| Parameter | v1 (current) | v2 (planned) |
|-----------|-------------|-------------|
| Loss | InfoNCE only | InfoNCE + 0.5 × CMP |
| CMP mask ratio | N/A | 30% of frames |
| Dataset | AVSpeech (21K) | AVSpeech (21K) + LRS2 (96K) |
| Epochs | 20 | 20 |
| Hardware | H200 | H200 |
| SLURM script | `slurm_train_pretrain.sh` | `scripts/slurm_pretrain.sh` |

### Code
- `CrossModalPredictionLoss` in `src/training/losses.py` — masks 30% of frames in one modality, uses MLP predictors (256→512→256) to reconstruct via L1 loss
- `PretrainLoss` updated to combine: L = L_InfoNCE + 0.5 × L_CMP
- Config: `training.pretrain.cross_modal_prediction: true`, `cmp_weight: 0.5`, `cmp_mask_ratio: 0.3`

### Expected Outcome
- Sync scores should still converge (≥ 0.70) since InfoNCE is retained
- CMP loss should decrease, indicating cross-modal predictability improves
- Downstream DFDC AUC after Phase 2 fine-tuning should improve from 0.5712 → target ≥ 0.72
