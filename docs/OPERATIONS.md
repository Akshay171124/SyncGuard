# OPERATIONS.md — When to Use What, Step-by-Step

**Project:** SyncGuard — Contrastive Audio-Visual Deepfake Detection

> This is your operating manual. It tells you what to do, when, and in what order.
> Keep this open while you work. Each task has the exact prompt or command to run.

---

## Prerequisites

Before starting, make sure you have:

| Item | Status | Location |
|------|--------|----------|
| CLAUDE.md | In `.claude/CLAUDE.md` | Claude Code reads this automatically |
| BASELINES.md | In `docs/` | Expected metrics and sanity checks |
| RESEARCH.md | In `docs/` | Technical rationale for design choices |
| EXECUTION_PLAN.md | In `docs/` | Day-by-day timeline with task ownership |
| Conda environment `syncguard` | On HPC | All dependencies installed |
| FakeAVCeleb dataset | In `data/raw/FakeAVCeleb/` | Obtained via author request |

---

## Phase 3A — Define & Setup (Mar 11–15)

### Task A1: Data Procurement

**Do this yourself (not Claude Code):**

```bash
# SSH into Explorer
ssh <username>@explorer.rc.northeastern.edu

# Set up project in /scratch
cd /scratch/$USER
git clone https://github.com/Akshay171124/SyncGuard.git
cd SyncGuard

# Activate environment
conda activate syncguard

# Set HuggingFace cache
export HF_HOME=/scratch/$USER/.cache/huggingface
echo 'export HF_HOME=/scratch/$USER/.cache/huggingface' >> ~/.bashrc

# Pre-download models (on CPU node, don't waste GPU time)
srun --partition=short --time=01:00:00 --mem=32GB --cpus-per-task=4 --pty /bin/bash
python -c "
from transformers import Wav2Vec2Model
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
print('Wav2Vec 2.0 downloaded')
"
exit
```

**Download datasets (Akshay, Ritik, Atharva in parallel):**
- FakeAVCeleb → `data/raw/FakeAVCeleb/` (Akshay)
- CelebDF-v2 → `data/raw/CelebDF-v2/` (Atharva)
- VoxCeleb2 subset → `data/raw/VoxCeleb2/` (Ritik — if infeasible, use FakeAVCeleb real subset)

### Task A2: Implement Model Architectures

**Tool:** Claude Code
**Where:** Terminal, inside SyncGuard project root
**Prompt:**

```
Read .claude/CLAUDE.md and docs/EXECUTION_PLAN.md.

Implement the model architectures in this order:

1. src/models/visual_encoder.py — AV-HuBERT visual frontend wrapper
   - Load pretrained weights from fairseq
   - Projection head: Linear → ReLU → Linear → L2-normalize to R^256
   - Input: (B, T, 1, 96, 96) grayscale mouth crops
   - Output: (B, T, 256) frame-level visual embeddings
   - freeze_pretrained configurable from config

2. src/models/audio_encoder.py — Wav2Vec 2.0 wrapper
   - Load facebook/wav2vec2-base-960h via transformers
   - Extract hidden states from configurable layer (default: 9)
   - Projection head: Linear → ReLU → Linear → L2-normalize to R^256
   - Input: (B, waveform_samples) raw audio at 16kHz
   - Output: (B, T, 256) frame-level audio embeddings at 49Hz
   - freeze_pretrained configurable (default: True)

3. src/models/classifier.py — Bi-LSTM temporal classifier
   - Input: (B, T) sync-scores
   - Bi-LSTM: input_size=1, hidden_size=128, num_layers=2, dropout=0.3
   - Pooling: concat mean + max pool → 512-dim → Linear(512,256) → ReLU → Linear(256,1) → Sigmoid
   - Output: (B, 1) real/fake probability

4. src/models/syncguard.py — Full model integration
   - Combines all three modules
   - Forward returns: sync_scores, logits, v_embeds, a_embeds
   - Sync-score: s(t) = cos(v_t, a_t)

5. src/models/__init__.py — Module exports

Write a quick __main__ test in syncguard.py that creates random inputs and verifies
shapes are correct and gradients flow. Use the config from configs/default.yaml.
```

### Task A3: Implement Loss Functions and Training Dataset

**Tool:** Claude Code
**Prompt:**

```
Read .claude/CLAUDE.md, docs/RESEARCH.md, and docs/BASELINES.md.

Implement:

1. src/training/losses.py — All loss functions:
   - InfoNCE (frame-level) with MoCo memory bank (queue_size=4096)
   - Temporal consistency loss (real-only, L2 on first derivatives)
   - Classification loss (BCE)
   - Combined loss: L_total = L_InfoNCE + γ * L_temp + δ * L_cls
   - Learnable temperature τ (init=0.07, clamp=[0.01, 0.5])

   Pay attention to RESEARCH.md Section 3 for design rationale.
   The MoCo queue must be FIFO, on correct device, no gradients.

2. src/training/dataset.py — Training dataset:
   - Load preprocessed samples (mouth_crops.npy, audio.wav, speech_mask.npy)
   - Speaker-disjoint train/val/test splits for FakeAVCeleb
   - Hard negative mining: same-speaker, different-time windows
   - Hard negative ratio annealed 0% → 20% over first 10 epochs
   - Collation with padding + attention masks for variable-length sequences

3. src/training/__init__.py — Module exports

Write a __main__ test in losses.py that creates random tensors and verifies:
(a) InfoNCE loss produces finite values
(b) Gradients flow through projection heads but not frozen encoders
(c) Temperature clamp works correctly
(d) MoCo queue updates correctly (FIFO, correct size)
```

### Task A4: Implement Training Loops

**Tool:** Claude Code
**Prompt:**

```
Read .claude/CLAUDE.md and docs/EXECUTION_PLAN.md.

Implement:

1. src/training/pretrain.py — Phase 1 contrastive pretraining loop:
   - Load visual + audio encoders, freeze pretrained if configured
   - InfoNCE loss only (no classification, no temporal consistency)
   - Cosine LR schedule with warmup
   - Save checkpoints every 5 epochs + best val loss
   - Log per-epoch: loss, lr, τ, mean sync-score on validation subset
   - Save metrics to outputs/logs/pretrain.json
   - Support --config yaml + CLI overrides

2. src/training/finetune.py — Phase 2 fine-tuning loop:
   - Load pretrained checkpoint
   - Combined loss: L_InfoNCE + γ * L_temp + δ * L_cls
   - Hard negative mining with annealing
   - Early stopping (patience=5 epochs on val AUC-ROC)
   - Save checkpoints every 5 epochs + best val AUC-ROC
   - Log per-epoch: all loss components, val AUC-ROC, val EER
   - Save metrics to outputs/logs/finetune.json
   - Support --config yaml + CLI overrides

3. CLI scripts:
   - scripts/train_pretrain.py — Entry point for Phase 1
   - scripts/train_finetune.py — Entry point for Phase 2
```

### Task A5: Implement Evaluation Framework

**Tool:** Claude Code
**Prompt:**

```
Read .claude/CLAUDE.md and docs/BASELINES.md (for expected metric ranges).

Implement:

1. src/evaluation/metrics.py — Metric computation:
   - AUC-ROC (sklearn)
   - EER (scipy interpolation on FPR/TPR)
   - pAUC at FPR < 0.1
   - Per-category AUC for FakeAVCeleb (RV-RA, FV-RA, RV-FA, FV-FA)
   - Results returned as dict, saved as JSON

2. src/evaluation/evaluate.py — Evaluation runner:
   - Load checkpoint + test dataset
   - Run inference, collect predictions + labels + sync-scores
   - Compute all metrics
   - Per-category breakdown for FakeAVCeleb
   - Save results JSON to outputs/logs/

3. src/evaluation/visualize.py — Visualization:
   - Plot s(t) curves for individual clips (real vs fake side-by-side)
   - Sync-score distribution histograms (real vs fake)
   - ROC curves per test set
   - Training loss curves over epochs
   - Ablation comparison bar charts
   - Follow the plotting standards in CLAUDE.md (colors, fonts, DPI)

4. src/evaluation/__init__.py — Module exports

5. scripts/evaluate.py — CLI entry point for evaluation
```

### Task A6: Smoke Test

**Tool:** Claude Code
**Prompt:**

```
Create scripts/smoke_test.sh that:
1. Creates a tiny synthetic dataset (10 random samples with fake mouth crops and audio)
2. Runs pretrain for 2 epochs on the synthetic data (batch_size=2, max_samples=10)
3. Runs finetune for 2 epochs loading the pretrain checkpoint
4. Runs evaluation on the synthetic test set
5. Prints a summary: losses, AUC (will be random on synthetic data), shapes verified

Also create scripts/slurm_smoke_test.sh — a SLURM wrapper that runs smoke_test.sh
on a single H200 GPU (gpu-interactive partition, 1 hour).

This should complete in <5 minutes. I want to verify the full pipeline works
before committing real GPU hours.
```

**Run it:**
```bash
srun --partition=gpu-interactive --nodes=1 --pty \
     --gres=gpu:h200:1 --cpus-per-task=8 --mem=64GB --time=01:00:00 /bin/bash
conda activate syncguard
cd /scratch/$USER/SyncGuard
bash scripts/smoke_test.sh
```

### Task A7: Preprocess Datasets

**Run on HPC (yourself):**

```bash
# Start interactive GPU session for preprocessing
srun --partition=gpu-interactive --nodes=1 --pty \
     --gres=gpu:h200:1 --cpus-per-task=8 --mem=64GB --time=04:00:00 /bin/bash

conda activate syncguard
cd /scratch/$USER/SyncGuard

# Preprocess FakeAVCeleb (primary — takes 1-3 hours)
python scripts/preprocess_dataset.py \
    --dataset fakeavceleb \
    --data_dir data/raw/FakeAVCeleb \
    --config configs/default.yaml

# Preprocess CelebDF-v2 (zero-shot eval)
python scripts/preprocess_dataset.py \
    --dataset celebdf \
    --data_dir data/raw/CelebDF-v2 \
    --config configs/default.yaml

# Quick test with 10 samples first
python scripts/preprocess_dataset.py \
    --dataset fakeavceleb \
    --data_dir data/raw/FakeAVCeleb \
    --max_samples 10
```

---

## Phase 3B — Contrastive Pretraining (Mar 16–21)

### Task B1: Launch Pretraining

**Tool:** Claude Code (to verify config, then submit via SLURM)
**Prompt:**

```
Create a SLURM job script scripts/slurm_pretrain.sh for Phase 1 pretraining:
- Partition: gpu
- GPU: 1x H200
- Time: 12 hours
- Memory: 64GB
- Run: python scripts/train_pretrain.py --config configs/default.yaml

Also verify that configs/default.yaml pretraining settings are correct per EXECUTION_PLAN.md:
20 epochs, batch_size=32, lr=1e-4, cosine schedule, 2-epoch warmup, MoCo queue 4096, τ=0.07.
```

**Submit:**
```bash
cd /scratch/$USER/SyncGuard
mkdir -p outputs/logs
sbatch scripts/slurm_pretrain.sh
squeue -u $USER  # monitor
```

### Task B2: Monitor Pretraining (Daily)

**Check these numbers against BASELINES.md Section 1:**

```bash
# Check latest metrics
python -c "
import json
with open('outputs/logs/pretrain.json') as f:
    metrics = json.load(f)
for m in metrics[-5:]:
    print(f'Epoch {m[\"epoch\"]}: loss={m[\"loss\"]:.3f}, tau={m.get(\"tau\", \"N/A\")}, sync_score={m.get(\"avg_sync_score\", \"N/A\")}')
"
```

**Expected by epoch 10:** Loss < 3.0, mean sync-score on real > 0.5, τ stable around 0.04–0.08.

### Task B3: Implement Evaluation + Visualization (Atharva, in parallel)

**Tool:** Claude Code
**Prompt:**

```
While pretraining runs, implement the remaining evaluation components.
Read docs/BASELINES.md for expected metric ranges and docs/RESEARCH.md for rationale.

1. Verify src/evaluation/evaluate.py handles all 4 test sets correctly
2. Verify src/evaluation/visualize.py generates all required plots per CLAUDE.md plotting standards
3. Create scripts/evaluate.py CLI that accepts --checkpoint and --test_set flags
4. Test the evaluation pipeline with a dummy checkpoint on a tiny dataset
```

---

## Phase 3C — Fine-tuning & Evaluation (Mar 22–30)

### Task C1: Launch Fine-tuning

**Tool:** Claude Code
**Prompt:**

```
Create scripts/slurm_finetune.sh for Phase 2 fine-tuning:
- Partition: gpu
- GPU: 1x H200
- Time: 16 hours
- Memory: 64GB
- Run: python scripts/train_finetune.py --config configs/default.yaml \
       --pretrain_ckpt outputs/checkpoints/pretrain_best.pt

Verify configs/default.yaml finetune settings match EXECUTION_PLAN.md:
30 epochs, batch_size=16, lr=5e-5, γ=0.5, δ=1.0, hard negatives 0%→20% over 10 epochs.
```

### Task C2: Monitor Fine-tuning (Daily)

**Check against BASELINES.md Section 2:**

```bash
python -c "
import json
with open('outputs/logs/finetune.json') as f:
    metrics = json.load(f)
for m in metrics[-5:]:
    print(f'Epoch {m[\"epoch\"]}: loss={m[\"loss_total\"]:.3f}, val_auc={m.get(\"val_auc\", \"N/A\"):.3f}, val_eer={m.get(\"val_eer\", \"N/A\"):.3f}')
"
```

**Expected by epoch 10:** Val AUC > 0.80. By epoch 20: Val AUC > 0.85.

### Task C3: Run All Evaluations

**Tool:** Claude Code
**Prompt:**

```
Run evaluation on all 4 test axes with the best fine-tuned checkpoint.
Check results against BASELINES.md Section 3.

python scripts/evaluate.py --checkpoint outputs/checkpoints/finetune_best.pt \
    --test_set fakeavceleb --config configs/default.yaml
python scripts/evaluate.py --checkpoint outputs/checkpoints/finetune_best.pt \
    --test_set celebdf --config configs/default.yaml
python scripts/evaluate.py --checkpoint outputs/checkpoints/finetune_best.pt \
    --test_set dfdc --config configs/default.yaml
python scripts/evaluate.py --checkpoint outputs/checkpoints/finetune_best.pt \
    --test_set wavlip_adversarial --config configs/default.yaml

After all complete, print a summary table of AUC-ROC, EER, pAUC for each test set.
Generate the ROC curve plots.
```

### Task C4: Run Ablation Studies

**Tool:** Claude Code
**Prompt:**

```
Create scripts/run_ablations.sh that submits all ablation runs to SLURM in parallel.
Each ablation: 15 epochs fine-tuning on FakeAVCeleb, evaluate on test split.

Ablation 1 — Visual encoder (Akshay):
  - AV-HuBERT (already done — use main model results)
  - ResNet-18: override model.visual_encoder.name=resnet18
  - SyncNet: override model.visual_encoder.name=syncnet

Ablation 2 — Wav2Vec layer (Ritik):
  - Layer 3, 5, 7, 9 (done), 11: override model.audio_encoder.layer=N

Ablation 3 — Classifier (Atharva):
  - Bi-LSTM (done), 1D-CNN: override model.classifier.name=cnn1d
  - Statistical: override model.classifier.name=statistical

Ablation 4 — Hard negatives (Ritik):
  - 20% (done), 0%: override training.finetune.hard_negative_ratio=0.0

After all ablations complete, generate bar chart comparisons per CLAUDE.md plotting standards.
Save results to outputs/logs/ablation_results.json.
```

### Task C5: Generate All Visualizations

**Tool:** Claude Code
**Prompt:**

```
Generate all publication-quality visualizations per CLAUDE.md plotting standards.

1. s(t) curves: pick 3 real + 3 fake clips from FakeAVCeleb test, 2 from CelebDF-v2, 2 from DFDC
   - Show real (green) vs fake (red) side-by-side
   - Mark phoneme boundaries if MFA timestamps available

2. Sync-score distribution histogram: real vs fake overlaid
3. ROC curves: one per test set, all overlaid on one plot
4. Training curves: pretraining loss, fine-tuning loss components, val AUC over epochs
5. Ablation bar charts: one per ablation study
6. Per-category AUC breakdown for FakeAVCeleb

Save all to outputs/visualizations/ as PNG (300 DPI) + PDF.
Use the exact color palette from CLAUDE.md.
```

---

## Phase 4 & 5 — Packaging (Mar 30 – Apr 13)

### Task D1: Update Lab Notebook with Final Results

**Tool:** Claude Code
**Prompt:**

```
Read all results JSON files in outputs/logs/.
Update docs/lab_notebook.md with a final summary entry:
- All metric tables (main results + ablations)
- Which hypotheses were confirmed
- Key findings in 3-5 sentences
- What worked, what didn't, what surprised you
```

### Task D2: Write Report Sections

**Manual task (team effort):**
- Use actual numbers from `outputs/logs/` JSON files
- Reference plots from `outputs/visualizations/`
- Follow IEEE conference format, 6–8 pages
- See EXECUTION_PLAN.md Phase 5 for section assignments

### Task D3: Create Demo Script

**Tool:** Claude Code
**Prompt:**

```
Create scripts/demo.py — a lightweight CLI demo:
- Input: path to video file
- Output: Real/Fake prediction, confidence score, s(t) curve saved as PNG
- Target: <10 seconds for a 30-second clip on CPU
- Load best fine-tuned checkpoint
- Run full pipeline: face detection → mouth crop → audio extraction → inference
- Save sync-score visualization next to input file

python scripts/demo.py --video path/to/video.mp4 \
    --checkpoint outputs/checkpoints/finetune_best.pt
```

### Task D4: Final README Polish

**Tool:** Claude Code
**Prompt:**

```
Polish README.md for public release:
- One-sentence problem statement
- Architecture diagram (ASCII or reference image)
- Results summary table (main + zero-shot)
- Key visualization (s(t) curves or reference plot)
- Setup and reproduction instructions
- Ablation summary
- Citation-ready reference list
- Team and course info
```

---

## Quick Reference: Commands

```bash
# Environment
conda activate syncguard

# Preprocessing
python scripts/preprocess_dataset.py --dataset fakeavceleb --config configs/default.yaml

# Phase 1: Pretrain
python scripts/train_pretrain.py --config configs/default.yaml

# Phase 2: Fine-tune
python scripts/train_finetune.py --config configs/default.yaml \
    --pretrain_ckpt outputs/checkpoints/pretrain_best.pt

# Evaluate
python scripts/evaluate.py --checkpoint outputs/checkpoints/finetune_best.pt \
    --test_set fakeavceleb

# Demo
python scripts/demo.py --video sample.mp4 --checkpoint outputs/checkpoints/finetune_best.pt

# SLURM
sbatch scripts/slurm_pretrain.sh          # Submit pretraining
sbatch scripts/slurm_finetune.sh          # Submit fine-tuning
squeue -u $USER                           # Check job queue
scancel <job_id>                           # Cancel job
tail -f outputs/logs/slurm_<job_id>.out   # Watch live output
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| OOM during pretraining | Batch size too large for GPU | Reduce `batch_size` 32→16→8, enable gradient accumulation |
| AV-HuBERT import fails | fairseq version conflict | Pin `fairseq==0.12.3`, install from source if needed |
| Sync-scores all ~0.0 | L2 normalization missing or wrong | Verify `F.normalize(x, dim=-1)` on both embeddings |
| Val AUC stuck at 0.50 | Labels are wrong or shuffled | Check dataset loader label assignment |
| Loss is NaN | Temperature → 0 or log(0) | Check τ clamp, add epsilon to log operations |
| SLURM job killed (OOM) | Exceeded --mem request | Increase `--mem` or reduce batch size |
| SLURM job killed (time) | Exceeded --time request | Increase time limit or save checkpoints more frequently |
| Preprocessing crashes on some videos | Corrupted video or no face detected | Add try/except, log skipped files, continue |
| Wav2Vec outputs wrong shape | Not using `output_hidden_states=True` | Check model call and layer indexing |
| Speaker leakage in splits | Same speaker in train and test | Verify split script groups by speaker ID |
