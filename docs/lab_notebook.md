# SyncGuard — Lab Notebook

> Ongoing journal of experiments, observations, and decisions.
> Append new entries at the bottom. Never delete old entries — they're the history.
> Each entry should be dated and attributed to the team member who ran the experiment.

---

## How to Use This Notebook

After every significant action (training run, evaluation, debugging session), add an entry:

```markdown
## [Date] — [Short Title]
**Owner:** [Name]
**Phase:** [Pretrain / Finetune / Evaluation / Ablation / Debug]

### What I Did
[1-3 sentences]

### Results
[Key numbers — loss, AUC, sync-score, or "N/A" if no metrics]

### Observations
[What surprised me? What matched BASELINES.md expectations?]

### Decision
[What I'll do next based on these results]

### Artifacts
[Paths to checkpoints, JSONs, plots generated]
```

---

## Mar 11, 2026 — Project Setup

**Owner:** Akshay
**Phase:** Setup

### What I Did
- Set up project structure: src/, configs/, scripts/, docs/
- Implemented preprocessing pipeline: RetinaFace face detection, MediaPipe mouth-ROI extraction, Silero-VAD, audio extraction
- Created `configs/default.yaml` with all hyperparameters
- Downloaded 20 AVSpeech sample clips for testing
- Finalized project proposal

### Results
- Preprocessing pipeline tested on 20 AVSpeech clips — all passed
- Face detection confidence threshold 0.8 works well on frontal talking-head videos
- Temporal alignment (25fps → 49Hz) produces matching sequence lengths

### Observations
- RetinaFace occasionally fails on low-resolution or profile faces — confidence threshold of 0.8 handles this by skipping
- Silero-VAD correctly identifies speech/non-speech segments in test clips
- Preprocessing takes ~2-3 seconds per 10-second clip on CPU

### Decision
- Proceed with data procurement (FakeAVCeleb, CelebDF-v2, VoxCeleb2)
- Next: implement model architectures (visual encoder, audio encoder, classifier)

### Artifacts
- `configs/default.yaml` — finalized config
- `data/raw/AVSpeech/clips/` — 20 test clips
- `docs/Final_Project_Proposal.md` — complete proposal

---

## Mar 14, 2026 — Documentation & Operational Framework

**Owner:** Akshay
**Phase:** Setup

### What I Did
- Created comprehensive project documentation:
  - Enhanced `.claude/CLAUDE.md` with HPC/SLURM details, results convention, experiment templates, plotting standards, common pitfalls
  - Created `docs/BASELINES.md` — expected metric ranges for every phase, sanity checks, red flags
  - Created `docs/RESEARCH.md` — technical foundations and design rationale for all architecture choices
  - Created `docs/OPERATIONS.md` — step-by-step operational guide with exact commands and prompts
  - Created `docs/lab_notebook.md` — this journal

### Results
N/A — documentation phase, no metrics

### Observations
- Having expected numerical ranges documented upfront (BASELINES.md) will save significant debugging time during experiments
- The operational guide maps directly to EXECUTION_PLAN.md phases but adds the "how" (exact commands, SLURM scripts, Claude Code prompts)

### Decision
- Ready for model implementation (Task A2 in OPERATIONS.md)
- Priority: visual encoder → audio encoder → classifier → integration → loss functions → training loops

### Artifacts
- `.claude/CLAUDE.md` — enhanced with HPC, results convention, plotting standards
- `docs/BASELINES.md` — expected metrics and sanity checks
- `docs/RESEARCH.md` — technical rationale
- `docs/OPERATIONS.md` — operational guide
- `docs/lab_notebook.md` — this file

---

## Mar 14, 2026 — Model Architectures Implemented (Task A2)

**Owner:** Akshay
**Phase:** Setup / Implementation

### What I Did
- Implemented all model architectures in `src/models/`:
  - **Visual encoder** (`visual_encoder.py`): AV-HuBERT (3D conv frontend + ResNet-18 trunk + projection head), ResNet-18 variant (ablation), SyncNet variant (ablation). All output (B, T, 256) L2-normalized embeddings. Supports loading pretrained AV-HuBERT weights from fairseq checkpoint and `freeze_pretrained` flag.
  - **Audio encoder** (`audio_encoder.py`): Wav2Vec 2.0 wrapper via HuggingFace transformers. Configurable hidden layer extraction (default layer 9 per Pasad et al. 2021). Frozen backbone by default, only projection head trains.
  - **Classifier** (`classifier.py`): Bi-LSTM (primary), 1D-CNN (ablation), Statistical baseline (ablation). All support variable-length inputs via packed sequences and masked pooling.
  - **SyncGuard** (`syncguard.py`): Full model integration. Handles temporal alignment between visual (25fps upsampled to 49Hz) and audio (native 49Hz) sequences. Returns `SyncGuardOutput` dataclass with logits, sync_scores, v_embeds, a_embeds.
  - **Factory functions**: `build_visual_encoder()`, `build_audio_encoder()`, `build_classifier()`, `build_syncguard()` — all driven by `configs/default.yaml`.
- Ran all `__main__` tests on CPU — all passed.

### Results
- **Total parameters:** 107,401,217 (107M)
- **Trainable parameters:** 13,029,505 (13M) — Wav2Vec frozen, visual backbone + projection heads + classifier trainable
- **Shape verification:** mouth_crops (2, 50, 1, 96, 96) → v_embeds (2, 50, 256), waveform (2, 32000) → a_embeds (2, 50, 256) → sync_scores (2, 50) → logits (2, 1)
- **L2 normalization:** Confirmed on both embedding streams
- **Gradient flow:** Confirmed through projection heads and classifier; frozen backbone has no gradients
- **Freeze mechanism:** Works correctly — backbone params frozen, projection heads trainable

### Observations
- Wav2Vec 2.0 base model downloads to `~/.cache/huggingface/` (~360MB). Will need to set `HF_HOME` on HPC to `/scratch/`.
- The `masked_spec_embed` warning from Wav2Vec is expected — we're not using masked prediction, just feature extraction.
- Sequence alignment (truncate to min length) works cleanly — off-by-one differences between visual and audio sequences are handled.
- All three visual encoder variants produce identical output shapes, making ablation swaps trivial via config.
- AV-HuBERT weight loading is implemented but untested (no checkpoint file locally). Will test on HPC when checkpoint is available.

### Decision
- Model architectures are complete. Next: Task A3 — loss functions (`src/training/losses.py`) and training dataset (`src/training/dataset.py`).
- Datasets are on Google Drive, will transfer to HPC for training. No HPC needed until training phase.

### Artifacts
- `src/models/visual_encoder.py` — 3 visual encoder variants + factory
- `src/models/audio_encoder.py` — Wav2Vec 2.0 wrapper + factory
- `src/models/classifier.py` — 3 classifier variants + factory
- `src/models/syncguard.py` — full model integration + SyncGuardOutput dataclass
- `src/models/__init__.py` — module exports

---

## Mar 14, 2026 — Loss Functions & Training Dataset (Task A3)

**Owner:** Akshay
**Phase:** Implementation

### What I Did
- Implemented all loss functions in `src/training/losses.py`: MoCoQueue (FIFO memory bank), InfoNCELoss (frame-level with MoCo negatives, learnable τ), TemporalConsistencyLoss (L2 on first derivatives, real-only), CombinedLoss (weighted sum of all three), PretrainLoss (InfoNCE-only wrapper).
- Implemented training dataset in `src/training/dataset.py`: SyncGuardDataset (loads preprocessed features), SyncGuardBatch (collated dataclass), collate_syncguard (variable-length padding with masks), build_dataloaders (speaker-disjoint splits), hard negative mining (same-speaker different-clip).
- Fixed MoCo queue test assertion — queue is capped at `queue_size`, not `B*T`.
- Updated `src/training/__init__.py` with all exports.

### Results
- **InfoNCE loss:** 5.64 (random embeddings, τ=0.07, queue_size=128) — matches expected range for random inputs
- **Temporal consistency:** 4.04 (random embeddings) — non-zero for random, exactly 0.0 for all-fake batch ✓
- **Combined loss:** total=8.40 (nce=5.64 + 0.5*temp=2.02 + cls=0.74) — component weights verified ✓
- **Temperature gradient:** Confirmed learnable τ receives gradients ✓
- **Queue no-grad:** Confirmed MoCo queue does not require gradients ✓
- **Dataset:** 12 synthetic samples loaded, variable-length collation correct, padding masks verified ✓
- **Hard negative mining:** Same-speaker different-clip selection working ✓

### Observations
- MoCo queue correctly caps at `queue_size` when batch exceeds capacity — important to test with realistic queue sizes (4096) during training
- InfoNCE loss ~5.6 for random embeddings with 128 negatives is reasonable (log(128) ≈ 4.85)
- Temporal consistency loss correctly returns 0.0 when `is_real` mask is all-False
- Variable-length collation pads to max length in batch — memory efficient vs global max padding

### Decision
- Task A3 complete. Next: Task A4 — training loops (`src/training/pretrain.py` and `src/training/finetune.py`)
- Training loops will wire together the model, losses, dataset, optimizer, scheduler, and checkpointing

### Artifacts
- `src/training/losses.py` — All loss functions + factory functions
- `src/training/dataset.py` — Dataset + collation + DataLoader factory
- `src/training/__init__.py` — Module exports

---

## Mar 15, 2026 — Training Loops Implemented (Task A4)

**Owner:** Akshay
**Phase:** Implementation

### What I Did
- Implemented Phase 1 contrastive pretraining loop (`src/training/pretrain.py`): encodes visual + audio embeddings, computes InfoNCE loss with MoCo queue, cosine LR with linear warmup, periodic + best-val-loss checkpointing, resume support, per-epoch JSON logging.
- Implemented Phase 2 fine-tuning loop (`src/training/finetune.py`): full forward pass through SyncGuard model, combined loss (InfoNCE + γ·L_temp + δ·L_cls), hard negative annealing (0%→20% over 10 epochs), early stopping (patience=5 on val AUC-ROC), AUC-ROC and EER computation, resume support.
- Created CLI entry points: `scripts/train_pretrain.py`, `scripts/train_finetune.py`.
- Fixed critical bug: Wav2Vec 2.0 frozen backbone produces NaN in train mode due to group normalization on zero-padded waveforms. Fix: force frozen backbone to eval mode during forward pass.
- Fixed scheduler ZeroDivisionError when warmup_steps >= total_steps.

### Results
- **Pretraining test (2 epochs, CPU):** loss 5.02→4.87 (decreasing ✓), sync_score -0.03→0.18 (increasing ✓), τ=0.07 (stable ✓)
- **Fine-tuning test (2 epochs, CPU):** loss 5.89→6.41, val_auc=0.0 (expected with random data), all loss components finite ✓
- **No NaN** after audio encoder fix ✓
- **Gradient clipping** (max_norm=1.0) applied ✓
- **Checkpoint saving** includes full state: model, optimizer, scheduler, criterion (MoCo queue + temperature) ✓

### Observations
- Wav2Vec 2.0 group normalization layers produce NaN when processing zero-padded regions in train mode. This is a known issue — frozen backbones must be kept in eval mode. This would have been a painful bug to debug during actual training on HPC.
- Pretraining loss ~5.0 for random data with MoCo queue is reasonable (log(4096) ≈ 8.3, but queue is partially filled)
- Both loops handle variable-length batches correctly through the collation + mask pipeline

### Decision
- Task A4 complete. Next: Task A5 — evaluation framework (`src/evaluation/metrics.py`, `evaluate.py`, `visualize.py`)

### Artifacts
- `src/training/pretrain.py` — Phase 1 pretraining loop
- `src/training/finetune.py` — Phase 2 fine-tuning loop
- `scripts/train_pretrain.py` — CLI entry point
- `scripts/train_finetune.py` — CLI entry point
- `src/models/audio_encoder.py` — Fixed frozen backbone eval mode

---

<!-- ADD NEW ENTRIES BELOW THIS LINE -->
