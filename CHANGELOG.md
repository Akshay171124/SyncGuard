# Changelog

All notable changes to SyncGuard will be documented in this file.

## [0.6.0] - 2026-03-18

### Added
- Evaluation framework (`src/evaluation/`):
  - `metrics.py` — AUC-ROC (sklearn), EER (from ROC curve), pAUC at FPR<0.1 and FPR<0.05, per-category FakeAVCeleb breakdown (RV-RA vs each fake category), bootstrapped 95% CI
  - `evaluate.py` — Inference runner: loads checkpoint, runs model on test DataLoaders, collects predictions + sync-score statistics, saves JSON results + .npz predictions per test set
  - `visualize.py` — Publication-quality plots (300 DPI PNG + PDF): ROC curves (single, multi-dataset, per-category), sync-score temporal profiles (real vs fake), sync-score distribution histograms, training loss curves (pretrain + finetune), ablation bar charts. Consistent color palette per plotting standards
  - `__init__.py` — Full module exports
- All tested on CPU with synthetic data — metrics and all 7 plot types verified

---

## [0.5.0] - 2026-03-15

### Added
- Training loops (`src/training/pretrain.py`, `src/training/finetune.py`):
  - Phase 1 contrastive pretraining: InfoNCE loss only, cosine LR with warmup, checkpoint saving (periodic + best val loss), resume support
  - Phase 2 fine-tuning: combined loss (InfoNCE + temporal + BCE), hard negative annealing (0%→20% over 10 epochs), early stopping (patience=5 on val AUC), AUC-ROC and EER computation
  - Both loops: gradient clipping (max_norm=1.0), per-epoch JSON logging, full state checkpointing (model + optimizer + scheduler + criterion)
- CLI scripts: `scripts/train_pretrain.py`, `scripts/train_finetune.py`
- Updated `src/training/__init__.py` with training loop exports

### Fixed
- Wav2Vec 2.0 NaN in train mode: frozen backbone now forced to eval mode to prevent group normalization NaN on zero-padded waveforms
- Cosine LR scheduler ZeroDivisionError when warmup_steps >= total_steps (T_max clamped to >= 1)

---

## [0.4.0] - 2026-03-14

### Added
- Training loss functions (`src/training/losses.py`):
  - `MoCoQueue` — FIFO memory bank for contrastive negatives (size 4096, no gradients)
  - `InfoNCELoss` — Frame-level InfoNCE with MoCo negatives and learnable temperature τ (clamped [0.01, 0.5])
  - `TemporalConsistencyLoss` — L2 on first derivatives of embeddings, real-only via `is_real` mask
  - `CombinedLoss` — L_InfoNCE + γ*L_temp + δ*L_cls with component-wise logging
  - `PretrainLoss` — InfoNCE-only wrapper for Phase 1
  - Factory functions: `build_pretrain_loss()`, `build_finetune_loss()`
- Training dataset (`src/training/dataset.py`):
  - `SyncGuardDataset` — Loads preprocessed mouth crops + audio, supports variable-length clips
  - `SyncGuardBatch` — Collated batch dataclass with padding masks
  - `collate_syncguard()` — Custom collation with padding and boolean masks
  - `build_dataloaders()` — Factory for speaker-disjoint train/val/test DataLoaders
  - Hard negative mining via same-speaker different-clip selection
- `src/training/__init__.py` — Module exports for all loss and dataset classes
- All tested on CPU with synthetic data — all assertions pass

---

## [0.3.0] - 2026-03-14

### Added
- Project documentation framework:
  - `docs/BASELINES.md` — Expected metric ranges, sanity checks, red flags, pre/post experiment checklists
  - `docs/RESEARCH.md` — Technical foundations and design rationale for all architecture choices
  - `docs/OPERATIONS.md` — Step-by-step operational guide with HPC commands and Claude Code prompts
  - `docs/lab_notebook.md` — Ongoing experiment journal
- Enhanced `.claude/CLAUDE.md` with HPC/SLURM details, results storage convention, experiment log template, plotting standards, 10 common pitfalls

---

## [0.2.0] - 2026-03-14

### Added
- Model architectures (`src/models/`):
  - `visual_encoder.py` — AV-HuBERT (3D frontend + ResNet trunk), ResNet-18, SyncNet; all with projection heads and L2 normalization
  - `audio_encoder.py` — Wav2Vec 2.0 wrapper with configurable hidden layer extraction and frozen backbone
  - `classifier.py` — Bi-LSTM (primary), 1D-CNN (ablation), Statistical baseline (ablation); all with masked pooling for variable-length sequences
  - `syncguard.py` — Full model integration with sequence alignment, SyncGuardOutput dataclass
  - `__init__.py` — Module exports and factory functions
- All models tested on CPU with `__main__` blocks: shape verification, L2 normalization, gradient flow, freeze mechanism

---

## [0.1.1] - 2026-03-11

### Added
- `docs/Final_Project_Proposal.md` — Complete finalized project proposal with all decided architecture, datasets, evaluation plan, schedule, and references

---

## [0.1.0] - 2026-03-11

### Added
- Initial project structure with modular directory layout
- GitHub repository created (https://github.com/Akshay171124/SyncGuard)
- Configuration system (`configs/default.yaml`) with all hyperparameters
- `.gitignore` for Python, data, checkpoints, and IDE files
- Preprocessing pipeline:
  - `src/preprocessing/face_detector.py` — RetinaFace + MediaPipe mouth-ROI extraction
  - `src/preprocessing/audio_extractor.py` — Audio extraction, resampling, Wav2Vec feature prep
  - `src/preprocessing/vad.py` — Silero-VAD speech gating
  - `src/preprocessing/dataset_loader.py` — FakeAVCeleb and CelebDF-v2 dataset loaders
  - `src/preprocessing/pipeline.py` — End-to-end preprocessing orchestrator
- Utility modules:
  - `src/utils/config.py` — YAML config loader
  - `src/utils/io.py` — Video/audio I/O helpers
- `requirements.txt` with all dependencies
- `README.md` with setup and usage instructions
- `scripts/preprocess_dataset.py` — CLI script to run preprocessing on datasets
