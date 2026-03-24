# Changelog

All notable changes to SyncGuard will be documented in this file.

## [2.1.0] - 2026-03-23

### Fixed
- **MediaPipe Tasks API migration:** Rewrote `face_detector.py` from deprecated `mp.solutions.face_mesh.FaceMesh` to `mp.tasks.vision.FaceLandmarker`. Required because mediapipe 0.10.33 + protobuf 7.x removed the `solutions` API entirely.
- **EGL segfault on GPU nodes:** MediaPipe Tasks API aggressively initializes EGL/OpenGL even with `Delegate.CPU`. Fix: `export __EGL_VENDOR_LIBRARY_DIRS=/dev/null` blocks vendor library loading.
- **LRS2 unique ID collision:** Speakers share filenames (00001.mp4, etc.) causing only 225/96K samples to be processed. Fix: `unique_id = f"{speaker_id}_{video_stem}"` for LRS2 dataset in both `pipeline.py` and `dataset.py`.
- **LRS2 config path:** `lrs2_dir` corrected to `data/raw/LRS2/mvlrs_v1/pretrain`

### Added
- **Multiprocessing for preprocessing:** `mp.Pool`-based parallel processing with per-worker pipeline instances. 15x speedup (~190 samples/min with 14 workers vs ~13/min single-threaded).
- `--workers` CLI argument in `preprocess_dataset.py`
- New SLURM training scripts: `slurm_pretrain.sh` (Phase 1 CMP, H200, auto-resubmit), `slurm_finetune.sh` (Phase 2 EAR, H200, auto-resubmit)

### Completed (HPC)
- **EAR extraction:** FakeAVCeleb (19,725 samples) + DFDC (1,334 samples) — all complete
- **LRS2 transfer:** ~50 GB, 144K videos transferred to HPC
- **LRS2 preprocessing:** In progress (~18,453/96,318 done, auto-resuming after job preemption)

---

## [2.0.0] - 2026-03-22

### Added
- **Cross-modal prediction pretraining (AVFF-style):** `CrossModalPredictionLoss` — masks 30% of frames, predicts across modalities via MLP. Combined with InfoNCE: L = L_InfoNCE + 0.5 * L_CMP
- **EAR (Eye Aspect Ratio) blink features:** Per-frame blink detection from MediaPipe eye landmarks (6 per eye). Extracted during preprocessing, fed to BiLSTM classifier as 2nd input channel
- **LRS2 dataset loader** (`LRS2Loader`) — real speech videos for expanded pretraining and extra real samples in fine-tuning
- **EAR-only extraction script** (`scripts/extract_ear_features.py`) — adds EAR to existing preprocessed data without full re-run
- **New SLURM scripts:** `slurm_preprocess_lrs2.sh`, `slurm_extract_ear.sh`, `slurm_train_pretrain_cmp.sh`, `slurm_train_finetune_ear.sh`

### Changed
- `PretrainLoss` now combines InfoNCE + cross-modal prediction (configurable via `cross_modal_prediction` flag)
- `BiLSTMClassifier` accepts `use_ear=True` to expand input from 1D (sync-scores) to 2D (sync-scores + EAR)
- `SyncGuard.forward()` accepts optional `ear_features` parameter
- `build_dataloaders()` Phase 1 loads AVSpeech + LRS2; Phase 2 adds LRS2 reals to FakeAVCeleb
- Preprocessing pipeline now saves `ear_features.npy` alongside mouth crops
- `preprocess_dataset.py` supports `lrs2` dataset

### Config
- `training.pretrain.cross_modal_prediction: true`
- `training.pretrain.cmp_weight: 0.5`
- `training.pretrain.cmp_mask_ratio: 0.3`
- `model.classifier.use_ear: true`
- `data.lrs2_dir: "data/raw/LRS2"`

---

## [1.2.0] - 2026-03-22

### Attempted
- **CelebDF-v2 cross-dataset evaluation** — preprocessed 921 clips, discovered entire dataset has no audio streams. Incompatible with AV sync-based methods. Dropped from evaluation.
- **DFDC cross-dataset evaluation** — downloaded Part 0 (1,334 clips), preprocessed, ran cascade evaluation with 5 strategies. All at random chance (best AUC: 0.5712 sync_only). Root cause: DFDC face-swaps preserve lip-sync, so sync-scores don't discriminate.
- **Raw sync-score thresholding** on DFDC — AUC 0.4378, confirming the encoder representations themselves don't generalize, not just the classifier.

### Added
- DFDC dataset loader (`src/preprocessing/dataset_loader.py`) and preprocessing SLURM script
- `raw_sync_score` evaluation strategy in `scripts/evaluate_cascade.py` — uses mean sync-score directly without Bi-LSTM

### Fixed
- protobuf version conflict on HPC: kaggle pulled protobuf 7.x breaking mediapipe. Pinned to `protobuf<5` (4.25.8)

### Results — DFDC Cross-Dataset (Zero-Shot)
| Strategy | AUC | EER |
|----------|-----|-----|
| sync_only | 0.5712 | 0.4535 |
| audio_only | 0.4857 | 0.5084 |
| max_fusion | 0.4960 | 0.5120 |
| avg_fusion | 0.5378 | 0.4649 |
| raw_sync_score | 0.4378 | 0.5563 |

### Next Steps
- Implement cross-modal prediction pretraining (AVFF-style) to learn deeper AV correspondence
- Add blink rate / EAR features for face-swap detection
- Retrain on expanded data (LRS2 + AVSpeech)

---

## [1.1.0] - 2026-03-20

### Added
- **Standalone audio classifier** (`src/models/audio_classifier.py`) — Wav2Vec2 (frozen, layer 9) → mean+max pool → MLP for detecting audio deepfakes (TTS, voice cloning)
- Training script for audio classifier (`scripts/train_audio_classifier.py`) with per-category AUC logging
- **Cascade evaluation** (`scripts/evaluate_cascade.py`) — runs SyncGuard + audio classifier, compares 4 fusion strategies (sync-only, audio-only, max, avg)
- SLURM scripts: `slurm_train_audio_clf.sh`, `slurm_evaluate_cascade.sh`

### Results
- Audio classifier standalone: val_auc=0.8909 (30 epochs, 426K trainable params)
- **Max-fusion cascade (final system):**
  - Overall AUC: **0.9458** (↑ from 0.9254 sync-only)
  - RV-FA AUC: **0.9278** (↑ from 0.5070 — fixed random-chance detection)
  - FV-FA AUC: **0.9902** (↑ from 0.9528)
  - pAUC@0.1: **0.7378** (↑ from 0.6097)
  - EER: **0.1445** (↓ from 0.1481)

---

## [1.0.0] - 2026-03-20

### Added
- **Phase 2 fine-tuning** on FakeAVCeleb — 4 experiment runs
- Audio-swap augmentation (`src/training/dataset.py`) — replaces 15% of fake samples with synthetic RV-FA (video from one real clip + audio from another)
- **AudioClassifier** head (`src/models/classifier.py`) for RV-FA detection via audio embeddings
- Dual-head architecture in `src/models/syncguard.py` — sync head + audio head with learnable fusion weight
- `audio_logits` support in `CombinedLoss` (`src/training/losses.py`)
- Evaluation CLI wrapper (`scripts/evaluate.py`) and SLURM script (`scripts/slurm_evaluate.sh`)

### Fixed
- `np.trapezoid` AttributeError on NumPy <1.25 — added fallback to `np.trapz`
- `evaluate.py` dataset loading: uses `build_dataloaders(phase="finetune")` instead of unsupported `phase="test"`

### Results
- **Run 1** (no augmentation): AUC=0.9112, EER=0.1726, RV-FA=0.5641
- **Run 2** (audio-swap on reals — bug): AUC collapsed to ~0.50, discarded
- **Run 3** (audio-swap on fakes, 15%): AUC=0.9254, EER=0.1481, RV-FA=0.5070 — **best sync-only model**
- **Run 4** (dual-head + fusion): AUC=0.5542, early stopped — learnable fusion destroyed sync signal, abandoned

### Lessons Learned
- Audio-swap augmentation helps overall metrics but cannot fix RV-FA — architectural limitation of sync-score approach
- Logit-level fusion of trained + untrained heads degrades both — never mix randomly initialized outputs with trained outputs
- RV-FA in FakeAVCeleb uses same-content voice cloning, not content-mismatched dubbing — phoneme-viseme mismatch signal is absent

---

## [0.9.0] - 2026-03-20

### Added
- Phase 1 contrastive pretraining completed — two runs compared (fixed τ vs learnable τ)
- A100 SLURM script (`scripts/slurm_train_pretrain_a100.sh`) for gpu-interactive partition
- Experiment comparison report (`outputs/logs/experiment_pretrain_comparison.md`)

### Fixed
- **Temperature τ not being optimized (critical bug):** `build_optimizer()` in `pretrain.py` only included `model.parameters()`, missing `criterion.parameters()` where the learnable `log_temperature` lives. Fixed by passing criterion to optimizer builder.

### Results
- **Run 2 (learnable τ) selected as winner** for Phase 2
  - Best val loss: 8.2561 (vs 8.2990 fixed τ)
  - Sync score: 0.7063 (vs 0.7005)
  - No overfitting (train-val gap 0.028 vs 0.071)
  - τ learned: 0.07 → 0.041
- Winner checkpoint: `outputs/checkpoints/pretrain_best.pt` (epoch 17)

---

## [0.8.0] - 2026-03-19

### Added
- Weights & Biases (wandb) integration in both training loops (`pretrain.py`, `finetune.py`)
  - Pretrain: logs train/val loss, sync-score, temperature, lr per epoch
  - Finetune: logs all loss components (infonce, temporal, cls), val AUC, val EER, hard negative ratio per epoch
  - Project: `SyncGuard`, runs tagged by phase
- Phase 1 contrastive pretraining SLURM script (`scripts/slurm_train_pretrain.sh`) — H200 GPU, 8hr limit, auto-resubmit with checkpoint resume
- Phase 2 fine-tuning SLURM script (`scripts/slurm_train_finetune.sh`) — ready to submit after pretraining
- AVSpeech support in `build_dataloaders()` for pretraining phase (85/15 train/val split)
- `avspeech_dir` config entry in `configs/default.yaml`

### Fixed
- `features_dir` in config pointed to `data/features` instead of `data/processed` — training could not find preprocessed data
- `_get_feature_path()` now tries pipeline output format (`dataset/category/video_stem`) first, matching actual preprocessing output
- `build_dataloaders()` no longer hardcoded to FakeAVCeleb — selects dataset based on phase (pretrain=AVSpeech, finetune=FakeAVCeleb)
- `_load_mouth_crops()` crashed on RGB data `(T, 96, 96, 3)` — added RGB→grayscale conversion before channel-dim expansion (caught via CPU dry run)

---

## [0.7.1] - 2026-03-19

### Added
- Full FakeAVCeleb dataset (21,544 clips, all 4 categories) uploaded to HPC
- Auto-resubmitting SLURM script for FakeAVCeleb preprocessing (`scripts/slurm_preprocess_fakeavceleb.sh`)

### Changed
- FakeAVCeleb preprocessing now covers all categories (RV-RA, FV-RA, RV-FA, FV-FA), not just FV-FA

---

## [0.7.0] - 2026-03-18

### Added
- AVSpeech dataset loader (`src/preprocessing/dataset_loader.py`) for pretraining data (flat folder of real speech clips)
- Resume support in preprocessing pipeline — skips already-processed samples via metadata.json check
- GPU smoke test script (`scripts/gpu_smoke_test.py`) — verifies model load, forward/backward pass, no NaN, GPU memory
- Auto-resubmitting SLURM script (`scripts/slurm_preprocess_avspeech.sh`) for long preprocessing jobs
- HPC environment fully provisioned on Northeastern Explorer cluster

### Fixed
- Speaker ID extraction for nested FakeAVCeleb directory structure (ethnicity/gender/speaker_id)
- Module import path in GPU smoke test (sys.path.insert for project root)
- MediaPipe/TensorFlow/protobuf dependency compatibility on HPC (mediapipe 0.10.14 + TF 2.16.2)

---

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
