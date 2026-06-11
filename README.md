# SyncGuard

**Contrastive Audio-Visual Deepfake Detection via Temporal Phoneme-Face Coherence**

CS 5330 Computer Vision & Pattern Recognition — Northeastern University, Khoury College of Computer Sciences

## Team

- **Akshay Prajapati** — Visual Encoder, Preprocessing, Integration Lead (prajapati.aksh@northeastern.edu)
- **Ritik Mahyavanshi** — Audio Encoder, Contrastive Pretraining (mahyavanshi.r@northeastern.edu)
- **Atharva Dhumal** — Temporal Classifier, Evaluation (dhumal.a@northeastern.edu)

## Overview

SyncGuard detects deepfake videos by measuring the temporal coherence between speech audio and facial motion. The system uses a two-stream contrastive learning architecture:

1. **Visual Encoder** (AV-HuBERT) extracts frame-level lip embeddings from mouth-ROI crops
2. **Audio Encoder** (Wav2Vec 2.0, frozen) extracts frame-level speech embeddings
3. **Sync-Score** `s(t) = cos(v_t, a_t)` measures per-frame audio-visual alignment
4. **Bi-LSTM Classifier** detects temporal dip patterns in sync-scores to classify real vs fake
5. **EAR Features** (Eye Aspect Ratio) detect unnatural blink patterns in face-swaps

### Training Pipeline

- **Phase 1 — Contrastive Pretraining:** InfoNCE + Cross-Modal Prediction (AVFF-style) on AVSpeech + LRS2 (~117K real clips)
- **Phase 2 — Fine-tuning:** Combined loss (InfoNCE + temporal consistency + BCE) on FakeAVCeleb with EAR features and hard negative mining

### Current Results (v3.2.0)

| Dataset | Model | AUC | EER | pAUC@0.1 |
|---------|-------|-----|-----|----------|
| FakeAVCeleb (in-domain) | v4+CA fused | **0.9628** | **0.0931** | **0.8607** |
| DFDC (zero-shot) | CA Stage 1+2 | 0.5263 | 0.4911 | 0.0644 |

Per-category AUC on FakeAVCeleb (v4+CA, epoch 17):
- FV-RA (face-swap, real audio): 0.9398
- RV-FA (real video, fake audio): **0.8949**
- FV-FA (both swapped): **0.9872**

DFDC cross-dataset generalization remains an open challenge — face-swaps that preserve lip-sync defeat AV correspondence signals. BN adaptation and frequency-domain features did not overcome this fundamental domain gap. See `docs/superpowers/specs/review-findings.md` for detailed analysis.

## Architecture

```
                    ┌─────────────────┐     ┌─────────────────┐
                    │  Mouth Crops     │     │  Raw Audio       │
                    │  (T, 1, 96, 96) │     │  (16kHz waveform)│
                    └────────┬────────┘     └────────┬────────┘
                             │                       │
                    ┌────────▼────────┐     ┌────────▼────────┐
                    │  AV-HuBERT      │     │  Wav2Vec 2.0    │
                    │  Visual Encoder  │     │  Audio Encoder   │
                    │  → (B, T, 256)  │     │  → (B, T, 256)  │
                    └────────┬────────┘     └────────┬────────┘
                             │                       │
                             └──────────┬────────────┘
                                        │
                              s(t) = cos(v_t, a_t)
                                        │
                             ┌──────────▼──────────┐
                             │   Bi-LSTM Classifier │
                             │   Input: [s(t), EAR] │
                             │   → Real / Fake      │
                             └─────────────────────┘
```

## Project Structure

```
SyncGuard/
├── configs/
│   └── default.yaml              # All hyperparameters and data paths
├── src/
│   ├── preprocessing/
│   │   ├── face_detector.py      # MediaPipe FaceLandmarker + RetinaFace + EAR
│   │   ├── audio_extractor.py    # Audio extraction & resampling
│   │   ├── vad.py                # Silero Voice Activity Detection
│   │   ├── dataset_loader.py     # FakeAVCeleb, AVSpeech, LRS2, CelebDF, DFDC
│   │   └── pipeline.py           # End-to-end preprocessing (multiprocessing)
│   ├── models/
│   │   ├── visual_encoder.py     # AV-HuBERT, ResNet-18, SyncNet
│   │   ├── audio_encoder.py      # Wav2Vec 2.0 wrapper
│   │   ├── classifier.py         # Bi-LSTM, 1D-CNN, Statistical baseline
│   │   ├── syncguard.py          # Full model integration
│   │   └── audio_classifier.py   # Standalone audio deepfake classifier
│   ├── training/
│   │   ├── losses.py             # InfoNCE, CMP, temporal consistency, combined
│   │   ├── dataset.py            # Dataset + collation + hard negative mining
│   │   ├── pretrain.py           # Phase 1 contrastive pretraining loop
│   │   └── finetune.py           # Phase 2 fine-tuning loop
│   ├── evaluation/
│   │   ├── metrics.py            # AUC-ROC, EER, pAUC, per-category breakdown
│   │   ├── evaluate.py           # Inference runner
│   │   └── visualize.py          # Publication-quality plots
│   └── utils/
│       ├── config.py             # YAML config loader
│       └── io.py                 # Video/audio I/O helpers
├── scripts/
│   ├── preprocess_dataset.py     # CLI for preprocessing
│   ├── train_pretrain.py         # CLI for Phase 1 pretraining
│   ├── train_finetune.py         # CLI for Phase 2 fine-tuning
│   ├── evaluate.py               # CLI for evaluation
│   ├── evaluate_cascade.py       # Cascade evaluation (sync + audio fusion)
│   ├── extract_ear_features.py   # Standalone EAR extraction
│   ├── gpu_smoke_test.py         # GPU verification
│   └── slurm_*.sh               # SLURM job scripts for HPC
├── docs/
│   ├── EXECUTION_PLAN.md         # Timeline and task tracking
│   ├── BASELINES.md              # Expected metric ranges
│   ├── RESEARCH.md               # Technical rationale
│   ├── OPERATIONS.md             # Step-by-step HPC guide
│   └── lab_notebook.md           # Experiment journal
├── outputs/
│   ├── checkpoints/              # Model weights (gitignored)
│   ├── logs/                     # Training metrics + experiment reports
│   └── visualizations/           # Plots (ROC, sync-scores, ablations)
├── notebooks/
│   └── download_avspeech_colab.ipynb
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Setup

### Local Development

```bash
git clone https://github.com/Akshay171124/SyncGuard.git
cd SyncGuard

conda create -n syncguard python=3.11 -y
conda activate syncguard
pip install -r requirements.txt

# Ensure ffmpeg is installed
brew install ffmpeg  # macOS
# sudo apt install ffmpeg  # Ubuntu
```

### HPC (Northeastern Explorer)

```bash
module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
```

## Dependencies

| Category | Packages |
|----------|----------|
| **Deep Learning** | PyTorch >= 2.0, torchaudio >= 2.0, torchvision >= 0.15 |
| **Pretrained Models** | transformers >= 4.30 (Wav2Vec 2.0), fairseq >= 0.12 (AV-HuBERT) |
| **Computer Vision** | opencv-python >= 4.8, mediapipe >= 0.10, retinaface >= 0.0.17 |
| **Audio** | soundfile >= 0.12, librosa >= 0.10 |
| **Evaluation & Viz** | scikit-learn >= 1.3, matplotlib >= 3.7, seaborn >= 0.12 |
| **Experiment Tracking** | wandb >= 0.15 |
| **Config** | pyyaml >= 6.0, numpy >= 1.24, scipy >= 1.10 |
| **System** | ffmpeg (must be installed separately via `brew install ffmpeg` or `apt install ffmpeg`) |

All Python dependencies are pinned in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing

```bash
# Preprocess FakeAVCeleb
python scripts/preprocess_dataset.py --dataset fakeavceleb --config configs/default.yaml

# Preprocess LRS2 with multiprocessing
python scripts/preprocess_dataset.py --dataset lrs2 --config configs/default.yaml --workers 14

# Extract EAR features for existing preprocessed data
python scripts/extract_ear_features.py --dataset fakeavceleb --config configs/default.yaml
```

### Training

```bash
# Phase 1: Contrastive pretraining (InfoNCE + Cross-Modal Prediction)
python scripts/train_pretrain.py --config configs/default.yaml

# Phase 2: Fine-tuning on FakeAVCeleb with EAR features
python scripts/train_finetune.py --config configs/default.yaml \
    --pretrain_ckpt outputs/checkpoints/pretrain_best.pt
```

### Evaluation

```bash
# Evaluate on FakeAVCeleb
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint outputs/checkpoints/finetune_best.pt --test_set fakeavceleb

# Cascade evaluation (sync + audio classifier fusion)
python scripts/evaluate_cascade.py --config configs/default.yaml \
    --sync_ckpt outputs/checkpoints/finetune_best.pt \
    --audio_ckpt outputs/checkpoints/audio_clf_best.pt
```

### SLURM (HPC)

```bash
# Submit Phase 1 pretraining (H200 GPU, auto-resubmit)
sbatch scripts/slurm_pretrain.sh

# Submit Phase 2 fine-tuning
sbatch scripts/slurm_finetune.sh
```

## Testing

Run the full test suite (219 tests, ~12 seconds on CPU):

```bash
python -m pytest tests/ -v
```

Tests cover all major components without requiring GPU or dataset downloads:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_metrics.py` | 18 | AUC-ROC, EER, pAUC, per-category breakdown, bootstrap CI |
| `test_losses.py` | 27 | MoCo queue, InfoNCE, temperature clamping, temporal consistency, combined loss |
| `test_models.py` | 45 | Visual encoders, classifiers, cross-attention, DCT extractor, factory functions |
| `test_syncguard.py` | 20 | Full model integration, sequence alignment, sync-score computation |
| `test_dataset.py` | 16 | Batch collation, padding, masking, variable-length sequences |
| `test_dataset_loader.py` | 17 | Dataset scanning, speaker-disjoint splits, category detection |
| `test_checkpoint.py` | 7 | Save/load round-trip for model, optimizer, scheduler, criterion |
| `test_audio_encoder.py` | 7 | Mocked Wav2Vec2, layer extraction, frozen backbone behavior |
| `test_augmentation.py` | 18 | Self-Blended Image blending, mask generation, sequence augmentation |
| `test_preprocessing.py` | 16 | Audio extraction helpers, upsampling, EAR computation |
| `test_config.py` | 8 | YAML config loading, device auto-detection |

Tests that require optional dependencies (e.g., mediapipe for EAR) are automatically skipped when the dependency is not installed.

## End-to-End Reproducibility Walkthrough

This section walks through reproducing the headline numbers from the final report (96.3% AUC on FakeAVCeleb, 52.6% AUC on DFDC zero-shot). Each step lists the command, expected runtime, and what you should see if it worked.

### Step 1 — Smoke test (no GPU, no data, ~30 seconds)

Verify the environment is set up correctly:

```bash
python -m pytest tests/ -v
```

Expected output:
```
=================== 215 passed, 4 skipped in 11.8s ===================
```

If this fails, do not proceed — fix the environment first. The 4 skipped tests are mediapipe-dependent EAR tests that require the optional `mediapipe` package.

### Step 2 — Download pretrained models (one-time, ~5 minutes)

Pre-fetch the Wav2Vec 2.0 weights to your HuggingFace cache to avoid wasting GPU time during training:

```bash
python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')"
```

Expected output: `Downloading … 100%` then silent success. Produces ~360 MB under `$HF_HOME` (default `~/.cache/huggingface`).

### Step 3 — Download datasets (~20–60 minutes depending on network)

Datasets are hosted on [Google Drive](https://drive.google.com/drive/folders/1wQ9cdWo5R9O8ZvwPO7XUnfMvVOgoIF1E?usp=drive_link). You can download either raw videos (to rerun preprocessing end-to-end) or the already-preprocessed `.npy` features (recommended — saves ~10 hours of preprocessing on HPC).

Place files so that:
- Raw videos live under `data/raw/<dataset>/…`
- Preprocessed features live under `data/processed/<dataset>/…`

Paths are configurable via `configs/default.yaml`.

### Step 4 — (Optional) Preprocess from raw (~6–10 hours on HPC)

Skip this step if you downloaded the preprocessed features directly.

```bash
python scripts/preprocess_dataset.py --dataset fakeavceleb --config configs/default.yaml --workers 14
```

Expected output (per sample, trimmed):
```
[12:03:17] fakeavceleb/real/id00123/00042.mp4  →  T=144 frames  audio=16000Hz  OK
[12:03:19] fakeavceleb/fake/id00456/00088.mp4  →  T=131 frames  SKIPPED (RetinaFace confidence < 0.8 on 18 frames)
…
Completed 21544/21544 samples — 19872 written, 1672 skipped
```

### Step 5 — Phase 1 pretraining (HPC, ~8 hours on H200)

```bash
sbatch scripts/slurm_pretrain.sh     # uses configs/pretrain_frozen.yaml by default
```

Track progress in W&B (`SyncGuard / phase1-pretrain`) or at `outputs/logs/pretrain.json`. Expected best metrics at epoch ~17:

| Metric | Value |
|---|---|
| Val InfoNCE loss | **8.06** |
| Val sync-score | **0.978** |

Checkpoint written to `outputs/checkpoints/pretrain_best.pt`.

### Step 6 — Phase 2 fine-tuning (HPC, ~6 hours on H200)

```bash
sbatch scripts/slurm_finetune.sh     # uses configs/finetune_v4_best.yaml
```

Expected best metrics on the FakeAVCeleb val set around epoch 17 (early stopping triggers):

| Metric | Value |
|---|---|
| Val AUC | **0.953** |
| Best checkpoint | `outputs/checkpoints/finetune_best.pt` |

### Step 7 — Evaluate on FakeAVCeleb (~5 minutes on a single H200)

```bash
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint outputs/checkpoints/finetune_best.pt --test_set fakeavceleb
```

Expected output:
```
===== FakeAVCeleb Test Set (n=2950) =====
AUC-ROC:  0.9628
EER:      0.0931
pAUC@0.1: 0.8607
Per-category:
  FV-RA (face-swap only):   AUC=0.9398
  RV-FA (voice-clone only): AUC=0.8949
  FV-FA (both fake):        AUC=0.9872
Results written to outputs/logs/eval_fakeavceleb.json
```

### Step 8 — Zero-shot cross-dataset evaluation on DFDC (~2 minutes)

```bash
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint outputs/checkpoints/finetune_best.pt --test_set dfdc
```

Expected output:
```
===== DFDC Part 0 Test Set (n=1343) =====
AUC-ROC:  0.5263
EER:      0.4911
pAUC@0.1: 0.0644
Note: zero-shot transfer — model never saw DFDC during training.
```

**Interpretation:** The DFDC number is intentionally near chance. This is a known limitation of sync-based methods against face-swap generators that preserve lip motion (see `docs/superpowers/specs/review-findings.md` for the full ablation).

### Sanity checks during a fresh run

Watch for these early signals to catch problems before wasting GPU-hours:

- **Loss explodes to NaN in first 100 steps:** Wav2Vec backbone accidentally unfrozen (check `config.audio.freeze_backbone`).
- **Val AUC stuck at 0.5 across 3 epochs:** Check speaker-disjoint split — train speakers leaking into val defeats the metric.
- **Sync-score saturates to 1.0 in pretraining:** Wav2Vec unfrozen during pretrain, causing representation collapse.
- **DataLoader crashes batch 0 with shape mismatch:** Collation assumes wrong channel count for CLIP+RGB pipeline (fixed in v3.3.0).

## Datasets

**Dataset Download:** All datasets (raw and preprocessed) are available on [Google Drive](https://drive.google.com/drive/folders/1wQ9cdWo5R9O8ZvwPO7XUnfMvVOgoIF1E?usp=drive_link).

| Dataset | Samples | Role | Status |
|---------|---------|------|--------|
| FakeAVCeleb | 21,544 | Primary train/val/test | Preprocessed |
| AVSpeech | 24,760 | Contrastive pretraining | Preprocessed |
| LRS2 | 96,318 | Expanded pretraining + extra reals | In progress |
| DFDC Part 0 | 1,334 | Cross-dataset zero-shot test | Preprocessed |
| CelebDF-v2 | 921 | Dropped (no audio streams) | N/A |

After downloading, place datasets under `data/raw/` and preprocessed features under `data/processed/`. Update paths in `configs/default.yaml` if your directory structure differs.

## Known Issues & Special Considerations

- **DFDC cross-dataset generalization:** Zero-shot AUC on DFDC is 52.6% (target: 72%). Face-swap methods that preserve lip-sync defeat audio-visual correspondence signals. CLIP ViT-L/14 + Self-Blended Image augmentation is being explored to address this.
- **Wav2Vec 2.0 must be frozen** during fine-tuning on small datasets (~21K samples). Unfreezing causes catastrophic forgetting of pretrained speech representations and degrades performance.
- **RetinaFace silent failures:** Frames with no detected face (low confidence, non-frontal, occluded) are silently skipped during preprocessing. The confidence threshold is set to 0.8 in `configs/default.yaml`.
- **GPU memory:** H200 (140GB) is recommended for training. A100 (40GB) works but may require reducing batch size from 32 to 16 for pretraining. AV-HuBERT + Wav2Vec 2.0 + Bi-LSTM accumulate large hidden states.
- **Pre-download pretrained models** before submitting GPU jobs to avoid wasting compute time:
  ```bash
  python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')"
  ```
- **Temporal alignment:** Visual features (25 fps) are upsampled to match Wav2Vec output rate (~49 Hz). Off-by-one frame differences are handled by truncating to the shorter sequence in `SyncGuard.align_sequences()`.

## Key References

- **AVFF** (CVPR 2024) — Cross-modal prediction for AV deepfake detection
- **AV-HuBERT** — Self-supervised audio-visual speech representation
- **Wav2Vec 2.0** — Self-supervised speech representation learning
- **FakeAVCeleb** — Audio-visual deepfake detection benchmark

## License

This project is for academic use as part of CS 5330 at Northeastern University.
