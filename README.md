# SyncGuard

**Contrastive Audio-Visual Deepfake Detection via Temporal Phoneme-Face Coherence**

CS 5330 Computer Vision & Pattern Recognition — Northeastern University

## Team
- Akshay Prajapati (prajapati.aksh@northeastern.edu)
- Ritik Mahyavanshi (mahyavanshi.r@northeastern.edu)
- Atharva Dhumal (dhumal.a@northeastern.edu)

## Overview

SyncGuard detects deepfake videos by measuring the temporal coherence between speech audio and facial motion. It computes a frame-level sync-score `s(t) = cos(v_t, a_t)` using contrastively trained visual and audio encoders, then classifies clips based on temporal dip patterns in this score sequence.

## Project Structure

```
SyncGuard/
├── configs/              # YAML configuration files
│   └── default.yaml
├── data/
│   ├── raw/              # Raw datasets (not tracked by git)
│   ├── processed/        # Preprocessed outputs (mouth crops, audio, masks)
│   └── features/         # Extracted embeddings
├── src/
│   ├── preprocessing/    # Data preprocessing pipeline
│   │   ├── face_detector.py      # RetinaFace + MediaPipe mouth-ROI
│   │   ├── audio_extractor.py    # Audio extraction & Wav2Vec prep
│   │   ├── vad.py                # Silero Voice Activity Detection
│   │   ├── dataset_loader.py     # FakeAVCeleb & CelebDF-v2 loaders
│   │   └── pipeline.py           # End-to-end preprocessing orchestrator
│   ├── models/           # Model architectures
│   ├── training/         # Training loops and losses
│   ├── evaluation/       # Metrics and evaluation scripts
│   └── utils/            # Config loader, I/O helpers
├── scripts/              # CLI scripts
│   └── preprocess_dataset.py
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/
│   ├── checkpoints/      # Saved model weights
│   ├── logs/             # Training logs
│   └── visualizations/   # Sync-score plots, figures
├── tests/                # Unit tests
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/Akshay171124/SyncGuard.git
cd SyncGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Ensure ffmpeg is installed
brew install ffmpeg  # macOS
# sudo apt install ffmpeg  # Ubuntu
```

## Preprocessing

```bash
# Preprocess FakeAVCeleb dataset
python scripts/preprocess_dataset.py \
    --dataset fakeavceleb \
    --data_dir data/raw/FakeAVCeleb

# Preprocess CelebDF-v2 dataset
python scripts/preprocess_dataset.py \
    --dataset celebdf \
    --data_dir data/raw/CelebDF-v2

# Test with a small subset
python scripts/preprocess_dataset.py \
    --dataset fakeavceleb \
    --data_dir data/raw/FakeAVCeleb \
    --max_samples 10
```

## Datasets

| Dataset | Role | Access |
|---------|------|--------|
| FakeAVCeleb | Primary training | Obtained via author request |
| VoxCeleb2 / LRS2-BBC | Contrastive pretraining | Pending |
| CelebDF-v2 | Cross-generator test | Public download |
| DFDC | In-the-wild test | Kaggle |
| Wav2Lip generated | Adversarial test | Self-generated |

## License

This project is for academic use as part of CS 5330 at Northeastern University.
