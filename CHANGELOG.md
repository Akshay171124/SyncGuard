# Changelog

All notable changes to SyncGuard will be documented in this file.

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
