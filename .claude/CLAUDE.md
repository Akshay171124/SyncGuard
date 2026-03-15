# Claude Rules for SyncGuard

## Communication Style
- Be concise. Short answers are preferred over verbose explanations.
- Don't ask unnecessary clarifying questions - make reasonable decisions.
- When the user provides feedback or a preference, apply it without lengthy justification.
- "clear" means acknowledged, move on.

## Code Style
- Include docstrings for functions with Args/Returns sections.
- Use descriptive variable names.
- Add comments only where logic isn't self-evident.
- Prefer general naming over overly specific (e.g., `output.log` not `training.log`).

## Git Workflow
- Commit messages: short summary line, optional body for context.
- Do NOT mention LLM assistance anywhere (commits, PRs, code comments, docs) unless explicitly requested. This includes Co-Authored-By, "Generated with", "AI-assisted", or similar attributions.
- Push when explicitly asked.
- **DCP**: When the user says "DCP", do: **D**ouble-check the code, **C**ommit, **P**ush.
- **Merging**: Always squash-and-merge branches into main.

## Project Context
- SyncGuard: Contrastive audio-visual deepfake detection via temporal phoneme-face coherence.
- Academic project for CS 5330 (Computer Vision) at Northeastern University, Khoury College.
- Team: Akshay (Visual Encoder & Preprocessing, Integration Lead), Ritik (Audio Encoder & Contrastive Pretraining), Atharva (Temporal Classifier & Evaluation).
- Architecture: Two-stream contrastive learning — AV-HuBERT (visual) + Wav2Vec 2.0 (audio) → frame-level sync-score s(t) → Bi-LSTM classifier.
- Training: Phase 1 = contrastive pretraining (real data, InfoNCE), Phase 2 = fine-tuning on FakeAVCeleb (InfoNCE + temporal consistency + BCE).
- Datasets: FakeAVCeleb (primary train/val), CelebDF-v2 & DFDC (zero-shot test), Wav2Lip (adversarial test), VoxCeleb2/LRS2 (pretraining).
- Key targets: AUC-ROC >= 0.88 (FakeAVCeleb), >= 0.72 (DFDC zero-shot).

## Project Structure
- `scripts/preprocess_dataset.py` — CLI entry point for preprocessing.
- `src/preprocessing/` — Pipeline, face detection (RetinaFace + MediaPipe), audio extraction, VAD (Silero).
- `src/models/` — Model architectures (AV-HuBERT visual encoder, Wav2Vec 2.0 audio encoder, Bi-LSTM classifier).
- `src/training/` — Training loops (contrastive pretraining, fine-tuning).
- `src/evaluation/` — Metrics (AUC-ROC, EER, pAUC) and visualization.
- `src/utils/` — Config loading (`configs/default.yaml`), I/O helpers.
- `configs/default.yaml` — Central config (data paths, model params, training hyperparameters).
- `data/` — raw/, processed/, features/ (all gitignored).
- `outputs/` — checkpoints/, logs/, visualizations/ (all gitignored).

## Python Environment
- Always activate conda environment before running Python commands:
  ```bash
  source ~/.zshrc && conda activate syncguard
  ```
- Dependencies in `requirements.txt`.
- **Never install packages directly** via `pip install`. Always:
  1. Add the dependency to `requirements.txt`
  2. Run `pip install -r requirements.txt` to install
- Core deps: PyTorch, torchaudio, torchvision, transformers, fairseq, opencv-python, mediapipe, retinaface, librosa, soundfile.

## Preprocessing
- Video: RetinaFace detection → MediaPipe FaceMesh (468 landmarks) → 96x96 mouth-ROI crop.
- Audio: ffmpeg extraction → 16 kHz mono PCM → Silero-VAD speech gating.
- Temporal alignment: visual features upsampled from 25 fps → 49 Hz (Wav2Vec native rate).
- Output per sample: `mouth_crops.npy`, `audio.wav`, `speech_mask.npy`, `metadata.json`.

## Testing & Hardware
- All experimentation runs on the Northeastern HPC cluster (connected directly).
- Watch for OOM — hidden states and attention maps from AV-HuBERT/Wav2Vec accumulate quickly.
- RetinaFace confidence threshold: 0.8 (drops low-quality / non-frontal frames).
- Device auto-detection: CUDA > MPS > CPU (see `src/utils/config.py:get_device()`).

## Dates
- The current year is 2026. Use this when writing dates in docs, logs, etc.

---

## Northeastern University HPC — Explorer Cluster

This project runs on Northeastern's Explorer cluster at MGHPCC.
Docs: https://rc-docs.northeastern.edu | Help: rchelp@northeastern.edu

### Cluster Overview

| Resource | Details |
|----------|---------|
| Cluster name | Explorer (replaced Discovery in 2025) |
| Location | MGHPCC, Holyoke, MA |
| OS | Rocky Linux 9.3 |
| Scheduler | SLURM |
| GPUs available | H200 (140GB), H100, A100, A6000, L40, V100, T4, P100 |
| Key GPUs for this project | **H200** (ideal, 140GB — fits both encoders + classifier) or **A100** (40/80GB) |
| Home directory | `/home/<username>` (limited quota, don't store data here) |
| Scratch directory | `/scratch/<username>` (large, NOT backed up, auto-purged after 28 days) |
| Module system | `module load <name>` |

### GPU Partitions

| Partition | GPUs per job | Max time | Access |
|-----------|-------------|----------|--------|
| `gpu-interactive` | 1 | 8 hours | Open — debugging and quick tests |
| `gpu-short` | 1 | 4 hours | Open — short experiments |
| `gpu` | 1 | 24 hours | Open — single-GPU training runs |
| `multigpu` | 2-4 | 24 hours | **Requires access request** — needed for multi-GPU training |

### Environment Setup (Run Once)

```bash
# Load modules
module load cuda/12.1
module load python/3.11

# Activate conda environment
conda activate syncguard

# Set HuggingFace cache to scratch (NOT /home — quota too small)
export HF_HOME=/scratch/$USER/.cache/huggingface
echo 'export HF_HOME=/scratch/$USER/.cache/huggingface' >> ~/.bashrc

# Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, Type: {torch.cuda.get_device_name(0)}')"
```

### SLURM Job Templates

#### Interactive Session (debugging)
```bash
srun --partition=gpu-interactive --nodes=1 --pty \
     --gres=gpu:h200:1 --ntasks=1 --cpus-per-task=8 \
     --mem=64GB --time=04:00:00 /bin/bash
```

#### Single-GPU Training Job
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=syncguard_pretrain
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err

module load cuda/12.1
conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/SyncGuard

python scripts/train_pretrain.py --config configs/default.yaml
```

### Monitoring Jobs

```bash
squeue -u $USER                    # Check job queue
seff <job_id>                      # Check job efficiency after completion
scancel <job_id>                   # Cancel a job
sinfo -p gpu --Format=partition,available,nodes,statecompact,gres  # GPU availability
```

### HPC Tips for This Project

1. **H200 (140GB) is ideal.** AV-HuBERT + Wav2Vec 2.0 + Bi-LSTM + batch data all fit comfortably on one H200 in bf16.
2. **A100 40GB works** but watch batch size — AV-HuBERT is memory-hungry with long video sequences. Reduce `batch_size` from 32→16 if OOM.
3. **Don't train on login nodes.** Login nodes are monitored. Use `srun` for interactive work.
4. **Scratch is purged.** Files in `/scratch/` may be deleted after 28 days of no access. Push results to git regularly.
5. **Pre-download models** before your first training job to avoid wasting GPU-hours:
   ```bash
   srun --partition=short --time=01:00:00 --mem=32GB --cpus-per-task=4 --pty /bin/bash
   python -c "
   from transformers import Wav2Vec2Model, Wav2Vec2Processor
   Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
   Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
   print('Wav2Vec 2.0 downloaded')
   "
   ```
6. **Create logs directory** before submitting: `mkdir -p outputs/logs`
7. **Tmux for long sessions:**
   ```bash
   tmux new -s syncguard
   srun --partition=gpu-interactive ...
   # Detach: Ctrl+B then D | Reattach: tmux attach -t syncguard
   ```

---

## Results Storage Convention

Each experiment/ablation run produces a directory under `outputs/`:

```
outputs/
├── checkpoints/
│   ├── pretrain_best.pt
│   ├── pretrain_epoch_5.pt
│   ├── finetune_best.pt
│   └── ablation_resnet18_best.pt
├── logs/
│   ├── pretrain.json              ← Per-epoch metrics: {epoch, loss, lr, tau, avg_sync_score}
│   ├── finetune.json              ← Per-epoch metrics: {epoch, loss_total, loss_infonce, loss_temp, loss_cls, val_auc, val_eer}
│   ├── eval_fakeavceleb.json      ← Evaluation results per test set
│   ├── eval_celebdf.json
│   ├── eval_dfdc.json
│   ├── eval_wavlip.json
│   └── ablation_results.json      ← All ablation results in one file
└── visualizations/
    ├── sync_scores/                ← s(t) curve plots (real vs fake)
    ├── roc_curves/                 ← ROC curves per test set
    ├── ablation_charts/            ← Bar charts comparing ablation variants
    └── training_curves/            ← Loss and metric curves over epochs
```

**Git rules:**
- Track: all `.json` result files, `.md` logs, `.png`/`.pdf` plots
- Gitignore: `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`, raw/processed data

---

## Experiment Log Template

Use this template for every experiment run. Store in `outputs/logs/` as markdown.

```markdown
# Experiment: [NAME]

**Date:** YYYY-MM-DD
**Owner:** [Team member name]

## Setup
- Phase: [pretrain / finetune / ablation / evaluation]
- Dataset: [name, sample count]
- Config overrides: [any changes from default.yaml]
- Hardware: [GPU type] × [count]
- Checkpoint loaded: [path or "from scratch"]

## Results
- Primary metric: [AUC-ROC / loss / etc.] = [value]
- Secondary metrics: [EER, pAUC, etc.]
- Training time: [hours]
- GPU memory peak: [GB]

## Observations
[What surprised you? What matched expectations? Any anomalies?]

## Decision
[Based on these results, the next step is... because...]

## Artifacts
- Checkpoint: [path]
- Metrics JSON: [path]
- Plots: [paths]
```

---

## Plotting Standards

All plots go in `outputs/visualizations/` and should be publication-quality for the poster and report.

- **Format:** PNG (300 DPI) + PDF (vector) for each plot
- **Font:** Arial or Helvetica, 12pt body, 14pt titles
- **Colors — consistent palette:**
  - Real clips: `#27AE60` (green)
  - Fake clips: `#E74C3C` (red)
  - FakeAVCeleb: `#1A5276` (dark blue)
  - CelebDF-v2: `#F39C12` (orange)
  - DFDC: `#8E44AD` (purple)
  - Wav2Lip adversarial: `#E67E22` (dark orange)
  - SyncGuard (ours): `#3498DB` (light blue)
  - Baselines: `#95A5A6` (gray)
- **Always include:** Axis labels, legend, title
- **Never include:** Gridlines on both axes (Y only if needed), chartjunk
- **Error bars:** Show standard deviation or 95% CI where applicable

**Required plots:**
1. `sync_score_real_vs_fake.png` — s(t) curves for real vs fake clips (the money shot)
2. `roc_fakeavceleb.png` — ROC curve on FakeAVCeleb (per-category)
3. `roc_cross_dataset.png` — ROC curves on CelebDF-v2, DFDC, Wav2Lip (zero-shot)
4. `training_loss_pretrain.png` — InfoNCE loss over pretraining epochs
5. `training_loss_finetune.png` — Composite loss components over fine-tuning epochs
6. `ablation_visual_encoder.png` — Bar chart: AV-HuBERT vs ResNet-18 vs SyncNet
7. `ablation_wav2vec_layer.png` — Bar chart: layers 3, 5, 7, 9, 11
8. `ablation_classifier.png` — Bar chart: Bi-LSTM vs 1D-CNN vs statistical
9. `sync_score_distribution.png` — Histogram of mean s(t) for real vs fake
10. `per_category_auc.png` — FakeAVCeleb 4-category AUC breakdown

---

## Common Pitfalls to Avoid

1. **AV-HuBERT loading.** AV-HuBERT uses fairseq, which has version conflicts with recent transformers. Pin `fairseq>=0.12.0` and test the import before starting GPU jobs.

2. **Temporal alignment mismatch.** Visual features are at 25 fps, Wav2Vec outputs at 49 Hz. The upsampling in `audio_extractor.py` must produce exactly matching sequence lengths. Verify `v_embeds.shape[1] == a_embeds.shape[1]` in the forward pass.

3. **RetinaFace on HPC.** RetinaFace may fail silently if no face is detected. Always check for empty detections and log skipped frames rather than crashing.

4. **MoCo queue on GPU.** The MoCo memory bank (4096 × 256) is ~4MB — negligible, but make sure it's on the correct device and updated correctly (FIFO, no gradients).

5. **Learnable temperature τ.** Clamp to [0.01, 0.5] EVERY step. If τ → 0, softmax becomes argmax and gradients vanish. If τ → ∞, all similarities flatten.

6. **Temporal consistency loss on fakes.** L_temp should ONLY be computed on real clips (`1[real]` mask). Applying it to fakes would teach the model that desynchronized temporal dynamics are normal.

7. **Speaker leakage.** Train/val/test splits MUST be speaker-disjoint. If the same speaker appears in train and test, the model learns speaker identity instead of sync patterns.

8. **Wav2Vec layer extraction.** Use `output_hidden_states=True` and index into `hidden_states[layer]`. The default output is the LAST layer, not layer 9.

9. **Batch collation with variable lengths.** Video clips have different durations. Pad sequences and use attention masks in the Bi-LSTM. Don't let padding corrupt the mean/max pooling.

10. **Checkpoint saving.** Save the full state dict including optimizer, scheduler, epoch, and best metric. This allows resuming training after SLURM job timeouts.
