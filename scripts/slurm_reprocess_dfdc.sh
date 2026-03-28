#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --job-name=reprocess_dfdc
#SBATCH --output=outputs/logs/reprocess_dfdc_%j.out
#SBATCH --error=outputs/logs/reprocess_dfdc_%j.err

# DFDC Reprocessing — runs after HP-1/2/3/4 fixes
# Clears old DFDC preprocessed data and re-runs with corrected pipeline:
#   HP-1: Proper label handling (skip unknowns instead of defaulting to REAL)
#   HP-2: Timestamp-based fps sampling (fixes 20% drift on 30fps videos)
#   HP-3: RetinaFace resolution normalization (fixes detection at 1080p)
#   HP-4: VAD params from config (configurable thresholds)

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs

echo "=== DFDC Reprocessing with pipeline fixes ($(date)) ==="

# Back up old preprocessed data before clearing
DFDC_PROCESSED="data/processed/dfdc"
if [ -d "$DFDC_PROCESSED" ]; then
    BACKUP="data/processed/dfdc_pre_fix_backup"
    echo "Backing up old DFDC data to $BACKUP"
    mv "$DFDC_PROCESSED" "$BACKUP"
fi

# Re-run preprocessing with 14 workers (matches HPC cpus-per-task)
python scripts/preprocess_dataset.py \
    --dataset dfdc \
    --config configs/default.yaml \
    --workers 14

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

# Count and validate processed samples
PROCESSED=$(find "$DFDC_PROCESSED" -name "mouth_crops.npy" 2>/dev/null | wc -l)
echo "Processed samples: $PROCESSED"

# Spot-check label distribution
python3 -c "
import json
from pathlib import Path
from collections import Counter

processed = Path('$DFDC_PROCESSED')
labels = []
for meta_file in sorted(processed.rglob('metadata.json')):
    with open(meta_file) as f:
        meta = json.load(f)
    labels.append(meta.get('label', -1))

counts = Counter(labels)
print(f'Label distribution: {dict(counts)}')
total = sum(counts.values())
if total > 0:
    print(f'  Real (0): {counts.get(0, 0)} ({counts.get(0, 0)/total:.1%})')
    print(f'  Fake (1): {counts.get(1, 0)} ({counts.get(1, 0)/total:.1%})')
else:
    print('WARNING: No processed samples found!')
"

# Spot-check frame counts (should be ~250 for 10s@25fps, not ~300)
python3 -c "
import numpy as np
from pathlib import Path

processed = Path('$DFDC_PROCESSED')
frame_counts = []
for npy in sorted(processed.rglob('mouth_crops.npy'))[:20]:
    crops = np.load(npy)
    frame_counts.append(crops.shape[0])

if frame_counts:
    print(f'Frame count spot-check (first 20 samples):')
    print(f'  Mean: {np.mean(frame_counts):.0f} (expect ~250 for 10s clips)')
    print(f'  Min: {np.min(frame_counts)}, Max: {np.max(frame_counts)}')
    if np.mean(frame_counts) > 280:
        print('  WARNING: Frame counts still high — fps fix may not be applied')
    else:
        print('  OK: Frame counts consistent with 25fps sampling')
"
