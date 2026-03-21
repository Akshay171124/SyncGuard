#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --job-name=preprocess_dfdc
#SBATCH --output=outputs/logs/preprocess_dfdc_%j.out
#SBATCH --error=outputs/logs/preprocess_dfdc_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
source activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs

echo "=== DFDC Preprocessing ($(date)) ==="

python scripts/preprocess_dataset.py \
    --dataset dfdc \
    --config configs/default.yaml

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

# Count processed samples
PROCESSED=$(find data/processed/dfdc/ -name "metadata.json" 2>/dev/null | wc -l)
echo "Processed samples: $PROCESSED"
