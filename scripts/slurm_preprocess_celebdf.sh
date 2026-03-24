#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --job-name=preprocess_celebdf
#SBATCH --output=outputs/logs/preprocess_celebdf_%j.out
#SBATCH --error=outputs/logs/preprocess_celebdf_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs

echo "=== CelebDF-v2 Preprocessing ($(date)) ==="

python scripts/preprocess_dataset.py \
    --dataset celebdf \
    --config configs/default.yaml

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

# Count processed samples
PROCESSED=$(find data/processed/celebdf/ -name "metadata.json" 2>/dev/null | wc -l)
echo "Processed samples: $PROCESSED"

# Auto-resubmit if not all done and no error
if [ $EXIT_CODE -eq 0 ] && [ $PROCESSED -lt 6229 ]; then
    echo "Resubmitting — $PROCESSED/6229 done"
    sbatch scripts/slurm_preprocess_celebdf.sh
fi
