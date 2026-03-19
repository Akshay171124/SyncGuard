#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=preprocess_avspeech
#SBATCH --output=outputs/logs/preprocess_avspeech_%j.out
#SBATCH --error=outputs/logs/preprocess_avspeech_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
source activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard

echo "=== Starting AVSpeech preprocessing ($(date)) ==="
echo "Already processed: $(ls data/processed/avspeech/real/ 2>/dev/null | wc -l) / 24760"

python scripts/preprocess_dataset.py \
    --dataset avspeech \
    --data_dir data/raw/AVSpeech \
    --config configs/default.yaml

EXIT_CODE=$?
PROCESSED=$(ls data/processed/avspeech/real/ 2>/dev/null | wc -l)
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="
echo "Processed so far: $PROCESSED / 24760"

# Auto-resubmit if not all clips are done
if [ "$PROCESSED" -lt 24000 ]; then
    echo "Not done yet — resubmitting..."
    sbatch scripts/slurm_preprocess_avspeech.sh
else
    echo "All clips processed!"
fi
