#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --job-name=syncguard_audio_clf
#SBATCH --output=outputs/logs/audio_clf_%j.out
#SBATCH --error=outputs/logs/audio_clf_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_API_KEY=$(grep password ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== Audio Classifier Training ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python scripts/train_audio_classifier.py --config configs/default.yaml

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="
