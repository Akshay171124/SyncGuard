#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --job-name=syncguard_cascade_eval
#SBATCH --output=outputs/logs/cascade_eval_%j.out
#SBATCH --error=outputs/logs/cascade_eval_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
source activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs

echo "=== Cascade Evaluation ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python scripts/evaluate_cascade.py \
    --config configs/default.yaml \
    --sync_checkpoint outputs/checkpoints/finetune_best_run3_audioswap.pt \
    --audio_checkpoint outputs/checkpoints/audio_clf_best.pt

echo "=== Finished with exit code $? ($(date)) ==="
