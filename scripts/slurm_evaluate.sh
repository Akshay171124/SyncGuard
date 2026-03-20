#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --job-name=syncguard_eval
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
source activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs

echo "=== SyncGuard Evaluation ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

CHECKPOINT="${1:-outputs/checkpoints/finetune_best.pt}"
echo "Checkpoint: $CHECKPOINT"

python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --test_sets fakeavceleb

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="
