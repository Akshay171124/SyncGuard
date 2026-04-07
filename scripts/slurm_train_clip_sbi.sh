#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=clip_sbi
#SBATCH --output=outputs/logs/clip_sbi_%j.out
#SBATCH --error=outputs/logs/clip_sbi_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== CLIP + SBI Training ($(date)) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Clear old checkpoints to avoid conflicts
mkdir -p outputs/checkpoints/pre_clip_backup
mv outputs/checkpoints/finetune_best.pt outputs/checkpoints/pre_clip_backup/ 2>/dev/null
mv outputs/checkpoints/finetune_epoch_*.pt outputs/checkpoints/pre_clip_backup/ 2>/dev/null

RESUME_ARG=""
LATEST=$(ls -t outputs/checkpoints/finetune_epoch_*.pt outputs/checkpoints/finetune_best.pt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_ARG="--resume $LATEST"
fi

python scripts/train_clip_sbi.py \
    --config configs/clip_sbi.yaml \
    $RESUME_ARG

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

# Evaluate on FakeAVCeleb + DFDC
if [ $EXIT_CODE -eq 0 ] && [ -f outputs/checkpoints/finetune_best.pt ]; then
    echo "=== Evaluating ==="
    python scripts/evaluate.py \
        --config configs/clip_sbi.yaml \
        --checkpoint outputs/checkpoints/finetune_best.pt \
        --test_sets fakeavceleb dfdc
fi
