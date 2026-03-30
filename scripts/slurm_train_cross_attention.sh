#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=syncguard_ca
#SBATCH --output=outputs/logs/cross_attention_%j.out
#SBATCH --error=outputs/logs/cross_attention_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== Cross-Attention Training ($(date)) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Stage 1: Train CA head only (uses finetune_best checkpoint)
FINETUNE_CKPT="${FINETUNE_CKPT:-outputs/checkpoints/finetune_v2_backup/finetune_best.pt}"
echo "Stage 1: Training cross-attention head"
echo "Base checkpoint: $FINETUNE_CKPT"

python scripts/train_cross_attention.py \
    --config configs/finetune_frozen.yaml \
    --checkpoint "$FINETUNE_CKPT" \
    --stage 1

echo ""
echo "Stage 2: End-to-end fusion fine-tuning"
CA_STAGE1="${CA_STAGE1:-outputs/checkpoints/ca_stage1_best.pt}"
echo "Stage 1 checkpoint: $CA_STAGE1"

python scripts/train_cross_attention.py \
    --config configs/finetune_frozen.yaml \
    --checkpoint "$CA_STAGE1" \
    --stage 2

echo ""
echo "=== Evaluating ==="
python scripts/evaluate.py \
    --config configs/finetune_frozen.yaml \
    --checkpoint outputs/checkpoints/ca_stage2_best.pt \
    --test_sets fakeavceleb dfdc

echo "=== Done ($(date)) ==="
