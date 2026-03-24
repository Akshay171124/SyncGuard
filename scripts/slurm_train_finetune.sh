#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=07:55:00
#SBATCH --job-name=syncguard_finetune
#SBATCH --output=outputs/logs/finetune_%j.out
#SBATCH --error=outputs/logs/finetune_%j.err
#SBATCH --signal=B:USR1@300

# Auto-resubmit on timeout
CHECKPOINT="outputs/checkpoints/finetune_best.pt"

resubmit() {
    echo "=== Time limit approaching, resubmitting with resume... ==="
    sbatch scripts/slurm_train_finetune.sh
    exit 0
}
trap resubmit USR1

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_API_KEY=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

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== Phase 2 Fine-tuning ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Resume from checkpoint if it exists
RESUME_FLAG=""
if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from $CHECKPOINT"
    RESUME_FLAG="--resume $CHECKPOINT"
fi

# Load pretrained checkpoint from Phase 1
PRETRAIN_CKPT="outputs/checkpoints/pretrain_best.pt"
PRETRAIN_FLAG=""
if [ -f "$PRETRAIN_CKPT" ]; then
    echo "Loading pretrained weights from $PRETRAIN_CKPT"
    PRETRAIN_FLAG="--pretrain_ckpt $PRETRAIN_CKPT"
fi

python scripts/train_finetune.py \
    --config configs/default.yaml \
    $PRETRAIN_FLAG \
    $RESUME_FLAG &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed or was interrupted — resubmitting..."
    sbatch scripts/slurm_train_finetune.sh
fi
