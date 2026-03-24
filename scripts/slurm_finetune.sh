#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=syncguard_finetune
#SBATCH --output=outputs/logs/finetune_%j.out
#SBATCH --error=outputs/logs/finetune_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

# Auto-resubmit on ANY termination signal (timeout, preemption, cancel)
RESUBMITTED=0
resubmit() {
    if [ $RESUBMITTED -eq 0 ]; then
        RESUBMITTED=1
        echo "Signal received — resubmitting... ($(date))"
        LATEST=$(ls -t outputs/checkpoints/finetune_epoch_*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            sbatch --export=RESUME_CKPT="$LATEST",PRETRAIN_CKPT="$PRETRAIN_CKPT" scripts/slurm_finetune.sh
        else
            sbatch --export=PRETRAIN_CKPT="$PRETRAIN_CKPT" scripts/slurm_finetune.sh
        fi
    fi
}
trap resubmit USR1 TERM INT HUP XCPU

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

# Default pretrain checkpoint
PRETRAIN_CKPT="${PRETRAIN_CKPT:-outputs/checkpoints/pretrain_best.pt}"

echo "=== Phase 2: Fine-tuning on H200 ($(date)) ==="
echo "Pretrain checkpoint: $PRETRAIN_CKPT"
echo "Dataset: FakeAVCeleb (with EAR features)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from: $RESUME_CKPT"
    RESUME_ARG="--resume $RESUME_CKPT"
fi

python scripts/train_finetune.py \
    --config configs/default.yaml \
    --pretrain_ckpt "$PRETRAIN_CKPT" \
    $RESUME_ARG &

CHILD_PID=$!
wait $CHILD_PID
EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

# Safety net: resubmit on non-zero exit if trap didn't fire
if [ $EXIT_CODE -ne 0 ] && [ $RESUBMITTED -eq 0 ]; then
    echo "Non-zero exit ($EXIT_CODE) — resubmitting as safety net..."
    resubmit
fi
