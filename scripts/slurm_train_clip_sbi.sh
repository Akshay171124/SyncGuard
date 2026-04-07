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

# Auto-resubmit on timeout/preemption with checkpoint resume
RESUBMITTED=0
resubmit() {
    if [ $RESUBMITTED -eq 0 ]; then
        RESUBMITTED=1
        echo "Signal received — saving and resubmitting... ($(date))"
        # Find latest checkpoint for resume
        LATEST=$(ls -t outputs/checkpoints/finetune_epoch_*.pt outputs/checkpoints/finetune_best.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "Will resume from: $LATEST"
        fi
        sbatch scripts/slurm_train_clip_sbi.sh
    fi
}
trap resubmit USR1 TERM INT HUP XCPU

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== CLIP + SBI Training ($(date)) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Only backup on first run (no checkpoint exists yet)
if [ ! -f outputs/checkpoints/finetune_best.pt ] && [ ! -f outputs/checkpoints/finetune_epoch_4.pt ]; then
    mkdir -p outputs/checkpoints/pre_clip_backup
    echo "First run — no checkpoints to backup"
fi

# Resume from latest checkpoint if available
RESUME_ARG=""
LATEST=$(ls -t outputs/checkpoints/finetune_epoch_*.pt outputs/checkpoints/finetune_best.pt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_ARG="--resume $LATEST"
fi

python scripts/train_clip_sbi.py \
    --config configs/clip_sbi.yaml \
    $RESUME_ARG &

CHILD_PID=$!
wait $CHILD_PID
EXIT_CODE=$?
echo "=== Training finished with exit code $EXIT_CODE ($(date)) ==="

# Evaluate on FakeAVCeleb + DFDC if training completed successfully
if [ $EXIT_CODE -eq 0 ] && [ -f outputs/checkpoints/finetune_best.pt ]; then
    echo "=== Evaluating ==="
    python scripts/evaluate.py \
        --config configs/clip_sbi.yaml \
        --checkpoint outputs/checkpoints/finetune_best.pt \
        --test_sets fakeavceleb dfdc
    echo "=== Evaluation done ($(date)) ==="
fi

# Safety net: resubmit on non-zero exit if trap didn't fire
if [ $EXIT_CODE -ne 0 ] && [ $RESUBMITTED -eq 0 ]; then
    echo "Non-zero exit ($EXIT_CODE) — resubmitting as safety net..."
    resubmit
fi
