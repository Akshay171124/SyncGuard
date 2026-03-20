#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:55:00
#SBATCH --job-name=syncguard_pretrain
#SBATCH --output=outputs/logs/pretrain_%j.out
#SBATCH --error=outputs/logs/pretrain_%j.err
#SBATCH --signal=B:USR1@120

# Auto-resubmit on timeout (signal sent 2 min before time limit)
CHECKPOINT="outputs/checkpoints/pretrain_best.pt"

resubmit() {
    echo "=== Time limit approaching, resubmitting with resume... ==="
    sbatch scripts/slurm_train_pretrain_a100.sh
    exit 0
}
trap resubmit USR1

module load miniconda3/24.11.1 FFmpeg/7.1.1
source activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_API_KEY=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

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== Phase 1 Contrastive Pretraining - Learnable Tau ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Resume from checkpoint if it exists
RESUME_FLAG=""
if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from $CHECKPOINT"
    RESUME_FLAG="--resume $CHECKPOINT"
fi

python scripts/train_pretrain.py \
    --config configs/default.yaml \
    $RESUME_FLAG &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed or was interrupted (exit code $EXIT_CODE)"
    echo "Check error log for details. NOT auto-resubmitting on failure."
fi
