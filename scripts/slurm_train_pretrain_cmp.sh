#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=pretrain_cmp
#SBATCH --output=outputs/logs/pretrain_cmp_%j.out
#SBATCH --error=outputs/logs/pretrain_cmp_%j.err
#SBATCH --signal=B:USR1@120

# Auto-resubmit on timeout (resumes from checkpoint)
resubmit() {
    echo "Time limit approaching — resubmitting with resume..."
    sbatch scripts/slurm_train_pretrain_cmp.sh
}
trap resubmit USR1

module load miniconda3/24.11.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_API_KEY=$(grep password ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== Phase 1: CMP Pretraining ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Resume from checkpoint if exists
RESUME_FLAG=""
if [ -f outputs/checkpoints/pretrain_best.pt ]; then
    RESUME_FLAG="--resume outputs/checkpoints/pretrain_best.pt"
    echo "Resuming from existing checkpoint"
fi

python scripts/train_pretrain.py \
    --config configs/default.yaml \
    $RESUME_FLAG &

wait $!
EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="
