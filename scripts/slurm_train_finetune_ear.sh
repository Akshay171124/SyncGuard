#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=finetune_ear
#SBATCH --output=outputs/logs/finetune_ear_%j.out
#SBATCH --error=outputs/logs/finetune_ear_%j.err
#SBATCH --signal=B:USR1@120

# Auto-resubmit on timeout
resubmit() {
    echo "Time limit approaching — resubmitting with resume..."
    sbatch scripts/slurm_train_finetune_ear.sh
}
trap resubmit USR1

module load miniconda3/24.11.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export WANDB_API_KEY=$(grep password ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== Phase 2: Fine-tuning with EAR + LRS2 ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Use CMP-pretrained checkpoint
PRETRAIN_CKPT="outputs/checkpoints/pretrain_best.pt"
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found at $PRETRAIN_CKPT"
    echo "Run Phase 1 pretraining first (slurm_train_pretrain_cmp.sh)"
    exit 1
fi

# Resume from finetune checkpoint if exists
RESUME_FLAG=""
if [ -f outputs/checkpoints/finetune_best.pt ]; then
    RESUME_FLAG="--resume outputs/checkpoints/finetune_best.pt"
    echo "Resuming from existing finetune checkpoint"
fi

python scripts/train_finetune.py \
    --config configs/default.yaml \
    --pretrain_ckpt $PRETRAIN_CKPT \
    $RESUME_FLAG &

wait $!
EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="
