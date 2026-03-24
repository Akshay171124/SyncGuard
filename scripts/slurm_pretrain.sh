#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=syncguard_pretrain
#SBATCH --output=outputs/logs/pretrain_%j.out
#SBATCH --error=outputs/logs/pretrain_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

# Auto-resubmit on timeout (resumes from latest checkpoint)
resubmit() {
    echo "Time limit approaching — resubmitting..."
    LATEST=$(ls -t outputs/checkpoints/pretrain_epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        sbatch --export=RESUME_CKPT="$LATEST" scripts/slurm_pretrain.sh
    else
        sbatch scripts/slurm_pretrain.sh
    fi
}
trap resubmit USR1

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== Phase 1: Contrastive Pretraining on H200 ($(date)) ==="
echo "Datasets: AVSpeech + LRS2"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from: $RESUME_CKPT"
    RESUME_ARG="--resume $RESUME_CKPT"
fi

python scripts/train_pretrain.py \
    --config configs/default.yaml \
    $RESUME_ARG

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="
