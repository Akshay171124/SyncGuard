#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=preprocess_lrs2
#SBATCH --output=outputs/logs/preprocess_lrs2_%j.out
#SBATCH --error=outputs/logs/preprocess_lrs2_%j.err
#SBATCH --signal=B:USR1@120

# Trap for auto-resubmit on timeout
resubmit() {
    echo "Time limit approaching — resubmitting..."
    sbatch scripts/slurm_preprocess_lrs2.sh
}
trap resubmit USR1

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs

# Prevent mediapipe EGL/GPU segfault on headless nodes
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=4.5

echo "=== LRS2 Preprocessing — smoke test on A100 ($(date)) ==="

# Process only pretrain split, cap at 40K samples
python scripts/preprocess_dataset.py \
    --dataset lrs2 \
    --data_dir data/raw/LRS2/mvlrs_v1/pretrain \
    --config configs/default.yaml \
    --max_samples 40000 &

wait $!
EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

# Count processed samples
PROCESSED=$(find data/processed/lrs2/ -name "metadata.json" 2>/dev/null | wc -l)
echo "Processed samples: $PROCESSED"
