#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=preprocess_lrs2_mp
#SBATCH --output=outputs/logs/preprocess_lrs2_%j.out
#SBATCH --error=outputs/logs/preprocess_lrs2_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

# Auto-resubmit on ANY termination signal (timeout, preemption, cancel)
RESUBMITTED=0
resubmit() {
    if [ $RESUBMITTED -eq 0 ]; then
        RESUBMITTED=1
        echo "Signal received — resubmitting job... ($(date))"
        sbatch scripts/slurm_preprocess_lrs2.sh
    fi
}
# Trap all relevant signals:
#   USR1  = SLURM --signal before timeout
#   TERM  = SLURM cancellation / preemption (SIGTERM)
#   INT   = Ctrl-C / scancel
#   HUP   = session hangup
#   XCPU  = CPU time limit exceeded
trap resubmit USR1 TERM INT HUP XCPU

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs

# Prevent mediapipe EGL/GPU segfault on headless nodes
export __EGL_VENDOR_LIBRARY_DIRS=/dev/null
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=4.5

echo "=== LRS2 Preprocessing — resuming with 14 workers ($(date)) ==="

python scripts/preprocess_dataset.py \
    --dataset lrs2 \
    --data_dir data/raw/LRS2/mvlrs_v1/pretrain \
    --config configs/default.yaml \
    --workers 14 &

# Wait for python process; if a signal arrives, wait returns early
# and the trap fires, resubmitting before we exit
CHILD_PID=$!
wait $CHILD_PID
EXIT_CODE=$?

echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

# Count processed samples
PROCESSED=$(ls data/processed/lrs2/real/ 2>/dev/null | wc -l)
echo "Processed samples: $PROCESSED / 96318"

# If killed by signal and trap didn't fire yet, resubmit now
if [ $EXIT_CODE -ne 0 ] && [ $RESUBMITTED -eq 0 ]; then
    echo "Non-zero exit ($EXIT_CODE) — resubmitting as safety net..."
    resubmit
fi
