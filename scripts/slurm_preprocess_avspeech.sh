#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:55:00
#SBATCH --job-name=preprocess_avspeech
#SBATCH --output=outputs/logs/preprocess_avspeech_%j.out
#SBATCH --error=outputs/logs/preprocess_avspeech_%j.err
#SBATCH --signal=B:USR1@120

# Resubmit on timeout signal (sent 120s before time limit)
resubmit() {
    echo "=== Time limit approaching, resubmitting... ==="
    PROCESSED=$(ls data/processed/avspeech/real/ 2>/dev/null | wc -l)
    echo "Processed so far: $PROCESSED / 24760"
    if [ "$PROCESSED" -lt 24000 ]; then
        sbatch scripts/slurm_preprocess_avspeech.sh
        echo "Resubmitted."
    fi
    # Kill the python process so job exits cleanly
    kill $PY_PID 2>/dev/null
    wait $PY_PID 2>/dev/null
    exit 0
}
trap resubmit USR1

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard

echo "=== Starting AVSpeech preprocessing ($(date)) ==="
echo "Already processed: $(ls data/processed/avspeech/real/ 2>/dev/null | wc -l) / 24760"

python scripts/preprocess_dataset.py \
    --dataset avspeech \
    --data_dir data/raw/AVSpeech \
    --config configs/default.yaml &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
PROCESSED=$(ls data/processed/avspeech/real/ 2>/dev/null | wc -l)
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="
echo "Processed so far: $PROCESSED / 24760"

# Auto-resubmit if not all clips are done
if [ "$PROCESSED" -lt 24000 ]; then
    echo "Not done yet — resubmitting..."
    sbatch scripts/slurm_preprocess_avspeech.sh
else
    echo "All clips processed!"
fi
