#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:55:00
#SBATCH --job-name=preprocess_fakeavceleb
#SBATCH --output=outputs/logs/preprocess_fakeavceleb_%j.out
#SBATCH --error=outputs/logs/preprocess_fakeavceleb_%j.err
#SBATCH --signal=B:USR1@120

# Total expected: 21,544 clips (500 RV-RA + 9709 FV-RA + 500 RV-FA + 10835 FV-FA)
EXPECTED=21544

# Resubmit on timeout signal (sent 120s before time limit)
resubmit() {
    echo "=== Time limit approaching, resubmitting... ==="
    PROCESSED=$(find data/processed/fakeavceleb/ -name "metadata.json" 2>/dev/null | wc -l)
    echo "Processed so far: $PROCESSED / $EXPECTED"
    if [ "$PROCESSED" -lt 21000 ]; then
        sbatch scripts/slurm_preprocess_fakeavceleb.sh
        echo "Resubmitted."
    fi
    kill $PY_PID 2>/dev/null
    wait $PY_PID 2>/dev/null
    exit 0
}
trap resubmit USR1

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard

echo "=== Starting FakeAVCeleb preprocessing ($(date)) ==="
PROCESSED=$(find data/processed/fakeavceleb/ -name "metadata.json" 2>/dev/null | wc -l)
echo "Already processed: $PROCESSED / $EXPECTED"

python scripts/preprocess_dataset.py \
    --dataset fakeavceleb \
    --data_dir data/raw/FakeAVCeleb \
    --config configs/default.yaml &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
PROCESSED=$(find data/processed/fakeavceleb/ -name "metadata.json" 2>/dev/null | wc -l)
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="
echo "Processed so far: $PROCESSED / $EXPECTED"

# Auto-resubmit if not all clips are done
if [ "$PROCESSED" -lt 21000 ]; then
    echo "Not done yet — resubmitting..."
    sbatch scripts/slurm_preprocess_fakeavceleb.sh
else
    echo "All clips processed!"
fi
