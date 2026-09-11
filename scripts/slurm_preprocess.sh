#!/bin/bash -l
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --job-name=preprocess
#SBATCH --output=outputs/logs/preprocess_%x_%j.out
#SBATCH --error=outputs/logs/preprocess_%x_%j.err

# Preprocess one dataset. Submit with:
#   sbatch --job-name=pre_lrs2 --time=2-00:00:00 \
#          --export=ALL,DATASET=lrs2 scripts/slurm_preprocess.sh
#
# CPU partition by design. MediaPipe is pinned to Delegate.CPU
# (src/preprocessing/face_detector.py:65) and the lab notebook measures
# ~2-3 s per clip on CPU, so a GPU buys nothing here and would compete with
# training for scarce H200 time. The `short` partition also allows 2 days,
# which removes the auto-resubmit chaining the old 1h55m GPU scripts needed.
set -uo pipefail

# Login shell (-l above) so the module system is initialised. Without it,
# `module load` is silently a no-op in a non-interactive shell and ffmpeg is
# absent, which costs every sample its audio track.
module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard

# Fail fast rather than process the whole dataset without audio. Audio
# extraction failures do not abort the run, so a missing ffmpeg would
# otherwise yield a full set of samples with no audio.wav — useless for
# audio-visual sync training.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FATAL: ffmpeg not on PATH. Audio extraction would fail for every sample." >&2
    exit 1
fi
echo "ffmpeg: $(command -v ffmpeg)"

cd /scratch/$USER/SyncGuard || exit 1
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs data/processed

: "${DATASET:?DATASET must be set, e.g. --export=ALL,DATASET=lrs2}"
WORKERS="${WORKERS:-14}"

echo "=== $(date +%F_%H:%M:%S) preprocessing $DATASET with $WORKERS workers ==="
python scripts/preprocess_dataset.py \
    --dataset "$DATASET" \
    --config configs/rebuild_pretrain.yaml \
    --workers "$WORKERS"
RC=$?
echo "=== $(date +%F_%H:%M:%S) exit=$RC ==="

echo "--- processed sample count ---"
find "data/processed/$DATASET" -name metadata.json 2>/dev/null | wc -l
