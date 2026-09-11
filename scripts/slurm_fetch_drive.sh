#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --job-name=fetch_drive
#SBATCH --output=outputs/logs/fetch_drive_%j.out
#SBATCH --error=outputs/logs/fetch_drive_%j.err

# Pull raw datasets from the Shared Drive onto scratch.
# Must run on a compute node: rclone is SIGKILLed by login-node enforcement.
set -uo pipefail
module load rclone/1.72.0

SRC="gdrive:5330 - CVPR"
DEST="/scratch/$USER/SyncGuard/data/raw"
mkdir -p "$DEST"

RC_OPTS="--transfers 8 --checkers 16 --drive-chunk-size 128M --stats 120s --stats-one-line --retries 5 --low-level-retries 20"

for f in "FakeAVCeleb_v1.2.zip" "Celeb DF (v2).zip" "lrs2_v1.tar"; do
    echo "=== $(date +%H:%M:%S) fetching archive: $f"
    rclone copy "$SRC/$f" "$DEST/" $RC_OPTS
    echo "    exit=$?"
done

echo "=== $(date +%H:%M:%S) fetching AVSpeech (24,766 files)"
rclone copy "$SRC/SyncGuard/data/raw/AVSpeech" "$DEST/AVSpeech" $RC_OPTS
echo "    exit=$?"

echo "=== done $(date +%H:%M:%S)"
du -sh "$DEST"/* 2>/dev/null
