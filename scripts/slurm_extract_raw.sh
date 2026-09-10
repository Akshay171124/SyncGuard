#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --job-name=extract_raw
#SBATCH --output=outputs/logs/extract_raw_%j.out
#SBATCH --error=outputs/logs/extract_raw_%j.err

# Extract the raw dataset archives fetched from Drive.
# Independent of the AVSpeech file-by-file transfer, so it can run alongside it.
set -uo pipefail
RAW="/scratch/$USER/SyncGuard/data/raw"
cd "$RAW" || exit 1

echo "=== $(date +%H:%M:%S) FakeAVCeleb ==="
mkdir -p fakeavceleb
unzip -q -o FakeAVCeleb_v1.2.zip -d fakeavceleb && echo "  ok" || echo "  FAILED exit=$?"
find fakeavceleb -type f \( -name '*.mp4' -o -name '*.avi' \) | wc -l

echo "=== $(date +%H:%M:%S) CelebDF-v2 ==="
mkdir -p celebdf
unzip -q -o "Celeb DF (v2).zip" -d celebdf && echo "  ok" || echo "  FAILED exit=$?"
find celebdf -type f \( -name '*.mp4' -o -name '*.avi' \) | wc -l

echo "=== $(date +%H:%M:%S) LRS2 ==="
mkdir -p lrs2
tar -xf lrs2_v1.tar -C lrs2 && echo "  ok" || echo "  FAILED exit=$?"
find lrs2 -type f \( -name '*.mp4' -o -name '*.avi' \) | wc -l

echo "=== $(date +%H:%M:%S) done ==="
du -sh "$RAW"/*/ 2>/dev/null
