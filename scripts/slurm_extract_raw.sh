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

# Directory names must match the *_dir keys in configs/*.yaml exactly, and the
# archives nest their own top-level folder, so each is flattened after
# extraction. A mismatch here fails every preprocessing job instantly.

echo "=== $(date +%H:%M:%S) FakeAVCeleb ==="
mkdir -p FakeAVCeleb
unzip -q -o FakeAVCeleb_v1.2.zip -d FakeAVCeleb && echo "  ok" || echo "  FAILED exit=$?"
# Archive nests FakeAVCeleb_v1.2/; the loader wants the category dirs at root.
if [ -d FakeAVCeleb/FakeAVCeleb_v1.2 ]; then
    mv FakeAVCeleb/FakeAVCeleb_v1.2/* FakeAVCeleb/ && rmdir FakeAVCeleb/FakeAVCeleb_v1.2
fi
find FakeAVCeleb -type f \( -name '*.mp4' -o -name '*.avi' \) | wc -l

echo "=== $(date +%H:%M:%S) CelebDF-v2 ==="
mkdir -p CelebDF-v2
unzip -q -o "Celeb DF (v2).zip" -d CelebDF-v2 && echo "  ok" || echo "  FAILED exit=$?"
find CelebDF-v2 -type f \( -name '*.mp4' -o -name '*.avi' \) | wc -l

echo "=== $(date +%H:%M:%S) LRS2 ==="
mkdir -p LRS2
tar -xf lrs2_v1.tar -C LRS2 && echo "  ok" || echo "  FAILED exit=$?"
find LRS2 -type f \( -name '*.mp4' -o -name '*.avi' \) | wc -l

echo "=== $(date +%H:%M:%S) done ==="
du -sh "$RAW"/*/ 2>/dev/null
