#!/bin/bash
#SBATCH --partition=courses
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=extract_ear
#SBATCH --output=outputs/logs/extract_ear_%j.out
#SBATCH --error=outputs/logs/extract_ear_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs

# Create a minimal isolated env for EAR extraction to avoid
# mediapipe/tensorflow/protobuf conflicts in the main env.
EAR_ENV=/home/prajapati.aksh/.conda/envs/ear_extract
# Recreate env to ensure correct mediapipe version
echo "=== Setting up ear_extract env ==="
conda remove -n ear_extract --all -y 2>/dev/null | tail -1
conda create -n ear_extract python=3.11 numpy pyyaml -c conda-forge -y 2>&1 | tail -3
$EAR_ENV/bin/pip install 'mediapipe==0.10.14' opencv-python-headless 2>&1 | tail -5
$EAR_ENV/bin/python -c "import mediapipe as mp; fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1); print('FaceMesh OK'); fm.close()"
PYTHON="$EAR_ENV/bin/python"

# Extract EAR features for FakeAVCeleb (21,544 samples)
echo "=== EAR Extraction: FakeAVCeleb ($(date)) ==="
$PYTHON scripts/extract_ear_features.py \
    --dataset fakeavceleb \
    --config configs/default.yaml

echo ""

# Extract EAR features for DFDC (1,334 samples)
echo "=== EAR Extraction: DFDC ($(date)) ==="
$PYTHON scripts/extract_ear_features.py \
    --dataset dfdc \
    --config configs/default.yaml

echo ""
echo "=== All EAR extraction done ($(date)) ==="

# Report counts
FAC_EAR=$(find data/processed/fakeavceleb/ -name "ear_features.npy" 2>/dev/null | wc -l)
DFDC_EAR=$(find data/processed/dfdc/ -name "ear_features.npy" 2>/dev/null | wc -l)
echo "FakeAVCeleb EAR: $FAC_EAR"
echo "DFDC EAR: $DFDC_EAR"
