#!/bin/bash
#SBATCH --partition=courses
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --job-name=rebuild_env
#SBATCH --output=outputs/logs/rebuild_env_%j.out
#SBATCH --error=outputs/logs/rebuild_env_%j.err

module load miniconda3/24.11.1

echo "=== Removing old env ==="
conda remove -n syncguard --all -y 2>&1 | tail -3

ENV=/home/prajapati.aksh/.conda/envs/syncguard
PIP="$ENV/bin/pip"
PYTHON="$ENV/bin/python"

echo "=== Creating conda env (step 1/4) ==="
conda create -n syncguard python=3.11 numpy scipy scikit-learn matplotlib seaborn h5py pyyaml -c conda-forge -y 2>&1 | tail -5

echo "=== Installing PyTorch (step 2/4) ==="
$PIP install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

echo "=== Installing remaining packages (step 3/4) ==="
$PIP install opencv-python-headless transformers soundfile gdown 2>&1 | tail -3
$PIP install "ml-dtypes<0.4" --no-build-isolation 2>&1 | tail -3
$PIP install mediapipe retina-face --no-build-isolation 2>&1 | tail -3
$PIP install wandb librosa --no-build-isolation 2>&1 | tail -3

echo "=== Verifying (step 4/4) ==="
$PYTHON -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
$PYTHON -c "import numpy; print(f'numpy {numpy.__version__}')"
$PYTHON -c "import sklearn; print(f'sklearn {sklearn.__version__}')"
$PYTHON -c "import cv2; print(f'opencv {cv2.__version__}')"
$PYTHON -c "import transformers; print(f'transformers {transformers.__version__}')"
$PYTHON -c "import mediapipe as mp; print(f'mediapipe {mp.__version__}')"
$PYTHON -c "import mediapipe as mp; fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1); print('FaceMesh OK'); fm.close()"
$PYTHON -c "from retinaface import RetinaFace; print('retina-face OK')"
$PYTHON -c "import google.protobuf; print(f'protobuf {google.protobuf.__version__}')"
$PYTHON -c "import tensorflow as tf; print(f'tensorflow {tf.__version__}')"
$PYTHON -c "print('ALL IMPORTS OK')"
echo "=== DONE ==="
