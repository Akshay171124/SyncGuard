#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --job-name=setup_env
#SBATCH --output=outputs/logs/setup_env_%j.out
#SBATCH --error=outputs/logs/setup_env_%j.err

module load miniconda3/24.11.1

# Use env python/pip directly to avoid activation issues
ENV=/home/prajapati.aksh/.conda/envs/syncguard
PIP="$ENV/bin/pip"
PYTHON="$ENV/bin/python"

echo "=== Creating conda env (step 1/3) ==="
conda create -n syncguard python=3.11 numpy scipy scikit-learn matplotlib seaborn h5py pyyaml -c conda-forge -y

echo "=== Installing PyTorch (step 2/3) ==="
$PIP install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing remaining packages (step 3/3) ==="
$PIP install opencv-python-headless transformers soundfile gdown 'protobuf<5'
$PIP install 'ml-dtypes<0.4' --no-build-isolation
$PIP install mediapipe retina-face --no-build-isolation
$PIP install wandb librosa --no-build-isolation

echo "=== Verifying ==="
$PYTHON -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
$PYTHON -c "import numpy; print(f'numpy {numpy.__version__}')"
$PYTHON -c "import sklearn; print(f'sklearn {sklearn.__version__}')"
$PYTHON -c "import cv2; print(f'opencv {cv2.__version__}')"
$PYTHON -c "import transformers; print(f'transformers {transformers.__version__}')"
$PYTHON -c "import mediapipe; print(f'mediapipe {mediapipe.__version__}')"
$PYTHON -c "from retinaface import RetinaFace; print('retina-face OK')"
$PYTHON -c "print('ALL IMPORTS OK')"
echo "=== DONE ==="
