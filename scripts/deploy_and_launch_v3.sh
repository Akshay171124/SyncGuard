#!/bin/bash
# =============================================================================
# SyncGuard v3 Deployment & Launch Script
# =============================================================================
# Run this ON the HPC login node after git pulling the latest fixes.
#
# What it does (in order):
#   1. Pulls latest code with all bug fixes
#   2. Spot-checks AVSpeech/LRS2 fps (decides if reprocessing needed)
#   3. Reprocesses DFDC (submits SLURM job)
#   4. Launches Phase 1 v3 pretraining (submits SLURM job)
#
# Usage:
#   ssh explorer
#   cd /scratch/$USER/SyncGuard
#   bash scripts/deploy_and_launch_v3.sh
# =============================================================================

set -e  # Exit on first error

echo "=============================================="
echo "  SyncGuard v3 Deployment"
echo "  $(date)"
echo "=============================================="

# --- Step 0: Verify we're on HPC ---
if [ ! -d "/scratch/$USER" ]; then
    echo "ERROR: Not on HPC (no /scratch/$USER). Run this on Explorer."
    exit 1
fi

cd /scratch/$USER/SyncGuard

# --- Step 1: Pull latest code ---
echo ""
echo ">>> Step 1: Pulling latest code..."
git pull origin main
echo "Latest commit: $(git log --oneline -1)"

# --- Step 2: Activate environment ---
echo ""
echo ">>> Step 2: Setting up environment..."
module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH

# --- Step 3: Spot-check fps on existing datasets ---
echo ""
echo ">>> Step 3: Checking source fps (determines if reprocessing needed)..."
echo ""
echo "--- AVSpeech ---"
python scripts/check_dataset_fps.py --dataset avspeech --max_samples 100 2>/dev/null || echo "  (AVSpeech not available locally)"
echo ""
echo "--- LRS2 ---"
python scripts/check_dataset_fps.py --dataset lrs2 --max_samples 100 2>/dev/null || echo "  (LRS2 not available locally)"

# --- Step 4: Submit DFDC reprocessing ---
echo ""
echo ">>> Step 4: Submitting DFDC reprocessing job..."
mkdir -p outputs/logs
DFDC_JOB=$(sbatch --parsable scripts/slurm_reprocess_dfdc.sh)
echo "  DFDC reprocessing job: $DFDC_JOB"

# --- Step 5: Submit Phase 1 v3 pretraining (starts after DFDC finishes or immediately) ---
# Pretraining doesn't depend on DFDC — it uses AVSpeech + LRS2
echo ""
echo ">>> Step 5: Submitting Phase 1 v3 pretraining..."

# Clear any old pretrain checkpoints from previous (buggy) runs
echo "  Backing up old pretrain checkpoints..."
if ls outputs/checkpoints/pretrain_*.pt 1> /dev/null 2>&1; then
    mkdir -p outputs/checkpoints/pre_v3_backup
    mv outputs/checkpoints/pretrain_*.pt outputs/checkpoints/pre_v3_backup/ 2>/dev/null || true
    echo "  Old checkpoints moved to outputs/checkpoints/pre_v3_backup/"
fi

PRETRAIN_JOB=$(sbatch --parsable scripts/slurm_pretrain.sh)
echo "  Phase 1 v3 pretrain job: $PRETRAIN_JOB"

# --- Summary ---
echo ""
echo "=============================================="
echo "  Deployment Complete"
echo "=============================================="
echo ""
echo "  Jobs submitted:"
echo "    DFDC reprocess:  $DFDC_JOB"
echo "    Phase 1 v3:      $PRETRAIN_JOB"
echo ""
echo "  Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f outputs/logs/reprocess_dfdc_${DFDC_JOB}.out"
echo "    tail -f outputs/logs/pretrain_${PRETRAIN_JOB}.out"
echo ""
echo "  After pretraining completes (~72h), launch Phase 2:"
echo "    sbatch scripts/slurm_finetune.sh"
echo ""
echo "  After DFDC reprocessing completes (~1h), verify with:"
echo "    python scripts/check_dataset_fps.py --dataset dfdc --max_samples 50"
echo ""
