#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --job-name=syncguard_verify
#SBATCH --output=outputs/logs/slurm_verify_%j.out
#SBATCH --error=outputs/logs/slurm_verify_%j.err

# Statistical verification of SyncGuard metrics.
# Recomputes AUC, EER, pAUC, per-category breakdown, and bootstrap CIs
# from saved .npz prediction files. No GPU required.

module load miniconda3/24.11.1
eval "$(conda shell.bash hook)"
conda activate syncguard

cd /scratch/$USER/SyncGuard

echo "Starting metric verification at $(date)"
echo "Python: $(which python)"

python scripts/verify_metrics.py \
    --predictions_dir outputs/logs \
    --n_bootstrap 5000 \
    --output outputs/logs/verification_results.json

echo "Verification complete at $(date)"
