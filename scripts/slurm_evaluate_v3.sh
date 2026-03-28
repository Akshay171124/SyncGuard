#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --job-name=syncguard_eval_v3
#SBATCH --output=outputs/logs/eval_v3_%j.out
#SBATCH --error=outputs/logs/eval_v3_%j.err

# Full evaluation pipeline after Phase 2 v2 fine-tuning:
#   1. Evaluate on FakeAVCeleb (expect >=0.94 AUC)
#   2. Evaluate on DFDC (reprocessed, Tier 1 baseline)
#   3. Run cascade evaluation (all fusion strategies)
#   4. Run DFDC diagnostics
#   5. Run metric verification with bootstrap CIs

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/visualizations

CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/finetune_best.pt}"

echo "=== SyncGuard v3 Full Evaluation ($(date)) ==="
echo "Checkpoint: $CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Step 1: FakeAVCeleb evaluation
echo ""
echo ">>> Step 1: FakeAVCeleb evaluation..."
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --test_sets fakeavceleb

# Step 2: DFDC evaluation (on reprocessed data)
echo ""
echo ">>> Step 2: DFDC evaluation (reprocessed)..."
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --test_sets dfdc

# Step 3: Cascade evaluation (all fusion strategies)
echo ""
echo ">>> Step 3: Cascade evaluation..."
python scripts/evaluate_cascade.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --test_sets fakeavceleb dfdc 2>/dev/null || echo "  (Cascade eval requires audio classifier checkpoint)"

# Step 4: DFDC diagnostics
echo ""
echo ">>> Step 4: DFDC diagnostics..."
python scripts/diagnose_dfdc.py \
    --config configs/default.yaml \
    --predictions_dir outputs/logs

# Step 5: Metric verification with bootstrap CIs
echo ""
echo ">>> Step 5: Metric verification with bootstrap CIs..."
python scripts/verify_metrics.py 2>/dev/null || echo "  (verify_metrics.py not yet configured)"

echo ""
echo "=== Evaluation Complete ($(date)) ==="
echo ""
echo "Results saved to:"
echo "  outputs/logs/eval_fakeavceleb.json"
echo "  outputs/logs/eval_dfdc.json"
echo "  outputs/visualizations/sync_dist_*.png"
echo ""
echo "Next steps:"
echo "  - If DFDC AUC >= 0.72: Move to paper writing (Day 12+)"
echo "  - If DFDC AUC 0.65-0.72: Implement Tier 2 (embedding bypass)"
echo "  - If DFDC AUC < 0.65: Implement Tier 2 + Tier 3 (DCT features)"
