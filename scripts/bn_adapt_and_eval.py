#!/usr/bin/env python3
"""Batch normalization adaptation + threshold recalibration for DFDC.

1. Load trained model
2. Forward pass through DFDC data with BN in training mode (updates running stats)
3. Evaluate with adapted BN stats
4. Find optimal threshold on DFDC predictions
5. Report results with and without recalibration

Usage:
    python scripts/bn_adapt_and_eval.py \
        --config configs/finetune_frozen.yaml \
        --checkpoint outputs/checkpoints/ca_stage2_best.pt \
        --dataset dfdc
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.syncguard import build_syncguard
from src.training.dataset import build_dataloaders, build_test_dataloader
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


def set_bn_train(model):
    """Set only BatchNorm and LayerNorm layers to training mode."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            module.training = True


def adapt_bn(model, dataloader, device, num_batches=50):
    """Forward pass through data to adapt BN running statistics."""
    # Set everything to inference, then selectively enable BN
    model.train()
    model.set_to_inference = True  # marker
    for m in model.modules():
        m.training = False
    set_bn_train(model)

    logger.info(f"Adapting BN statistics on {num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            mouth_crops = batch.mouth_crops.to(device)
            waveforms = batch.waveforms.to(device)
            lengths = batch.lengths.to(device)
            ear = batch.ear_features.to(device) if batch.ear_features is not None else None

            _ = model(mouth_crops, waveforms, lengths=lengths, ear_features=ear)

            if (i + 1) % 10 == 0:
                logger.info(f"  Adapted {i+1}/{num_batches} batches")

    # Back to full inference
    for m in model.modules():
        m.training = False
    logger.info("BN adaptation complete")


def find_optimal_threshold(labels, scores):
    """Find threshold that maximizes accuracy."""
    best_acc = 0
    best_thresh = 0.5
    for thresh in np.arange(0.01, 0.99, 0.01):
        preds = (scores >= thresh).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def compute_auc(labels, scores):
    """Simple AUC-ROC computation."""
    if len(set(labels)) < 2:
        return 0.5
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tp, fp, auc, prev_fpr = 0, 0, 0.0, 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp / n_pos) * (fp / n_neg - prev_fpr)
            prev_fpr = fp / n_neg
    return auc


def compute_eer(labels, scores):
    """Compute Equal Error Rate."""
    if len(set(labels)) < 2:
        return 0.5
    thresholds = sorted(set(scores))
    min_diff = float("inf")
    eer = 0.5
    for thresh in thresholds:
        preds = [1 if s >= thresh else 0 for s in scores]
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        n_neg = sum(1 for l in labels if l == 0)
        n_pos = sum(1 for l in labels if l == 1)
        fpr = fp / max(n_neg, 1)
        fnr = fn / max(n_pos, 1)
        diff = abs(fpr - fnr)
        if diff < min_diff:
            min_diff = diff
            eer = (fpr + fnr) / 2
    return eer


def run_inference(model, dataloader, device):
    """Run inference and return scores + labels."""
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            mouth_crops = batch.mouth_crops.to(device)
            waveforms = batch.waveforms.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels
            ear = batch.ear_features.to(device) if batch.ear_features is not None else None

            output = model(mouth_crops, waveforms, lengths=lengths, ear_features=ear)
            probs = torch.sigmoid(output.logits.squeeze(-1))
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return labels, scores


def main():
    parser = argparse.ArgumentParser(description="BN adaptation + threshold recalibration")
    parser.add_argument("--config", type=str, default="configs/finetune_frozen.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dfdc", choices=["dfdc", "celebdf"])
    parser.add_argument("--bn_batches", type=int, default=50,
                        help="Number of batches for BN adaptation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config(args.config)
    device = get_device(config)

    # Build model and load checkpoint
    model = build_syncguard(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info(f"Loaded checkpoint: {args.checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})")

    # Build DFDC dataloader
    test_loader = build_test_dataloader(config, args.dataset)
    logger.info(f"Loaded {args.dataset}: {len(test_loader.dataset)} samples")

    # === Step 1: Evaluate WITHOUT BN adaptation (baseline) ===
    print("\n" + "=" * 60)
    print("  Step 1: Baseline (no BN adaptation)")
    print("=" * 60)
    for m in model.modules():
        m.training = False
    labels, scores_baseline = run_inference(model, test_loader, device)
    auc_base = compute_auc(labels.tolist(), scores_baseline.tolist())
    eer_base = compute_eer(labels.tolist(), scores_baseline.tolist())
    print(f"  AUC: {auc_base:.4f}")
    print(f"  EER: {eer_base:.4f}")

    # === Step 2: BN adaptation ===
    print("\n" + "=" * 60)
    print("  Step 2: BN Adaptation")
    print("=" * 60)
    adapt_bn(model, test_loader, device, num_batches=args.bn_batches)

    # === Step 3: Evaluate WITH BN adaptation ===
    print("\n" + "=" * 60)
    print("  Step 3: After BN Adaptation")
    print("=" * 60)
    labels, scores_adapted = run_inference(model, test_loader, device)
    auc_adapt = compute_auc(labels.tolist(), scores_adapted.tolist())
    eer_adapt = compute_eer(labels.tolist(), scores_adapted.tolist())
    print(f"  AUC: {auc_adapt:.4f}")
    print(f"  EER: {eer_adapt:.4f}")

    # === Step 4: Threshold recalibration ===
    print("\n" + "=" * 60)
    print("  Step 4: Threshold Recalibration")
    print("=" * 60)
    thresh_adapted, acc_adapted = find_optimal_threshold(labels, scores_adapted)
    thresh_baseline, acc_baseline = find_optimal_threshold(labels, scores_baseline)
    print(f"  Baseline optimal threshold: {thresh_baseline:.2f} (acc={acc_baseline:.4f})")
    print(f"  Adapted optimal threshold:  {thresh_adapted:.2f} (acc={acc_adapted:.4f})")

    # Score distribution analysis
    real_mask = labels == 0
    fake_mask = labels == 1
    print(f"\n  Score distributions (adapted):")
    print(f"    Real: mean={scores_adapted[real_mask].mean():.4f}, std={scores_adapted[real_mask].std():.4f}")
    print(f"    Fake: mean={scores_adapted[fake_mask].mean():.4f}, std={scores_adapted[fake_mask].std():.4f}")
    gap = scores_adapted[fake_mask].mean() - scores_adapted[real_mask].mean()
    print(f"    Gap (fake-real):  {gap:.4f} ({'correct direction' if gap > 0 else 'INVERTED'})")

    # === Summary ===
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Baseline':>10} {'BN Adapted':>12} {'Delta':>8}")
    print(f"  {'AUC':<20} {auc_base:>10.4f} {auc_adapt:>12.4f} {auc_adapt - auc_base:>+8.4f}")
    print(f"  {'EER':<20} {eer_base:>10.4f} {eer_adapt:>12.4f} {eer_adapt - eer_base:>+8.4f}")
    print(f"  {'Opt. Accuracy':<20} {acc_baseline:>10.4f} {acc_adapted:>12.4f} {acc_adapted - acc_baseline:>+8.4f}")

    # Save results
    output_path = Path("outputs/logs") / f"bn_adapt_{args.dataset}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "bn_batches": args.bn_batches,
        "baseline": {"auc": auc_base, "eer": eer_base},
        "adapted": {"auc": auc_adapt, "eer": eer_adapt},
        "threshold_baseline": thresh_baseline,
        "threshold_adapted": thresh_adapted,
        "accuracy_baseline": acc_baseline,
        "accuracy_adapted": acc_adapted,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
