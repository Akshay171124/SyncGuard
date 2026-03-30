"""Evaluation runner for SyncGuard.

Loads a trained checkpoint and runs inference on one or more test sets,
collecting predictions and computing metrics.

Usage:
    python -m src.evaluation.evaluate --config configs/default.yaml \
        --checkpoint outputs/checkpoints/finetune_best.pt \
        --test_sets fakeavceleb celebdf dfdc
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.syncguard import SyncGuard, SyncGuardOutput, build_syncguard
from src.training.dataset import SyncGuardBatch
from src.evaluation.metrics import compute_all_metrics, EvaluationResult
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(
    model: SyncGuard,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run model inference on a dataloader.

    Collects predictions, labels, sync-score sequences, and optional
    category labels for downstream metric computation.

    Args:
        model: Trained SyncGuard model.
        dataloader: Test DataLoader.
        device: Torch device.

    Returns:
        Dict with keys:
            - scores: (N,) prediction scores (sigmoid of logits)
            - labels: (N,) ground truth binary labels
            - sync_scores: list of (T_i,) mean sync-score per sample
            - categories: (N,) category strings (if available)
    """
    model.eval()

    all_scores = []
    all_labels = []
    all_sync_means = []
    all_categories = []

    n_batches = 0
    t_start = time.time()

    for batch in dataloader:
        batch: SyncGuardBatch
        mouth_crops = batch.mouth_crops.to(device)
        waveforms = batch.waveforms.to(device)
        lengths = batch.lengths.to(device)
        labels = batch.labels
        ear = batch.ear_features.to(device) if batch.ear_features is not None else None

        output: SyncGuardOutput = model(mouth_crops, waveforms, lengths=lengths, ear_features=ear)

        # Sigmoid scores for binary classification
        probs = torch.sigmoid(output.logits.squeeze(-1))  # (B,)
        all_scores.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

        # Mean sync-score per sample (masked if variable length)
        sync = output.sync_scores  # (B, T)
        if lengths is not None:
            T = sync.shape[1]
            lengths_clamped = lengths.clamp(max=T)
            mask = torch.arange(T, device=device).unsqueeze(0) < lengths_clamped.unsqueeze(1)
            # Masked mean
            sync_masked = sync * mask.float()
            means = sync_masked.sum(dim=1) / lengths_clamped.float().clamp(min=1)
        else:
            means = sync.mean(dim=1)
        all_sync_means.append(means.cpu().numpy())

        # Categories (if dataset provides them)
        if hasattr(batch, "categories") and batch.categories is not None:
            all_categories.extend(batch.categories)

        n_batches += 1

    elapsed = time.time() - t_start
    logger.info(f"Inference: {n_batches} batches in {elapsed:.1f}s")

    result = {
        "scores": np.concatenate(all_scores),
        "labels": np.concatenate(all_labels),
        "sync_means": np.concatenate(all_sync_means),
    }

    if all_categories:
        result["categories"] = np.array(all_categories)

    return result


def evaluate_test_set(
    model: SyncGuard,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str,
    output_dir: Path,
) -> EvaluationResult:
    """Evaluate model on a single test set and save results.

    Args:
        model: Trained SyncGuard model.
        dataloader: Test DataLoader.
        device: Torch device.
        dataset_name: Name for logging and file output.
        output_dir: Directory to save JSON results.

    Returns:
        EvaluationResult with all metrics.
    """
    logger.info(f"Evaluating on {dataset_name}...")

    predictions = run_inference(model, dataloader, device)

    categories = predictions.get("categories", None)
    result = compute_all_metrics(
        predictions["labels"],
        predictions["scores"],
        categories=categories,
    )

    # Save results JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"eval_{dataset_name}.json"
    result_dict = result.to_dict()
    result_dict["dataset"] = dataset_name

    # Include sync-score statistics
    sync_means = predictions["sync_means"]
    real_mask = predictions["labels"] == 0
    fake_mask = predictions["labels"] == 1
    result_dict["sync_score_stats"] = {
        "real_mean": float(np.mean(sync_means[real_mask])) if real_mask.any() else None,
        "real_std": float(np.std(sync_means[real_mask])) if real_mask.any() else None,
        "fake_mean": float(np.mean(sync_means[fake_mask])) if fake_mask.any() else None,
        "fake_std": float(np.std(sync_means[fake_mask])) if fake_mask.any() else None,
    }

    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"Results saved to {result_path}")

    # Save raw predictions for plotting
    preds_path = output_dir / f"predictions_{dataset_name}.npz"
    save_dict = {
        "scores": predictions["scores"],
        "labels": predictions["labels"],
        "sync_means": predictions["sync_means"],
        "fpr": result.fpr,
        "tpr": result.tpr,
    }
    if categories is not None:
        save_dict["categories"] = categories
    np.savez(preds_path, **save_dict)
    logger.info(f"Predictions saved to {preds_path}")

    return result


def load_checkpoint(
    config: dict,
    checkpoint_path: str,
    device: torch.device,
) -> SyncGuard:
    """Load a trained SyncGuard model from checkpoint.

    Args:
        config: Full config dict.
        checkpoint_path: Path to .pt checkpoint file.
        device: Torch device.

    Returns:
        SyncGuard model in eval mode.
    """
    model = build_syncguard(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    epoch = ckpt.get("epoch", "?")
    val_metrics = ckpt.get("val_metrics", {})
    logger.info(
        f"Loaded checkpoint from epoch {epoch}: "
        f"val_auc={val_metrics.get('val_auc', '?')}"
    )

    model.eval()
    return model


def evaluate(
    config: dict,
    checkpoint_path: str,
    test_set_names: list[str],
    test_loaders: dict[str, DataLoader],
) -> dict[str, EvaluationResult]:
    """Run evaluation on multiple test sets.

    Args:
        config: Full config dict.
        checkpoint_path: Path to trained checkpoint.
        test_set_names: List of test set names to evaluate.
        test_loaders: Dict mapping test set name to DataLoader.

    Returns:
        Dict mapping test set name to EvaluationResult.
    """
    device = get_device(config)
    output_dir = Path("outputs/logs")

    model = load_checkpoint(config, checkpoint_path, device)

    results = {}
    for name in test_set_names:
        if name not in test_loaders:
            logger.warning(f"No dataloader for {name}, skipping")
            continue

        result = evaluate_test_set(
            model, test_loaders[name], device, name, output_dir
        )
        results[name] = result

        logger.info(
            f"  {name}: AUC={result.auc_roc:.4f} "
            f"EER={result.eer:.4f} "
            f"pAUC@0.1={result.pauc_fpr01:.4f}"
        )

    # Save combined summary
    summary = {
        name: r.to_dict() for name, r in results.items()
    }
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return results


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="SyncGuard Evaluation")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--test_sets", nargs="+",
        default=["fakeavceleb"],
        help="Test set names to evaluate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Inference batch size",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)

    from src.training.dataset import build_dataloaders, build_test_dataloader

    test_loaders = {}
    for name in args.test_sets:
        if name == "fakeavceleb":
            # FakeAVCeleb uses speaker-disjoint test split
            loaders = build_dataloaders(config, phase="finetune")
            test_loaders[name] = loaders["test"]
        elif name in ("dfdc", "celebdf"):
            # Cross-dataset evaluation — load entire dataset
            try:
                test_loaders[name] = build_test_dataloader(config, name)
                logger.info(f"Loaded {name} test set: {len(test_loaders[name].dataset)} samples")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
        else:
            logger.warning(f"Unknown dataset: {name}")

    evaluate(
        config=config,
        checkpoint_path=args.checkpoint,
        test_set_names=list(test_loaders.keys()),
        test_loaders=test_loaders,
    )


if __name__ == "__main__":
    main()
