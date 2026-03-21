"""Cascade evaluation: SyncGuard (sync) + standalone audio classifier.

Runs both models on the test set and combines predictions via max-score
fusion. This gives the sync model's strength on FV-RA/FV-FA and the
audio model's strength on RV-FA, without either hurting the other.

Usage:
    python scripts/evaluate_cascade.py \
        --config configs/default.yaml \
        --sync_checkpoint outputs/checkpoints/finetune_best_run3_audioswap.pt \
        --audio_checkpoint outputs/checkpoints/audio_clf_best.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.syncguard import build_syncguard, SyncGuardOutput
from src.models.audio_classifier import build_standalone_audio_classifier
from src.training.dataset import build_dataloaders, build_test_dataloader, SyncGuardBatch
from src.evaluation.metrics import compute_all_metrics
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_cascade_inference(
    sync_model,
    audio_model,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run both models and collect per-sample scores.

    Args:
        sync_model: Trained SyncGuard model.
        audio_model: Trained standalone audio classifier.
        dataloader: Test DataLoader.
        device: Torch device.

    Returns:
        Dict with sync_scores, audio_scores, max_scores, labels, categories.
    """
    sync_model.eval()
    audio_model.eval()

    all_sync_scores = []
    all_audio_scores = []
    all_labels = []
    all_categories = []

    for batch in dataloader:
        batch: SyncGuardBatch
        mouth_crops = batch.mouth_crops.to(device)
        waveforms = batch.waveforms.to(device)
        lengths = batch.lengths.to(device)
        labels = batch.labels

        # Sync model
        output: SyncGuardOutput = sync_model(mouth_crops, waveforms, lengths=lengths)
        sync_probs = torch.sigmoid(output.logits.squeeze(-1))  # (B,)

        # Audio model
        audio_logits = audio_model(waveforms)
        audio_probs = torch.sigmoid(audio_logits.squeeze(-1))  # (B,)

        all_sync_scores.append(sync_probs.cpu().numpy())
        all_audio_scores.append(audio_probs.cpu().numpy())
        all_labels.append(labels.numpy())
        if hasattr(batch, "categories"):
            all_categories.extend(batch.categories)

    return {
        "sync_scores": np.concatenate(all_sync_scores),
        "audio_scores": np.concatenate(all_audio_scores),
        "labels": np.concatenate(all_labels),
        "categories": np.array(all_categories) if all_categories else None,
    }


def evaluate_cascade(predictions: dict, output_dir: Path, dataset_name: str = "fakeavceleb"):
    """Evaluate cascade fusion strategies and print results.

    Args:
        predictions: Dict from run_cascade_inference.
        output_dir: Directory to save results.
        dataset_name: Name used for output file naming.
    """
    labels = predictions["labels"]
    sync_scores = predictions["sync_scores"]
    audio_scores = predictions["audio_scores"]
    categories = predictions.get("categories")

    # Fusion strategies
    max_scores = np.maximum(sync_scores, audio_scores)
    avg_scores = (sync_scores + audio_scores) / 2

    strategies = {
        "sync_only": sync_scores,
        "audio_only": audio_scores,
        "max_fusion": max_scores,
        "avg_fusion": avg_scores,
    }

    results = {}
    for name, scores in strategies.items():
        result = compute_all_metrics(labels, scores, categories=categories)
        results[name] = result.to_dict()

        logger.info(
            f"\n{'='*60}\n"
            f"[{dataset_name}] Strategy: {name}\n"
            f"  Overall AUC: {result.auc_roc:.4f}\n"
            f"  EER:         {result.eer:.4f}\n"
            f"  pAUC@0.1:    {result.pauc_fpr01:.4f}\n"
            f"  pAUC@0.05:   {result.pauc_fpr05:.4f}"
        )

        # Per-category breakdown
        if result.per_category:
            for cat, auc in sorted(result.per_category.items()):
                logger.info(f"  {cat}: AUC={auc:.4f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"eval_cascade_{dataset_name}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {result_path}")

    # Save raw predictions
    save_dict = {
        "sync_scores": sync_scores,
        "audio_scores": audio_scores,
        "max_scores": max_scores,
        "labels": labels,
    }
    if categories is not None:
        save_dict["categories"] = categories
    np.savez(output_dir / f"predictions_cascade_{dataset_name}.npz", **save_dict)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SyncGuard Cascade Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sync_checkpoint", type=str, required=True,
                        help="Path to trained SyncGuard checkpoint")
    parser.add_argument("--audio_checkpoint", type=str, required=True,
                        help="Path to trained audio classifier checkpoint")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["fakeavceleb"],
                        help="Datasets to evaluate on (e.g., fakeavceleb celebdf dfdc)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    device = get_device(config)

    # Disable audio_head so SyncGuard loads without dual-head
    config["model"]["audio_head"] = False

    # Load sync model
    sync_model = build_syncguard(config).to(device)
    sync_ckpt = torch.load(args.sync_checkpoint, map_location=device, weights_only=False)
    sync_model.load_state_dict(sync_ckpt["model_state_dict"], strict=False)
    sync_model.eval()
    logger.info(f"Loaded sync model from {args.sync_checkpoint}")

    # Load audio model
    audio_model = build_standalone_audio_classifier(config).to(device)
    audio_ckpt = torch.load(args.audio_checkpoint, map_location=device, weights_only=False)
    audio_model.load_state_dict(audio_ckpt["model_state_dict"])
    audio_model.eval()
    logger.info(f"Loaded audio model from {args.audio_checkpoint}")

    output_dir = Path("outputs/logs")

    for dataset_name in args.datasets:
        logger.info(f"\n{'#'*60}\n# Evaluating on: {dataset_name}\n{'#'*60}")

        if dataset_name == "fakeavceleb":
            # In-domain: use speaker-disjoint test split
            dataloaders = build_dataloaders(config, phase="finetune")
            test_loader = dataloaders["test"]
        else:
            # Cross-dataset: load entire dataset for zero-shot eval
            try:
                test_loader = build_test_dataloader(config, dataset_name)
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Skipping {dataset_name}: {e}")
                continue

        logger.info(f"Test set ({dataset_name}): {len(test_loader.dataset)} samples")

        if len(test_loader.dataset) == 0:
            logger.warning(f"No preprocessed samples found for {dataset_name}, skipping")
            continue

        # Run inference
        predictions = run_cascade_inference(sync_model, audio_model, test_loader, device)

        # Evaluate all fusion strategies
        evaluate_cascade(predictions, output_dir, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
