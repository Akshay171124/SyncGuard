"""Train standalone audio-only deepfake classifier.

Trains a Wav2Vec2-based binary classifier on FakeAVCeleb audio tracks.
Used for inference-time cascade with the sync-based SyncGuard model.

Usage:
    python scripts/train_audio_classifier.py --config configs/default.yaml
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import wandb

from src.models.audio_classifier import build_standalone_audio_classifier
from src.training.dataset import build_dataloaders, SyncGuardBatch
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


def compute_auc_roc(labels: list[int], scores: list[float]) -> float:
    """Compute AUC-ROC from labels and prediction scores."""
    if len(set(labels)) < 2:
        return 0.5

    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr

    return auc


def compute_eer(labels: list[int], scores: list[float]) -> float:
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


def validate(model, val_loader, criterion, device):
    """Run validation and compute metrics.

    Args:
        model: Audio classifier.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Dict with val_loss, val_auc, val_eer, and per-category AUCs.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels = []
    all_scores = []
    all_categories = []

    with torch.no_grad():
        for batch in val_loader:
            batch: SyncGuardBatch
            waveforms = batch.waveforms.to(device)
            labels = batch.labels.to(device)

            logits = model(waveforms)
            loss = criterion(logits.squeeze(-1), labels.float())

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits.squeeze(-1))
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())
            all_categories.extend(batch.categories)

    model.train()

    if n_batches == 0:
        return {"val_loss": 0.0, "val_auc": 0.5, "val_eer": 0.5}

    # Overall metrics
    val_auc = compute_auc_roc(all_labels, all_scores)
    val_eer = compute_eer(all_labels, all_scores)

    # Per-category AUC
    cat_aucs = {}
    for cat in set(all_categories):
        cat_labels = [l for l, c in zip(all_labels, all_categories) if c == cat]
        cat_scores = [s for s, c in zip(all_scores, all_categories) if c == cat]
        if len(set(cat_labels)) >= 2:
            cat_aucs[cat] = compute_auc_roc(cat_labels, cat_scores)

    return {
        "val_loss": total_loss / n_batches,
        "val_auc": val_auc,
        "val_eer": val_eer,
        "cat_aucs": cat_aucs,
    }


def train(config: dict):
    """Train the standalone audio classifier.

    Args:
        config: Full config dict.
    """
    device = get_device(config)

    # Hyperparameters
    epochs = 30
    lr = 1e-4
    weight_decay = 1e-4
    warmup_epochs = 3
    patience = 7
    batch_size = 32

    checkpoint_dir = Path("outputs/checkpoints")
    log_path = Path("outputs/logs/audio_classifier.json")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_standalone_audio_classifier(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Audio classifier: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer (only trainable params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)

    # Build dataloaders (reuse FakeAVCeleb splits, no audio-swap needed)
    # Override batch size for audio-only training
    config_copy = {**config}
    config_copy["training"] = {**config["training"]}
    config_copy["training"]["finetune"] = {
        **config["training"]["finetune"],
        "batch_size": batch_size,
        "audio_swap_ratio": 0.0,
        "hard_negative_ratio": 0.0,
    }
    dataloaders = build_dataloaders(config_copy, phase="finetune")
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Scheduler
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    cosine_steps = max(total_steps - warmup_steps, 1)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # wandb
    wandb.init(
        project="SyncGuard",
        name="audio-classifier-standalone",
        config={
            "phase": "audio_classifier",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "trainable_params": trainable_params,
        },
        tags=["audio-classifier", "fakeavceleb"],
    )

    logger.info(f"Starting audio classifier training: {epochs} epochs, lr={lr}, device={device}")

    best_val_auc = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        for batch in train_loader:
            batch: SyncGuardBatch
            waveforms = batch.waveforms.to(device)
            labels = batch.labels.to(device)

            logits = model(waveforms)
            loss = criterion(logits.squeeze(-1), labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        # Per-category log string
        cat_str = " | ".join(
            f"{k}={v:.4f}" for k, v in sorted(val_metrics.get("cat_aucs", {}).items())
        )

        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_metrics["val_loss"],
            "val_auc": val_metrics["val_auc"],
            "val_eer": val_metrics["val_eer"],
            "cat_aucs": val_metrics.get("cat_aucs", {}),
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(epoch_log)

        logger.info(
            f"Epoch {epoch}/{epochs-1} ({epoch_time:.0f}s) | "
            f"loss={avg_loss:.4f} | "
            f"val_auc={val_metrics['val_auc']:.4f} val_eer={val_metrics['val_eer']:.4f} | "
            f"lr={current_lr:.2e} | {cat_str}"
        )

        # wandb logging
        log_dict = {
            "epoch": epoch,
            "train/loss": avg_loss,
            "val/loss": val_metrics["val_loss"],
            "val/auc": val_metrics["val_auc"],
            "val/eer": val_metrics["val_eer"],
            "lr": current_lr,
        }
        for cat, auc in val_metrics.get("cat_aucs", {}).items():
            log_dict[f"val/auc_{cat}"] = auc
        wandb.log(log_dict)

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = checkpoint_dir / f"audio_clf_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        # Best checkpoint
        if val_metrics["val_auc"] > best_val_auc:
            best_val_auc = val_metrics["val_auc"]
            epochs_without_improvement = 0
            ckpt_path = checkpoint_dir / "audio_clf_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            logger.info(f"  New best val_auc: {best_val_auc:.4f}")
        else:
            epochs_without_improvement += 1

        # Save log
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping at epoch {epoch} — no improvement for {patience} epochs")
            break

    logger.info(f"Audio classifier training complete. Best val_auc: {best_val_auc:.4f}")
    wandb.finish()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train standalone audio deepfake classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
