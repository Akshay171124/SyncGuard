"""Phase 2: Fine-tuning loop for SyncGuard.

Loads pretrained encoders and trains the full pipeline (encoders + classifier)
with combined loss: L_InfoNCE + γ*L_temp + δ*L_cls.
Includes hard negative annealing and early stopping.

Usage:
    python -m src.training.finetune --config configs/default.yaml \
        --pretrain_ckpt outputs/checkpoints/pretrain_best.pt
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import wandb

from src.models.syncguard import SyncGuard, build_syncguard
from src.training.losses import CombinedLoss, build_finetune_loss
from src.training.dataset import SyncGuardBatch
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


def compute_auc_roc(labels: list[int], scores: list[float]) -> float:
    """Compute AUC-ROC from labels and prediction scores.

    Simple trapezoidal implementation to avoid sklearn dependency during training.

    Args:
        labels: Binary ground truth labels.
        scores: Prediction scores (higher = more likely fake).

    Returns:
        AUC-ROC value.
    """
    if len(set(labels)) < 2:
        return 0.5  # Undefined if only one class

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
    """Compute Equal Error Rate.

    Args:
        labels: Binary ground truth labels.
        scores: Prediction scores (higher = more likely fake).

    Returns:
        EER value.
    """
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


def get_hard_negative_ratio(epoch: int, config: dict) -> float:
    """Compute annealed hard negative ratio for current epoch.

    Linearly anneals from 0% to target ratio over anneal_epochs.

    Args:
        epoch: Current epoch number.
        config: Full config dict.

    Returns:
        Hard negative ratio for this epoch.
    """
    ft_cfg = config["training"]["finetune"]
    target_ratio = ft_cfg.get("hard_negative_ratio", 0.2)
    anneal_epochs = ft_cfg.get("hard_negative_anneal_epochs", 10)

    if anneal_epochs <= 0:
        return target_ratio

    return min(target_ratio, target_ratio * epoch / anneal_epochs)


def build_optimizer(model: SyncGuard, criterion: CombinedLoss, config: dict) -> AdamW:
    """Build optimizer for fine-tuning.

    Includes both model parameters and learnable temperature from criterion.

    Args:
        model: SyncGuard model.
        criterion: CombinedLoss (has learnable temperature).
        config: Full config dict.

    Returns:
        AdamW optimizer.
    """
    ft_cfg = config["training"]["finetune"]
    params = list(model.parameters()) + list(criterion.parameters())
    trainable = [p for p in params if p.requires_grad]
    return AdamW(
        trainable,
        lr=ft_cfg["lr"],
        weight_decay=ft_cfg["weight_decay"],
    )


def build_scheduler(
    optimizer: AdamW, config: dict, steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build cosine LR scheduler with linear warmup.

    Args:
        optimizer: AdamW optimizer.
        config: Full config dict.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        LR scheduler (step-level).
    """
    ft_cfg = config["training"]["finetune"]
    warmup_epochs = ft_cfg.get("warmup_epochs", 3)
    total_epochs = ft_cfg["epochs"]

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    cosine_steps = max(total_steps - warmup_steps, 1)

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=max(warmup_steps, 1),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def validate(
    model: SyncGuard,
    criterion: CombinedLoss,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and compute metrics.

    Args:
        model: SyncGuard model.
        criterion: CombinedLoss instance.
        val_loader: Validation DataLoader.
        device: Torch device.

    Returns:
        Dict with avg losses, val_auc, val_eer, temperature.
    """
    model.eval()
    total_loss = 0.0
    total_nce = 0.0
    total_temp = 0.0
    total_cls = 0.0
    n_batches = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in val_loader:
            batch: SyncGuardBatch
            mouth_crops = batch.mouth_crops.to(device)
            waveforms = batch.waveforms.to(device)
            mask = batch.mask.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels.to(device)
            ear = batch.ear_features.to(device) if batch.ear_features is not None else None

            # Full forward pass
            output = model(mouth_crops, waveforms, lengths=lengths, ear_features=ear)

            # Align mask to model output length
            T = output.v_embeds.shape[1]
            mask_aligned = mask[:, :T]

            # Compute loss — skip queue update during validation
            loss_dict = criterion(
                output.v_embeds, output.a_embeds,
                output.logits, labels,
                mask=mask_aligned,
                audio_logits=output.audio_logits,
                update_queue=False,
            )

            total_loss += loss_dict["loss"].item()
            total_nce += loss_dict["loss_infonce"].item()
            total_temp += loss_dict["loss_temp"].item()
            total_cls += loss_dict["loss_cls"].item()
            n_batches += 1

            # Collect predictions for AUC
            probs = torch.sigmoid(output.logits.squeeze(-1))
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())

    model.train()

    if n_batches == 0:
        return {
            "avg_loss": 0.0, "avg_loss_infonce": 0.0,
            "avg_loss_temp": 0.0, "avg_loss_cls": 0.0,
            "val_auc": 0.5, "val_eer": 0.5,
            "temperature": 0.0,
        }

    return {
        "avg_loss": total_loss / n_batches,
        "avg_loss_infonce": total_nce / n_batches,
        "avg_loss_temp": total_temp / n_batches,
        "avg_loss_cls": total_cls / n_batches,
        "val_auc": compute_auc_roc(all_labels, all_scores),
        "val_eer": compute_eer(all_labels, all_scores),
        "temperature": criterion.temperature.item(),
    }


def save_checkpoint(
    model: SyncGuard,
    optimizer: AdamW,
    scheduler,
    criterion: CombinedLoss,
    epoch: int,
    val_metrics: dict,
    path: Path,
):
    """Save training checkpoint.

    Args:
        model: SyncGuard model.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        criterion: Loss function state.
        epoch: Current epoch.
        val_metrics: Validation metrics dict.
        path: Save path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": val_metrics,
        },
        tmp_path,
    )
    os.replace(str(tmp_path), str(path))  # Atomic on POSIX
    logger.info(f"Checkpoint saved: {path}")


def train(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pretrain_ckpt: str = None,
    resume_from: str = None,
):
    """Run Phase 2 fine-tuning.

    Args:
        config: Full config dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        pretrain_ckpt: Path to Phase 1 pretrained checkpoint.
        resume_from: Optional checkpoint path to resume fine-tuning.
    """
    device = get_device(config)
    ft_cfg = config["training"]["finetune"]
    epochs = ft_cfg["epochs"]
    patience = 5
    checkpoint_dir = Path("outputs/checkpoints")
    log_path = Path("outputs/logs/finetune.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # CB-5: Set random seeds for reproducibility
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

    # Build model and load pretrained weights
    model = build_syncguard(config).to(device)

    if pretrain_ckpt:
        ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
        # HP-6: Log missing/unexpected keys to verify transfer
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        encoder_missing = [k for k in missing if "classifier" not in k and "audio_classifier" not in k and "fusion_weight" not in k]
        if encoder_missing:
            logger.error(f"ENCODER weights missing from pretrained checkpoint: {encoder_missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint (ignored): {unexpected}")
        logger.info(
            f"Loaded pretrained weights from {pretrain_ckpt} "
            f"(missing={len(missing)} keys, unexpected={len(unexpected)} keys)"
        )

    criterion = build_finetune_loss(config).to(device)
    optimizer = build_optimizer(model, criterion, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    start_epoch = 0
    best_val_auc = 0.0
    epochs_without_improvement = 0
    history = []

    # Resume from fine-tuning checkpoint
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        criterion.load_state_dict(ckpt["criterion_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt["val_metrics"].get("val_auc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_auc={best_val_auc:.4f}")

    # Initialize wandb
    audio_swap_ratio = ft_cfg.get("audio_swap_ratio", 0.0)
    use_audio_head = config["model"].get("audio_head", False)

    # Determine wandb run name
    run_name = "phase2-finetune"
    if use_audio_head:
        run_name = "phase2-finetune-dualhead"
    elif audio_swap_ratio > 0:
        run_name = "phase2-finetune-audioswap"

    wandb.init(
        project="SyncGuard",
        name=run_name,
        config={
            "phase": "finetune",
            "epochs": epochs,
            "batch_size": ft_cfg["batch_size"],
            "lr": ft_cfg["lr"],
            "weight_decay": ft_cfg["weight_decay"],
            "warmup_epochs": ft_cfg.get("warmup_epochs", 3),
            "gamma": ft_cfg["gamma"],
            "delta": ft_cfg["delta"],
            "hard_negative_ratio": ft_cfg.get("hard_negative_ratio", 0.2),
            "audio_swap_ratio": audio_swap_ratio,
            "dataset": "fakeavceleb",
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "pretrain_ckpt": pretrain_ckpt or "none",
        },
        tags=["finetune", "fakeavceleb", "audio-swap"] if audio_swap_ratio > 0 else ["finetune", "fakeavceleb"],
    )

    logger.info(
        f"Starting fine-tuning: {epochs} epochs, "
        f"batch_size={ft_cfg['batch_size']}, lr={ft_cfg['lr']}, "
        f"γ={ft_cfg['gamma']}, δ={ft_cfg['delta']}, "
        f"device={device}"
    )

    nan_skips = 0

    for epoch in range(start_epoch, epochs):
        model.train()

        # Anneal hard negative ratio
        hn_ratio = get_hard_negative_ratio(epoch, config)
        if hasattr(train_loader.dataset, "hard_negative_ratio"):
            train_loader.dataset.hard_negative_ratio = hn_ratio

        epoch_loss = 0.0
        epoch_nce = 0.0
        epoch_temp = 0.0
        epoch_cls = 0.0
        n_batches = 0
        t_start = time.time()

        for batch in train_loader:
            batch: SyncGuardBatch
            mouth_crops = batch.mouth_crops.to(device)
            waveforms = batch.waveforms.to(device)
            mask = batch.mask.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels.to(device)
            ear = batch.ear_features.to(device) if batch.ear_features is not None else None

            # Full forward pass
            output = model(mouth_crops, waveforms, lengths=lengths, ear_features=ear)

            # Align mask
            T = output.v_embeds.shape[1]
            mask_aligned = mask[:, :T]

            # Combined loss
            loss_dict = criterion(
                output.v_embeds, output.a_embeds,
                output.logits, labels,
                mask=mask_aligned,
                audio_logits=output.audio_logits,
            )
            loss = loss_dict["loss"]

            # CB-6: NaN guard — skip batch instead of corrupting weights
            if not torch.isfinite(loss):
                nan_skips += 1
                logger.warning(
                    f"Non-finite loss at epoch {epoch}, batch {n_batches}: "
                    f"loss={loss.item()}, tau={criterion.temperature.item():.6f}. "
                    f"Skipping batch. Total skips: {nan_skips}"
                )
                if nan_skips > 50:
                    logger.error(f"Too many NaN batches ({nan_skips}). Halting.")
                    raise RuntimeError(f"Training halted: {nan_skips} NaN batches")
                scheduler.step()
                continue

            # Backward — HP-5: clip both model and criterion (learnable temperature)
            optimizer.zero_grad()
            loss.backward()
            all_params = list(model.parameters()) + list(criterion.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_nce += loss_dict["loss_infonce"].item()
            epoch_temp += loss_dict["loss_temp"].item()
            epoch_cls += loss_dict["loss_cls"].item()
            n_batches += 1

        # Epoch metrics
        epoch_time = time.time() - t_start
        n = max(n_batches, 1)

        # Validation
        val_metrics = validate(model, criterion, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch,
            "train_loss": epoch_loss / n,
            "train_loss_infonce": epoch_nce / n,
            "train_loss_temp": epoch_temp / n,
            "train_loss_cls": epoch_cls / n,
            "val_loss": val_metrics["avg_loss"],
            "val_loss_infonce": val_metrics["avg_loss_infonce"],
            "val_loss_temp": val_metrics["avg_loss_temp"],
            "val_loss_cls": val_metrics["avg_loss_cls"],
            "val_auc": val_metrics["val_auc"],
            "val_eer": val_metrics["val_eer"],
            "temperature": val_metrics["temperature"],
            "lr": current_lr,
            "hard_negative_ratio": hn_ratio,
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(epoch_log)

        logger.info(
            f"Epoch {epoch}/{epochs-1} ({epoch_time:.0f}s) | "
            f"loss={epoch_loss/n:.4f} "
            f"(nce={epoch_nce/n:.4f} temp={epoch_temp/n:.4f} cls={epoch_cls/n:.4f}) | "
            f"val_auc={val_metrics['val_auc']:.4f} "
            f"val_eer={val_metrics['val_eer']:.4f} | "
            f"τ={val_metrics['temperature']:.4f} lr={current_lr:.2e} hn={hn_ratio:.2f}"
        )

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": epoch_loss / n,
            "train/loss_infonce": epoch_nce / n,
            "train/loss_temp": epoch_temp / n,
            "train/loss_cls": epoch_cls / n,
            "val/loss": val_metrics["avg_loss"],
            "val/loss_infonce": val_metrics["avg_loss_infonce"],
            "val/loss_temp": val_metrics["avg_loss_temp"],
            "val/loss_cls": val_metrics["avg_loss_cls"],
            "val/auc": val_metrics["val_auc"],
            "val/eer": val_metrics["val_eer"],
            "temperature": val_metrics["temperature"],
            "lr": current_lr,
            "hard_negative_ratio": hn_ratio,
            "epoch_time_s": round(epoch_time, 1),
        })

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, criterion,
                epoch, val_metrics,
                checkpoint_dir / f"finetune_epoch_{epoch}.pt",
            )

        # Save best checkpoint (by val AUC)
        if val_metrics["val_auc"] > best_val_auc:
            best_val_auc = val_metrics["val_auc"]
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, scheduler, criterion,
                epoch, val_metrics,
                checkpoint_dir / "finetune_best.pt",
            )
            logger.info(f"  New best val_auc: {best_val_auc:.4f}")
        else:
            epochs_without_improvement += 1

        # Save metrics log
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} — "
                f"no improvement in val_auc for {patience} epochs"
            )
            break

    logger.info(f"Fine-tuning complete. Best val_auc: {best_val_auc:.4f}")
    wandb.finish()
    return history


def main():
    """CLI entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="SyncGuard Phase 2: Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--pretrain_ckpt", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume fine-tuning")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)

    from src.training.dataset import build_dataloaders

    dataloaders = build_dataloaders(config, phase="finetune")

    train(
        config=config,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        pretrain_ckpt=args.pretrain_ckpt,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
