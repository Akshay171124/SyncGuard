"""Phase 1: Contrastive pretraining loop for SyncGuard.

Trains visual + audio encoders with InfoNCE loss and MoCo memory bank.
No classification head or temporal consistency loss in this phase.

Usage:
    python -m src.training.pretrain --config configs/default.yaml
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

from src.models.syncguard import SyncGuard, build_syncguard
from src.training.losses import PretrainLoss, build_pretrain_loss
from src.training.dataset import SyncGuardDataset, collate_syncguard, SyncGuardBatch
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


def build_optimizer(model: SyncGuard, criterion: PretrainLoss, config: dict) -> AdamW:
    """Build optimizer for pretraining (model + criterion trainable parameters).

    Args:
        model: SyncGuard model.
        criterion: PretrainLoss instance (contains learnable temperature).
        config: Full config dict.

    Returns:
        AdamW optimizer.
    """
    pt_cfg = config["training"]["pretrain"]
    params = (
        [p for p in model.parameters() if p.requires_grad]
        + [p for p in criterion.parameters() if p.requires_grad]
    )
    return AdamW(
        params,
        lr=pt_cfg["lr"],
        weight_decay=pt_cfg["weight_decay"],
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
    pt_cfg = config["training"]["pretrain"]
    warmup_epochs = pt_cfg.get("warmup_epochs", 2)
    total_epochs = pt_cfg["epochs"]

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
    criterion: PretrainLoss,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and compute average metrics.

    Args:
        model: SyncGuard model.
        criterion: PretrainLoss instance.
        val_loader: Validation DataLoader.
        device: Torch device.

    Returns:
        Dict with avg_loss, avg_sync_score, temperature.
    """
    model.eval()
    total_loss = 0.0
    total_sync_score = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch: SyncGuardBatch
            mouth_crops = batch.mouth_crops.to(device)
            waveforms = batch.waveforms.to(device)
            mask = batch.mask.to(device)
            lengths = batch.lengths.to(device)

            # Encode only (no classifier in pretraining)
            v_embeds = model.encode_visual(mouth_crops)
            a_embeds = model.encode_audio(waveforms)

            # Align sequences
            v_embeds, a_embeds = model.align_sequences(v_embeds, a_embeds)

            # Truncate mask to aligned length
            T = v_embeds.shape[1]
            mask_aligned = mask[:, :T]

            # Compute loss (don't update queue during validation)
            loss_dict = criterion(v_embeds, a_embeds, mask=mask_aligned)

            # Compute mean sync-score
            sync_scores = model.compute_sync_scores(v_embeds, a_embeds)
            masked_scores = sync_scores * mask_aligned.float()
            avg_score = masked_scores.sum() / mask_aligned.float().sum().clamp(min=1)

            total_loss += loss_dict["loss"].item()
            total_sync_score += avg_score.item()
            n_batches += 1

    model.train()

    if n_batches == 0:
        return {"avg_loss": 0.0, "avg_sync_score": 0.0, "temperature": 0.0}

    return {
        "avg_loss": total_loss / n_batches,
        "avg_sync_score": total_sync_score / n_batches,
        "temperature": criterion.temperature.item(),
    }


def save_checkpoint(
    model: SyncGuard,
    optimizer: AdamW,
    scheduler,
    criterion: PretrainLoss,
    epoch: int,
    val_metrics: dict,
    path: Path,
):
    """Save training checkpoint.

    Args:
        model: SyncGuard model.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        criterion: Loss function state (includes MoCo queue + temperature).
        epoch: Current epoch.
        val_metrics: Validation metrics dict.
        path: Save path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": val_metrics,
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")


def train(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    resume_from: str = None,
):
    """Run Phase 1 contrastive pretraining.

    Args:
        config: Full config dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        resume_from: Optional checkpoint path to resume from.
    """
    device = get_device(config)
    pt_cfg = config["training"]["pretrain"]
    epochs = pt_cfg["epochs"]
    checkpoint_dir = Path("outputs/checkpoints")
    log_path = Path("outputs/logs/pretrain.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    use_cmp = pt_cfg.get("cross_modal_prediction", True)
    wandb.init(
        project="SyncGuard",
        name="phase1-pretrain-cmp" if use_cmp else "phase1-pretrain-learnable-tau",
        config={
            "phase": "pretrain",
            "epochs": epochs,
            "batch_size": pt_cfg["batch_size"],
            "lr": pt_cfg["lr"],
            "weight_decay": pt_cfg["weight_decay"],
            "warmup_epochs": pt_cfg.get("warmup_epochs", 2),
            "moco_queue_size": pt_cfg.get("moco_queue_size", 4096),
            "temperature": pt_cfg.get("temperature", 0.07),
            "cross_modal_prediction": use_cmp,
            "cmp_weight": pt_cfg.get("cmp_weight", 0.5),
            "cmp_mask_ratio": pt_cfg.get("cmp_mask_ratio", 0.3),
            "dataset": "avspeech",
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        },
        tags=["pretrain", "contrastive", "cross-modal-prediction"] if use_cmp else ["pretrain", "contrastive", "avspeech"],
    )

    # Build model, loss, optimizer, scheduler
    model = build_syncguard(config).to(device)
    criterion = build_pretrain_loss(config).to(device)
    criterion.infonce.queue.to(device)
    optimizer = build_optimizer(model, criterion, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    start_epoch = 0
    best_val_loss = float("inf")
    history = []

    # Resume from checkpoint
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        criterion.load_state_dict(ckpt["criterion_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_metrics"].get("avg_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    logger.info(
        f"Starting pretraining: {epochs} epochs, "
        f"batch_size={pt_cfg['batch_size']}, lr={pt_cfg['lr']}, "
        f"device={device}"
    )

    total_batches = len(train_loader)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_infonce = 0.0
        epoch_cmp = 0.0
        epoch_sync_score = 0.0
        n_batches = 0
        t_start = time.time()

        for batch in train_loader:
            batch: SyncGuardBatch
            mouth_crops = batch.mouth_crops.to(device)
            waveforms = batch.waveforms.to(device)
            mask = batch.mask.to(device)

            # Forward: encode only (no classifier)
            v_embeds = model.encode_visual(mouth_crops)
            a_embeds = model.encode_audio(waveforms)
            v_embeds, a_embeds = model.align_sequences(v_embeds, a_embeds)

            T = v_embeds.shape[1]
            mask_aligned = mask[:, :T]

            # Compute loss (InfoNCE + cross-modal prediction)
            loss_dict = criterion(v_embeds, a_embeds, mask=mask_aligned)
            loss = loss_dict["loss"]

            # Backward — clip grads for both model and CMP predictor heads
            optimizer.zero_grad()
            loss.backward()
            all_params = list(model.parameters()) + list(criterion.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            with torch.no_grad():
                sync_scores = model.compute_sync_scores(v_embeds, a_embeds)
                masked_scores = sync_scores * mask_aligned.float()
                avg_score = masked_scores.sum() / mask_aligned.float().sum().clamp(min=1)

            epoch_loss += loss.item()
            epoch_infonce += loss_dict["loss_infonce"].item()
            epoch_cmp += loss_dict["loss_cmp"].item()
            epoch_sync_score += avg_score.item()
            n_batches += 1

            # Per-batch progress logging
            if n_batches % 100 == 0 or n_batches == 1:
                elapsed = time.time() - t_start
                eta_epoch = (elapsed / n_batches) * (total_batches - n_batches)
                logger.info(
                    f"[Epoch {epoch+1}/{epochs}] Batch {n_batches}/{total_batches} "
                    f"loss={loss.item():.4f} infonce={loss_dict['loss_infonce'].item():.4f} "
                    f"cmp={loss_dict['loss_cmp'].item():.4f} sync={avg_score.item():.3f} "
                    f"tau={criterion.tau.item():.4f} "
                    f"ETA={eta_epoch/60:.0f}min"
                )

        # Epoch metrics
        epoch_time = time.time() - t_start
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_infonce = epoch_infonce / max(n_batches, 1)
        avg_train_cmp = epoch_cmp / max(n_batches, 1)
        avg_train_sync = epoch_sync_score / max(n_batches, 1)

        # Validation
        val_metrics = validate(model, criterion, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_infonce": avg_train_infonce,
            "train_cmp": avg_train_cmp,
            "train_sync_score": avg_train_sync,
            "val_loss": val_metrics["avg_loss"],
            "val_sync_score": val_metrics["avg_sync_score"],
            "temperature": val_metrics["temperature"],
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(epoch_log)

        logger.info(
            f"Epoch {epoch}/{epochs-1} ({epoch_time:.0f}s) | "
            f"loss={avg_train_loss:.4f} (nce={avg_train_infonce:.4f} cmp={avg_train_cmp:.4f}) | "
            f"val_loss={val_metrics['avg_loss']:.4f} | "
            f"sync={val_metrics['avg_sync_score']:.4f} | "
            f"τ={val_metrics['temperature']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/infonce": avg_train_infonce,
            "train/cmp": avg_train_cmp,
            "train/sync_score": avg_train_sync,
            "val/loss": val_metrics["avg_loss"],
            "val/sync_score": val_metrics["avg_sync_score"],
            "temperature": val_metrics["temperature"],
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 1),
        })

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, criterion,
                epoch, val_metrics,
                checkpoint_dir / f"pretrain_epoch_{epoch}.pt",
            )

        # Save best checkpoint
        if val_metrics["avg_loss"] < best_val_loss:
            best_val_loss = val_metrics["avg_loss"]
            save_checkpoint(
                model, optimizer, scheduler, criterion,
                epoch, val_metrics,
                checkpoint_dir / "pretrain_best.pt",
            )
            logger.info(f"  New best val_loss: {best_val_loss:.4f}")

        # Save metrics log
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    logger.info(f"Pretraining complete. Best val_loss: {best_val_loss:.4f}")
    wandb.finish()
    return history


def main():
    """CLI entry point for pretraining."""
    parser = argparse.ArgumentParser(description="SyncGuard Phase 1: Contrastive Pretraining")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)

    # Build dataloaders
    from src.training.dataset import build_dataloaders

    dataloaders = build_dataloaders(config, phase="pretrain")

    train(
        config=config,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
