#!/usr/bin/env python3
"""Train cross-attention embedding bypass for SyncGuard.

Stage 1: Train cross-attention head only (encoders + Bi-LSTM frozen)
Stage 2: End-to-end fusion fine-tuning (encoders frozen, Bi-LSTM + CA trainable)

Usage:
    python scripts/train_cross_attention.py --config configs/finetune_frozen.yaml \
        --checkpoint outputs/checkpoints/finetune_best.pt \
        --stage 1

    python scripts/train_cross_attention.py --config configs/finetune_frozen.yaml \
        --checkpoint outputs/checkpoints/ca_stage1_best.pt \
        --stage 2
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
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.syncguard import build_syncguard
from src.training.dataset import SyncGuardBatch, build_dataloaders
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


def compute_auc_roc(labels, scores):
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


def freeze_for_stage(model, stage):
    """Freeze/unfreeze parameters based on training stage."""
    # Always freeze encoders
    for p in model.visual_encoder.parameters():
        p.requires_grad = False
    for p in model.audio_encoder.parameters():
        p.requires_grad = False

    if stage == 1:
        # Freeze sync path (Bi-LSTM), only train cross-attention + embed classifier
        for p in model.classifier.parameters():
            p.requires_grad = False
        if hasattr(model, 'ca_fusion_weight'):
            model.ca_fusion_weight.requires_grad = False
        for p in model.cross_attn.parameters():
            p.requires_grad = True
        for p in model.embed_classifier.parameters():
            p.requires_grad = True
    elif stage == 2:
        # Unfreeze Bi-LSTM + cross-attention + fusion weight
        for p in model.classifier.parameters():
            p.requires_grad = True
        if hasattr(model, 'ca_fusion_weight'):
            model.ca_fusion_weight.requires_grad = True
        for p in model.cross_attn.parameters():
            p.requires_grad = True
        for p in model.embed_classifier.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Stage {stage}: {trainable:,} trainable / {total:,} total parameters")


def validate(model, val_loader, device, stage):
    """Run validation, return metrics dict."""
    model.eval()
    all_labels, all_scores = [], []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in val_loader:
            batch: SyncGuardBatch
            mc = batch.mouth_crops.to(device)
            wf = batch.waveforms.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels.to(device)
            ear = batch.ear_features.to(device) if batch.ear_features is not None else None

            output = model(mc, wf, lengths=lengths, ear_features=ear)

            if stage == 1:
                logits = output.embed_logits.squeeze(-1)
            else:
                logits = output.logits.squeeze(-1)

            loss = criterion(logits, labels.float())
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())

    model.train()
    auc = compute_auc_roc(all_labels, all_scores)
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_auc": auc,
    }


def train(config, checkpoint_path, stage, resume_from=None):
    """Train cross-attention bypass."""
    device = get_device(config)
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Seeds
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Stage config
    if stage == 1:
        epochs, lr, patience = 20, 1e-4, 5
        warmup_epochs = 2
        prefix = "ca_stage1"
    else:
        epochs, lr, patience = 10, 5e-5, 3
        warmup_epochs = 1
        prefix = "ca_stage2"

    # Build model
    config["model"].setdefault("cross_attention", {})["enabled"] = True
    model = build_syncguard(config).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Cross-attention keys will be missing (new module) — that's expected
    ca_missing = [k for k in missing if "cross_attn" not in k and "embed_classifier" not in k and "ca_fusion" not in k]
    if ca_missing:
        logger.error(f"Non-CA keys missing: {ca_missing}")
    logger.info(f"Loaded checkpoint from {checkpoint_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    # Freeze for stage
    freeze_for_stage(model, stage)

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Scheduler
    dataloaders = build_dataloaders(config, phase="finetune")
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    cosine_steps = max(total_steps - warmup_steps, 1)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    criterion = nn.BCEWithLogitsLoss()

    # Wandb
    wandb.init(
        project="SyncGuard",
        name=f"cross-attention-stage{stage}",
        config={"stage": stage, "lr": lr, "epochs": epochs, "checkpoint": checkpoint_path},
        tags=["cross-attention", f"stage{stage}"],
    )

    logger.info(f"Starting cross-attention Stage {stage}: {epochs} epochs, lr={lr}, device={device}")

    best_val_auc = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        # Re-freeze encoders (model.train() may have changed modes)
        model.visual_encoder.eval()
        model.audio_encoder.eval()

        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        for batch in train_loader:
            batch: SyncGuardBatch
            mc = batch.mouth_crops.to(device)
            wf = batch.waveforms.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels.to(device)
            ear = batch.ear_features.to(device) if batch.ear_features is not None else None

            output = model(mc, wf, lengths=lengths, ear_features=ear)

            if stage == 1:
                logits = output.embed_logits.squeeze(-1)
            else:
                logits = output.logits.squeeze(-1)

            loss = criterion(logits, labels.float())

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss at epoch {epoch}, batch {n_batches}. Skipping.")
                scheduler.step()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_metrics = validate(model, val_loader, device, stage)

        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_metrics["val_loss"],
            "val_auc": val_metrics["val_auc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(epoch_time, 1),
        }
        if hasattr(model, 'ca_fusion_weight'):
            epoch_log["fusion_weight"] = torch.sigmoid(model.ca_fusion_weight).item()

        history.append(epoch_log)

        logger.info(
            f"Epoch {epoch}/{epochs-1} ({epoch_time:.0f}s) | "
            f"loss={avg_loss:.4f} | val_auc={val_metrics['val_auc']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            + (f" | fusion_w={torch.sigmoid(model.ca_fusion_weight).item():.3f}" if hasattr(model, 'ca_fusion_weight') and model.ca_fusion_weight.requires_grad else "")
        )

        wandb.log(epoch_log)

        # Checkpoint
        if val_metrics["val_auc"] > best_val_auc:
            best_val_auc = val_metrics["val_auc"]
            epochs_without_improvement = 0
            save_path = checkpoint_dir / f"{prefix}_best.pt"
            tmp_path = save_path.with_suffix(".pt.tmp")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "stage": stage,
            }, tmp_path)
            os.replace(str(tmp_path), str(save_path))
            logger.info(f"  New best val_auc: {best_val_auc:.4f} → {save_path}")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            save_path = checkpoint_dir / f"{prefix}_epoch_{epoch}.pt"
            tmp_path = save_path.with_suffix(".pt.tmp")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "stage": stage,
            }, tmp_path)
            os.replace(str(tmp_path), str(save_path))

        # Save history
        with open(log_dir / f"{prefix}.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping at epoch {epoch} — no improvement for {patience} epochs")
            break

    logger.info(f"Stage {stage} complete. Best val_auc: {best_val_auc:.4f}")
    wandb.finish()
    return history


def main():
    parser = argparse.ArgumentParser(description="Train cross-attention embedding bypass")
    parser.add_argument("--config", type=str, default="configs/finetune_frozen.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to finetune checkpoint (Stage 1) or ca_stage1_best.pt (Stage 2)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Training stage: 1=train CA head only, 2=end-to-end fusion")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    train(config, args.checkpoint, args.stage)


if __name__ == "__main__":
    main()
