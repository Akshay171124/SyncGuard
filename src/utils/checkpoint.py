"""Shared checkpoint saving.

One saver for every training stage. `config` is required so that no stage
can produce a checkpoint without provenance. The save is atomic: write to
a temporary file, then os.replace, which is atomic on POSIX.
"""

import logging
import os
from pathlib import Path

import torch

from src.utils.provenance import build_provenance

logger = logging.getLogger(__name__)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    criterion,
    epoch: int,
    val_metrics: dict,
    path,
    config: dict,
    wandb_run_id: str | None = None,
    dataset_manifest: str | None = None,
) -> Path:
    """Atomically save a training checkpoint with embedded provenance.

    Args:
        model: Model whose state_dict is saved.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        criterion: Loss state (MoCo queue and temperature for pretraining).
        epoch: Current epoch.
        val_metrics: Validation metrics for this epoch.
        path: Destination path.
        config: Resolved run config, hashed into the provenance block.
        wandb_run_id: W&B run id, if a run is active.
        dataset_manifest: Hash of the dataset manifest, if available.

    Returns:
        The path written.
    """
    path = Path(path)
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
            "provenance": build_provenance(config, wandb_run_id, dataset_manifest),
        },
        tmp_path,
    )
    os.replace(str(tmp_path), str(path))  # Atomic on POSIX
    logger.info(f"Checkpoint saved: {path}")
    return path
