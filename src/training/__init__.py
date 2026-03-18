"""SyncGuard training module.

Loss functions, dataset, training loops, and utilities.
"""

from src.training.losses import (
    MoCoQueue,
    InfoNCELoss,
    TemporalConsistencyLoss,
    CombinedLoss,
    PretrainLoss,
    build_pretrain_loss,
    build_finetune_loss,
)
from src.training.dataset import (
    SyncGuardBatch,
    SyncGuardDataset,
    collate_syncguard,
    build_dataloaders,
)
from src.training.pretrain import train as pretrain
from src.training.finetune import train as finetune

__all__ = [
    # Losses
    "MoCoQueue",
    "InfoNCELoss",
    "TemporalConsistencyLoss",
    "CombinedLoss",
    "PretrainLoss",
    "build_pretrain_loss",
    "build_finetune_loss",
    # Dataset
    "SyncGuardBatch",
    "SyncGuardDataset",
    "collate_syncguard",
    "build_dataloaders",
    # Training loops
    "pretrain",
    "finetune",
]
