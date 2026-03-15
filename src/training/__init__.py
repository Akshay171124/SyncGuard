"""SyncGuard training module.

Loss functions, dataset, and training utilities.
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
]
