#!/usr/bin/env python3
"""CLI entry point for SyncGuard Phase 2: Fine-tuning.

Usage:
    python scripts/train_finetune.py --config configs/default.yaml \
        --pretrain_ckpt outputs/checkpoints/pretrain_best.pt

    python scripts/train_finetune.py --config configs/default.yaml \
        --resume outputs/checkpoints/finetune_epoch_14.pt
"""

from src.training.finetune import main

if __name__ == "__main__":
    main()
