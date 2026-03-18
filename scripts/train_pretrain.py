#!/usr/bin/env python3
"""CLI entry point for SyncGuard Phase 1: Contrastive Pretraining.

Usage:
    python scripts/train_pretrain.py --config configs/default.yaml
    python scripts/train_pretrain.py --config configs/default.yaml --resume outputs/checkpoints/pretrain_epoch_9.pt
"""

from src.training.pretrain import main

if __name__ == "__main__":
    main()
