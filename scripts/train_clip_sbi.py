#!/usr/bin/env python3
"""Train SyncGuard with CLIP backbone + SBI augmentation.

Single-phase training — skip contrastive pretraining.
CLIP is already pretrained, go directly to supervised fine-tuning.

Usage:
    python scripts/train_clip_sbi.py --config configs/clip_sbi.yaml
    python scripts/train_clip_sbi.py --config configs/clip_sbi.yaml --resume outputs/checkpoints/finetune_best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.finetune import main

if __name__ == "__main__":
    main()
