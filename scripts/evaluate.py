#!/usr/bin/env python
"""CLI wrapper to run SyncGuard evaluation.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint outputs/checkpoints/finetune_best.pt
"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluate import main

if __name__ == "__main__":
    main()
