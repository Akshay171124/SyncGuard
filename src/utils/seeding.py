"""Deterministic seeding for every training entry point.

April's pretrain and finetune runs were unseeded, which is one reason their
checkpoints cannot be regenerated. Every entry point calls seed_everything()
before constructing models or dataloaders.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all RNGs used by the training pipeline.

    Args:
        seed: Value applied to python, numpy, and torch RNGs.
        deterministic: Force cuDNN into deterministic mode. Costs some
            throughput but makes runs comparable across launches.

    Returns:
        The seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
