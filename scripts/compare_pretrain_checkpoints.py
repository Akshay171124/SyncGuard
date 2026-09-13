"""Compare pretraining checkpoints under one fixed-temperature protocol.

Both raw InfoNCE loss and sync-score depend on tau, so neither can be compared
across runs that trained at different temperatures. InfoNCE forms logits as
similarity/tau, and the encoder's embedding geometry adapts: at a smaller tau
matched pairs need less absolute similarity to dominate the negatives, so mean
matched-pair similarity — which is what sync-score measures — settles lower.

Evaluating every checkpoint on the same validation set at the same reference
tau removes that confound. Raw loss and the checkpoint's own tau are reported
alongside for context, but only the fixed-tau column is comparable.

Usage:
    python scripts/compare_pretrain_checkpoints.py --config configs/rebuild_pretrain.yaml \
        --checkpoints a.pt b.pt --max-batches 60
"""

import argparse
import logging
from pathlib import Path

import torch

from src.models.syncguard import build_syncguard
from src.training.losses import build_pretrain_loss
from src.training.pretrain import validate
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


def evaluate(ckpt_path: str, config: dict, val_loader, device, ref_tau: float) -> dict:
    """Load one checkpoint and validate it at the reference temperature.

    Args:
        ckpt_path: Path to the checkpoint file.
        config: Resolved config, used to rebuild the model and loss.
        val_loader: Shared validation DataLoader, identical for every checkpoint.
        device: Torch device.
        ref_tau: Fixed temperature used for the comparable loss.

    Returns:
        Dict with the checkpoint's epoch, its own tau, and its metrics.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_syncguard(config).to(device)
    model.load_state_dict(ck["model_state_dict"])

    criterion = build_pretrain_loss(config).to(device)
    if "criterion_state_dict" in ck:
        criterion.load_state_dict(ck["criterion_state_dict"])

    metrics = validate(model, criterion, val_loader, device, selection_temperature=ref_tau)
    return {
        "name": Path(ckpt_path).name,
        "epoch": ck.get("epoch"),
        "own_tau": criterion.temperature.item(),
        "fixed_tau_loss": metrics["avg_loss_fixed_tau"],
        "raw_loss": metrics["avg_loss"],
        "sync": metrics["avg_sync_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rebuild_pretrain.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--ref-tau", type=float, default=0.07)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = load_config(args.config)
    device = get_device(config)

    from src.training.dataset import build_dataloaders

    val_loader = build_dataloaders(config, phase="pretrain")["val"]

    print(f"\nValidation set: {len(val_loader.dataset)} samples")
    print(f"Reference tau : {args.ref_tau}\n")
    header = f"{'checkpoint':<32}{'ep':>4}{'own_tau':>9}{'FIXED-TAU':>12}{'raw':>9}{'sync':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for path in args.checkpoints:
        if not Path(path).exists():
            print(f"{Path(path).name:<32}  MISSING")
            continue
        r = evaluate(path, config, val_loader, device, args.ref_tau)
        rows.append(r)
        print(f"{r['name']:<32}{r['epoch']:>4}{r['own_tau']:>9.4f}"
              f"{r['fixed_tau_loss']:>12.4f}{r['raw_loss']:>9.4f}{r['sync']:>8.4f}")

    if rows:
        best = min(rows, key=lambda r: r["fixed_tau_loss"])
        print(f"\nBest by fixed-tau loss: {best['name']} "
              f"(epoch {best['epoch']}, {best['fixed_tau_loss']:.4f})")
        print("Only the FIXED-TAU column is comparable across runs.")


if __name__ == "__main__":
    main()
