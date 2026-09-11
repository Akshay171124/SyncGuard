"""Print a checkpoint's epoch, validation metrics, and provenance block."""
import sys
from pathlib import Path

import torch


def describe(path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    prov = ck.get("provenance", {})
    name = Path(path).name
    print(f"{name}")
    print(f"   epoch        : {ck.get('epoch')}")
    print(f"   val_metrics  : {ck.get('val_metrics')}")
    print(f"   git_sha      : {prov.get('git_sha', 'MISSING')}")
    print(f"   config_sha256: {prov.get('config_sha256', 'MISSING')[:16]}")
    print(f"   wandb_run_id : {prov.get('wandb_run_id', 'MISSING')}")
    print(f"   timestamp    : {prov.get('timestamp', 'MISSING')}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        describe(p)
