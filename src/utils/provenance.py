"""Provenance metadata embedded in every checkpoint.

April's checkpoints carried no link back to the commit or config that
produced them, and the configs themselves lived only on scratch. Embedding
provenance inside the checkpoint makes artifacts self-describing, so the
link survives files being moved, renamed, or restored from backup.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone


def _git_sha() -> str:
    """Current HEAD commit, or "unknown" outside a git checkout.

    Returns:
        str: 40-character commit SHA, or "unknown" if git is unavailable,
             fails, or times out.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def config_sha256(config: dict) -> str:
    """Stable hash of a resolved config.

    Sorts keys so logically identical configs hash identically, and coerces
    non-JSON values (such as Path) via str so hashing never raises.

    Args:
        config: Dictionary of configuration parameters to hash.

    Returns:
        str: 64-character hexadecimal SHA256 digest of the sorted config.
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def build_provenance(
    config: dict,
    wandb_run_id: str | None = None,
    dataset_manifest: str | None = None,
) -> dict:
    """Build the provenance block stored alongside checkpoint weights.

    Args:
        config: Resolved configuration dict to hash for reproducibility.
        wandb_run_id: Optional Weights & Biases run ID; defaults to "none".
        dataset_manifest: Optional dataset identifier or checksum; defaults to "none".

    Returns:
        dict: Provenance metadata dict with keys:
            - git_sha: Current HEAD commit (40 hex chars or "unknown").
            - config_sha256: SHA256 hash of the config (64 hex chars).
            - wandb_run_id: W&B run ID or "none".
            - dataset_manifest: Dataset identifier or "none".
            - timestamp: ISO-format UTC timestamp of provenance creation.
    """
    return {
        "git_sha": _git_sha(),
        "config_sha256": config_sha256(config),
        "wandb_run_id": wandb_run_id or "none",
        "dataset_manifest": dataset_manifest or "none",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
