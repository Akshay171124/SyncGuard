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
    """Current HEAD commit, or "unknown" outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_sha256(config: dict) -> str:
    """Stable hash of a resolved config.

    Sorts keys so logically identical configs hash identically, and coerces
    non-JSON values (such as Path) via str so hashing never raises.
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def build_provenance(
    config: dict,
    wandb_run_id: str | None = None,
    dataset_manifest: str | None = None,
) -> dict:
    """Build the provenance block stored alongside checkpoint weights."""
    return {
        "git_sha": _git_sha(),
        "config_sha256": config_sha256(config),
        "wandb_run_id": wandb_run_id or "none",
        "dataset_manifest": dataset_manifest or "none",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
