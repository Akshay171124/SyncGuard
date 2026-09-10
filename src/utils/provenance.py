"""Provenance metadata embedded in every checkpoint.

April's checkpoints carried no link back to the commit or config that
produced them, and the configs themselves lived only on scratch. Embedding
provenance inside the checkpoint makes artifacts self-describing, so the
link survives files being moved, renamed, or restored from backup.
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Anchor for git commands so provenance is correct regardless of the
# process's current working directory (e.g. a SLURM job that cd's into
# /scratch before running, or a test invoked from an unrelated directory).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    """Current HEAD commit, with a "-dirty" suffix if the tree has uncommitted changes.

    Anchored to the repo root (derived from this module's own location) so
    the result does not depend on the process's current working directory.

    Returns:
        str: 40-character commit SHA, optionally suffixed with "-dirty" if
             there are uncommitted changes, or "unknown" if git is
             unavailable, fails, or times out.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
            cwd=_REPO_ROOT,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        if status.strip():
            sha = f"{sha}-dirty"
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Could not determine git SHA ({e!r}); provenance will record 'unknown'.")
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
            - git_sha: Current HEAD commit (40 hex chars, optionally suffixed
              with "-dirty", or "unknown").
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
