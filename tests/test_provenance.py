"""Tests for checkpoint provenance metadata."""

import re
import subprocess
import tempfile

from src.utils import provenance
from src.utils.provenance import build_provenance, config_sha256

PROVENANCE_KEYS = {
    "git_sha",
    "config_sha256",
    "wandb_run_id",
    "dataset_manifest",
    "timestamp",
}


def test_config_hash_is_order_independent():
    a = {"lr": 1e-4, "epochs": 20}
    b = {"epochs": 20, "lr": 1e-4}
    assert config_sha256(a) == config_sha256(b)


def test_config_hash_changes_with_content():
    assert config_sha256({"lr": 1e-4}) != config_sha256({"lr": 2e-4})


def test_config_hash_handles_nested_and_nonjson():
    from pathlib import Path
    h = config_sha256({"model": {"name": "av_hubert"}, "path": Path("/tmp/x")})
    assert re.fullmatch(r"[0-9a-f]{64}", h)


def test_build_provenance_has_all_keys():
    p = build_provenance({"lr": 1e-4})
    assert set(p) == PROVENANCE_KEYS


def test_git_sha_is_sha_or_unknown():
    sha = build_provenance({})["git_sha"]
    # A dirty working tree (uncommitted changes) suffixes the SHA with "-dirty".
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}(-dirty)?", sha)


def _make_git_repo(path, dirty: bool):
    """Create a throwaway git repo at `path`, optionally leaving it dirty.

    Args:
        path: Directory to initialize as a git repo.
        dirty: If True, leave an uncommitted modification in the working tree.

    Returns:
        Path: The same `path`, now containing a one-commit git repo.
    """
    run = lambda *args: subprocess.run(  # noqa: E731
        args, cwd=path, check=True, capture_output=True, text=True
    )
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    tracked = path / "file.txt"
    tracked.write_text("hello")
    run("git", "add", "file.txt")
    run("git", "commit", "-q", "-m", "init")
    if dirty:
        tracked.write_text("modified, not committed")
    return path


def test_git_sha_dirty_tree_gets_suffix(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, dirty=True)
    monkeypatch.setattr(provenance, "_REPO_ROOT", repo)
    sha = provenance._git_sha()
    assert re.fullmatch(r"[0-9a-f]{40}-dirty", sha)


def test_git_sha_clean_tree_has_no_suffix(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, dirty=False)
    monkeypatch.setattr(provenance, "_REPO_ROOT", repo)
    sha = provenance._git_sha()
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def test_git_sha_is_independent_of_process_cwd(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, dirty=False)
    monkeypatch.setattr(provenance, "_REPO_ROOT", repo)
    # Run from a directory with no relation to the repo at all.
    monkeypatch.chdir(tempfile.gettempdir())
    sha = provenance._git_sha()
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def test_optional_fields_default_to_none_string():
    p = build_provenance({})
    assert p["wandb_run_id"] == "none"
    assert p["dataset_manifest"] == "none"


def test_optional_fields_are_recorded():
    p = build_provenance({}, wandb_run_id="abc123", dataset_manifest="deadbeef")
    assert p["wandb_run_id"] == "abc123"
    assert p["dataset_manifest"] == "deadbeef"
