"""Tests for checkpoint provenance metadata."""

import re

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
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)


def test_optional_fields_default_to_none_string():
    p = build_provenance({})
    assert p["wandb_run_id"] == "none"
    assert p["dataset_manifest"] == "none"


def test_optional_fields_are_recorded():
    p = build_provenance({}, wandb_run_id="abc123", dataset_manifest="deadbeef")
    assert p["wandb_run_id"] == "abc123"
    assert p["dataset_manifest"] == "deadbeef"
