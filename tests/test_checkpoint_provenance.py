"""Tests for the shared checkpoint saver and its provenance block."""

import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.checkpoint import save_checkpoint


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


def _state():
    model = _Tiny()
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = CosineAnnealingLR(opt, T_max=10)
    crit = torch.nn.BCEWithLogitsLoss()
    return model, opt, sched, crit


def test_saves_all_state_keys():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        save_checkpoint(model, opt, sched, crit, 3, {"val_auc": 0.9}, p,
                        config={"lr": 1e-3})
        ck = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("epoch", "model_state_dict", "optimizer_state_dict",
              "scheduler_state_dict", "criterion_state_dict", "val_metrics"):
        assert k in ck


def test_embeds_provenance_block():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        save_checkpoint(model, opt, sched, crit, 1, {}, p,
                        config={"lr": 1e-3}, wandb_run_id="run42")
        ck = torch.load(p, map_location="cpu", weights_only=False)
    prov = ck["provenance"]
    assert set(prov) == {"git_sha", "config_sha256", "wandb_run_id",
                         "dataset_manifest", "timestamp"}
    assert prov["wandb_run_id"] == "run42"


def test_creates_parent_directory():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nested" / "deeper" / "ckpt.pt"
        save_checkpoint(model, opt, sched, crit, 0, {}, p, config={})
        assert p.exists()


def test_leaves_no_tmp_file():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        save_checkpoint(model, opt, sched, crit, 0, {}, p, config={})
        assert list(Path(d).glob("*.tmp")) == []


def test_returns_path():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        assert save_checkpoint(model, opt, sched, crit, 0, {}, p,
                               config={}) == p
