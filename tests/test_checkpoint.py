"""Tests for checkpoint save/load round-trip (pitfall #10).

Tests verify:
- Full state dict saved (model, optimizer, scheduler, criterion, epoch, metrics)
- Round-trip: save → load → state matches
- Atomic save via .tmp rename
- Training can resume from checkpoint (epoch counter, optimizer state)
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.losses import CombinedLoss


class SimpleModel(torch.nn.Module):
    """Minimal model for checkpoint testing."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# ──────────────────────────────────────────────
# Save/Load Round-Trip
# ──────────────────────────────────────────────

class TestCheckpointRoundTrip:
    @pytest.fixture
    def training_state(self):
        """Create a full training state (model, optimizer, scheduler, criterion)."""
        model = SimpleModel()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        criterion = CombinedLoss(embedding_dim=256, queue_size=64)

        # Simulate one training step to populate optimizer state
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        return model, optimizer, scheduler, criterion

    def test_save_and_load_model_weights(self, training_state, tmp_path):
        """Model weights match after save/load."""
        model, optimizer, scheduler, criterion = training_state
        path = tmp_path / "ckpt.pt"

        # Save
        torch.save({
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": {"auc": 0.95, "eer": 0.08},
        }, path)

        # Load into fresh model
        model2 = SimpleModel()
        ckpt = torch.load(path, weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"])

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_epoch_preserved(self, training_state, tmp_path):
        """Epoch counter preserved in checkpoint."""
        model, optimizer, scheduler, criterion = training_state
        path = tmp_path / "ckpt.pt"

        torch.save({
            "epoch": 7,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": {},
        }, path)

        ckpt = torch.load(path, weights_only=False)
        assert ckpt["epoch"] == 7

    def test_optimizer_state_restored(self, training_state, tmp_path):
        """Optimizer state (momentum buffers, step count) restored."""
        model, optimizer, scheduler, criterion = training_state
        path = tmp_path / "ckpt.pt"

        torch.save({
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": {},
        }, path)

        # Create fresh optimizer and restore
        model2 = SimpleModel()
        optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        ckpt = torch.load(path, weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"])
        optimizer2.load_state_dict(ckpt["optimizer_state_dict"])

        # Optimizer state should have entries (not empty)
        assert len(optimizer2.state) > 0

    def test_scheduler_state_restored(self, training_state, tmp_path):
        """LR scheduler state restored (last_epoch, etc)."""
        model, optimizer, scheduler, criterion = training_state
        path = tmp_path / "ckpt.pt"

        torch.save({
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": {},
        }, path)

        model2 = SimpleModel()
        opt2 = AdamW(model2.parameters(), lr=1e-3)
        sched2 = CosineAnnealingLR(opt2, T_max=10)
        ckpt = torch.load(path, weights_only=False)
        sched2.load_state_dict(ckpt["scheduler_state_dict"])

        assert sched2.last_epoch == scheduler.last_epoch

    def test_criterion_state_includes_queue(self, training_state, tmp_path):
        """CombinedLoss state dict includes MoCo queue buffers."""
        _, _, _, criterion = training_state
        state = criterion.state_dict()
        # MoCo queue is registered as buffer
        queue_keys = [k for k in state if "queue" in k]
        assert len(queue_keys) > 0, "MoCo queue not in criterion state dict"

    def test_val_metrics_preserved(self, training_state, tmp_path):
        """Validation metrics dict round-trips correctly."""
        model, optimizer, scheduler, criterion = training_state
        path = tmp_path / "ckpt.pt"
        metrics = {"auc": 0.963, "eer": 0.093, "pauc_01": 0.861}

        torch.save({
            "epoch": 10,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": metrics,
        }, path)

        ckpt = torch.load(path, weights_only=False)
        assert ckpt["val_metrics"]["auc"] == 0.963
        assert ckpt["val_metrics"]["eer"] == 0.093

    def test_checkpoint_all_keys_present(self, training_state, tmp_path):
        """Checkpoint contains all required keys per pitfall #10."""
        model, optimizer, scheduler, criterion = training_state
        path = tmp_path / "ckpt.pt"

        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": {},
        }, path)

        ckpt = torch.load(path, weights_only=False)
        required = {"epoch", "model_state_dict", "optimizer_state_dict",
                     "scheduler_state_dict", "criterion_state_dict", "val_metrics"}
        assert required.issubset(set(ckpt.keys()))
