"""Tests for deterministic seeding across RNGs."""

import numpy as np
import torch

from src.utils.seeding import seed_everything


def test_torch_rng_is_reproducible():
    seed_everything(1234)
    a = torch.randn(5)
    seed_everything(1234)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_numpy_rng_is_reproducible():
    seed_everything(99)
    a = np.random.rand(5)
    seed_everything(99)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    seed_everything(1)
    a = torch.randn(5)
    seed_everything(2)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_returns_applied_seed():
    assert seed_everything(7) == 7


def test_sets_cudnn_deterministic():
    seed_everything(42, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
