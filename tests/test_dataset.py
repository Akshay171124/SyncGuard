"""Tests for dataset and collation (src/training/dataset.py).

Tests verify:
- Collation pads to max length in batch
- Boolean masks correctly flag valid vs padded frames
- Batch tensor shapes are correct
- Variable-length sequences handled properly
- Edge cases: single sample, uniform lengths
"""

import numpy as np
import pytest
import torch

from src.training.dataset import SyncGuardBatch, collate_syncguard


def _make_sample(num_frames: int, audio_len: int, label: int = 0,
                 category: str = "RV-RA", channels: int = 1) -> dict:
    """Create a synthetic sample dict matching SyncGuardDataset.__getitem__ output."""
    return {
        "mouth_crops": torch.randn(num_frames, channels, 96, 96),
        "waveform": torch.randn(audio_len),
        "label": label,
        "is_real": label == 0,
        "category": category,
        "speaker_id": "spk_001",
        "sample_id": f"sample_{num_frames}",
        "num_frames": num_frames,
        "ear_features": torch.randn(num_frames),
    }


# ──────────────────────────────────────────────
# Collation
# ──────────────────────────────────────────────

class TestCollateSyncguard:
    def test_returns_syncguard_batch(self):
        """Collation returns SyncGuardBatch dataclass."""
        batch = [_make_sample(30, 16000), _make_sample(40, 20000)]
        result = collate_syncguard(batch)
        assert isinstance(result, SyncGuardBatch)

    def test_pads_to_max_frames(self):
        """Mouth crops padded to max T in batch."""
        batch = [_make_sample(20, 16000), _make_sample(40, 16000)]
        result = collate_syncguard(batch)
        assert result.mouth_crops.shape[1] == 40  # max T

    def test_pads_to_max_audio_length(self):
        """Waveforms padded to max audio length in batch."""
        batch = [_make_sample(30, 10000), _make_sample(30, 20000)]
        result = collate_syncguard(batch)
        assert result.waveforms.shape[1] == 20000

    def test_mask_valid_frames_true(self):
        """Mask is True for valid (non-padded) frames."""
        batch = [_make_sample(20, 16000), _make_sample(40, 16000)]
        result = collate_syncguard(batch)
        # First sample: 20 valid, rest padded
        assert result.mask[0, :20].all()
        assert not result.mask[0, 20:].any()
        # Second sample: all 40 valid
        assert result.mask[1, :40].all()

    def test_lengths_correct(self):
        """Lengths tensor matches num_frames per sample."""
        batch = [_make_sample(25, 16000), _make_sample(50, 16000), _make_sample(35, 16000)]
        result = collate_syncguard(batch)
        assert result.lengths.tolist() == [25, 50, 35]

    def test_labels_correct(self):
        """Labels match input."""
        batch = [
            _make_sample(30, 16000, label=0),
            _make_sample(30, 16000, label=1),
            _make_sample(30, 16000, label=1),
        ]
        result = collate_syncguard(batch)
        assert result.labels.tolist() == [0, 1, 1]

    def test_is_real_correct(self):
        """is_real flag matches labels."""
        batch = [
            _make_sample(30, 16000, label=0),
            _make_sample(30, 16000, label=1),
        ]
        result = collate_syncguard(batch)
        assert result.is_real[0].item() is True
        assert result.is_real[1].item() is False

    def test_categories_preserved(self):
        """Category strings preserved in list."""
        batch = [
            _make_sample(30, 16000, category="RV-RA"),
            _make_sample(30, 16000, category="FV-FA"),
        ]
        result = collate_syncguard(batch)
        assert result.categories == ["RV-RA", "FV-FA"]

    def test_ear_features_padded(self):
        """EAR features padded to max T."""
        batch = [_make_sample(20, 16000), _make_sample(40, 16000)]
        result = collate_syncguard(batch)
        assert result.ear_features.shape == (2, 40)
        # Padded region should be zero
        assert result.ear_features[0, 20:].sum() == 0.0

    def test_batch_shape_five_dim(self):
        """mouth_crops is 5D: (B, T, C, H, W)."""
        batch = [_make_sample(30, 16000)]
        result = collate_syncguard(batch)
        assert result.mouth_crops.ndim == 5

    def test_padded_regions_are_zero(self):
        """Padded mouth_crops regions are zero-filled."""
        batch = [_make_sample(10, 16000), _make_sample(20, 16000)]
        result = collate_syncguard(batch)
        # First sample: frames 10-19 should be zero
        assert result.mouth_crops[0, 10:].abs().sum() == 0.0

    def test_rgb_channels(self):
        """Handles 3-channel (RGB/CLIP) input correctly."""
        batch = [
            _make_sample(20, 16000, channels=3),
            _make_sample(30, 16000, channels=3),
        ]
        result = collate_syncguard(batch)
        assert result.mouth_crops.shape == (2, 30, 3, 96, 96)


# ──────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────

class TestCollateEdgeCases:
    def test_single_sample(self):
        """Batch of one sample works correctly."""
        batch = [_make_sample(30, 16000)]
        result = collate_syncguard(batch)
        assert result.mouth_crops.shape[0] == 1
        assert result.mask.all()  # No padding needed

    def test_uniform_lengths(self):
        """All samples same length → no padding needed."""
        batch = [_make_sample(30, 16000) for _ in range(4)]
        result = collate_syncguard(batch)
        assert result.mask.all()
        assert (result.lengths == 30).all()

    def test_very_short_sequence(self):
        """Single-frame sequence works."""
        batch = [_make_sample(1, 320), _make_sample(10, 3200)]
        result = collate_syncguard(batch)
        assert result.mouth_crops.shape[1] == 10
        assert result.lengths[0] == 1
        assert result.mask[0, 0].item() is True
        assert not result.mask[0, 1:].any()

    def test_sample_ids_preserved(self):
        """sample_id strings preserved."""
        batch = [_make_sample(30, 16000), _make_sample(40, 16000)]
        result = collate_syncguard(batch)
        assert len(result.sample_ids) == 2
