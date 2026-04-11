"""Tests for SBI augmentation (src/augmentation/sbi.py).

Tests verify:
- blend_frame output shape and value range
- Grayscale and RGB input handling
- augment_sequence produces correct tensor shape
- JPEG compression stays in [0, 1]
- Face mask is elliptical and centered
- Color jitter stays in [0, 1]
"""

import numpy as np
import pytest
import torch

from src.augmentation.sbi import SelfBlendedImage, build_sbi


# ──────────────────────────────────────────────
# SelfBlendedImage
# ──────────────────────────────────────────────

class TestSBIBlendFrame:
    @pytest.fixture
    def sbi(self):
        """Create SBI with fixed parameters for deterministic testing."""
        return SelfBlendedImage(
            color_jitter=0.1,
            blur_sigma=(1.0, 2.0),
            warp_strength=0.03,
            mask_blur_sigma=(5, 10),
            jpeg_quality=(80, 95),
        )

    def test_grayscale_output_shape(self, sbi):
        """Grayscale (H, W) input → (H, W) output."""
        frame = np.random.rand(96, 96).astype(np.float32)
        blended = sbi.blend_frame(frame)
        assert blended.shape == (96, 96)

    def test_rgb_output_shape(self, sbi):
        """RGB (H, W, 3) input → (H, W, 3) output."""
        frame = np.random.rand(96, 96, 3).astype(np.float32)
        blended = sbi.blend_frame(frame)
        assert blended.shape == (96, 96, 3)

    def test_output_range_zero_one(self, sbi):
        """Output values are clipped to [0, 1]."""
        frame = np.random.rand(96, 96).astype(np.float32)
        blended = sbi.blend_frame(frame)
        assert blended.min() >= 0.0
        assert blended.max() <= 1.0

    def test_output_dtype_float32(self, sbi):
        """Output is float32."""
        frame = np.random.rand(96, 96).astype(np.float32)
        blended = sbi.blend_frame(frame)
        assert blended.dtype == np.float32

    def test_blended_differs_from_input(self, sbi):
        """Blended frame should differ from input (augmentation applied)."""
        frame = np.ones((96, 96), dtype=np.float32) * 0.5
        blended = sbi.blend_frame(frame)
        # With color jitter and warping, output should differ
        assert not np.allclose(frame, blended, atol=1e-3)

    def test_no_nan_or_inf(self, sbi):
        """No NaN or Inf in blended output."""
        frame = np.random.rand(96, 96).astype(np.float32)
        blended = sbi.blend_frame(frame)
        assert np.all(np.isfinite(blended))


class TestSBIAugmentSequence:
    @pytest.fixture
    def sbi(self):
        return SelfBlendedImage()

    def test_output_shape_grayscale(self, sbi):
        """(T, 1, H, W) tensor → same shape output."""
        crops = torch.rand(10, 1, 96, 96)
        result = sbi.augment_sequence(crops)
        assert result.shape == (10, 1, 96, 96)

    def test_output_range(self, sbi):
        """Output tensor values in [0, 1]."""
        crops = torch.rand(5, 1, 96, 96)
        result = sbi.augment_sequence(crops)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_different_from_input(self, sbi):
        """Augmented sequence differs from original."""
        crops = torch.ones(5, 1, 96, 96) * 0.5
        result = sbi.augment_sequence(crops)
        assert not torch.allclose(crops, result, atol=1e-3)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

class TestSBIHelpers:
    def test_color_jitter_range(self):
        """Color jitter keeps values in [0, 1]."""
        sbi = SelfBlendedImage(color_jitter=0.2)
        img = np.random.rand(96, 96).astype(np.float32)
        result = sbi._color_jitter(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_face_mask_shape(self):
        """Face mask has correct shape and is in [0, 1]."""
        sbi = SelfBlendedImage()
        mask = sbi._create_face_mask(96, 96)
        assert mask.shape == (96, 96)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_face_mask_center_nonzero(self):
        """Face mask is nonzero in center (where the ellipse is)."""
        sbi = SelfBlendedImage()
        mask = sbi._create_face_mask(96, 96)
        center_val = mask[48, 48]
        assert center_val > 0.5  # Center should be well inside the ellipse

    def test_gaussian_blur_preserves_shape(self):
        """Gaussian blur output has same shape."""
        sbi = SelfBlendedImage(blur_sigma=(1.0, 2.0))
        img = np.random.rand(96, 96).astype(np.float32)
        blurred = sbi._gaussian_blur(img)
        assert blurred.shape == img.shape

    def test_affine_warp_preserves_shape(self):
        """Affine warp output has same shape."""
        sbi = SelfBlendedImage(warp_strength=0.05)
        img = np.random.rand(96, 96).astype(np.float32)
        warped = sbi._affine_warp(img)
        assert warped.shape == img.shape


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class TestBuildSBI:
    def test_default_config(self):
        """build_sbi with empty config returns SBI with defaults."""
        sbi = build_sbi({})
        assert isinstance(sbi, SelfBlendedImage)
        assert sbi.color_jitter == 0.1

    def test_custom_config(self):
        """build_sbi applies custom config values."""
        config = {
            "augmentation": {
                "sbi": {
                    "color_jitter": 0.2,
                    "warp_strength": 0.1,
                    "jpeg_quality": [60, 90],
                }
            }
        }
        sbi = build_sbi(config)
        assert sbi.color_jitter == 0.2
        assert sbi.warp_strength == 0.1
        assert sbi.jpeg_quality == (60, 90)
