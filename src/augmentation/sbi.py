# src/augmentation/sbi.py
"""Self-Blended Image (SBI) augmentation for deepfake detection.

Generates synthetic face-swap training data by blending face regions with
transformed versions of themselves. Creates blending boundary artifacts
common to ALL face-swap methods, enabling cross-dataset generalization.

Based on: Shiohara & Yamasaki, "Detecting Deepfakes with Self-Blended Images" (CVPR 2022)
"""

import logging
import random
from io import BytesIO

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SelfBlendedImage:
    """Self-Blended Image augmentation.

    Takes a real face frame and creates a synthetic fake by blending
    the original with a transformed version of itself using a face-shaped mask.

    Args:
        color_jitter: Color jitter strength (default: 0.1 = ±10%).
        blur_sigma: Range of Gaussian blur sigma for target (default: [1.0, 3.0]).
        warp_strength: Affine warp strength (default: 0.05 = ±5%).
        mask_blur_sigma: Range of Gaussian blur sigma for mask feathering (default: [5, 15]).
        jpeg_quality: Range of JPEG compression quality (default: [70, 95]).
    """

    def __init__(
        self,
        color_jitter: float = 0.1,
        blur_sigma: tuple[float, float] = (1.0, 3.0),
        warp_strength: float = 0.05,
        mask_blur_sigma: tuple[int, int] = (5, 15),
        jpeg_quality: tuple[int, int] = (70, 95),
    ):
        self.color_jitter = color_jitter
        self.blur_sigma = blur_sigma
        self.warp_strength = warp_strength
        self.mask_blur_sigma = mask_blur_sigma
        self.jpeg_quality = jpeg_quality

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply random brightness/contrast jitter."""
        factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        return np.clip(img * factor, 0, 1).astype(np.float32)

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply random Gaussian blur."""
        sigma = random.uniform(*self.blur_sigma)
        ksize = int(sigma * 4) | 1  # Ensure odd kernel size
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def _affine_warp(self, img: np.ndarray) -> np.ndarray:
        """Apply slight random affine transformation."""
        h, w = img.shape[:2]
        strength = self.warp_strength
        # Random affine: slight rotation + scale + translation
        center = (w / 2, h / 2)
        angle = random.uniform(-5, 5)  # ±5 degrees
        scale = 1.0 + random.uniform(-strength, strength)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # Add slight translation
        M[0, 2] += random.uniform(-w * strength, w * strength)
        M[1, 2] += random.uniform(-h * strength, h * strength)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _create_face_mask(self, h: int, w: int) -> np.ndarray:
        """Create an elliptical face mask centered on the frame.

        Since we're working with mouth crops (already tightly cropped faces),
        the mask is a centered ellipse covering ~70% of the frame.
        """
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.35), int(h * 0.40))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        # Feather the mask edges
        blur_sigma = random.randint(*self.mask_blur_sigma)
        ksize = blur_sigma * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), blur_sigma)
        return mask

    def _jpeg_compress(self, img: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        quality = random.randint(*self.jpeg_quality)
        # Convert to uint8 for JPEG encoding
        img_uint8 = (img * 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", img_uint8, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE)
        return decoded.astype(np.float32) / 255.0

    def blend_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply SBI augmentation to a single frame.

        Args:
            frame: (H, W) or (H, W, C) float32 frame in [0, 1].

        Returns:
            Blended frame with same shape, containing blending boundary artifacts.
        """
        is_gray = frame.ndim == 2
        if is_gray:
            frame = frame[:, :, np.newaxis]

        h, w, c = frame.shape

        # Create transformed target
        target = frame.copy()
        target = self._color_jitter(target)
        target = self._gaussian_blur(target)
        target = self._affine_warp(target)

        # Create face mask with feathered edges
        mask = self._create_face_mask(h, w)
        mask = mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting

        # Alpha blend: source outside mask, target inside mask
        blended = frame * (1 - mask) + target * mask

        # Optional JPEG compression
        if random.random() < 0.5:
            # _jpeg_compress expects (H, W) or (H, W, C)
            blend_2d = blended[:, :, 0] if is_gray else blended
            blend_2d = self._jpeg_compress(blend_2d)
            if is_gray:
                blended = blend_2d[:, :, np.newaxis] if blend_2d.ndim == 2 else blend_2d[:, :, :1]
            else:
                blended = blend_2d

        if is_gray:
            blended = blended[:, :, 0] if blended.ndim == 3 else blended

        return np.clip(blended, 0, 1).astype(np.float32)

    def augment_sequence(self, mouth_crops: torch.Tensor) -> torch.Tensor:
        """Apply SBI to a sequence of mouth crop frames.

        Args:
            mouth_crops: (T, C, H, W) tensor of mouth crops in [0, 1].

        Returns:
            (T, C, H, W) tensor with SBI augmentation applied to each frame.
        """
        T, C, H, W = mouth_crops.shape
        result = mouth_crops.clone()

        for t in range(T):
            frame = mouth_crops[t].permute(1, 2, 0).numpy()  # (H, W, C)
            if C == 1:
                frame = frame.squeeze(-1)  # (H, W)
            blended = self.blend_frame(frame)
            if C == 1 and blended.ndim == 2:
                blended = blended[:, :, np.newaxis]
            result[t] = torch.from_numpy(blended).permute(2, 0, 1) if blended.ndim == 3 else torch.from_numpy(blended).unsqueeze(0)

        return result


def build_sbi(config: dict) -> SelfBlendedImage:
    """Build SBI augmentation from config."""
    sbi_cfg = config.get("augmentation", {}).get("sbi", {})
    return SelfBlendedImage(
        color_jitter=sbi_cfg.get("color_jitter", 0.1),
        blur_sigma=tuple(sbi_cfg.get("blur_sigma", [1.0, 3.0])),
        warp_strength=sbi_cfg.get("warp_strength", 0.05),
        mask_blur_sigma=tuple(sbi_cfg.get("mask_blur_sigma", [5, 15])),
        jpeg_quality=tuple(sbi_cfg.get("jpeg_quality", [70, 95])),
    )
