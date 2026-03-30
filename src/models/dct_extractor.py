"""Learnable DCT feature extractor for face-swap artifact detection.

Applies 2D DCT to mouth crops, then uses a lightweight CNN to learn
discriminative frequency-domain features. Face-swap blending artifacts
leave traces in the frequency domain that are invisible in pixel space.

Architecture:
    mouth_crop (B, T, 1, 96, 96) → per-frame 2D DCT → (B*T, 1, 96, 96)
    → Conv2d(1,8) → ReLU → MaxPool
    → Conv2d(8,16) → ReLU → MaxPool
    → AdaptiveAvgPool(4,4) → Flatten → Linear(256, output_dim)
    → (B, T, output_dim)
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def dct2d(x: torch.Tensor) -> torch.Tensor:
    """Compute 2D DCT using FFT (Type-II DCT approximation).

    Args:
        x: (B, 1, H, W) input tensor

    Returns:
        (B, 1, H, W) DCT coefficients
    """
    # DCT via FFT: mirror the signal, take real part of FFT
    B, C, H, W = x.shape

    # DCT along height
    v = torch.cat([x, x.flip(dims=[2])], dim=2)
    V = torch.fft.rfft(v, dim=2)
    dct_h = V[:, :, :H, :].real

    # DCT along width
    v2 = torch.cat([dct_h, dct_h.flip(dims=[3])], dim=3)
    V2 = torch.fft.rfft(v2, dim=3)
    dct_hw = V2[:, :, :, :W].real

    return dct_hw


class DCTFeatureExtractor(nn.Module):
    """Learnable frequency-domain feature extractor.

    Converts mouth crops to DCT domain, then applies a lightweight CNN
    to learn discriminative frequency patterns for face-swap detection.

    Args:
        output_dim: Per-frame feature dimension (default: 16).
        dropout: Dropout rate (default: 0.2).
    """

    def __init__(self, output_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.output_dim = output_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96→48

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48→24

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # →4x4
        )
        # 32 * 4 * 4 = 512
        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, mouth_crops: torch.Tensor) -> torch.Tensor:
        """Extract DCT features from mouth crops.

        Args:
            mouth_crops: (B, T, 1, 96, 96) grayscale mouth crops

        Returns:
            (B, T, output_dim) per-frame DCT features
        """
        B, T, C, H, W = mouth_crops.shape

        # Reshape to process all frames at once
        x = mouth_crops.reshape(B * T, C, H, W)

        # Apply 2D DCT
        x = dct2d(x)

        # Log-scale the DCT coefficients (compress dynamic range)
        x = torch.log1p(x.abs())

        # CNN feature extraction
        x = self.cnn(x)  # (B*T, 32, 4, 4)
        x = x.flatten(1)  # (B*T, 512)
        x = self.fc(x)  # (B*T, output_dim)

        # Reshape back to sequence
        return x.reshape(B, T, self.output_dim)


def build_dct_extractor(config: dict) -> DCTFeatureExtractor:
    """Build DCT feature extractor from config."""
    dct_cfg = config["model"].get("dct_extractor", {})
    extractor = DCTFeatureExtractor(
        output_dim=dct_cfg.get("output_dim", 16),
        dropout=dct_cfg.get("dropout", 0.2),
    )
    total_params = sum(p.numel() for p in extractor.parameters())
    logger.info(f"DCTFeatureExtractor: {total_params:,} parameters, output_dim={extractor.output_dim}")
    return extractor
