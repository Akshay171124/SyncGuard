"""CLIP ViT-L/14 visual encoder with LayerNorm-only tuning.

Processes mouth crop frames independently through CLIP's vision transformer,
extracts CLS token per frame, and projects to the shared AV embedding space.
Only LayerNorm parameters are trainable (~90K of 300M) to prevent overfitting
to source dataset artifacts while preserving CLIP's general visual understanding.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Linear -> ReLU -> Linear -> L2-normalize projection head."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, dim=-1, eps=1e-6)
        return x


class CLIPVisualEncoder(nn.Module):
    """CLIP ViT-L/14 visual encoder for frame-level face embeddings.

    Processes each frame independently through CLIP's frozen vision transformer.
    Only LayerNorm parameters are trainable (following GenD, WACV 2026).

    Args:
        model_id: HuggingFace model identifier (default: openai/clip-vit-large-patch14).
        embedding_dim: Output embedding dimension (default: 256).
        tune_layernorm: If True, only LayerNorm params are trainable (default: True).
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        embedding_dim: int = 256,
        tune_layernorm: bool = True,
    ):
        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained(model_id)
        hidden_size = self.clip_vision.config.hidden_size  # 768 for ViT-L/14

        self.projection = ProjectionHead(in_dim=hidden_size, out_dim=embedding_dim)

        # Freeze everything first
        for param in self.clip_vision.parameters():
            param.requires_grad = False

        # Selectively unfreeze LayerNorm parameters
        if tune_layernorm:
            ln_count = 0
            for name, param in self.clip_vision.named_parameters():
                if "layernorm" in name.lower() or "layer_norm" in name.lower():
                    param.requires_grad = True
                    ln_count += param.numel()
            total = sum(p.numel() for p in self.clip_vision.parameters())
            logger.info(
                f"CLIP LayerNorm tuning: {ln_count:,} trainable / {total:,} total "
                f"({ln_count/total*100:.2f}%)"
            )

    def forward(self, mouth_crops: torch.Tensor) -> torch.Tensor:
        """Extract frame-level visual embeddings from mouth crops.

        Args:
            mouth_crops: (B, T, C, H, W) mouth crops. C=3 (RGB), H=W=224.

        Returns:
            (B, T, embedding_dim) L2-normalized visual embeddings.
        """
        B, T, C, H, W = mouth_crops.shape

        # Reshape to process all frames at once: (B*T, C, H, W)
        frames = mouth_crops.reshape(B * T, C, H, W)

        # CLIP vision encoder expects pixel_values in [0, 1]
        outputs = self.clip_vision(pixel_values=frames)

        # Extract CLS token: (B*T, hidden_size)
        cls_features = outputs.pooler_output

        # Project and L2-normalize: (B*T, embedding_dim)
        embeddings = self.projection(cls_features)

        # Reshape back to sequence: (B, T, embedding_dim)
        return embeddings.reshape(B, T, -1)


def build_clip_visual_encoder(config: dict) -> CLIPVisualEncoder:
    """Build CLIP visual encoder from config."""
    ve_cfg = config["model"]["visual_encoder"]
    encoder = CLIPVisualEncoder(
        model_id=ve_cfg.get("model_id", "openai/clip-vit-large-patch14"),
        embedding_dim=ve_cfg.get("embedding_dim", 256),
        tune_layernorm=ve_cfg.get("tune_layernorm", True),
    )
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"CLIPVisualEncoder: {trainable:,} trainable / {total_params:,} total")
    return encoder
