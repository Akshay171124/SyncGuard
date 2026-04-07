import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from src.models.clip_visual_encoder import build_clip_visual_encoder

logger = logging.getLogger(__name__)


class Frontend3D(nn.Module):
    """3D convolutional frontend for processing temporal mouth-ROI sequences.

    Matches AV-HuBERT's visual frontend architecture:
    Conv3d → BatchNorm3d → PReLU → MaxPool3d
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.PReLU(out_channels)
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process temporal mouth-ROI input.

        Args:
            x: (B, 1, T, 96, 96) grayscale mouth crops

        Returns:
            (B, 64, T, 24, 24) feature maps
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class ResNetTrunk(nn.Module):
    """Modified ResNet-18 trunk that accepts 64-channel input from Frontend3D.

    Processes each frame independently through 2D convolutions.
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        base = resnet18(weights=None)

        # Replace first conv to accept in_channels instead of 3
        # Frontend3D output is already downsampled, so use kernel=1, stride=1
        self.adapt_conv = nn.Conv2d(
            in_channels, 64,
            kernel_size=1, stride=1, bias=False,
        )
        self.adapt_bn = nn.BatchNorm2d(64)

        # Use ResNet layers 1-4 (skip the original conv1/bn1/relu/maxpool)
        self.layer1 = base.layer1  # 64 → 64
        self.layer2 = base.layer2  # 64 → 128
        self.layer3 = base.layer3  # 128 → 256
        self.layer4 = base.layer4  # 256 → 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process per-frame features.

        Args:
            x: (B*T, 64, 24, 24) per-frame feature maps

        Returns:
            (B*T, 512) per-frame feature vectors
        """
        x = self.adapt_conv(x)
        x = self.adapt_bn(x)
        x = F.relu(x, inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


class ProjectionHead(nn.Module):
    """Linear → ReLU → Linear → L2-normalize projection head."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Use larger eps to prevent NaN from near-zero vectors (SF-3 fix)
        x = F.normalize(x, dim=-1, eps=1e-6)
        return x


class AVHubertVisualEncoder(nn.Module):
    """AV-HuBERT visual frontend + ResNet trunk + projection head.

    Architecture:
        (B, T, 1, 96, 96) → Frontend3D → ResNetTrunk → ProjectionHead → (B, T, embedding_dim)

    Args:
        embedding_dim: Output embedding dimension (default: 256).
        freeze_pretrained: If True, freeze Frontend3D and ResNetTrunk (only train projection).
    """

    def __init__(self, embedding_dim: int = 256, freeze_pretrained: bool = False):
        super().__init__()
        self.frontend3d = Frontend3D(in_channels=1, out_channels=64)
        self.trunk = ResNetTrunk(in_channels=64)
        self.projection = ProjectionHead(in_dim=512, out_dim=embedding_dim)
        self.embedding_dim = embedding_dim

        if freeze_pretrained:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze frontend3d and trunk, only train projection head."""
        for param in self.frontend3d.parameters():
            param.requires_grad = False
        for param in self.trunk.parameters():
            param.requires_grad = False
        logger.info("Froze AV-HuBERT visual backbone (frontend3d + trunk)")

    def load_av_hubert_weights(self, checkpoint_path: str):
        """Load pretrained weights from a fairseq AV-HuBERT checkpoint.

        Args:
            checkpoint_path: Path to the AV-HuBERT .pt checkpoint file.
        """
        try:
            import fairseq
            models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [checkpoint_path]
            )
            av_hubert = models[0]

            # Map AV-HuBERT visual frontend weights to our architecture
            src_state = av_hubert.state_dict()
            frontend_map = {}
            trunk_map = {}

            for k, v in src_state.items():
                if "feature_extractor" in k and "video" in k.lower():
                    # Frontend3D weights
                    local_key = k.split("feature_extractor.")[-1]
                    if local_key.startswith("frontend3D."):
                        mapped = local_key.replace("frontend3D.", "")
                        frontend_map[mapped] = v
                    elif local_key.startswith("trunk."):
                        mapped = local_key.replace("trunk.", "")
                        trunk_map[mapped] = v

            if frontend_map:
                self.frontend3d.load_state_dict(frontend_map, strict=False)
                logger.info(f"Loaded {len(frontend_map)} frontend3D weight tensors")
            if trunk_map:
                self.trunk.load_state_dict(trunk_map, strict=False)
                logger.info(f"Loaded {len(trunk_map)} trunk weight tensors")

            if not frontend_map and not trunk_map:
                logger.warning(
                    "No matching visual frontend weights found in checkpoint. "
                    "Using random initialization."
                )
        except ImportError:
            logger.warning("fairseq not installed. Using random initialization.")
        except Exception as e:
            logger.warning(f"Failed to load AV-HuBERT weights: {e}. Using random init.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frame-level visual embeddings from mouth crops.

        Args:
            x: (B, T, 1, 96, 96) grayscale mouth crops (float, normalized to [0,1])

        Returns:
            (B, T, embedding_dim) L2-normalized frame-level visual embeddings
        """
        B, T = x.shape[:2]

        # Frontend3D expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, 1, T, 96, 96)
        x = self.frontend3d(x)  # (B, 64, T, 24, 24)

        # Reshape for per-frame 2D processing
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, 64, 24, 24)
        x = x.reshape(B * T, 64, x.shape[3], x.shape[4])  # (B*T, 64, 24, 24)

        x = self.trunk(x)  # (B*T, 512)
        x = self.projection(x)  # (B*T, embedding_dim)
        x = x.reshape(B, T, self.embedding_dim)  # (B, T, embedding_dim)
        return x


class ResNet18VisualEncoder(nn.Module):
    """Simple ResNet-18 visual encoder for ablation comparison.

    Processes each frame independently through a standard ResNet-18.

    Args:
        embedding_dim: Output embedding dimension (default: 256).
        freeze_pretrained: If True, freeze ResNet-18 backbone.
    """

    def __init__(self, embedding_dim: int = 256, freeze_pretrained: bool = False):
        super().__init__()
        self.backbone = resnet18(weights="DEFAULT")
        # Replace first conv for grayscale input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove the final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projection = ProjectionHead(in_dim=in_features, out_dim=embedding_dim)
        self.embedding_dim = embedding_dim

        if freeze_pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze ResNet-18 backbone")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frame-level visual embeddings.

        Args:
            x: (B, T, 1, 96, 96) grayscale mouth crops

        Returns:
            (B, T, embedding_dim) L2-normalized frame-level visual embeddings
        """
        B, T = x.shape[:2]
        x = x.reshape(B * T, 1, 96, 96)  # (B*T, 1, 96, 96)
        x = self.backbone(x)  # (B*T, 512)
        x = self.projection(x)  # (B*T, embedding_dim)
        x = x.reshape(B, T, self.embedding_dim)
        return x


class SyncNetVisualEncoder(nn.Module):
    """SyncNet-style visual encoder for ablation comparison.

    Simplified 2D CNN designed for lip-sync detection (Chung & Zisserman, 2016).

    Args:
        embedding_dim: Output embedding dimension (default: 256).
    """

    def __init__(self, embedding_dim: int = 256, freeze_pretrained: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = ProjectionHead(in_dim=256, out_dim=embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frame-level visual embeddings.

        Args:
            x: (B, T, 1, 96, 96) grayscale mouth crops

        Returns:
            (B, T, embedding_dim) L2-normalized frame-level visual embeddings
        """
        B, T = x.shape[:2]
        x = x.reshape(B * T, 1, 96, 96)
        x = self.features(x)
        x = x.flatten(1)  # (B*T, 256)
        x = self.projection(x)  # (B*T, embedding_dim)
        x = x.reshape(B, T, self.embedding_dim)
        return x


def build_visual_encoder(config: dict) -> nn.Module:
    """Factory function to build visual encoder from config.

    Args:
        config: Full config dict (reads model.visual_encoder section).

    Returns:
        Visual encoder module.
    """
    ve_cfg = config["model"]["visual_encoder"]
    name = ve_cfg["name"]
    embedding_dim = ve_cfg.get("embedding_dim", 256)
    freeze = ve_cfg.get("freeze_pretrained", False)

    if name == "av_hubert":
        encoder = AVHubertVisualEncoder(
            embedding_dim=embedding_dim,
            freeze_pretrained=freeze,
        )
        ckpt = ve_cfg.get("checkpoint_path")
        if ckpt:
            encoder.load_av_hubert_weights(ckpt)
        return encoder
    elif name == "resnet18":
        return ResNet18VisualEncoder(
            embedding_dim=embedding_dim,
            freeze_pretrained=freeze,
        )
    elif name == "syncnet":
        return SyncNetVisualEncoder(
            embedding_dim=embedding_dim,
            freeze_pretrained=freeze,
        )
    elif name == "clip":
        return build_clip_visual_encoder(config)
    else:
        raise ValueError(f"Unknown visual encoder: {name}")


if __name__ == "__main__":
    # Quick shape test
    B, T = 2, 10
    x = torch.randn(B, T, 1, 96, 96)

    for name, Encoder in [
        ("av_hubert", AVHubertVisualEncoder),
        ("resnet18", ResNet18VisualEncoder),
        ("syncnet", SyncNetVisualEncoder),
    ]:
        enc = Encoder(embedding_dim=256)
        out = enc(x)
        assert out.shape == (B, T, 256), f"{name}: expected {(B, T, 256)}, got {out.shape}"
        # Verify L2-normalized
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"{name}: not L2-normalized"
        print(f"  {name}: input {x.shape} → output {out.shape} ✓ (L2-normalized)")

    # Verify gradients flow through projection but can be frozen for backbone
    enc = AVHubertVisualEncoder(embedding_dim=256, freeze_pretrained=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert enc.projection.fc1.weight.grad is not None, "Projection grad missing"
    assert all(p.grad is None for p in enc.frontend3d.parameters()), "Frozen backbone has grad"
    print("  freeze_pretrained: projection grads ✓, backbone frozen ✓")

    print("All visual encoder tests passed.")
