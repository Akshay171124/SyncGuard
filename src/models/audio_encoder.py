import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

logger = logging.getLogger(__name__)


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


class Wav2Vec2AudioEncoder(nn.Module):
    """Wav2Vec 2.0 audio encoder with configurable layer extraction.

    Loads facebook/wav2vec2-base-960h, extracts hidden states from a specific
    layer, and projects to the shared embedding space.

    Architecture:
        raw waveform → Wav2Vec2Model → hidden_states[layer] → ProjectionHead → (B, T, embedding_dim)

    Args:
        model_id: HuggingFace model identifier.
        layer: Which hidden layer to extract (0-12). Default 9 per Pasad et al. (2021).
        embedding_dim: Output embedding dimension (default: 256).
        freeze_pretrained: If True, freeze Wav2Vec 2.0 weights (default: True).
    """

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base-960h",
        layer: int = 9,
        embedding_dim: int = 256,
        freeze_pretrained: bool = True,
    ):
        super().__init__()
        self.layer = layer
        self.embedding_dim = embedding_dim

        # Load pretrained Wav2Vec 2.0
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_id)
        hidden_size = self.wav2vec2.config.hidden_size  # 768 for base

        self.projection = ProjectionHead(in_dim=hidden_size, out_dim=embedding_dim)

        if freeze_pretrained:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all Wav2Vec 2.0 parameters."""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        logger.info("Froze Wav2Vec 2.0 backbone")

    def train(self, mode: bool = True):
        """Override to keep entire Wav2Vec backbone in inference mode (SF-6 fix).

        Wav2Vec2 produces NaN on padded waveforms when normalization layers run
        in training mode. The entire backbone stays in inference mode but with
        requires_grad=True, so gradients still flow for fine-tuning.
        Only the projection head follows the normal train/inference toggle.
        """
        super().train(mode)
        # Keep entire wav2vec2 in inference mode (gradients still flow)
        self.wav2vec2.training = False
        for module in self.wav2vec2.modules():
            module.training = False
        return self

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Extract frame-level audio embeddings from raw waveform.

        Args:
            waveform: (B, num_samples) raw audio at 16kHz
            attention_mask: (B, num_samples) optional mask for padded waveforms

        Returns:
            (B, T, embedding_dim) L2-normalized frame-level audio embeddings at ~49Hz
        """
        with torch.set_grad_enabled(any(
            p.requires_grad for p in self.wav2vec2.parameters()
        )):
            outputs = self.wav2vec2(
                waveform,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract the specified hidden layer
        # hidden_states is a tuple of (num_layers + 1) tensors, each (B, T, hidden_size)
        # Index 0 = output of feature extractor, 1-12 = transformer layers
        hidden_states = outputs.hidden_states
        if self.layer >= len(hidden_states):
            raise ValueError(
                f"Requested layer {self.layer} but model only has {len(hidden_states)} layers"
            )
        features = hidden_states[self.layer]  # (B, T, 768)

        # Project and L2-normalize
        embeddings = self.projection(features)  # (B, T, embedding_dim)
        return embeddings


def build_audio_encoder(config: dict) -> nn.Module:
    """Factory function to build audio encoder from config.

    Args:
        config: Full config dict (reads model.audio_encoder section).

    Returns:
        Audio encoder module.
    """
    ae_cfg = config["model"]["audio_encoder"]
    return Wav2Vec2AudioEncoder(
        model_id=ae_cfg.get("model_id", "facebook/wav2vec2-base-960h"),
        layer=ae_cfg.get("layer", 9),
        embedding_dim=ae_cfg.get("embedding_dim", 256),
        freeze_pretrained=ae_cfg.get("freeze_pretrained", True),
    )


if __name__ == "__main__":
    print("Testing Wav2Vec2AudioEncoder...")

    B = 2
    num_samples = 16000 * 3  # 3 seconds at 16kHz
    waveform = torch.randn(B, num_samples)

    encoder = Wav2Vec2AudioEncoder(
        model_id="facebook/wav2vec2-base-960h",
        layer=9,
        embedding_dim=256,
        freeze_pretrained=True,
    )

    out = encoder(waveform)
    T_expected = num_samples // 320  # Wav2Vec stride ~320 samples → ~49 Hz
    print(f"  Input: {waveform.shape}")
    print(f"  Output: {out.shape} (expected T ≈ {T_expected})")
    assert out.shape[0] == B
    assert out.shape[2] == 256
    # T should be close to num_samples // 320 (≈ 150 for 3s)
    assert abs(out.shape[1] - T_expected) < 5, f"T={out.shape[1]}, expected ≈{T_expected}"

    # Verify L2-normalized
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Not L2-normalized"
    print("  L2-normalized ✓")

    # Verify backbone is frozen, projection is trainable
    loss = out.sum()
    loss.backward()
    assert encoder.projection.fc1.weight.grad is not None, "Projection grad missing"
    assert all(
        p.grad is None for p in encoder.wav2vec2.parameters()
    ), "Frozen backbone has grads"
    print("  freeze_pretrained: projection grads ✓, backbone frozen ✓")

    print("All audio encoder tests passed.")
