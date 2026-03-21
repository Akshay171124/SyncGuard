"""Standalone audio-only deepfake classifier.

Uses Wav2Vec 2.0 embeddings to detect fake audio (TTS, voice conversion)
independent of visual input. Designed for inference-time cascade with
the sync-based SyncGuard model.

Architecture:
    waveform → Wav2Vec2 (frozen, layer 9) → mean+max pool → MLP → (B, 1)
"""

import logging

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

logger = logging.getLogger(__name__)


class StandaloneAudioClassifier(nn.Module):
    """Wav2Vec2-based audio deepfake classifier.

    Extracts hidden states from a specified Wav2Vec2 layer, pools temporally,
    and classifies as real or fake audio.

    Args:
        model_id: HuggingFace model identifier.
        layer: Which hidden layer to extract (0-12).
        hidden_dim: MLP hidden dimension.
        dropout: Dropout rate.
        freeze_backbone: If True, freeze Wav2Vec2 weights.
    """

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base-960h",
        layer: int = 9,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.layer = layer

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_id)
        self.hidden_size = self.wav2vec2.config.hidden_size  # 768

        if freeze_backbone:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            logger.info("Froze Wav2Vec2 backbone")

        # mean + max pool → 2 * hidden_size
        pool_dim = self.hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Classify audio as real or fake.

        Args:
            waveform: (B, num_samples) raw audio at 16kHz.
            attention_mask: (B, num_samples) optional mask for padded audio.

        Returns:
            (B, 1) logits (pre-sigmoid).
        """
        # Run frozen backbone in eval mode
        backbone_training = self.wav2vec2.training
        if not any(p.requires_grad for p in self.wav2vec2.parameters()):
            self.wav2vec2.eval()

        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.wav2vec2.parameters())
        ):
            outputs = self.wav2vec2(
                waveform,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        if backbone_training:
            self.wav2vec2.train(backbone_training)

        features = outputs.hidden_states[self.layer]  # (B, T, 768)

        # Mean + max pooling
        mean_pool = features.mean(dim=1)  # (B, 768)
        max_pool, _ = features.max(dim=1)  # (B, 768)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 1536)

        return self.classifier(pooled)  # (B, 1)


def build_standalone_audio_classifier(config: dict) -> StandaloneAudioClassifier:
    """Build standalone audio classifier from config.

    Args:
        config: Full config dict.

    Returns:
        StandaloneAudioClassifier instance.
    """
    ae_cfg = config["model"]["audio_encoder"]
    return StandaloneAudioClassifier(
        model_id=ae_cfg.get("model_id", "facebook/wav2vec2-base-960h"),
        layer=ae_cfg.get("layer", 9),
        freeze_backbone=ae_cfg.get("freeze_pretrained", True),
    )
