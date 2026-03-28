import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.visual_encoder import build_visual_encoder
from src.models.audio_encoder import build_audio_encoder
from src.models.classifier import build_classifier, AudioClassifier

logger = logging.getLogger(__name__)


@dataclass
class SyncGuardOutput:
    """Container for SyncGuard model outputs.

    Attributes:
        logits: (B, 1) fused real/fake classification logits (pre-sigmoid)
        sync_scores: (B, T) frame-level cosine similarities s(t)
        v_embeds: (B, T, D) visual embeddings (L2-normalized)
        a_embeds: (B, T, D) audio embeddings (L2-normalized)
        sync_logits: (B, 1) sync-based classifier logits
        audio_logits: (B, 1) audio-only classifier logits (None if disabled)
    """
    logits: torch.Tensor
    sync_scores: torch.Tensor
    v_embeds: torch.Tensor
    a_embeds: torch.Tensor
    sync_logits: torch.Tensor = None
    audio_logits: torch.Tensor = None


class SyncGuard(nn.Module):
    """Full SyncGuard model: two-stream contrastive AV deepfake detection.

    Combines visual encoder + audio encoder → frame-level sync-score →
    temporal classifier → real/fake prediction.

    Architecture:
        mouth_crops → VisualEncoder → v_t (B, T, D)
        waveform    → AudioEncoder  → a_t (B, T, D)
        s(t) = cos(v_t, a_t)       → (B, T)
        s(t) → Classifier           → logits (B, 1)

    Args:
        config: Full config dict with model.visual_encoder, model.audio_encoder,
                model.classifier sections.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.visual_encoder = build_visual_encoder(config)
        self.audio_encoder = build_audio_encoder(config)
        self.classifier = build_classifier(config)

        embedding_dim = config["model"]["visual_encoder"]["embedding_dim"]

        # Audio-only classification head (for RV-FA detection)
        self.use_audio_head = config["model"].get("audio_head", False)
        if self.use_audio_head:
            dropout = config["model"]["classifier"].get("dropout", 0.3)
            self.audio_classifier = AudioClassifier(
                embedding_dim=embedding_dim, dropout=dropout,
            )
            # Learnable fusion weight (sigmoid → [0,1])
            self.fusion_weight = nn.Parameter(torch.tensor(0.0))
            logger.info("Audio-only classification head enabled")

        logger.info(
            f"SyncGuard initialized: "
            f"visual={config['model']['visual_encoder']['name']}, "
            f"audio={config['model']['audio_encoder']['name']}, "
            f"classifier={config['model']['classifier']['name']}, "
            f"embedding_dim={embedding_dim}"
        )

    def compute_sync_scores(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute frame-level cosine similarity between visual and audio embeddings.

        Since both embeddings are L2-normalized, cosine similarity = dot product.

        Args:
            v_embeds: (B, T, D) L2-normalized visual embeddings
            a_embeds: (B, T, D) L2-normalized audio embeddings

        Returns:
            (B, T) cosine similarities in [-1, 1]
        """
        return (v_embeds * a_embeds).sum(dim=-1)

    def align_sequences(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align visual and audio embedding sequences to the same length.

        Truncates the longer sequence to match the shorter one.
        Visual features are upsampled to ~49Hz during preprocessing, so
        lengths should be close, but off-by-one differences can occur.

        Args:
            v_embeds: (B, T_v, D) visual embeddings
            a_embeds: (B, T_a, D) audio embeddings

        Returns:
            Aligned (B, T, D) visual and audio embeddings where T = min(T_v, T_a)
        """
        T_v = v_embeds.shape[1]
        T_a = a_embeds.shape[1]
        T = min(T_v, T_a)
        return v_embeds[:, :T], a_embeds[:, :T]

    def forward(
        self,
        mouth_crops: torch.Tensor,
        waveform: torch.Tensor,
        audio_attention_mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
        ear_features: torch.Tensor = None,
    ) -> SyncGuardOutput:
        """Full forward pass: mouth crops + waveform → real/fake prediction.

        Args:
            mouth_crops: (B, T_v, 1, 96, 96) grayscale mouth crops (float, [0,1])
            waveform: (B, num_samples) raw audio at 16kHz
            audio_attention_mask: (B, num_samples) optional mask for padded audio
            lengths: (B,) actual sync-score sequence lengths (for padded batches)
            ear_features: (B, T) per-frame EAR values (optional, for blink detection)

        Returns:
            SyncGuardOutput with logits, sync_scores, v_embeds, a_embeds
        """
        # Extract embeddings
        v_embeds = self.visual_encoder(mouth_crops)  # (B, T_v, D)
        a_embeds = self.audio_encoder(waveform, audio_attention_mask)  # (B, T_a, D)

        # Align temporal dimensions
        v_embeds, a_embeds = self.align_sequences(v_embeds, a_embeds)

        # Compute sync-scores
        sync_scores = self.compute_sync_scores(v_embeds, a_embeds)  # (B, T)

        # Truncate EAR features and clamp lengths to match aligned length
        T = sync_scores.shape[1]
        ear_aligned = None
        if ear_features is not None:
            ear_aligned = ear_features[:, :T]
        if lengths is not None:
            lengths = lengths.clamp(max=T)

        # Sync-based classification (with optional EAR features)
        sync_logits = self.classifier(sync_scores, lengths=lengths, ear_features=ear_aligned)  # (B, 1)

        # Audio-only classification (for RV-FA detection)
        audio_logits = None
        if self.use_audio_head:
            audio_logits = self.audio_classifier(a_embeds, lengths=lengths)  # (B, 1)
            # Fuse: learnable weighted average of sync and audio logits
            w = torch.sigmoid(self.fusion_weight)
            logits = (1 - w) * sync_logits + w * audio_logits
        else:
            logits = sync_logits

        return SyncGuardOutput(
            logits=logits,
            sync_scores=sync_scores,
            v_embeds=v_embeds,
            a_embeds=a_embeds,
            sync_logits=sync_logits,
            audio_logits=audio_logits,
        )

    def encode_visual(self, mouth_crops: torch.Tensor) -> torch.Tensor:
        """Extract visual embeddings only (for pretraining or analysis).

        Args:
            mouth_crops: (B, T, 1, 96, 96) grayscale mouth crops

        Returns:
            (B, T, D) L2-normalized visual embeddings
        """
        return self.visual_encoder(mouth_crops)

    def encode_audio(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Extract audio embeddings only (for pretraining or analysis).

        Args:
            waveform: (B, num_samples) raw audio at 16kHz
            attention_mask: (B, num_samples) optional mask

        Returns:
            (B, T, D) L2-normalized audio embeddings
        """
        return self.audio_encoder(waveform, attention_mask)


def build_syncguard(config: dict) -> SyncGuard:
    """Build a SyncGuard model from config.

    Args:
        config: Full config dict.

    Returns:
        SyncGuard model instance.
    """
    return SyncGuard(config)


if __name__ == "__main__":
    import yaml

    # Load config
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    print("Testing SyncGuard full model...")

    B = 2
    T_v = 50  # ~2 seconds at 25fps
    num_samples = 16000 * 2  # 2 seconds at 16kHz

    mouth_crops = torch.randn(B, T_v, 1, 96, 96)
    waveform = torch.randn(B, num_samples)

    model = SyncGuard(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Forward pass
    output = model(mouth_crops, waveform)

    print(f"  mouth_crops: {mouth_crops.shape}")
    print(f"  waveform: {waveform.shape}")
    print(f"  v_embeds: {output.v_embeds.shape}")
    print(f"  a_embeds: {output.a_embeds.shape}")
    print(f"  sync_scores: {output.sync_scores.shape}")
    print(f"  logits: {output.logits.shape}")

    # Verify shapes
    T = output.sync_scores.shape[1]
    assert output.v_embeds.shape == (B, T, 256)
    assert output.a_embeds.shape == (B, T, 256)
    assert output.sync_scores.shape == (B, T)
    assert output.logits.shape == (B, 1)
    print("  Shapes ✓")

    # Verify sync-scores are in [-1, 1] (cosine similarity of L2-normalized vectors)
    assert output.sync_scores.min() >= -1.0 - 1e-6
    assert output.sync_scores.max() <= 1.0 + 1e-6
    print("  Sync-scores in [-1, 1] ✓")

    # Verify L2-normalization of embeddings
    v_norms = output.v_embeds.norm(dim=-1)
    a_norms = output.a_embeds.norm(dim=-1)
    assert torch.allclose(v_norms, torch.ones_like(v_norms), atol=1e-5)
    assert torch.allclose(a_norms, torch.ones_like(a_norms), atol=1e-5)
    print("  Embeddings L2-normalized ✓")

    # Verify gradients flow
    loss = output.logits.sum() + output.sync_scores.sum()
    loss.backward()

    # Projection heads should have gradients
    assert model.visual_encoder.projection.fc1.weight.grad is not None
    assert model.audio_encoder.projection.fc1.weight.grad is not None
    # Classifier should have gradients
    assert any(p.grad is not None for p in model.classifier.parameters())
    print("  Gradients flow ✓")

    # Test encode-only methods
    v_only = model.encode_visual(mouth_crops)
    a_only = model.encode_audio(waveform)
    assert v_only.shape == (B, T_v, 256)
    print(f"  encode_visual: {v_only.shape} ✓")
    print(f"  encode_audio: {a_only.shape} ✓")

    print("All SyncGuard tests passed.")
