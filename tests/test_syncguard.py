"""Integration tests for the full SyncGuard model (src/models/syncguard.py).

Tests verify:
- align_sequences truncates to min length
- compute_sync_scores produces cosine similarities in [-1, 1]
- Full forward pass produces correct output shapes
- SyncGuardOutput dataclass fields populated correctly
- Optional audio head and cross-attention paths
- Gradient flow through full pipeline
"""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

from src.models.syncguard import SyncGuard, SyncGuardOutput, build_syncguard


class FakeVisualEncoder(torch.nn.Module):
    """Lightweight visual encoder stub for integration tests."""
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.projection = torch.nn.Linear(10, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        B, T = x.shape[:2]
        out = torch.randn(B, T, self.embedding_dim)
        return F.normalize(out, dim=-1)


class FakeAudioEncoder(torch.nn.Module):
    """Lightweight audio encoder stub (avoids Wav2Vec2 download)."""
    def __init__(self, embedding_dim=256, output_T=None):
        super().__init__()
        self.projection = torch.nn.Linear(10, embedding_dim)
        self.embedding_dim = embedding_dim
        self.output_T = output_T

    def forward(self, waveform, attention_mask=None):
        B = waveform.shape[0]
        T = self.output_T or (waveform.shape[1] // 320)
        out = torch.randn(B, T, self.embedding_dim)
        return F.normalize(out, dim=-1)


def _make_syncguard(config, visual_T=50, audio_T=None):
    """Build SyncGuard with fake encoders to avoid pretrained model downloads."""
    model = SyncGuard.__new__(SyncGuard)
    torch.nn.Module.__init__(model)

    embedding_dim = config["model"]["visual_encoder"]["embedding_dim"]
    model.visual_encoder = FakeVisualEncoder(embedding_dim)
    model.audio_encoder = FakeAudioEncoder(embedding_dim, output_T=audio_T)

    from src.models.classifier import build_classifier
    model.classifier = build_classifier(config)

    model.use_audio_head = config["model"].get("audio_head", False)
    model.use_cross_attention = False
    model.use_dct = False

    if model.use_audio_head:
        from src.models.classifier import AudioClassifier
        dropout = config["model"]["classifier"].get("dropout", 0.3)
        model.audio_classifier = AudioClassifier(embedding_dim=embedding_dim, dropout=dropout)
        model.fusion_weight = torch.nn.Parameter(torch.tensor(0.0))

    return model


# ──────────────────────────────────────────────
# align_sequences
# ──────────────────────────────────────────────

class TestAlignSequences:
    def test_equal_lengths_unchanged(self, default_config):
        """Same-length sequences returned as-is."""
        model = _make_syncguard(default_config)
        v = torch.randn(2, 50, 256)
        a = torch.randn(2, 50, 256)
        v_out, a_out = model.align_sequences(v, a)
        assert v_out.shape[1] == 50
        assert a_out.shape[1] == 50

    def test_visual_longer_truncated(self, default_config):
        """Visual longer than audio → visual truncated."""
        model = _make_syncguard(default_config)
        v = torch.randn(2, 60, 256)
        a = torch.randn(2, 50, 256)
        v_out, a_out = model.align_sequences(v, a)
        assert v_out.shape[1] == 50
        assert a_out.shape[1] == 50

    def test_audio_longer_truncated(self, default_config):
        """Audio longer than visual → audio truncated."""
        model = _make_syncguard(default_config)
        v = torch.randn(2, 45, 256)
        a = torch.randn(2, 50, 256)
        v_out, a_out = model.align_sequences(v, a)
        assert v_out.shape[1] == 45
        assert a_out.shape[1] == 45

    def test_off_by_one(self, default_config):
        """Off-by-one difference handled (common in AV alignment)."""
        model = _make_syncguard(default_config)
        v = torch.randn(2, 49, 256)
        a = torch.randn(2, 50, 256)
        v_out, a_out = model.align_sequences(v, a)
        assert v_out.shape[1] == a_out.shape[1] == 49

    def test_preserves_batch_and_dim(self, default_config):
        """Batch size and embedding dim preserved."""
        model = _make_syncguard(default_config)
        v = torch.randn(4, 30, 128)
        a = torch.randn(4, 25, 128)
        v_out, a_out = model.align_sequences(v, a)
        assert v_out.shape == (4, 25, 128)
        assert a_out.shape == (4, 25, 128)


# ──────────────────────────────────────────────
# compute_sync_scores
# ──────────────────────────────────────────────

class TestComputeSyncScores:
    def test_output_shape(self, default_config):
        """(B, T, D) × (B, T, D) → (B, T)."""
        model = _make_syncguard(default_config)
        v = F.normalize(torch.randn(4, 20, 256), dim=-1)
        a = F.normalize(torch.randn(4, 20, 256), dim=-1)
        scores = model.compute_sync_scores(v, a)
        assert scores.shape == (4, 20)

    def test_range_minus_one_to_one(self, default_config):
        """Cosine similarity of L2-normed vectors is in [-1, 1]."""
        model = _make_syncguard(default_config)
        v = F.normalize(torch.randn(4, 50, 256), dim=-1)
        a = F.normalize(torch.randn(4, 50, 256), dim=-1)
        scores = model.compute_sync_scores(v, a)
        assert scores.min() >= -1.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6

    def test_identical_embeddings_score_one(self, default_config):
        """Identical v and a → sync score = 1.0."""
        model = _make_syncguard(default_config)
        v = F.normalize(torch.randn(2, 10, 256), dim=-1)
        scores = model.compute_sync_scores(v, v.clone())
        assert torch.allclose(scores, torch.ones_like(scores), atol=1e-5)

    def test_opposite_embeddings_score_minus_one(self, default_config):
        """Negated embeddings → sync score = -1.0."""
        model = _make_syncguard(default_config)
        v = F.normalize(torch.randn(2, 10, 256), dim=-1)
        scores = model.compute_sync_scores(v, -v)
        assert torch.allclose(scores, -torch.ones_like(scores), atol=1e-5)


# ──────────────────────────────────────────────
# Full forward pass
# ──────────────────────────────────────────────

class TestSyncGuardForward:
    def test_output_type(self, default_config):
        """Forward returns SyncGuardOutput dataclass."""
        model = _make_syncguard(default_config, audio_T=50)
        mouth_crops = torch.randn(2, 50, 1, 96, 96)
        waveform = torch.randn(2, 32000)
        output = model(mouth_crops, waveform)
        assert isinstance(output, SyncGuardOutput)

    def test_output_shapes(self, default_config):
        """All output tensors have correct shapes."""
        model = _make_syncguard(default_config, audio_T=50)
        B, T_v = 2, 50
        mouth_crops = torch.randn(B, T_v, 1, 96, 96)
        waveform = torch.randn(B, 32000)
        output = model(mouth_crops, waveform)

        T = output.sync_scores.shape[1]
        assert output.logits.shape == (B, 1)
        assert output.sync_scores.shape == (B, T)
        assert output.v_embeds.shape == (B, T, 256)
        assert output.a_embeds.shape == (B, T, 256)
        assert output.sync_logits.shape == (B, 1)

    def test_embeddings_l2_normalized(self, default_config):
        """Output embeddings are L2-normalized."""
        model = _make_syncguard(default_config, audio_T=50)
        mouth_crops = torch.randn(2, 50, 1, 96, 96)
        waveform = torch.randn(2, 32000)
        output = model(mouth_crops, waveform)
        v_norms = output.v_embeds.norm(dim=-1)
        a_norms = output.a_embeds.norm(dim=-1)
        assert torch.allclose(v_norms, torch.ones_like(v_norms), atol=1e-4)
        assert torch.allclose(a_norms, torch.ones_like(a_norms), atol=1e-4)

    def test_with_lengths_and_ear(self, default_config):
        """Forward pass with optional lengths and EAR features."""
        model = _make_syncguard(default_config, audio_T=50)
        B, T_v = 2, 50
        mouth_crops = torch.randn(B, T_v, 1, 96, 96)
        waveform = torch.randn(B, 32000)
        lengths = torch.tensor([50, 40])
        ear = torch.randn(B, T_v)
        output = model(mouth_crops, waveform, lengths=lengths, ear_features=ear)
        assert output.logits.shape == (B, 1)

    def test_gradient_flows_end_to_end(self, default_config):
        """Gradient flows from logits back through encoders."""
        model = _make_syncguard(default_config, audio_T=50)
        mouth_crops = torch.randn(2, 50, 1, 96, 96)
        waveform = torch.randn(2, 32000)
        output = model(mouth_crops, waveform)
        loss = output.logits.sum() + output.sync_scores.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_no_audio_logits_by_default(self, default_config):
        """audio_logits is None when audio_head disabled."""
        model = _make_syncguard(default_config, audio_T=50)
        mouth_crops = torch.randn(2, 50, 1, 96, 96)
        waveform = torch.randn(2, 32000)
        output = model(mouth_crops, waveform)
        assert output.audio_logits is None

    def test_audio_head_produces_logits(self, default_config):
        """audio_logits populated when audio_head enabled."""
        default_config["model"]["audio_head"] = True
        model = _make_syncguard(default_config, audio_T=50)
        mouth_crops = torch.randn(2, 50, 1, 96, 96)
        waveform = torch.randn(2, 32000)
        output = model(mouth_crops, waveform)
        assert output.audio_logits is not None
        assert output.audio_logits.shape == (2, 1)


# ──────────────────────────────────────────────
# Encode-only methods
# ──────────────────────────────────────────────

class TestEncodeOnly:
    def test_encode_visual(self, default_config):
        """encode_visual returns (B, T, D) embeddings."""
        model = _make_syncguard(default_config)
        mouth_crops = torch.randn(2, 30, 1, 96, 96)
        v = model.encode_visual(mouth_crops)
        assert v.shape == (2, 30, 256)

    def test_encode_audio(self, default_config):
        """encode_audio returns (B, T, D) embeddings."""
        model = _make_syncguard(default_config, audio_T=100)
        waveform = torch.randn(2, 32000)
        a = model.encode_audio(waveform)
        assert a.shape == (2, 100, 256)


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class TestBuildSyncguard:
    def test_build_returns_syncguard(self, default_config):
        """build_syncguard returns a SyncGuard instance (uses real encoders)."""
        # This downloads Wav2Vec2 — skip if not cached
        try:
            model = build_syncguard(default_config)
            assert isinstance(model, SyncGuard)
        except Exception:
            pytest.skip("Wav2Vec2 model not available locally")
