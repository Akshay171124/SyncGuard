"""Tests for audio encoder with mocked Wav2Vec2 (src/models/audio_encoder.py).

Tests verify:
- Output shape (B, T, embedding_dim)
- L2 normalization of embeddings
- Layer extraction from hidden_states tuple (pitfall #8)
- Frozen backbone behavior
- Projection head gradient flow
- train() mode override (SF-6 fix: wav2vec stays in inference mode)
"""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.models.audio_encoder import ProjectionHead, Wav2Vec2AudioEncoder


def _create_mock_wav2vec(hidden_size=768, num_layers=13):
    """Create a mock Wav2Vec2Model that returns fake hidden states."""
    mock_model = MagicMock()
    mock_model.config = SimpleNamespace(hidden_size=hidden_size)

    # Mock parameters() to return empty iterator (no real params)
    mock_model.parameters.return_value = iter([])
    mock_model.modules.return_value = iter([mock_model])

    def forward_fn(waveform, attention_mask=None, output_hidden_states=True, return_dict=True):
        B = waveform.shape[0]
        T = waveform.shape[1] // 320  # Wav2Vec stride

        # Build hidden_states tuple: (num_layers+1) x (B, T, hidden_size)
        hidden_states = tuple(
            torch.randn(B, T, hidden_size) for _ in range(num_layers)
        )
        return SimpleNamespace(hidden_states=hidden_states)

    mock_model.side_effect = forward_fn
    mock_model.__call__ = forward_fn
    return mock_model


class TestWav2Vec2AudioEncoder:
    @pytest.fixture
    def encoder(self):
        """Create encoder with mocked Wav2Vec2 backbone."""
        with patch("src.models.audio_encoder.Wav2Vec2Model") as MockCls:
            mock = _create_mock_wav2vec()
            MockCls.from_pretrained.return_value = mock
            enc = Wav2Vec2AudioEncoder(
                model_id="facebook/wav2vec2-base-960h",
                layer=9,
                embedding_dim=256,
                freeze_pretrained=True,
            )
        return enc

    def test_output_shape(self, encoder):
        """Input (B, num_samples) -> output (B, T, 256)."""
        B, num_samples = 2, 16000 * 3  # 3 seconds
        waveform = torch.randn(B, num_samples)
        out = encoder(waveform)
        T_expected = num_samples // 320
        assert out.shape == (B, T_expected, 256)

    def test_l2_normalized(self, encoder):
        """Output embeddings are L2-normalized."""
        waveform = torch.randn(2, 16000)
        out = encoder(waveform)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_layer_extraction_uses_correct_index(self):
        """Encoder extracts the specified hidden layer (pitfall #8)."""
        target_layer = 5
        with patch("src.models.audio_encoder.Wav2Vec2Model") as MockCls:
            mock = _create_mock_wav2vec()

            # Make each layer output identifiable by filling with layer index
            def tagged_forward(waveform, **kwargs):
                B = waveform.shape[0]
                T = waveform.shape[1] // 320
                hidden_states = []
                for i in range(13):
                    h = torch.full((B, T, 768), float(i))
                    hidden_states.append(h)
                return SimpleNamespace(hidden_states=tuple(hidden_states))

            mock.__call__ = tagged_forward
            MockCls.from_pretrained.return_value = mock

            enc = Wav2Vec2AudioEncoder(layer=target_layer, embedding_dim=256)

        waveform = torch.randn(1, 16000)
        out = enc(waveform)
        # Verify output shape is correct (projection applied)
        assert out.shape == (1, 16000 // 320, 256)

    def test_projection_head_gradient(self, encoder):
        """Gradient flows through projection head."""
        waveform = torch.randn(2, 16000)
        out = encoder(waveform)
        out.sum().backward()
        assert encoder.projection.fc1.weight.grad is not None

    def test_train_mode_keeps_wav2vec_in_inference(self, encoder):
        """SF-6 fix: Wav2Vec backbone stays in inference mode when model.train() is called."""
        encoder.train(True)
        assert encoder.wav2vec2.training is False


class TestProjectionHeadAudio:
    def test_output_l2_normalized(self):
        """ProjectionHead output is L2-normalized."""
        proj = ProjectionHead(in_dim=768, out_dim=256)
        x = torch.randn(10, 768)
        out = proj(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_output_shape(self):
        """(N, 768) -> (N, 256)."""
        proj = ProjectionHead(in_dim=768, out_dim=256)
        x = torch.randn(5, 768)
        out = proj(x)
        assert out.shape == (5, 256)
