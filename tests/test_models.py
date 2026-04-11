"""Tests for model architectures (src/models/).

Tests verify:
- Output shapes match expected (B, T, D) or (B, 1)
- Embeddings are L2-normalized (norms ≈ 1.0)
- Freeze/unfreeze behavior works correctly
- Factory functions instantiate correct classes
- Classifier handles variable-length input with masking
- Cross-attention preserves shape and allows gradients
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.visual_encoder import (
    AVHubertVisualEncoder,
    Frontend3D,
    ProjectionHead,
    ResNet18VisualEncoder,
    ResNetTrunk,
    SyncNetVisualEncoder,
    build_visual_encoder,
)
from src.models.classifier import (
    AudioClassifier,
    BiLSTMClassifier,
    CNN1DClassifier,
    StatisticalClassifier,
    build_classifier,
)
from src.models.cross_attention import (
    CrossAttentionModule,
    EmbedClassifier,
)
from src.models.dct_extractor import DCTFeatureExtractor, dct2d


# ──────────────────────────────────────────────
# Frontend3D + ResNetTrunk (sub-components)
# ──────────────────────────────────────────────

class TestFrontend3D:
    def test_output_shape(self):
        """(B, 1, T, 96, 96) → (B, 64, T, 24, 24)."""
        B, T = 2, 10
        x = torch.randn(B, 1, T, 96, 96)
        frontend = Frontend3D()
        out = frontend(x)
        assert out.shape == (B, 64, T, 24, 24)


class TestResNetTrunk:
    def test_output_shape(self):
        """(B*T, 64, 24, 24) → (B*T, 512)."""
        N = 20  # B * T
        x = torch.randn(N, 64, 24, 24)
        trunk = ResNetTrunk()
        out = trunk(x)
        assert out.shape == (N, 512)


class TestProjectionHead:
    def test_output_shape(self):
        """(N, in_dim) → (N, out_dim)."""
        proj = ProjectionHead(in_dim=512, out_dim=256)
        x = torch.randn(10, 512)
        out = proj(x)
        assert out.shape == (10, 256)

    def test_l2_normalized(self):
        """Output is L2-normalized (norms ≈ 1.0)."""
        proj = ProjectionHead(in_dim=512, out_dim=256)
        x = torch.randn(10, 512)
        out = proj(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# ──────────────────────────────────────────────
# Visual Encoders
# ──────────────────────────────────────────────

class TestVisualEncoders:
    """Parametrized tests across all visual encoder variants."""

    @pytest.fixture(params=[
        ("av_hubert", AVHubertVisualEncoder),
        ("resnet18", ResNet18VisualEncoder),
        ("syncnet", SyncNetVisualEncoder),
    ])
    def encoder_pair(self, request):
        """Yields (name, encoder_instance)."""
        name, cls = request.param
        return name, cls(embedding_dim=256)

    def test_output_shape(self, encoder_pair):
        """Input (B, T, 1, 96, 96) → output (B, T, 256)."""
        name, encoder = encoder_pair
        B, T = 2, 8
        x = torch.randn(B, T, 1, 96, 96)
        out = encoder(x)
        assert out.shape == (B, T, 256), f"{name}: got {out.shape}"

    def test_l2_normalized_output(self, encoder_pair):
        """Embeddings are L2-normalized (norms ≈ 1.0)."""
        name, encoder = encoder_pair
        x = torch.randn(2, 8, 1, 96, 96)
        out = encoder(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"{name}: not normalized"

    def test_gradient_flows(self, encoder_pair):
        """Loss backward produces gradients in projection head."""
        name, encoder = encoder_pair
        x = torch.randn(2, 4, 1, 96, 96)
        out = encoder(x)
        out.sum().backward()
        assert encoder.projection.fc1.weight.grad is not None, f"{name}: no grad"

    def test_batch_size_one(self, encoder_pair):
        """Works with batch size 1."""
        name, encoder = encoder_pair
        x = torch.randn(1, 4, 1, 96, 96)
        out = encoder(x)
        assert out.shape == (1, 4, 256)


class TestAVHubertFreeze:
    def test_freeze_backbone(self):
        """Frozen backbone parameters should not require grad."""
        encoder = AVHubertVisualEncoder(embedding_dim=256, freeze_pretrained=True)
        for p in encoder.frontend3d.parameters():
            assert not p.requires_grad
        for p in encoder.trunk.parameters():
            assert not p.requires_grad

    def test_projection_still_trainable_when_frozen(self):
        """Projection head remains trainable when backbone is frozen."""
        encoder = AVHubertVisualEncoder(embedding_dim=256, freeze_pretrained=True)
        for p in encoder.projection.parameters():
            assert p.requires_grad

    def test_frozen_backbone_no_grad(self):
        """Backward pass with frozen backbone → no grad on backbone params."""
        encoder = AVHubertVisualEncoder(embedding_dim=256, freeze_pretrained=True)
        x = torch.randn(2, 4, 1, 96, 96)
        out = encoder(x)
        out.sum().backward()
        for p in encoder.frontend3d.parameters():
            assert p.grad is None
        assert encoder.projection.fc1.weight.grad is not None


# ──────────────────────────────────────────────
# Visual Encoder Factory
# ──────────────────────────────────────────────

class TestBuildVisualEncoder:
    @pytest.mark.parametrize("name,expected_cls", [
        ("av_hubert", AVHubertVisualEncoder),
        ("resnet18", ResNet18VisualEncoder),
        ("syncnet", SyncNetVisualEncoder),
    ])
    def test_factory_returns_correct_class(self, default_config, name, expected_cls):
        """Factory instantiates correct encoder class."""
        default_config["model"]["visual_encoder"]["name"] = name
        encoder = build_visual_encoder(default_config)
        assert isinstance(encoder, expected_cls)

    def test_unknown_encoder_raises(self, default_config):
        """Unknown encoder name raises ValueError."""
        default_config["model"]["visual_encoder"]["name"] = "nonexistent"
        with pytest.raises(ValueError, match="Unknown visual encoder"):
            build_visual_encoder(default_config)


# ──────────────────────────────────────────────
# Classifiers
# ──────────────────────────────────────────────

class TestClassifiers:
    """Parametrized tests across all classifier variants."""

    @pytest.fixture(params=[
        ("bilstm", BiLSTMClassifier, {"hidden_size": 64, "num_layers": 1}),
        ("cnn1d", CNN1DClassifier, {"hidden_size": 64}),
        ("statistical", StatisticalClassifier, {}),
    ])
    def classifier_pair(self, request):
        name, cls, kwargs = request.param
        return name, cls(**kwargs)

    def test_output_shape(self, classifier_pair):
        """Input (B, T) → output (B, 1)."""
        name, clf = classifier_pair
        B, T = 4, 50
        sync_scores = torch.randn(B, T)
        logits = clf(sync_scores)
        assert logits.shape == (B, 1), f"{name}: got {logits.shape}"

    def test_with_lengths(self, classifier_pair):
        """Works with sequence lengths for masked pooling."""
        name, clf = classifier_pair
        B, T = 4, 50
        sync_scores = torch.randn(B, T)
        lengths = torch.tensor([50, 40, 30, 20])
        logits = clf(sync_scores, lengths=lengths)
        assert logits.shape == (B, 1)

    def test_gradient_flows(self, classifier_pair):
        """Loss backward produces gradients."""
        name, clf = classifier_pair
        sync_scores = torch.randn(4, 50)
        logits = clf(sync_scores)
        logits.sum().backward()
        has_grad = any(p.grad is not None for p in clf.parameters())
        assert has_grad, f"{name}: no gradients"

    def test_batch_size_one(self, classifier_pair):
        """Works with batch size 1."""
        name, clf = classifier_pair
        sync_scores = torch.randn(1, 50)
        logits = clf(sync_scores)
        assert logits.shape == (1, 1)


class TestBiLSTMWithEAR:
    def test_ear_input_accepted(self):
        """BiLSTM with use_ear=True accepts ear_features."""
        clf = BiLSTMClassifier(hidden_size=64, num_layers=1, use_ear=True)
        B, T = 4, 50
        sync_scores = torch.randn(B, T)
        ear = torch.randn(B, T)
        logits = clf(sync_scores, ear_features=ear)
        assert logits.shape == (B, 1)


class TestAudioClassifier:
    def test_output_shape(self):
        """Input (B, T, D) → output (B, 1)."""
        clf = AudioClassifier(embedding_dim=256)
        a_embeds = torch.randn(4, 50, 256)
        logits = clf(a_embeds)
        assert logits.shape == (4, 1)

    def test_with_lengths(self):
        """Masked pooling with sequence lengths."""
        clf = AudioClassifier(embedding_dim=256)
        a_embeds = torch.randn(4, 50, 256)
        lengths = torch.tensor([50, 40, 30, 20])
        logits = clf(a_embeds, lengths=lengths)
        assert logits.shape == (4, 1)


class TestBuildClassifier:
    @pytest.mark.parametrize("name,expected_cls", [
        ("bilstm", BiLSTMClassifier),
        ("cnn1d", CNN1DClassifier),
        ("statistical", StatisticalClassifier),
    ])
    def test_factory_returns_correct_class(self, default_config, name, expected_cls):
        """Factory instantiates correct classifier class."""
        default_config["model"]["classifier"]["name"] = name
        clf = build_classifier(default_config)
        assert isinstance(clf, expected_cls)

    def test_unknown_classifier_raises(self, default_config):
        """Unknown classifier name raises ValueError."""
        default_config["model"]["classifier"]["name"] = "transformer"
        with pytest.raises(ValueError, match="Unknown classifier"):
            build_classifier(default_config)


# ──────────────────────────────────────────────
# Cross-Attention
# ──────────────────────────────────────────────

class TestCrossAttentionModule:
    def test_output_shape(self):
        """Preserves input shape: (B, T, D) → (B, T, D) for both modalities."""
        B, T, D = 4, 20, 256
        ca = CrossAttentionModule(embed_dim=D, num_heads=2)
        v = torch.randn(B, T, D)
        a = torch.randn(B, T, D)
        v_att, a_att = ca(v, a)
        assert v_att.shape == (B, T, D)
        assert a_att.shape == (B, T, D)

    def test_with_padding_mask(self):
        """Works with key_padding_mask for padded sequences."""
        B, T, D = 4, 20, 256
        ca = CrossAttentionModule(embed_dim=D, num_heads=2)
        v = torch.randn(B, T, D)
        a = torch.randn(B, T, D)
        # True = padding position
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[-1, 15:] = True
        v_att, a_att = ca(v, a, key_padding_mask=mask)
        assert v_att.shape == (B, T, D)

    def test_gradient_flows(self):
        """Gradients flow through attention layers."""
        ca = CrossAttentionModule(embed_dim=64, num_heads=2)
        v = torch.randn(2, 10, 64, requires_grad=True)
        a = torch.randn(2, 10, 64)
        v_att, a_att = ca(v, a)
        (v_att.sum() + a_att.sum()).backward()
        assert v.grad is not None


class TestEmbedClassifier:
    def test_output_shape(self):
        """Input (B, T, D) × 2 → output (B, 1)."""
        clf = EmbedClassifier(embed_dim=256, hidden_dim=128)
        v = torch.randn(4, 20, 256)
        a = torch.randn(4, 20, 256)
        logits = clf(v, a)
        assert logits.shape == (4, 1)

    def test_with_lengths(self):
        """Masked pooling with sequence lengths."""
        clf = EmbedClassifier(embed_dim=256, hidden_dim=128)
        v = torch.randn(4, 20, 256)
        a = torch.randn(4, 20, 256)
        lengths = torch.tensor([20, 15, 10, 5])
        logits = clf(v, a, lengths=lengths)
        assert logits.shape == (4, 1)

    def test_with_dct_features(self):
        """Accepts optional DCT features."""
        dct_dim = 32
        clf = EmbedClassifier(embed_dim=256, hidden_dim=128, dct_dim=dct_dim)
        v = torch.randn(4, 20, 256)
        a = torch.randn(4, 20, 256)
        dct = torch.randn(4, 20, dct_dim)
        logits = clf(v, a, dct_features=dct)
        assert logits.shape == (4, 1)


# ──────────────────────────────────────────────
# DCT Feature Extractor
# ──────────────────────────────────────────────

class TestDCT2D:
    def test_output_shape(self):
        """dct2d preserves input shape: (B, 1, H, W) -> (B, 1, H, W)."""
        x = torch.randn(4, 1, 96, 96)
        out = dct2d(x)
        assert out.shape == (4, 1, 96, 96)

    def test_output_is_real(self):
        """DCT output is real-valued (no imaginary components)."""
        x = torch.randn(2, 1, 96, 96)
        out = dct2d(x)
        assert out.dtype in (torch.float32, torch.float64)
        assert torch.all(torch.isfinite(out))

    def test_dc_component_nonzero(self):
        """DC component (top-left) should be nonzero for nonzero input."""
        x = torch.ones(1, 1, 32, 32)
        out = dct2d(x)
        assert out[0, 0, 0, 0].abs() > 0


class TestDCTFeatureExtractor:
    def test_output_shape(self):
        """(B, T, 1, 96, 96) -> (B, T, output_dim)."""
        ext = DCTFeatureExtractor(output_dim=16)
        x = torch.randn(2, 10, 1, 96, 96)
        out = ext(x)
        assert out.shape == (2, 10, 16)

    def test_different_output_dims(self):
        """Output dimension matches configuration."""
        for dim in [8, 16, 32]:
            ext = DCTFeatureExtractor(output_dim=dim)
            x = torch.randn(1, 5, 1, 96, 96)
            out = ext(x)
            assert out.shape[2] == dim

    def test_gradient_flows(self):
        """Gradient flows through CNN layers."""
        ext = DCTFeatureExtractor(output_dim=16)
        x = torch.randn(2, 5, 1, 96, 96)
        out = ext(x)
        out.sum().backward()
        has_grad = any(p.grad is not None for p in ext.parameters())
        assert has_grad

    def test_no_nan_output(self):
        """No NaN in output for random input."""
        ext = DCTFeatureExtractor(output_dim=16)
        x = torch.randn(2, 5, 1, 96, 96)
        out = ext(x)
        assert torch.all(torch.isfinite(out))

    def test_batch_size_one(self):
        """Works with batch size 1."""
        ext = DCTFeatureExtractor(output_dim=16)
        x = torch.randn(1, 3, 1, 96, 96)
        out = ext(x)
        assert out.shape == (1, 3, 16)
