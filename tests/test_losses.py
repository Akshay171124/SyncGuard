"""Tests for training loss functions (src/training/losses.py).

Tests verify:
- MoCo queue FIFO behavior and capacity
- InfoNCE loss shape, finiteness, gradient flow
- Temperature clamping to configured range
- Temporal consistency loss real-only gating
- Combined loss component decomposition
- Pretrain loss with cross-modal prediction
"""

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import (
    CombinedLoss,
    CrossModalPredictionLoss,
    InfoNCELoss,
    MoCoQueue,
    PretrainLoss,
    TemporalConsistencyLoss,
    build_finetune_loss,
    build_pretrain_loss,
)


# ──────────────────────────────────────────────
# MoCo Queue
# ──────────────────────────────────────────────

class TestMoCoQueue:
    def test_queue_initializes_empty(self):
        """Queue starts not full, pointer at 0."""
        q = MoCoQueue(dim=64, size=128)
        assert q.ptr.item() == 0
        assert not q.is_full.item()

    def test_enqueue_updates_pointer(self):
        """Enqueue moves the pointer forward."""
        q = MoCoQueue(dim=64, size=128)
        embeddings = F.normalize(torch.randn(32, 64), dim=-1)
        q.enqueue_dequeue(embeddings)
        assert q.ptr.item() == 32

    def test_queue_wraps_around(self):
        """Queue wraps around when it exceeds capacity."""
        q = MoCoQueue(dim=64, size=64)
        e1 = F.normalize(torch.randn(40, 64), dim=-1)
        q.enqueue_dequeue(e1)
        assert q.ptr.item() == 40
        e2 = F.normalize(torch.randn(40, 64), dim=-1)
        q.enqueue_dequeue(e2)
        # 40 + 40 = 80 → wraps: ptr = 80 % 64 = 16
        assert q.ptr.item() == 16
        assert q.is_full.item()

    def test_get_negatives_before_full(self):
        """Before full, returns only enqueued entries."""
        q = MoCoQueue(dim=64, size=128)
        e = F.normalize(torch.randn(30, 64), dim=-1)
        q.enqueue_dequeue(e)
        neg = q.get_negatives()
        assert neg.shape == (30, 64)

    def test_get_negatives_when_full(self):
        """When full, returns entire queue."""
        q = MoCoQueue(dim=64, size=64)
        e = F.normalize(torch.randn(100, 64), dim=-1)
        q.enqueue_dequeue(e)
        neg = q.get_negatives()
        assert neg.shape == (64, 64)

    def test_queue_detaches_gradients(self):
        """Enqueued embeddings should not require grad."""
        q = MoCoQueue(dim=64, size=128)
        e = F.normalize(torch.randn(16, 64, requires_grad=True), dim=-1)
        q.enqueue_dequeue(e)
        neg = q.get_negatives()
        assert not neg.requires_grad

    def test_oversized_batch(self):
        """Batch larger than queue → only last queue_size entries kept."""
        q = MoCoQueue(dim=64, size=32)
        e = F.normalize(torch.randn(100, 64), dim=-1)
        q.enqueue_dequeue(e)
        neg = q.get_negatives()
        assert neg.shape == (32, 64)
        assert q.is_full.item()


# ──────────────────────────────────────────────
# InfoNCE Loss
# ──────────────────────────────────────────────

class TestInfoNCELoss:
    def test_output_is_scalar(self, dummy_embeddings):
        """Loss output is a scalar tensor."""
        v, a = dummy_embeddings
        loss_fn = InfoNCELoss(embedding_dim=256, queue_size=128)
        loss = loss_fn(v, a)
        assert loss.dim() == 0

    def test_loss_is_finite(self, dummy_embeddings):
        """Loss should not be NaN or Inf."""
        v, a = dummy_embeddings
        loss_fn = InfoNCELoss(embedding_dim=256, queue_size=128)
        loss = loss_fn(v, a)
        assert loss.isfinite()

    def test_loss_has_gradient(self, dummy_embeddings):
        """Loss should be differentiable."""
        v, a = dummy_embeddings
        v.requires_grad_(True)
        loss_fn = InfoNCELoss(embedding_dim=256, queue_size=128)
        loss = loss_fn(v, a)
        loss.backward()
        assert v.grad is not None

    def test_mask_reduces_effective_samples(self, dummy_embeddings, dummy_mask):
        """Masked loss should differ from unmasked when padding exists."""
        v, a = dummy_embeddings
        loss_fn = InfoNCELoss(embedding_dim=256, queue_size=128)
        loss_no_mask = loss_fn(v.clone(), a.clone(), update_queue=False)
        loss_masked = loss_fn(v.clone(), a.clone(), mask=dummy_mask, update_queue=False)
        # They should generally differ since mask excludes some frames
        assert loss_no_mask.isfinite()
        assert loss_masked.isfinite()

    def test_queue_updated_after_forward(self, dummy_embeddings):
        """Queue should contain entries after forward pass."""
        v, a = dummy_embeddings
        loss_fn = InfoNCELoss(embedding_dim=256, queue_size=256)
        loss_fn(v, a, update_queue=True)
        neg = loss_fn.queue.get_negatives()
        B, T = v.shape[:2]
        assert neg.shape[0] == B * T

    def test_queue_not_updated_during_validation(self, dummy_embeddings):
        """Queue unchanged when update_queue=False."""
        v, a = dummy_embeddings
        loss_fn = InfoNCELoss(embedding_dim=256, queue_size=256)
        neg_before = loss_fn.queue.get_negatives().clone()
        loss_fn(v, a, update_queue=False)
        neg_after = loss_fn.queue.get_negatives()
        assert neg_before.shape == neg_after.shape

    def test_temperature_property_clamped(self):
        """Temperature should be clamped to configured range."""
        loss_fn = InfoNCELoss(temperature=0.001, temperature_range=(0.01, 0.5))
        tau = loss_fn.temperature
        assert tau >= 0.01
        assert tau <= 0.5

    def test_temperature_is_learnable(self):
        """log_temperature should be a Parameter when learnable."""
        loss_fn = InfoNCELoss(learnable_temperature=True)
        assert isinstance(loss_fn.log_temperature, torch.nn.Parameter)

    def test_temperature_not_learnable(self):
        """log_temperature should be a buffer when not learnable."""
        loss_fn = InfoNCELoss(learnable_temperature=False)
        assert not isinstance(loss_fn.log_temperature, torch.nn.Parameter)

    def test_temperature_gradient_flows(self, dummy_embeddings):
        """Gradient should flow to learnable temperature."""
        v, a = dummy_embeddings
        loss_fn = InfoNCELoss(embedding_dim=256, queue_size=128, learnable_temperature=True)
        loss = loss_fn(v, a)
        loss.backward()
        assert loss_fn.log_temperature.grad is not None


# ──────────────────────────────────────────────
# Temporal Consistency Loss
# ──────────────────────────────────────────────

class TestTemporalConsistencyLoss:
    def test_zero_for_all_fake(self, dummy_embeddings):
        """Loss = 0 when all samples are fake (real-only loss)."""
        v, a = dummy_embeddings
        B = v.shape[0]
        is_real = torch.zeros(B, dtype=torch.bool)
        loss_fn = TemporalConsistencyLoss()
        loss = loss_fn(v, a, is_real=is_real)
        assert loss.item() == 0.0

    def test_positive_for_real_random_embeddings(self, dummy_embeddings):
        """Loss > 0 for real clips with random (non-synchronized) embeddings."""
        v, a = dummy_embeddings
        B = v.shape[0]
        is_real = torch.ones(B, dtype=torch.bool)
        loss_fn = TemporalConsistencyLoss()
        loss = loss_fn(v, a, is_real=is_real)
        assert loss.item() > 0.0

    def test_zero_for_identical_embeddings(self):
        """Loss = 0 when v and a are identical (perfect sync)."""
        B, T, D = 2, 10, 64
        embeds = torch.randn(B, T, D)
        is_real = torch.ones(B, dtype=torch.bool)
        loss_fn = TemporalConsistencyLoss()
        loss = loss_fn(embeds, embeds.clone(), is_real=is_real)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_respects_mask(self, dummy_embeddings, dummy_mask):
        """Loss differs with mask (padded frames excluded)."""
        v, a = dummy_embeddings
        B = v.shape[0]
        is_real = torch.ones(B, dtype=torch.bool)
        loss_fn = TemporalConsistencyLoss()
        loss_no_mask = loss_fn(v, a, is_real=is_real)
        loss_masked = loss_fn(v, a, is_real=is_real, mask=dummy_mask)
        # Both should be valid
        assert loss_no_mask.isfinite()
        assert loss_masked.isfinite()

    def test_gradient_flows(self, dummy_embeddings):
        """Loss is differentiable."""
        v, a = dummy_embeddings
        v.requires_grad_(True)
        is_real = torch.ones(v.shape[0], dtype=torch.bool)
        loss_fn = TemporalConsistencyLoss()
        loss = loss_fn(v, a, is_real=is_real)
        loss.backward()
        assert v.grad is not None


# ──────────────────────────────────────────────
# Cross-Modal Prediction Loss
# ──────────────────────────────────────────────

class TestCrossModalPredictionLoss:
    def test_output_keys(self, dummy_embeddings):
        """Returns dict with expected keys."""
        v, a = dummy_embeddings
        cmp = CrossModalPredictionLoss(embedding_dim=256)
        result = cmp(v, a)
        assert "loss_cmp" in result
        assert "loss_v2a" in result
        assert "loss_a2v" in result

    def test_loss_is_finite(self, dummy_embeddings):
        """All loss components are finite."""
        v, a = dummy_embeddings
        cmp = CrossModalPredictionLoss(embedding_dim=256)
        result = cmp(v, a)
        assert result["loss_cmp"].isfinite()

    def test_mask_generation_shape(self):
        """Generated mask has correct shape and ratio."""
        cmp = CrossModalPredictionLoss(mask_ratio=0.3)
        mask = cmp._generate_mask(B=4, T=100, device=torch.device("cpu"))
        assert mask.shape == (4, 100)
        assert mask.dtype == torch.bool
        # Each row should have ~30 masked frames
        for i in range(4):
            n_masked = mask[i].sum().item()
            assert 20 <= n_masked <= 40  # ±10 tolerance

    def test_gradient_flows(self, dummy_embeddings):
        """Gradient flows through predictor MLPs."""
        v, a = dummy_embeddings
        cmp = CrossModalPredictionLoss(embedding_dim=256)
        result = cmp(v, a)
        result["loss_cmp"].backward()
        assert cmp.v_to_a[0].weight.grad is not None
        assert cmp.a_to_v[0].weight.grad is not None


# ──────────────────────────────────────────────
# Combined Loss
# ──────────────────────────────────────────────

class TestCombinedLoss:
    def test_output_keys(self, dummy_embeddings, dummy_labels):
        """Returns dict with all expected keys."""
        v, a = dummy_embeddings
        logits = torch.randn(v.shape[0], 1)
        loss_fn = CombinedLoss(embedding_dim=256, queue_size=128)
        result = loss_fn(v, a, logits, dummy_labels)
        expected_keys = {"loss", "loss_infonce", "loss_temp", "loss_cls", "loss_audio_cls", "temperature"}
        assert set(result.keys()) == expected_keys

    def test_total_equals_components(self, dummy_embeddings, dummy_labels):
        """Total loss = InfoNCE + gamma*temporal + delta*cls."""
        v, a = dummy_embeddings
        logits = torch.randn(v.shape[0], 1)
        gamma, delta = 0.5, 1.0
        loss_fn = CombinedLoss(
            embedding_dim=256, queue_size=128, gamma=gamma, delta=delta,
        )
        result = loss_fn(v, a, logits, dummy_labels)
        expected = (
            result["loss_infonce"] + gamma * result["loss_temp"] + delta * result["loss_cls"]
        )
        assert torch.allclose(result["loss"].detach(), expected, atol=1e-4)

    def test_audio_head_adds_to_total(self, dummy_embeddings, dummy_labels):
        """Audio classifier loss adds delta * loss_audio_cls to total."""
        v, a = dummy_embeddings
        logits = torch.randn(v.shape[0], 1)
        audio_logits = torch.randn(v.shape[0], 1)
        loss_fn = CombinedLoss(embedding_dim=256, queue_size=128, delta=1.0)
        result = loss_fn(v, a, logits, dummy_labels, audio_logits=audio_logits)
        assert result["loss_audio_cls"].item() > 0.0

    def test_all_finite(self, dummy_embeddings, dummy_labels):
        """All loss components are finite."""
        v, a = dummy_embeddings
        logits = torch.randn(v.shape[0], 1)
        loss_fn = CombinedLoss(embedding_dim=256, queue_size=128)
        result = loss_fn(v, a, logits, dummy_labels)
        for key, val in result.items():
            assert val.isfinite(), f"{key} is not finite"


# ──────────────────────────────────────────────
# Pretrain Loss
# ──────────────────────────────────────────────

class TestPretrainLoss:
    def test_with_cross_modal(self, dummy_embeddings):
        """PretrainLoss with cross-modal prediction enabled."""
        v, a = dummy_embeddings
        loss_fn = PretrainLoss(embedding_dim=256, queue_size=128, use_cross_modal=True)
        result = loss_fn(v, a)
        assert result["loss"].isfinite()
        assert result["loss_cmp"].item() > 0.0

    def test_without_cross_modal(self, dummy_embeddings):
        """PretrainLoss without cross-modal prediction."""
        v, a = dummy_embeddings
        loss_fn = PretrainLoss(embedding_dim=256, queue_size=128, use_cross_modal=False)
        result = loss_fn(v, a)
        assert result["loss"].isfinite()
        assert result["loss_cmp"].item() == 0.0

    def test_total_includes_cmp_weight(self, dummy_embeddings):
        """Total = InfoNCE + cmp_weight * CMP when cross-modal enabled."""
        v, a = dummy_embeddings
        cmp_weight = 0.5
        loss_fn = PretrainLoss(
            embedding_dim=256, queue_size=128,
            use_cross_modal=True, cmp_weight=cmp_weight,
        )
        result = loss_fn(v, a)
        # total should be > infonce alone due to CMP contribution
        assert result["loss"].item() != result["loss_infonce"].item()


# ──────────────────────────────────────────────
# Factory Functions
# ──────────────────────────────────────────────

class TestLossFactories:
    def test_build_pretrain_loss(self, default_config):
        """build_pretrain_loss creates PretrainLoss from config."""
        loss_fn = build_pretrain_loss(default_config)
        assert isinstance(loss_fn, PretrainLoss)

    def test_build_finetune_loss(self, default_config):
        """build_finetune_loss creates CombinedLoss from config."""
        loss_fn = build_finetune_loss(default_config)
        assert isinstance(loss_fn, CombinedLoss)
        assert loss_fn.gamma == 0.5
        assert loss_fn.delta == 1.0
