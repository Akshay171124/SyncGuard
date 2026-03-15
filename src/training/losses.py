import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MoCoQueue:
    """MoCo-style FIFO memory bank for contrastive learning negatives.

    Maintains a fixed-size queue of past audio embeddings (detached, no grad).
    Updated after each batch via enqueue_dequeue.

    Args:
        dim: Embedding dimension (default: 256).
        size: Queue capacity (default: 4096).
    """

    def __init__(self, dim: int = 256, size: int = 4096):
        self.size = size
        self.dim = dim
        self.queue = torch.randn(size, dim)
        self.queue = F.normalize(self.queue, dim=-1)
        self.ptr = 0
        self.full = False

    def to(self, device: torch.device) -> "MoCoQueue":
        """Move queue to device."""
        self.queue = self.queue.to(device)
        return self

    @torch.no_grad()
    def enqueue_dequeue(self, embeddings: torch.Tensor):
        """Add new embeddings to the queue (FIFO).

        Args:
            embeddings: (N, dim) L2-normalized embeddings to enqueue.
        """
        embeddings = embeddings.detach()
        batch_size = embeddings.shape[0]

        if batch_size >= self.size:
            # If batch is larger than queue, just take the last `size` entries
            self.queue = embeddings[-self.size:].clone()
            self.ptr = 0
            self.full = True
            return

        end = self.ptr + batch_size
        if end <= self.size:
            self.queue[self.ptr:end] = embeddings
        else:
            # Wrap around
            overflow = end - self.size
            self.queue[self.ptr:] = embeddings[:batch_size - overflow]
            self.queue[:overflow] = embeddings[batch_size - overflow:]
            self.full = True

        self.ptr = end % self.size
        if end >= self.size:
            self.full = True

    def get_negatives(self) -> torch.Tensor:
        """Return current queue contents as negative samples.

        Returns:
            (K, dim) tensor where K = size if full, else ptr
        """
        if self.full:
            return self.queue.clone()
        return self.queue[:self.ptr].clone()


class InfoNCELoss(nn.Module):
    """Frame-level InfoNCE loss with MoCo memory bank.

    For each visual frame embedding v_t, computes contrastive loss against:
    - Positive: the matching audio embedding a_t
    - Negatives: all audio embeddings in the MoCo queue

    L = -(1/T) * sum_t log[ exp(cos(v_t, a_t)/τ) / (exp(cos(v_t, a_t)/τ) + sum_neg) ]

    Args:
        embedding_dim: Embedding dimension (default: 256).
        queue_size: MoCo queue size (default: 4096).
        temperature: Initial temperature value (default: 0.07).
        temperature_range: Min/max clamp for learnable temperature.
        learnable_temperature: Whether τ is learnable (default: True).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        queue_size: int = 4096,
        temperature: float = 0.07,
        temperature_range: tuple[float, float] = (0.01, 0.5),
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.queue = MoCoQueue(dim=embedding_dim, size=queue_size)
        self.temp_range = temperature_range

        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(temperature).log()
            )

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature, clamped to range."""
        return self.log_temperature.exp().clamp(*self.temp_range)

    def forward(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute frame-level InfoNCE loss.

        Args:
            v_embeds: (B, T, D) L2-normalized visual embeddings
            a_embeds: (B, T, D) L2-normalized audio embeddings
            mask: (B, T) optional mask (True = valid frame, False = padding)

        Returns:
            Scalar InfoNCE loss
        """
        B, T, D = v_embeds.shape
        tau = self.temperature

        # Flatten to (B*T, D) for processing
        v_flat = v_embeds.reshape(-1, D)  # (N, D) where N = B*T
        a_flat = a_embeds.reshape(-1, D)  # (N, D)

        # Positive similarities: cos(v_t, a_t) for matched pairs
        pos_sim = (v_flat * a_flat).sum(dim=-1) / tau  # (N,)

        # Negative similarities from MoCo queue
        negatives = self.queue.get_negatives()  # (K, D)
        if negatives.shape[0] > 0:
            neg_sim = torch.mm(v_flat, negatives.T) / tau  # (N, K)
            # Logits: positive in first column, negatives after
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (N, 1+K)
        else:
            # No negatives yet (first batch) — use in-batch negatives
            all_sim = torch.mm(v_flat, a_flat.T) / tau  # (N, N)
            logits = all_sim

        # Cross-entropy with target = 0 (positive is first column)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction="none")  # (N,)

        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1).float()  # (N,)
            loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        else:
            loss = loss.mean()

        # Update queue with current batch audio embeddings
        self.queue.enqueue_dequeue(a_flat)

        return loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss on rate-of-change of embeddings.

    Penalizes divergence between first derivatives of visual and audio embeddings.
    Applied ONLY to real clips.

    L_temp = sum_t ||(v_{t+1} - v_t) - (a_{t+1} - a_t)||²

    See RESEARCH.md Section 3.3 for design rationale.
    """

    def forward(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
        is_real: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute temporal consistency loss (real clips only).

        Args:
            v_embeds: (B, T, D) visual embeddings
            a_embeds: (B, T, D) audio embeddings
            is_real: (B,) boolean mask — True for real clips
            mask: (B, T) optional frame-level mask

        Returns:
            Scalar temporal consistency loss
        """
        if not is_real.any():
            return torch.tensor(0.0, device=v_embeds.device, requires_grad=True)

        # First derivatives
        v_diff = v_embeds[:, 1:] - v_embeds[:, :-1]  # (B, T-1, D)
        a_diff = a_embeds[:, 1:] - a_embeds[:, :-1]  # (B, T-1, D)

        # L2 distance between derivatives
        diff = (v_diff - a_diff).pow(2).sum(dim=-1)  # (B, T-1)

        # Mask: only real clips
        real_mask = is_real.float().unsqueeze(1)  # (B, 1)

        if mask is not None:
            # Adjust mask for derivatives (T-1 length)
            frame_mask = mask[:, 1:].float()  # (B, T-1)
            combined_mask = real_mask * frame_mask  # (B, T-1)
        else:
            combined_mask = real_mask.expand_as(diff)

        # Masked mean
        loss = (diff * combined_mask).sum() / combined_mask.sum().clamp(min=1)
        return loss


class CombinedLoss(nn.Module):
    """Combined loss for SyncGuard training.

    L_total = L_InfoNCE + γ * L_temp + δ * L_cls

    Args:
        embedding_dim: Embedding dimension (default: 256).
        queue_size: MoCo queue size (default: 4096).
        temperature: Initial temperature (default: 0.07).
        temperature_range: Clamp range for learnable temperature.
        gamma: Weight for temporal consistency loss (default: 0.5).
        delta: Weight for classification loss (default: 1.0).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        queue_size: int = 4096,
        temperature: float = 0.07,
        temperature_range: tuple[float, float] = (0.01, 0.5),
        gamma: float = 0.5,
        delta: float = 1.0,
    ):
        super().__init__()
        self.infonce = InfoNCELoss(
            embedding_dim=embedding_dim,
            queue_size=queue_size,
            temperature=temperature,
            temperature_range=temperature_range,
        )
        self.temporal = TemporalConsistencyLoss()
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.delta = delta

    @property
    def temperature(self) -> torch.Tensor:
        return self.infonce.temperature

    def forward(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            v_embeds: (B, T, D) visual embeddings
            a_embeds: (B, T, D) audio embeddings
            logits: (B, 1) classification logits (pre-sigmoid)
            labels: (B,) binary labels (0 = real, 1 = fake)
            mask: (B, T) optional frame-level mask

        Returns:
            Dict with 'loss' (total), 'loss_infonce', 'loss_temp', 'loss_cls', 'temperature'
        """
        is_real = (labels == 0)  # (B,)

        loss_infonce = self.infonce(v_embeds, a_embeds, mask=mask)
        loss_temp = self.temporal(v_embeds, a_embeds, is_real=is_real, mask=mask)
        loss_cls = self.cls_criterion(logits.squeeze(-1), labels.float())

        loss_total = loss_infonce + self.gamma * loss_temp + self.delta * loss_cls

        return {
            "loss": loss_total,
            "loss_infonce": loss_infonce.detach(),
            "loss_temp": loss_temp.detach(),
            "loss_cls": loss_cls.detach(),
            "temperature": self.temperature.detach(),
        }


class PretrainLoss(nn.Module):
    """Loss for Phase 1 contrastive pretraining (InfoNCE only, no classification).

    Args:
        embedding_dim: Embedding dimension (default: 256).
        queue_size: MoCo queue size (default: 4096).
        temperature: Initial temperature (default: 0.07).
        temperature_range: Clamp range for learnable temperature.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        queue_size: int = 4096,
        temperature: float = 0.07,
        temperature_range: tuple[float, float] = (0.01, 0.5),
    ):
        super().__init__()
        self.infonce = InfoNCELoss(
            embedding_dim=embedding_dim,
            queue_size=queue_size,
            temperature=temperature,
            temperature_range=temperature_range,
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.infonce.temperature

    def forward(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Compute pretraining loss.

        Args:
            v_embeds: (B, T, D) visual embeddings
            a_embeds: (B, T, D) audio embeddings
            mask: (B, T) optional frame-level mask

        Returns:
            Dict with 'loss', 'loss_infonce', 'temperature'
        """
        loss_infonce = self.infonce(v_embeds, a_embeds, mask=mask)
        return {
            "loss": loss_infonce,
            "loss_infonce": loss_infonce.detach(),
            "temperature": self.temperature.detach(),
        }


def build_pretrain_loss(config: dict) -> PretrainLoss:
    """Build loss for Phase 1 pretraining from config."""
    pt_cfg = config["training"]["pretrain"]
    emb_dim = config["model"]["visual_encoder"]["embedding_dim"]
    return PretrainLoss(
        embedding_dim=emb_dim,
        queue_size=pt_cfg.get("moco_queue_size", 4096),
        temperature=pt_cfg.get("temperature", 0.07),
        temperature_range=tuple(pt_cfg.get("temperature_range", [0.01, 0.5])),
    )


def build_finetune_loss(config: dict) -> CombinedLoss:
    """Build loss for Phase 2 fine-tuning from config."""
    pt_cfg = config["training"]["pretrain"]
    ft_cfg = config["training"]["finetune"]
    emb_dim = config["model"]["visual_encoder"]["embedding_dim"]
    return CombinedLoss(
        embedding_dim=emb_dim,
        queue_size=pt_cfg.get("moco_queue_size", 4096),
        temperature=pt_cfg.get("temperature", 0.07),
        temperature_range=tuple(pt_cfg.get("temperature_range", [0.01, 0.5])),
        gamma=ft_cfg.get("gamma", 0.5),
        delta=ft_cfg.get("delta", 1.0),
    )


if __name__ == "__main__":
    print("Testing loss functions...")

    B, T, D = 4, 50, 256

    v_embeds = F.normalize(torch.randn(B, T, D), dim=-1)
    a_embeds = F.normalize(torch.randn(B, T, D), dim=-1)
    logits = torch.randn(B, 1)
    labels = torch.tensor([0, 1, 0, 1])  # 2 real, 2 fake
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[3, 40:] = False  # Simulate padding

    # Test InfoNCE
    infonce = InfoNCELoss(embedding_dim=D, queue_size=128, temperature=0.07)
    loss_nce = infonce(v_embeds, a_embeds, mask=mask)
    assert loss_nce.isfinite(), "InfoNCE loss is not finite"
    assert loss_nce.requires_grad, "InfoNCE loss has no grad"
    print(f"  InfoNCE: {loss_nce.item():.4f} ✓")

    # Verify queue updated
    neg = infonce.queue.get_negatives()
    expected_queue = min(B * T, 128)  # Queue capped at queue_size
    assert neg.shape[0] == expected_queue, f"Queue should have {expected_queue} entries, got {neg.shape[0]}"
    print(f"  MoCo queue: {neg.shape[0]} entries (capped at queue_size=128) ✓")

    # Run a second forward to test with filled queue
    v2 = F.normalize(torch.randn(B, T, D), dim=-1)
    a2 = F.normalize(torch.randn(B, T, D), dim=-1)
    loss_nce2 = infonce(v2, a2)
    assert loss_nce2.isfinite(), "InfoNCE loss (2nd forward) not finite"
    print(f"  InfoNCE (with queue): {loss_nce2.item():.4f} ✓")

    # Test temperature clamp
    tau = infonce.temperature
    assert 0.01 <= tau.item() <= 0.5, f"Temperature {tau.item()} out of range"
    print(f"  Temperature: {tau.item():.4f} (clamped to [0.01, 0.5]) ✓")

    # Test TemporalConsistencyLoss
    temp_loss_fn = TemporalConsistencyLoss()
    is_real = torch.tensor([True, False, True, False])
    loss_temp = temp_loss_fn(v_embeds, a_embeds, is_real=is_real, mask=mask)
    assert loss_temp.isfinite(), "Temporal loss is not finite"
    assert loss_temp.item() > 0, "Temporal loss should be > 0 for random data"
    print(f"  Temporal consistency: {loss_temp.item():.6f} ✓")

    # Temporal loss with all-fake batch should be 0
    all_fake = torch.tensor([False, False, False, False])
    loss_temp_fake = temp_loss_fn(v_embeds, a_embeds, is_real=all_fake)
    assert loss_temp_fake.item() == 0.0, "Temporal loss should be 0 for all-fake batch"
    print("  Temporal (all-fake = 0.0) ✓")

    # Test CombinedLoss
    combined = CombinedLoss(
        embedding_dim=D, queue_size=128,
        gamma=0.5, delta=1.0,
    )
    result = combined(v_embeds, a_embeds, logits, labels, mask=mask)
    assert result["loss"].isfinite(), "Combined loss not finite"
    assert result["loss"].requires_grad, "Combined loss no grad"
    expected_total = (
        result["loss_infonce"] + 0.5 * result["loss_temp"] + 1.0 * result["loss_cls"]
    )
    assert torch.allclose(result["loss"].detach(), expected_total, atol=1e-5), "Total != sum of components"
    print(f"  Combined: total={result['loss'].item():.4f} "
          f"(nce={result['loss_infonce'].item():.4f} + "
          f"0.5*temp={0.5*result['loss_temp'].item():.4f} + "
          f"cls={result['loss_cls'].item():.4f}) ✓")

    # Test PretrainLoss
    pt_loss = PretrainLoss(embedding_dim=D, queue_size=128)
    pt_result = pt_loss(v_embeds, a_embeds)
    assert pt_result["loss"].isfinite(), "Pretrain loss not finite"
    print(f"  PretrainLoss: {pt_result['loss'].item():.4f} ✓")

    # Verify gradient flow: temperature should have grad
    result["loss"].backward()
    assert combined.infonce.log_temperature.grad is not None, "Temperature has no gradient"
    print("  Temperature gradient ✓")

    # Verify no grad through queue negatives
    for p in [infonce.queue.queue]:
        assert not p.requires_grad, "Queue should not require grad"
    print("  Queue no-grad ✓")

    print("All loss function tests passed.")
