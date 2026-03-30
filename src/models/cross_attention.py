import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossAttentionModule(nn.Module):
    """Bidirectional cross-modal attention between visual and audio embeddings.

    Visual attends to audio (Q=v, K=a, V=a) and audio attends to visual
    (Q=a, K=v, V=v). Both use residual connections and layer normalization.

    Args:
        embed_dim: Embedding dimension (must match encoder output, default: 256).
        num_heads: Number of attention heads (default: 2).
        dropout: Dropout on attention weights (default: 0.1).
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.v_to_a_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.a_to_v_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_a = nn.LayerNorm(embed_dim)

    def forward(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional cross-attention.

        Args:
            v_embeds: (B, T, D) visual embeddings
            a_embeds: (B, T, D) audio embeddings
            key_padding_mask: (B, T) True = padding position to ignore

        Returns:
            v_attended: (B, T, D) visual embeddings attended to audio
            a_attended: (B, T, D) audio embeddings attended to visual
        """
        v_attn_out, _ = self.v_to_a_attn(
            query=v_embeds, key=a_embeds, value=a_embeds,
            key_padding_mask=key_padding_mask,
        )
        v_attended = self.norm_v(v_embeds + v_attn_out)

        a_attn_out, _ = self.a_to_v_attn(
            query=a_embeds, key=v_embeds, value=v_embeds,
            key_padding_mask=key_padding_mask,
        )
        a_attended = self.norm_a(a_embeds + a_attn_out)

        return v_attended, a_attended


class EmbedClassifier(nn.Module):
    """Classifier on cross-attended AV embeddings with temporal pooling.

    Optionally fuses DCT frequency-domain features for face-swap detection.

    Args:
        embed_dim: Per-modality embedding dimension (default: 256).
        hidden_dim: MLP hidden dimension (default: 256).
        dropout: MLP dropout (default: 0.3).
        dct_dim: DCT feature dimension per frame (0 = no DCT, default: 0).
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.3, dct_dim: int = 0):
        super().__init__()
        self.dct_dim = dct_dim
        # Input: concat of [v_attended; a_attended; dct_features] mean+max pooled
        # = (2 * embed_dim + dct_dim) * 2 (mean + max)
        pool_dim = (embed_dim * 2 + dct_dim) * 2
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        v_attended: torch.Tensor,
        a_attended: torch.Tensor,
        lengths: torch.Tensor = None,
        dct_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Classify cross-attended embeddings.

        Args:
            v_attended: (B, T, D) cross-attended visual embeddings
            a_attended: (B, T, D) cross-attended audio embeddings
            lengths: (B,) valid sequence lengths for masked pooling
            dct_features: (B, T, dct_dim) DCT features (optional)

        Returns:
            (B, 1) classification logits (pre-sigmoid)
        """
        parts = [v_attended, a_attended]
        if dct_features is not None and self.dct_dim > 0:
            parts.append(dct_features)
        combined = torch.cat(parts, dim=-1)

        if lengths is not None:
            mask = torch.arange(combined.shape[1], device=combined.device).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1)

            combined_masked = combined * mask
            mean_pool = combined_masked.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)

            combined_for_max = combined.masked_fill(~mask, float("-inf"))
            max_pool, _ = combined_for_max.max(dim=1)
        else:
            mean_pool = combined.mean(dim=1)
            max_pool, _ = combined.max(dim=1)

        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.classifier(pooled)


def build_cross_attention(config: dict) -> tuple[CrossAttentionModule, EmbedClassifier]:
    """Build cross-attention module and embed classifier from config."""
    ca_cfg = config["model"].get("cross_attention", {})
    embed_dim = config["model"]["visual_encoder"]["embedding_dim"]
    dropout = config["model"]["classifier"].get("dropout", 0.3)

    cross_attn = CrossAttentionModule(
        embed_dim=embed_dim,
        num_heads=ca_cfg.get("num_heads", 2),
        dropout=ca_cfg.get("dropout", 0.1),
    )
    dct_cfg = config["model"].get("dct_extractor", {})
    dct_dim = dct_cfg.get("output_dim", 0) if dct_cfg.get("enabled", False) else 0

    embed_clf = EmbedClassifier(
        embed_dim=embed_dim,
        hidden_dim=ca_cfg.get("embed_classifier_hidden", 256),
        dropout=dropout,
        dct_dim=dct_dim,
    )

    total_params = sum(p.numel() for p in cross_attn.parameters()) + sum(p.numel() for p in embed_clf.parameters())
    logger.info(f"CrossAttention + EmbedClassifier: {total_params:,} parameters")

    return cross_attn, embed_clf
