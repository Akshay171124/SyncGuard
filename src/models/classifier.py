import logging

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class BiLSTMClassifier(nn.Module):
    """Bi-directional LSTM classifier over sync-score sequences.

    Takes frame-level sync-scores s(t) = cos(v_t, a_t) and optional EAR
    (Eye Aspect Ratio) features, and classifies the full clip as real or fake.

    Architecture:
        (B, T, input_size) → Bi-LSTM → mean+max pool → MLP → sigmoid → (B, 1)
        input_size = 1 (sync-scores only) or 2 (sync-scores + EAR)

    Args:
        hidden_size: LSTM hidden dimension per direction (default: 128).
        num_layers: Number of stacked LSTM layers (default: 2).
        dropout: Dropout between LSTM layers (default: 0.3).
        use_ear: Whether to accept EAR features as additional input (default: False).
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_ear: bool = False,
    ):
        super().__init__()
        self.use_ear = use_ear
        input_size = 2 if use_ear else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        # Concat mean-pool + max-pool → 2 * (2 * hidden_size) = 512
        pool_dim = 2 * hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        sync_scores: torch.Tensor,
        lengths: torch.Tensor = None,
        ear_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Classify sync-score sequence as real/fake.

        Args:
            sync_scores: (B, T) frame-level cosine similarities
            lengths: (B,) actual sequence lengths before padding (optional).
            ear_features: (B, T) per-frame EAR values (optional, used if use_ear=True).

        Returns:
            (B, 1) real/fake logits (pre-sigmoid for BCEWithLogitsLoss)
        """
        # LSTM expects (B, T, input_size)
        x = sync_scores.unsqueeze(-1)  # (B, T, 1)

        if self.use_ear and ear_features is not None:
            ear = ear_features.unsqueeze(-1)  # (B, T, 1)
            x = torch.cat([x, ear], dim=-1)  # (B, T, 2)

        # Pack padded sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        hidden, _ = self.lstm(x)  # (B, T, 2*hidden_size) or PackedSequence

        if lengths is not None:
            hidden, _ = nn.utils.rnn.pad_packed_sequence(
                hidden, batch_first=True
            )  # (B, T, 2*hidden_size)

        # Masked pooling
        if lengths is not None:
            mask = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)  # (B, T)
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            # Mean pool (ignoring padding)
            hidden_masked = hidden * mask
            mean_pool = hidden_masked.sum(dim=1) / lengths.unsqueeze(1).float()
            # Max pool (set padding to -inf)
            hidden_for_max = hidden.masked_fill(~mask, float("-inf"))
            max_pool, _ = hidden_for_max.max(dim=1)
        else:
            mean_pool = hidden.mean(dim=1)  # (B, 2*hidden_size)
            max_pool, _ = hidden.max(dim=1)  # (B, 2*hidden_size)

        # Concatenate mean + max pool
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 4*hidden_size)

        logits = self.classifier(pooled)  # (B, 1)
        return logits


class CNN1DClassifier(nn.Module):
    """1D-CNN classifier for ablation comparison.

    Captures local temporal patterns in sync-score sequences.

    Args:
        hidden_size: Channel dimension for conv layers (default: 128).
        dropout: Dropout rate (default: 0.3).
    """

    def __init__(self, hidden_size: int = 128, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        sync_scores: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Classify sync-score sequence.

        Args:
            sync_scores: (B, T) frame-level cosine similarities
            lengths: (B,) unused for CNN, accepted for interface compatibility

        Returns:
            (B, 1) real/fake logits
        """
        x = sync_scores.unsqueeze(1)  # (B, 1, T)
        x = self.features(x)  # (B, hidden*2, 1)
        x = x.squeeze(-1)  # (B, hidden*2)
        return self.classifier(x)


class StatisticalClassifier(nn.Module):
    """Statistical baseline classifier for ablation comparison.

    Computes hand-crafted statistics (mean, std, skewness, kurtosis, min, max)
    from sync-score sequences and feeds them to an MLP.

    Args:
        dropout: Dropout rate (default: 0.3).
    """

    def __init__(self, dropout: float = 0.3, **kwargs):
        super().__init__()
        # 6 statistical features
        self.classifier = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        sync_scores: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Classify based on sync-score statistics.

        Args:
            sync_scores: (B, T) frame-level cosine similarities
            lengths: (B,) unused, accepted for interface compatibility

        Returns:
            (B, 1) real/fake logits
        """
        mean = sync_scores.mean(dim=1, keepdim=True)
        std = sync_scores.std(dim=1, keepdim=True)
        min_val, _ = sync_scores.min(dim=1, keepdim=True)
        max_val, _ = sync_scores.max(dim=1, keepdim=True)

        # Skewness and kurtosis
        centered = sync_scores - mean
        std_safe = std.clamp(min=1e-8)
        normalized = centered / std_safe
        skewness = normalized.pow(3).mean(dim=1, keepdim=True)
        kurtosis = normalized.pow(4).mean(dim=1, keepdim=True) - 3.0

        features = torch.cat([mean, std, min_val, max_val, skewness, kurtosis], dim=-1)
        return self.classifier(features)


class AudioClassifier(nn.Module):
    """Audio-only classification head for detecting fake audio (RV-FA).

    Operates on audio embeddings directly, independent of visual input.
    Catches audio-only manipulations that sync-score-based classifiers miss.

    Architecture:
        (B, T, D) audio embeddings → mean+max pool → MLP → (B, 1)

    Args:
        embedding_dim: Audio embedding dimension (default: 256).
        dropout: Dropout rate (default: 0.3).
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        pool_dim = embedding_dim * 2  # mean + max pool
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        a_embeds: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Classify audio embeddings as real/fake.

        Args:
            a_embeds: (B, T, D) audio embeddings.
            lengths: (B,) actual sequence lengths (optional).

        Returns:
            (B, 1) real/fake logits.
        """
        if lengths is not None:
            T = a_embeds.shape[1]
            mask = torch.arange(T, device=a_embeds.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            a_masked = a_embeds * mask
            mean_pool = a_masked.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
            a_for_max = a_embeds.masked_fill(~mask, float("-inf"))
            max_pool, _ = a_for_max.max(dim=1)
        else:
            mean_pool = a_embeds.mean(dim=1)
            max_pool, _ = a_embeds.max(dim=1)

        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*D)
        return self.classifier(pooled)


def build_classifier(config: dict) -> nn.Module:
    """Factory function to build classifier from config.

    Args:
        config: Full config dict (reads model.classifier section).

    Returns:
        Classifier module.
    """
    cls_cfg = config["model"]["classifier"]
    name = cls_cfg["name"]
    hidden_size = cls_cfg.get("hidden_size", 128)
    num_layers = cls_cfg.get("num_layers", 2)
    dropout = cls_cfg.get("dropout", 0.3)

    use_ear = cls_cfg.get("use_ear", False)

    if name == "bilstm":
        return BiLSTMClassifier(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_ear=use_ear,
        )
    elif name == "cnn1d":
        return CNN1DClassifier(hidden_size=hidden_size, dropout=dropout)
    elif name == "statistical":
        return StatisticalClassifier(dropout=dropout)
    else:
        raise ValueError(f"Unknown classifier: {name}")


if __name__ == "__main__":
    B, T = 4, 100
    sync_scores = torch.randn(B, T) * 0.3 + 0.5  # Simulate sync-scores around 0.5
    lengths = torch.tensor([100, 80, 60, 50])

    for name, Cls, kwargs in [
        ("bilstm", BiLSTMClassifier, {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}),
        ("cnn1d", CNN1DClassifier, {"hidden_size": 128, "dropout": 0.3}),
        ("statistical", StatisticalClassifier, {"dropout": 0.3}),
    ]:
        cls = Cls(**kwargs)
        # Test without lengths
        logits = cls(sync_scores)
        assert logits.shape == (B, 1), f"{name}: expected {(B, 1)}, got {logits.shape}"

        # Test with lengths (masked pooling)
        logits_masked = cls(sync_scores, lengths=lengths)
        assert logits_masked.shape == (B, 1), f"{name} masked: shape mismatch"

        # Verify gradients
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in cls.parameters())
        assert has_grad, f"{name}: no gradients"

        print(f"  {name}: input ({B}, {T}) → output {logits.shape} ✓")

    print("All classifier tests passed.")
