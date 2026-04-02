# Cross-Attention Embedding Bypass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel cross-attention classification path that operates on full AV embeddings, bypassing the sync-score bottleneck, to improve DFDC generalization from 0.46 to 0.60-0.75 AUC.

**Architecture:** Bidirectional cross-attention (V→A, A→V) with residual connections and layer norm, followed by masked temporal pooling and MLP classifier. Runs parallel to existing sync-score Bi-LSTM path. Outputs fused via learnable weight.

**Tech Stack:** PyTorch (nn.MultiheadAttention), existing SyncGuard infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/cross_attention.py` | **Create** | CrossAttentionModule + EmbedClassifier classes |
| `src/models/syncguard.py` | **Modify** | Add cross-attention path + fusion to SyncGuard + SyncGuardOutput |
| `configs/finetune_frozen.yaml` | **Modify** | Add cross_attention config section |
| `scripts/train_cross_attention.py` | **Create** | Stage 1 + Stage 2 training CLI |
| `scripts/slurm_train_cross_attention.sh` | **Create** | SLURM job for HPC |

---

### Task 1: Create CrossAttentionModule and EmbedClassifier

**Files:**
- Create: `src/models/cross_attention.py`

- [ ] **Step 1: Create the cross-attention module**

```python
# src/models/cross_attention.py
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
        # Visual attends to audio: Q=v, K=a, V=a
        v_attn_out, _ = self.v_to_a_attn(
            query=v_embeds, key=a_embeds, value=a_embeds,
            key_padding_mask=key_padding_mask,
        )
        v_attended = self.norm_v(v_embeds + v_attn_out)  # Residual + LayerNorm

        # Audio attends to visual: Q=a, K=v, V=v
        a_attn_out, _ = self.a_to_v_attn(
            query=a_embeds, key=v_embeds, value=v_embeds,
            key_padding_mask=key_padding_mask,
        )
        a_attended = self.norm_a(a_embeds + a_attn_out)  # Residual + LayerNorm

        return v_attended, a_attended


class EmbedClassifier(nn.Module):
    """Classifier on cross-attended AV embeddings with temporal pooling.

    Takes bidirectional cross-attention output, applies masked mean+max pooling,
    then classifies via MLP.

    Args:
        embed_dim: Per-modality embedding dimension (default: 256).
        hidden_dim: MLP hidden dimension (default: 256).
        dropout: MLP dropout (default: 0.3).
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # Input: concat of [v_attended; a_attended] mean+max pooled
        # = 2 * embed_dim * 2 (mean + max) = 1024 for embed_dim=256
        pool_dim = embed_dim * 2 * 2
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
    ) -> torch.Tensor:
        """Classify cross-attended embeddings.

        Args:
            v_attended: (B, T, D) cross-attended visual embeddings
            a_attended: (B, T, D) cross-attended audio embeddings
            lengths: (B,) valid sequence lengths for masked pooling

        Returns:
            (B, 1) classification logits (pre-sigmoid)
        """
        # Concatenate both attended streams: (B, T, 2*D)
        combined = torch.cat([v_attended, a_attended], dim=-1)

        # Masked temporal pooling
        if lengths is not None:
            mask = torch.arange(combined.shape[1], device=combined.device).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)  # (B, T)
            mask = mask.unsqueeze(-1)  # (B, T, 1)

            # Mean pool (ignoring padding)
            combined_masked = combined * mask
            mean_pool = combined_masked.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)

            # Max pool (set padding to -inf)
            combined_for_max = combined.masked_fill(~mask, float("-inf"))
            max_pool, _ = combined_for_max.max(dim=1)
        else:
            mean_pool = combined.mean(dim=1)
            max_pool, _ = combined.max(dim=1)

        # Concat mean + max: (B, 4*D) = (B, 1024)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)

        return self.classifier(pooled)  # (B, 1)


def build_cross_attention(config: dict) -> tuple[CrossAttentionModule, EmbedClassifier]:
    """Build cross-attention module and embed classifier from config.

    Args:
        config: Full config dict with model.cross_attention section.

    Returns:
        (CrossAttentionModule, EmbedClassifier) tuple.
    """
    ca_cfg = config["model"].get("cross_attention", {})
    embed_dim = config["model"]["visual_encoder"]["embedding_dim"]
    dropout = config["model"]["classifier"].get("dropout", 0.3)

    cross_attn = CrossAttentionModule(
        embed_dim=embed_dim,
        num_heads=ca_cfg.get("num_heads", 2),
        dropout=ca_cfg.get("dropout", 0.1),
    )
    embed_clf = EmbedClassifier(
        embed_dim=embed_dim,
        hidden_dim=ca_cfg.get("embed_classifier_hidden", 256),
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in cross_attn.parameters()) + sum(p.numel() for p in embed_clf.parameters())
    logger.info(f"CrossAttention + EmbedClassifier: {total_params:,} parameters")

    return cross_attn, embed_clf
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/models/cross_attention.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/models/cross_attention.py
git commit -m "Add CrossAttentionModule and EmbedClassifier for embedding bypass"
```

---

### Task 2: Add cross-attention path to SyncGuard model

**Files:**
- Modify: `src/models/syncguard.py`

- [ ] **Step 1: Add import and embed_logits to SyncGuardOutput**

In `src/models/syncguard.py`, add the import at the top (after existing imports):

```python
from src.models.cross_attention import build_cross_attention
```

Add `embed_logits` field to `SyncGuardOutput`:

```python
@dataclass
class SyncGuardOutput:
    """Container for SyncGuard model outputs."""
    logits: torch.Tensor
    sync_scores: torch.Tensor
    v_embeds: torch.Tensor
    a_embeds: torch.Tensor
    sync_logits: torch.Tensor = None
    audio_logits: torch.Tensor = None
    embed_logits: torch.Tensor = None  # Cross-attention path output
```

- [ ] **Step 2: Add cross-attention initialization to SyncGuard.__init__**

After the `audio_head` block (after line 69), add:

```python
        # Cross-attention embedding bypass (for face-swap detection)
        self.use_cross_attention = config["model"].get("cross_attention", {}).get("enabled", False)
        if self.use_cross_attention:
            self.cross_attn, self.embed_classifier = build_cross_attention(config)
            ca_cfg = config["model"].get("cross_attention", {})
            self.ca_fusion_weight = nn.Parameter(
                torch.tensor(float(ca_cfg.get("fusion_init", 0.0)))
            )
            logger.info("Cross-attention embedding bypass enabled")
```

- [ ] **Step 3: Add cross-attention forward path**

In `SyncGuard.forward()`, after the sync_logits computation and before the audio_head block, add:

```python
        # Cross-attention embedding bypass
        embed_logits = None
        if self.use_cross_attention:
            # Build padding mask for attention (True = padding)
            attn_padding_mask = None
            if lengths is not None:
                attn_padding_mask = torch.arange(T, device=v_embeds.device).unsqueeze(0) >= lengths.unsqueeze(1)

            v_attended, a_attended = self.cross_attn(
                v_embeds, a_embeds, key_padding_mask=attn_padding_mask,
            )
            embed_logits = self.embed_classifier(v_attended, a_attended, lengths=lengths)

            # Fuse sync path and embed path
            w = torch.sigmoid(self.ca_fusion_weight)
            logits = w * sync_logits + (1 - w) * embed_logits
```

Update the `else` branch and the audio_head fusion to account for cross-attention:

Replace the existing logits assignment block (lines 160-168) with:

```python
        # Determine base logits from sync path (+ optional audio head)
        if self.use_audio_head:
            audio_logits = self.audio_classifier(a_embeds, lengths=lengths)
            w_audio = torch.sigmoid(self.fusion_weight)
            logits = (1 - w_audio) * sync_logits + w_audio * audio_logits
        else:
            logits = sync_logits

        # Cross-attention embedding bypass (overrides logits with fused output)
        embed_logits = None
        if self.use_cross_attention:
            attn_padding_mask = None
            if lengths is not None:
                attn_padding_mask = torch.arange(T, device=v_embeds.device).unsqueeze(0) >= lengths.unsqueeze(1)

            v_attended, a_attended = self.cross_attn(
                v_embeds, a_embeds, key_padding_mask=attn_padding_mask,
            )
            embed_logits = self.embed_classifier(v_attended, a_attended, lengths=lengths)

            w_ca = torch.sigmoid(self.ca_fusion_weight)
            logits = w_ca * sync_logits + (1 - w_ca) * embed_logits
```

Update the return to include embed_logits:

```python
        return SyncGuardOutput(
            logits=logits,
            sync_scores=sync_scores,
            v_embeds=v_embeds,
            a_embeds=a_embeds,
            sync_logits=sync_logits,
            audio_logits=audio_logits,
            embed_logits=embed_logits,
        )
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/models/syncguard.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/models/syncguard.py
git commit -m "Add cross-attention bypass path to SyncGuard forward pass"
```

---

### Task 3: Add config section and run smoke test

**Files:**
- Modify: `configs/finetune_frozen.yaml` (on HPC)
- Test: CPU smoke test

- [ ] **Step 1: Add cross_attention config**

Add to `configs/finetune_frozen.yaml` under the `model:` section (after `audio_head: false`):

```yaml
  # Cross-attention embedding bypass for face-swap detection
  cross_attention:
    enabled: true
    num_heads: 2
    num_layers: 1
    dropout: 0.1
    embed_classifier_hidden: 256
    fusion_init: 0.0  # sigmoid(0) = 0.5
```

- [ ] **Step 2: Run CPU smoke test**

```bash
python3 -c "
import torch, yaml
torch.manual_seed(42)

with open('configs/finetune_frozen.yaml') as f:
    config = yaml.safe_load(f)

# Enable cross-attention
config.setdefault('model', {}).setdefault('cross_attention', {})['enabled'] = True

from src.models.syncguard import build_syncguard
model = build_syncguard(config)
model.train()

B, T_v = 2, 50
mc = torch.randn(B, T_v, 1, 96, 96)
wf = torch.randn(B, 16000 * 2)
lengths = torch.tensor([T_v, T_v - 5])

output = model(mc, wf, lengths=lengths)
print(f'logits: {output.logits.shape}, nan={output.logits.isnan().any()}')
print(f'sync_logits: {output.sync_logits.shape}')
print(f'embed_logits: {output.embed_logits.shape}')
print(f'fusion_weight: {torch.sigmoid(model.ca_fusion_weight).item():.3f}')

# Verify backward
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    output.embed_logits.squeeze(-1), torch.tensor([0.0, 1.0])
)
loss.backward()

ca_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.cross_attn.parameters())
print(f'Cross-attention has gradients: {ca_has_grad}')
print('SMOKE TEST PASSED')
"
```

Expected output:
```
logits: torch.Size([2, 1]), nan=False
sync_logits: torch.Size([2, 1])
embed_logits: torch.Size([2, 1])
fusion_weight: 0.500
Cross-attention has gradients: True
SMOKE TEST PASSED
```

- [ ] **Step 3: Commit**

```bash
git add configs/finetune_frozen.yaml
git commit -m "Add cross_attention config section to finetune_frozen.yaml"
```

---

### Task 4: Create training script for Stage 1 + Stage 2

**Files:**
- Create: `scripts/train_cross_attention.py`

- [ ] **Step 1: Write the training script**

```python
#!/usr/bin/env python3
"""Train cross-attention embedding bypass for SyncGuard.

Stage 1: Train cross-attention head only (encoders + Bi-LSTM frozen)
Stage 2: End-to-end fusion fine-tuning (encoders frozen, Bi-LSTM + CA trainable)

Usage:
    python scripts/train_cross_attention.py --config configs/finetune_frozen.yaml \
        --checkpoint outputs/checkpoints/finetune_best.pt \
        --stage 1

    python scripts/train_cross_attention.py --config configs/finetune_frozen.yaml \
        --checkpoint outputs/checkpoints/ca_stage1_best.pt \
        --stage 2
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import wandb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.syncguard import build_syncguard
from src.training.dataset import SyncGuardBatch, build_dataloaders
from src.utils.config import load_config, get_device

logger = logging.getLogger(__name__)


def compute_auc_roc(labels, scores):
    """Simple AUC-ROC computation."""
    if len(set(labels)) < 2:
        return 0.5
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tp, fp, auc, prev_fpr = 0, 0, 0.0, 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp / n_pos) * (fp / n_neg - prev_fpr)
            prev_fpr = fp / n_neg
    return auc


def freeze_for_stage(model, stage):
    """Freeze/unfreeze parameters based on training stage."""
    # Always freeze encoders
    for p in model.visual_encoder.parameters():
        p.requires_grad = False
    for p in model.audio_encoder.parameters():
        p.requires_grad = False

    if stage == 1:
        # Freeze sync path (Bi-LSTM), only train cross-attention + embed classifier
        for p in model.classifier.parameters():
            p.requires_grad = False
        if hasattr(model, 'ca_fusion_weight'):
            model.ca_fusion_weight.requires_grad = False
        for p in model.cross_attn.parameters():
            p.requires_grad = True
        for p in model.embed_classifier.parameters():
            p.requires_grad = True
    elif stage == 2:
        # Unfreeze Bi-LSTM + cross-attention + fusion weight
        for p in model.classifier.parameters():
            p.requires_grad = True
        if hasattr(model, 'ca_fusion_weight'):
            model.ca_fusion_weight.requires_grad = True
        for p in model.cross_attn.parameters():
            p.requires_grad = True
        for p in model.embed_classifier.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Stage {stage}: {trainable:,} trainable / {total:,} total parameters")


def validate(model, val_loader, device, stage):
    """Run validation, return metrics dict."""
    model.eval()
    all_labels, all_scores = [], []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in val_loader:
            batch: SyncGuardBatch
            mc = batch.mouth_crops.to(device)
            wf = batch.waveforms.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels.to(device)
            ear = batch.ear_features.to(device) if batch.ear_features is not None else None

            output = model(mc, wf, lengths=lengths, ear_features=ear)

            if stage == 1:
                logits = output.embed_logits.squeeze(-1)
            else:
                logits = output.logits.squeeze(-1)

            loss = criterion(logits, labels.float())
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())

    model.train()
    auc = compute_auc_roc(all_labels, all_scores)
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_auc": auc,
    }


def train(config, checkpoint_path, stage, resume_from=None):
    """Train cross-attention bypass."""
    device = get_device(config)
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Seeds
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Stage config
    if stage == 1:
        epochs, lr, patience = 20, 1e-4, 5
        warmup_epochs = 2
        prefix = "ca_stage1"
    else:
        epochs, lr, patience = 10, 5e-5, 3
        warmup_epochs = 1
        prefix = "ca_stage2"

    # Build model
    config["model"].setdefault("cross_attention", {})["enabled"] = True
    model = build_syncguard(config).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Cross-attention keys will be missing (new module) — that's expected
    ca_missing = [k for k in missing if "cross_attn" not in k and "embed_classifier" not in k and "ca_fusion" not in k]
    if ca_missing:
        logger.error(f"Non-CA keys missing: {ca_missing}")
    logger.info(f"Loaded checkpoint from {checkpoint_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    # Freeze for stage
    freeze_for_stage(model, stage)

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Scheduler
    dataloaders = build_dataloaders(config, phase="finetune")
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    cosine_steps = max(total_steps - warmup_steps, 1)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    criterion = nn.BCEWithLogitsLoss()

    # Wandb
    wandb.init(
        project="SyncGuard",
        name=f"cross-attention-stage{stage}",
        config={"stage": stage, "lr": lr, "epochs": epochs, "checkpoint": checkpoint_path},
        tags=["cross-attention", f"stage{stage}"],
    )

    logger.info(f"Starting cross-attention Stage {stage}: {epochs} epochs, lr={lr}, device={device}")

    best_val_auc = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        # Re-freeze encoders (model.train() may have changed modes)
        model.visual_encoder.eval()
        model.audio_encoder.eval()

        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        for batch in train_loader:
            batch: SyncGuardBatch
            mc = batch.mouth_crops.to(device)
            wf = batch.waveforms.to(device)
            lengths = batch.lengths.to(device)
            labels = batch.labels.to(device)
            ear = batch.ear_features.to(device) if batch.ear_features is not None else None

            output = model(mc, wf, lengths=lengths, ear_features=ear)

            if stage == 1:
                logits = output.embed_logits.squeeze(-1)
            else:
                logits = output.logits.squeeze(-1)

            loss = criterion(logits, labels.float())

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss at epoch {epoch}, batch {n_batches}. Skipping.")
                scheduler.step()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_metrics = validate(model, val_loader, device, stage)

        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": val_metrics["val_loss"],
            "val_auc": val_metrics["val_auc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(epoch_time, 1),
        }
        if hasattr(model, 'ca_fusion_weight'):
            epoch_log["fusion_weight"] = torch.sigmoid(model.ca_fusion_weight).item()

        history.append(epoch_log)

        logger.info(
            f"Epoch {epoch}/{epochs-1} ({epoch_time:.0f}s) | "
            f"loss={avg_loss:.4f} | val_auc={val_metrics['val_auc']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            + (f" | fusion_w={torch.sigmoid(model.ca_fusion_weight).item():.3f}" if hasattr(model, 'ca_fusion_weight') and model.ca_fusion_weight.requires_grad else "")
        )

        wandb.log(epoch_log)

        # Checkpoint
        if val_metrics["val_auc"] > best_val_auc:
            best_val_auc = val_metrics["val_auc"]
            epochs_without_improvement = 0
            save_path = checkpoint_dir / f"{prefix}_best.pt"
            tmp_path = save_path.with_suffix(".pt.tmp")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "stage": stage,
            }, tmp_path)
            os.replace(str(tmp_path), str(save_path))
            logger.info(f"  New best val_auc: {best_val_auc:.4f} → {save_path}")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            save_path = checkpoint_dir / f"{prefix}_epoch_{epoch}.pt"
            tmp_path = save_path.with_suffix(".pt.tmp")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "stage": stage,
            }, tmp_path)
            os.replace(str(tmp_path), str(save_path))

        # Save history
        with open(log_dir / f"{prefix}.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping at epoch {epoch} — no improvement for {patience} epochs")
            break

    logger.info(f"Stage {stage} complete. Best val_auc: {best_val_auc:.4f}")
    wandb.finish()
    return history


def main():
    parser = argparse.ArgumentParser(description="Train cross-attention embedding bypass")
    parser.add_argument("--config", type=str, default="configs/finetune_frozen.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to finetune checkpoint (Stage 1) or ca_stage1_best.pt (Stage 2)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Training stage: 1=train CA head only, 2=end-to-end fusion")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    train(config, args.checkpoint, args.stage)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('scripts/train_cross_attention.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/train_cross_attention.py
git commit -m "Add cross-attention training script with Stage 1 + Stage 2 support"
```

---

### Task 5: Create SLURM job script

**Files:**
- Create: `scripts/slurm_train_cross_attention.sh`

- [ ] **Step 1: Write the SLURM script**

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=syncguard_ca
#SBATCH --output=outputs/logs/cross_attention_%j.out
#SBATCH --error=outputs/logs/cross_attention_%j.err

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== Cross-Attention Training ($(date)) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Stage 1: Train CA head only (uses finetune_best checkpoint)
FINETUNE_CKPT="${FINETUNE_CKPT:-outputs/checkpoints/finetune_v2_backup/finetune_best.pt}"
echo "Stage 1: Training cross-attention head"
echo "Base checkpoint: $FINETUNE_CKPT"

python scripts/train_cross_attention.py \
    --config configs/finetune_frozen.yaml \
    --checkpoint "$FINETUNE_CKPT" \
    --stage 1

echo ""
echo "Stage 2: End-to-end fusion fine-tuning"
CA_STAGE1="${CA_STAGE1:-outputs/checkpoints/ca_stage1_best.pt}"
echo "Stage 1 checkpoint: $CA_STAGE1"

python scripts/train_cross_attention.py \
    --config configs/finetune_frozen.yaml \
    --checkpoint "$CA_STAGE1" \
    --stage 2

echo ""
echo "=== Evaluating ==="
python scripts/evaluate.py \
    --config configs/finetune_frozen.yaml \
    --checkpoint outputs/checkpoints/ca_stage2_best.pt \
    --test_sets fakeavceleb dfdc

echo "=== Done ($(date)) ==="
```

- [ ] **Step 2: Commit**

```bash
git add scripts/slurm_train_cross_attention.sh
git commit -m "Add SLURM script for cross-attention training + evaluation pipeline"
```

---

### Task 6: Deploy to HPC, push, and submit

- [ ] **Step 1: Push all changes to GitHub**

```bash
git push origin main
```

- [ ] **Step 2: Pull on HPC**

```bash
ssh explorer "cd /scratch/prajapati.aksh/SyncGuard && git fetch origin && git checkout origin/main -- src/models/cross_attention.py src/models/syncguard.py scripts/train_cross_attention.py scripts/slurm_train_cross_attention.sh"
```

- [ ] **Step 3: Add cross_attention config to HPC config**

```bash
ssh explorer "cd /scratch/prajapati.aksh/SyncGuard && cat >> configs/finetune_frozen.yaml << 'EOF'
  # Cross-attention embedding bypass for face-swap detection
  cross_attention:
    enabled: true
    num_heads: 2
    num_layers: 1
    dropout: 0.1
    embed_classifier_hidden: 256
    fusion_init: 0.0
EOF"
```

- [ ] **Step 4: Submit SLURM job**

```bash
ssh explorer "cd /scratch/prajapati.aksh/SyncGuard && sbatch scripts/slurm_train_cross_attention.sh"
```

- [ ] **Step 5: Monitor**

```bash
ssh explorer "squeue -u prajapati.aksh && tail -5 /scratch/prajapati.aksh/SyncGuard/outputs/logs/cross_attention_*.err"
```
