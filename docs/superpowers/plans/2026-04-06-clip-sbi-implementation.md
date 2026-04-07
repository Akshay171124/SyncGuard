# CLIP + SBI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AV-HuBERT visual backbone with CLIP ViT-L/14 (LayerNorm-only tuning) and add Self-Blended Image augmentation to improve DFDC zero-shot AUC from 0.526 to 0.75-0.88.

**Architecture:** Frozen CLIP ViT-L/14 (300M params, ~90K trainable LayerNorm) replaces AV-HuBERT. SBI augmentation generates synthetic face-swaps from real frames during training. Everything downstream (sync-score path, cross-attention, fusion) remains identical.

**Tech Stack:** PyTorch, HuggingFace transformers (CLIPModel), OpenCV (SBI blending), existing SyncGuard infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/clip_visual_encoder.py` | **Create** | CLIP ViT-L/14 wrapper with LayerNorm-only tuning + ProjectionHead |
| `src/augmentation/__init__.py` | **Create** | Module init |
| `src/augmentation/sbi.py` | **Create** | Self-Blended Image augmentation |
| `src/models/visual_encoder.py` | **Modify** | Add `clip` to `build_visual_encoder()` factory |
| `src/training/dataset.py` | **Modify** | Add SBI augmentation + 224×224 RGB resize in `__getitem__()` |
| `configs/clip_sbi.yaml` | **Create** | Full config for CLIP + SBI training |
| `scripts/train_clip_sbi.py` | **Create** | Training script (single-phase, direct fine-tuning) |
| `scripts/slurm_train_clip_sbi.sh` | **Create** | SLURM job for HPC |

---

### Task 1: Create CLIP Visual Encoder

**Files:**
- Create: `src/models/clip_visual_encoder.py`

- [ ] **Step 1: Create the CLIP visual encoder module**

```python
# src/models/clip_visual_encoder.py
"""CLIP ViT-L/14 visual encoder with LayerNorm-only tuning.

Processes mouth crop frames independently through CLIP's vision transformer,
extracts CLS token per frame, and projects to the shared AV embedding space.
Only LayerNorm parameters are trainable (~90K of 300M) to prevent overfitting
to source dataset artifacts while preserving CLIP's general visual understanding.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """Linear -> ReLU -> Linear -> L2-normalize projection head."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, dim=-1, eps=1e-6)
        return x


class CLIPVisualEncoder(nn.Module):
    """CLIP ViT-L/14 visual encoder for frame-level face embeddings.

    Processes each frame independently through CLIP's frozen vision transformer.
    Only LayerNorm parameters are trainable (following GenD, WACV 2026).

    Args:
        model_id: HuggingFace model identifier (default: openai/clip-vit-large-patch14).
        embedding_dim: Output embedding dimension (default: 256).
        tune_layernorm: If True, only LayerNorm params are trainable (default: True).
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        embedding_dim: int = 256,
        tune_layernorm: bool = True,
    ):
        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained(model_id)
        hidden_size = self.clip_vision.config.hidden_size  # 768 for ViT-L/14

        self.projection = ProjectionHead(in_dim=hidden_size, out_dim=embedding_dim)

        # Freeze everything first
        for param in self.clip_vision.parameters():
            param.requires_grad = False

        # Selectively unfreeze LayerNorm parameters
        if tune_layernorm:
            ln_count = 0
            for name, param in self.clip_vision.named_parameters():
                if "layernorm" in name.lower() or "layer_norm" in name.lower():
                    param.requires_grad = True
                    ln_count += param.numel()
            total = sum(p.numel() for p in self.clip_vision.parameters())
            logger.info(
                f"CLIP LayerNorm tuning: {ln_count:,} trainable / {total:,} total "
                f"({ln_count/total*100:.2f}%)"
            )

    def forward(self, mouth_crops: torch.Tensor) -> torch.Tensor:
        """Extract frame-level visual embeddings from mouth crops.

        Args:
            mouth_crops: (B, T, C, H, W) mouth crops. C=3 (RGB), H=W=224.

        Returns:
            (B, T, embedding_dim) L2-normalized visual embeddings.
        """
        B, T, C, H, W = mouth_crops.shape

        # Reshape to process all frames at once: (B*T, C, H, W)
        frames = mouth_crops.reshape(B * T, C, H, W)

        # CLIP vision encoder expects pixel_values in [-1, 1] or [0, 1]
        # Our mouth crops are already [0, 1] float
        outputs = self.clip_vision(pixel_values=frames)

        # Extract CLS token: (B*T, hidden_size)
        cls_features = outputs.pooler_output

        # Project and L2-normalize: (B*T, embedding_dim)
        embeddings = self.projection(cls_features)

        # Reshape back to sequence: (B, T, embedding_dim)
        return embeddings.reshape(B, T, -1)


def build_clip_visual_encoder(config: dict) -> CLIPVisualEncoder:
    """Build CLIP visual encoder from config."""
    ve_cfg = config["model"]["visual_encoder"]
    encoder = CLIPVisualEncoder(
        model_id=ve_cfg.get("model_id", "openai/clip-vit-large-patch14"),
        embedding_dim=ve_cfg.get("embedding_dim", 256),
        tune_layernorm=ve_cfg.get("tune_layernorm", True),
    )
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"CLIPVisualEncoder: {trainable:,} trainable / {total_params:,} total")
    return encoder
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/models/clip_visual_encoder.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/models/clip_visual_encoder.py
git commit -m "Add CLIP ViT-L/14 visual encoder with LayerNorm-only tuning"
```

---

### Task 2: Create SBI Augmentation Module

**Files:**
- Create: `src/augmentation/__init__.py`
- Create: `src/augmentation/sbi.py`

- [ ] **Step 1: Create the augmentation module init**

```python
# src/augmentation/__init__.py
from src.augmentation.sbi import SelfBlendedImage
```

- [ ] **Step 2: Create the SBI augmentation**

```python
# src/augmentation/sbi.py
"""Self-Blended Image (SBI) augmentation for deepfake detection.

Generates synthetic face-swap training data by blending face regions with
transformed versions of themselves. Creates blending boundary artifacts
common to ALL face-swap methods, enabling cross-dataset generalization.

Based on: Shiohara & Yamasaki, "Detecting Deepfakes with Self-Blended Images" (CVPR 2022)
"""

import logging
import random
from io import BytesIO

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SelfBlendedImage:
    """Self-Blended Image augmentation.

    Takes a real face frame and creates a synthetic fake by blending
    the original with a transformed version of itself using a face-shaped mask.

    Args:
        color_jitter: Color jitter strength (default: 0.1 = ±10%).
        blur_sigma: Range of Gaussian blur sigma for target (default: [1.0, 3.0]).
        warp_strength: Affine warp strength (default: 0.05 = ±5%).
        mask_blur_sigma: Range of Gaussian blur sigma for mask feathering (default: [5, 15]).
        jpeg_quality: Range of JPEG compression quality (default: [70, 95]).
    """

    def __init__(
        self,
        color_jitter: float = 0.1,
        blur_sigma: tuple[float, float] = (1.0, 3.0),
        warp_strength: float = 0.05,
        mask_blur_sigma: tuple[int, int] = (5, 15),
        jpeg_quality: tuple[int, int] = (70, 95),
    ):
        self.color_jitter = color_jitter
        self.blur_sigma = blur_sigma
        self.warp_strength = warp_strength
        self.mask_blur_sigma = mask_blur_sigma
        self.jpeg_quality = jpeg_quality

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply random brightness/contrast jitter."""
        factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        return np.clip(img * factor, 0, 1).astype(np.float32)

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply random Gaussian blur."""
        sigma = random.uniform(*self.blur_sigma)
        ksize = int(sigma * 4) | 1  # Ensure odd kernel size
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def _affine_warp(self, img: np.ndarray) -> np.ndarray:
        """Apply slight random affine transformation."""
        h, w = img.shape[:2]
        strength = self.warp_strength
        # Random affine: slight rotation + scale + translation
        center = (w / 2, h / 2)
        angle = random.uniform(-5, 5)  # ±5 degrees
        scale = 1.0 + random.uniform(-strength, strength)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # Add slight translation
        M[0, 2] += random.uniform(-w * strength, w * strength)
        M[1, 2] += random.uniform(-h * strength, h * strength)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _create_face_mask(self, h: int, w: int) -> np.ndarray:
        """Create an elliptical face mask centered on the frame.

        Since we're working with mouth crops (already tightly cropped faces),
        the mask is a centered ellipse covering ~70% of the frame.
        """
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.35), int(h * 0.40))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        # Feather the mask edges
        blur_sigma = random.randint(*self.mask_blur_sigma)
        ksize = blur_sigma * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), blur_sigma)
        return mask

    def _jpeg_compress(self, img: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        quality = random.randint(*self.jpeg_quality)
        # Convert to uint8 for JPEG encoding
        img_uint8 = (img * 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", img_uint8, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE)
        return decoded.astype(np.float32) / 255.0

    def blend_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply SBI augmentation to a single frame.

        Args:
            frame: (H, W) or (H, W, C) float32 frame in [0, 1].

        Returns:
            Blended frame with same shape, containing blending boundary artifacts.
        """
        is_gray = frame.ndim == 2
        if is_gray:
            frame = frame[:, :, np.newaxis]

        h, w, c = frame.shape

        # Create transformed target
        target = frame.copy()
        target = self._color_jitter(target)
        target = self._gaussian_blur(target)
        target = self._affine_warp(target)

        # Create face mask with feathered edges
        mask = self._create_face_mask(h, w)
        mask = mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting

        # Alpha blend: source outside mask, target inside mask
        blended = frame * (1 - mask) + target * mask

        # Optional JPEG compression
        if random.random() < 0.5:
            if is_gray:
                blended = blended.squeeze(-1)
            blended = self._jpeg_compress(blended)
            if is_gray and blended.ndim == 2:
                blended = blended[:, :, np.newaxis]

        if is_gray:
            blended = blended.squeeze(-1)

        return np.clip(blended, 0, 1).astype(np.float32)

    def augment_sequence(self, mouth_crops: torch.Tensor) -> torch.Tensor:
        """Apply SBI to a sequence of mouth crop frames.

        Args:
            mouth_crops: (T, C, H, W) tensor of mouth crops in [0, 1].

        Returns:
            (T, C, H, W) tensor with SBI augmentation applied to each frame.
        """
        T, C, H, W = mouth_crops.shape
        result = mouth_crops.clone()

        for t in range(T):
            frame = mouth_crops[t].permute(1, 2, 0).numpy()  # (H, W, C)
            if C == 1:
                frame = frame.squeeze(-1)  # (H, W)
            blended = self.blend_frame(frame)
            if C == 1 and blended.ndim == 2:
                blended = blended[:, :, np.newaxis]
            result[t] = torch.from_numpy(blended).permute(2, 0, 1) if blended.ndim == 3 else torch.from_numpy(blended).unsqueeze(0)

        return result


def build_sbi(config: dict) -> SelfBlendedImage:
    """Build SBI augmentation from config."""
    sbi_cfg = config.get("augmentation", {}).get("sbi", {})
    return SelfBlendedImage(
        color_jitter=sbi_cfg.get("color_jitter", 0.1),
        blur_sigma=tuple(sbi_cfg.get("blur_sigma", [1.0, 3.0])),
        warp_strength=sbi_cfg.get("warp_strength", 0.05),
        mask_blur_sigma=tuple(sbi_cfg.get("mask_blur_sigma", [5, 15])),
        jpeg_quality=tuple(sbi_cfg.get("jpeg_quality", [70, 95])),
    )
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/augmentation/sbi.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/augmentation/__init__.py src/augmentation/sbi.py
git commit -m "Add Self-Blended Image augmentation for cross-dataset generalization"
```

---

### Task 3: Add CLIP to Visual Encoder Factory

**Files:**
- Modify: `src/models/visual_encoder.py` (the `build_visual_encoder` function at line 313)

- [ ] **Step 1: Add CLIP import and factory case**

At the top of `src/models/visual_encoder.py`, add the import:

```python
from src.models.clip_visual_encoder import build_clip_visual_encoder
```

In the `build_visual_encoder()` function, after the `syncnet` elif block (around line 345), add:

```python
    elif name == "clip":
        return build_clip_visual_encoder(config)
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/models/visual_encoder.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/models/visual_encoder.py
git commit -m "Add CLIP option to visual encoder factory"
```

---

### Task 4: Add SBI Augmentation + RGB Resize to Dataset

**Files:**
- Modify: `src/training/dataset.py`

- [ ] **Step 1: Add SBI import and initialization**

At the top of `src/training/dataset.py`, add:

```python
from src.augmentation.sbi import SelfBlendedImage, build_sbi
```

In `SyncGuardDataset.__init__()`, after the existing `audio_swap_ratio` initialization, add:

```python
        # SBI augmentation for cross-dataset generalization
        sbi_cfg = config.get("augmentation", {}).get("sbi", {})
        self.sbi_enabled = sbi_cfg.get("enabled", False)
        self.sbi_ratio = sbi_cfg.get("ratio", 0.3)
        self.sbi = build_sbi(config) if self.sbi_enabled else None

        # Visual input format from config
        ve_cfg = config.get("model", {}).get("visual_encoder", {})
        self.visual_input_size = ve_cfg.get("input_size", 96)
        self.visual_channels = 3 if ve_cfg.get("name") == "clip" else 1
```

- [ ] **Step 2: Add SBI augmentation in `__getitem__`**

In `SyncGuardDataset.__getitem__()`, after the audio-swap augmentation block and before `if self.transform is not None:`, add:

```python
        # SBI augmentation: for REAL samples, with some probability,
        # create synthetic face-swap by self-blending the mouth crops.
        if (
            self.sbi_enabled
            and self.sbi is not None
            and label == 0  # Only augment real samples
            and random.random() < self.sbi_ratio
        ):
            mouth_crops = self.sbi.augment_sequence(mouth_crops)
            label = 1  # Now it's a synthetic fake
            category = "SBI-aug"
```

After the EAR loading and before the return statement, add resize and RGB conversion:

```python
        # Resize and convert channels if needed (e.g., for CLIP backbone)
        if self.visual_input_size != mouth_crops.shape[-1]:
            T, C, H, W = mouth_crops.shape
            mouth_crops = F.interpolate(
                mouth_crops, size=(self.visual_input_size, self.visual_input_size),
                mode="bilinear", align_corners=False,
            )
        if self.visual_channels == 3 and mouth_crops.shape[1] == 1:
            mouth_crops = mouth_crops.repeat(1, 3, 1, 1)  # Grayscale -> RGB
```

Add the import for F at the top of the file if not already present:

```python
import torch.nn.functional as F
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/training/dataset.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/training/dataset.py
git commit -m "Add SBI augmentation and CLIP input preprocessing to dataset"
```

---

### Task 5: Create Config File

**Files:**
- Create: `configs/clip_sbi.yaml`

- [ ] **Step 1: Create the config**

```yaml
# SyncGuard Configuration — CLIP + SBI for DFDC Generalization

# Data paths
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  features_dir: "data/processed"
  fakeavceleb_dir: "data/raw/FakeAVCeleb"
  avspeech_dir: "data/raw/AVSpeech"
  celebdf_dir: "data/raw/CelebDF-v2"
  dfdc_dir: "data/raw/DFDC/dfdc_train_part_0"
  lrs2_dir: "data/raw/LRS2/mvlrs_v1/pretrain"

# Preprocessing
preprocessing:
  video:
    fps: 25
    mouth_crop_size: 96
    face_detection_confidence: 0.8
    face_detection_backend: "retinaface"
  audio:
    sample_rate: 16000
    target_fps: 49
  vad:
    threshold: 0.5
    min_speech_duration_ms: 250
    min_silence_duration_ms: 100

# Model
model:
  visual_encoder:
    name: "clip"
    model_id: "openai/clip-vit-large-patch14"
    embedding_dim: 256
    freeze_pretrained: true
    tune_layernorm: true
    input_size: 224
  audio_encoder:
    name: "wav2vec2"
    model_id: "facebook/wav2vec2-base-960h"
    layer: 9
    embedding_dim: 256
    freeze_pretrained: true
  classifier:
    name: "bilstm"
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    use_ear: true
  audio_head: false
  cross_attention:
    enabled: true
    num_heads: 2
    num_layers: 1
    dropout: 0.1
    embed_classifier_hidden: 256
    fusion_init: 0.0
  dct_extractor:
    enabled: false

# Augmentation
augmentation:
  sbi:
    enabled: true
    ratio: 0.3
    color_jitter: 0.1
    blur_sigma: [1.0, 3.0]
    warp_strength: 0.05
    mask_blur_sigma: [5, 15]
    jpeg_quality: [70, 95]

# Training — single phase, no pretraining
training:
  pretrain:
    epochs: 20
    batch_size: 16
    lr: 1.0e-4
    weight_decay: 1.0e-5
    scheduler: "cosine"
    warmup_epochs: 2
    moco_queue_size: 4096
    temperature: 0.07
    temperature_range: [0.03, 0.5]
    cross_modal_prediction: true
    cmp_weight: 100.0
    cmp_mask_ratio: 0.3
  finetune:
    epochs: 30
    batch_size: 16
    lr: 5.0e-5
    weight_decay: 1.0e-4
    scheduler: "cosine"
    warmup_epochs: 3
    gamma: 0.5
    delta: 1.0
    hard_negative_ratio: 0.2
    hard_negative_anneal_epochs: 10
    audio_swap_ratio: 0.15

seed: 42
```

- [ ] **Step 2: Commit**

```bash
git add configs/clip_sbi.yaml
git commit -m "Add CLIP + SBI config for DFDC generalization"
```

---

### Task 6: Create Training Script

**Files:**
- Create: `scripts/train_clip_sbi.py`

- [ ] **Step 1: Create the training script**

```python
#!/usr/bin/env python3
"""Train SyncGuard with CLIP backbone + SBI augmentation.

Single-phase training — skip contrastive pretraining.
CLIP is already pretrained, go directly to supervised fine-tuning.

Usage:
    python scripts/train_clip_sbi.py --config configs/clip_sbi.yaml
    python scripts/train_clip_sbi.py --config configs/clip_sbi.yaml --resume outputs/checkpoints/finetune_best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.finetune import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('scripts/train_clip_sbi.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/train_clip_sbi.py
git commit -m "Add CLIP+SBI training script (delegates to finetune.main)"
```

---

### Task 7: Create SLURM Job Script

**Files:**
- Create: `scripts/slurm_train_clip_sbi.sh`

- [ ] **Step 1: Create the SLURM script**

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=clip_sbi
#SBATCH --output=outputs/logs/clip_sbi_%j.out
#SBATCH --error=outputs/logs/clip_sbi_%j.err
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

module load miniconda3/24.11.1 FFmpeg/7.1.1
eval "$(conda shell.bash hook)" && conda activate syncguard
export HF_HOME=/scratch/$USER/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/SyncGuard
export PYTHONPATH=/scratch/$USER/SyncGuard:$PYTHONPATH
mkdir -p outputs/logs outputs/checkpoints

echo "=== CLIP + SBI Training ($(date)) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Clear old checkpoints to avoid conflicts
mkdir -p outputs/checkpoints/pre_clip_backup
mv outputs/checkpoints/finetune_best.pt outputs/checkpoints/pre_clip_backup/ 2>/dev/null
mv outputs/checkpoints/finetune_epoch_*.pt outputs/checkpoints/pre_clip_backup/ 2>/dev/null

RESUME_ARG=""
LATEST=$(ls -t outputs/checkpoints/finetune_epoch_*.pt outputs/checkpoints/finetune_best.pt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_ARG="--resume $LATEST"
fi

python scripts/train_clip_sbi.py \
    --config configs/clip_sbi.yaml \
    $RESUME_ARG

EXIT_CODE=$?
echo "=== Finished with exit code $EXIT_CODE ($(date)) ==="

# Evaluate on FakeAVCeleb + DFDC
if [ $EXIT_CODE -eq 0 ] && [ -f outputs/checkpoints/finetune_best.pt ]; then
    echo "=== Evaluating ==="
    python scripts/evaluate.py \
        --config configs/clip_sbi.yaml \
        --checkpoint outputs/checkpoints/finetune_best.pt \
        --test_sets fakeavceleb dfdc
fi
```

- [ ] **Step 2: Commit**

```bash
git add scripts/slurm_train_clip_sbi.sh
git commit -m "Add SLURM script for CLIP+SBI training + evaluation"
```

---

### Task 8: Smoke Test and Deploy

- [ ] **Step 1: Run CPU smoke test**

```bash
python3 -c "
import torch, yaml
torch.manual_seed(42)

with open('configs/clip_sbi.yaml') as f:
    config = yaml.safe_load(f)

# Test CLIP encoder
from src.models.clip_visual_encoder import CLIPVisualEncoder
enc = CLIPVisualEncoder(embedding_dim=256)
B, T = 2, 5
frames = torch.randn(B, T, 3, 224, 224)
v = enc(frames)
print(f'CLIP encoder: {v.shape}, nan={v.isnan().any()}')
assert v.shape == (B, T, 256)

# Test SBI augmentation
from src.augmentation.sbi import SelfBlendedImage
sbi = SelfBlendedImage()
frame = torch.rand(10, 1, 96, 96)
aug = sbi.augment_sequence(frame)
print(f'SBI augment: {aug.shape}, range=[{aug.min():.2f}, {aug.max():.2f}]')

# Test build_visual_encoder with clip
from src.models.visual_encoder import build_visual_encoder
ve = build_visual_encoder(config)
print(f'Visual encoder type: {type(ve).__name__}')
assert type(ve).__name__ == 'CLIPVisualEncoder'

print('SMOKE TEST PASSED')
"
```

Expected: `SMOKE TEST PASSED`

- [ ] **Step 2: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 3: Pull on HPC**

```bash
ssh explorer "cd /scratch/prajapati.aksh/SyncGuard && git fetch origin && git checkout origin/main -- src/models/clip_visual_encoder.py src/augmentation/sbi.py src/augmentation/__init__.py src/models/visual_encoder.py src/training/dataset.py configs/clip_sbi.yaml scripts/train_clip_sbi.py scripts/slurm_train_clip_sbi.sh"
```

- [ ] **Step 4: Pre-download CLIP model on HPC**

```bash
ssh explorer "cd /scratch/prajapati.aksh/SyncGuard && module load miniconda3/24.11.1 && conda activate syncguard && python3 -c 'from transformers import CLIPVisionModel; CLIPVisionModel.from_pretrained(\"openai/clip-vit-large-patch14\"); print(\"CLIP downloaded\")'"
```

- [ ] **Step 5: Submit SLURM job**

```bash
ssh explorer "cd /scratch/prajapati.aksh/SyncGuard && sbatch scripts/slurm_train_clip_sbi.sh"
```
