"""Pretrain smoke test for SyncGuard Phase 1 (InfoNCE + CMP).

Exercises the full pretraining forward/backward pipeline with synthetic data:
1. Build model + PretrainLoss (InfoNCE + CrossModalPrediction)
2. Encode visual + audio (separate, like pretrain loop)
3. Align sequences
4. Compute combined loss (InfoNCE + CMP)
5. Backward + gradient clip + optimizer step
6. Verify no NaN, correct shapes, gradient flow

Works on CPU (local), MPS (Mac), and CUDA (HPC).

Usage:
    python scripts/pretrain_smoke_test.py --config configs/default.yaml
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pretrain_smoke_test")


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="SyncGuard Pretrain Smoke Test")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=3, help="Number of training steps")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()

    logger.info("=" * 60)
    logger.info("SyncGuard Pretrain Smoke Test (InfoNCE + CMP)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} "
                     f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ─── Step 1: Build model ───────────────────────────────────────────
    logger.info("Building SyncGuard model...")
    from src.models.syncguard import build_syncguard
    model = build_syncguard(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total, {trainable_params:,} trainable")

    # ─── Step 2: Build pretrain loss (InfoNCE + CMP) ───────────────────
    logger.info("Building PretrainLoss (InfoNCE + CMP)...")
    from src.training.losses import build_pretrain_loss
    criterion = build_pretrain_loss(config).to(device)

    # Move MoCo queue to device
    criterion.infonce.queue = criterion.infonce.queue.to(device)

    cmp_enabled = config["training"]["pretrain"].get("cross_modal_prediction", True)
    cmp_weight = config["training"]["pretrain"].get("cmp_weight", 0.5)
    logger.info(f"CMP enabled: {cmp_enabled}, weight: {cmp_weight}")

    crit_params = sum(p.numel() for p in criterion.parameters() if p.requires_grad)
    logger.info(f"Criterion trainable params: {crit_params:,}")

    # ─── Step 3: Build optimizer ───────────────────────────────────────
    all_params = (
        [p for p in model.parameters() if p.requires_grad]
        + [p for p in criterion.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=1e-5)
    logger.info(f"Optimizer: AdamW, {len(all_params)} param groups")

    # ─── Step 4: Run training steps ────────────────────────────────────
    B = 2  # Small batch for smoke test
    T_v = 30  # ~1.2 seconds at 25fps
    num_audio_samples = 16000 * 2  # 2 seconds at 16kHz

    logger.info(f"\nRunning {args.steps} training steps (B={B}, T_v={T_v})...")
    model.train()

    for step in range(args.steps):
        t0 = time.time()

        # Synthetic data
        mouth_crops = torch.randn(B, T_v, 1, 96, 96, device=device)
        waveform = torch.randn(B, num_audio_samples, device=device)

        # Encode (same as pretrain.py loop)
        v_embeds = model.encode_visual(mouth_crops)
        a_embeds = model.encode_audio(waveform)
        v_embeds, a_embeds = model.align_sequences(v_embeds, a_embeds)

        T = v_embeds.shape[1]
        mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Compute loss
        loss_dict = criterion(v_embeds, a_embeds, mask=mask)
        loss = loss_dict["loss"]

        # Check for NaN
        for name, val in loss_dict.items():
            if isinstance(val, torch.Tensor) and torch.isnan(val).any():
                logger.error(f"NaN in {name} at step {step}!")
                sys.exit(1)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_params = list(model.parameters()) + list(criterion.parameters())
        torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
        optimizer.step()

        dt = time.time() - t0

        logger.info(
            f"  Step {step}: loss={loss.item():.4f} "
            f"(infonce={loss_dict['loss_infonce'].item():.4f}, "
            f"cmp={loss_dict['loss_cmp'].item():.4f}) "
            f"τ={loss_dict['temperature'].item():.4f} "
            f"v_shape={v_embeds.shape} a_shape={a_embeds.shape} "
            f"time={dt:.2f}s"
        )

        # Verify sync scores
        with torch.no_grad():
            sync_scores = model.compute_sync_scores(v_embeds, a_embeds)
            logger.info(f"         sync_scores: mean={sync_scores.mean().item():.4f}, "
                        f"std={sync_scores.std().item():.4f}, shape={sync_scores.shape}")

    # ─── Step 5: Gradient flow check ──────────────────────────────────
    logger.info("\nChecking gradient flow...")
    checked = 0
    no_grad = []
    for name, param in list(model.named_parameters()) + list(criterion.named_parameters()):
        if param.requires_grad:
            if param.grad is None:
                no_grad.append(name)
            elif torch.isnan(param.grad).any():
                logger.error(f"NaN gradient in {name}!")
                sys.exit(1)
            else:
                checked += 1

    logger.info(f"Gradient check: {checked} params OK")
    if no_grad:
        logger.warning(f"No gradient for {len(no_grad)} params: {no_grad[:5]}...")

    # ─── Step 6: Memory summary ────────────────────────────────────────
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU peak memory: {peak:.2f} GB / {total:.1f} GB")

    # ─── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PRETRAIN SMOKE TEST PASSED ✓")
    logger.info(f"  Device: {device}")
    logger.info(f"  CMP: {'enabled' if cmp_enabled else 'disabled'} (weight={cmp_weight})")
    logger.info(f"  Model params: {trainable_params:,} trainable")
    logger.info(f"  Criterion params: {crit_params:,} trainable")
    logger.info(f"  Steps completed: {args.steps}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
