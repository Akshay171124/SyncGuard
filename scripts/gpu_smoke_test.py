"""GPU smoke test for SyncGuard.

Runs one batch through the full pipeline on GPU to verify:
1. Model loads and moves to GPU
2. Forward pass produces correct shapes
3. Loss computes without NaN
4. Backward pass and optimizer step work
5. No OOM errors

Usage:
    python scripts/gpu_smoke_test.py --config configs/default.yaml
"""

import argparse
import logging
import sys
import time

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smoke_test")


def main():
    parser = argparse.ArgumentParser(description="SyncGuard GPU Smoke Test")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ─── Step 1: Check GPU ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SyncGuard GPU Smoke Test")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available! Run this on a GPU node.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ─── Step 2: Build model ─────────────────────────────────────────────
    logger.info("Building SyncGuard model...")
    from src.models.syncguard import build_syncguard

    model = build_syncguard(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info(f"GPU memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ─── Step 3: Create synthetic batch ──────────────────────────────────
    B = 4
    T_v = 50  # ~2 seconds at 25fps
    num_samples = 16000 * 2  # 2 seconds at 16kHz

    logger.info(f"Creating synthetic batch: B={B}, T_v={T_v}, audio={num_samples} samples")
    mouth_crops = torch.randn(B, T_v, 1, 96, 96, device=device)
    waveform = torch.randn(B, num_samples, device=device)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32, device=device)
    lengths = torch.tensor([T_v, T_v, T_v, T_v], device=device)

    logger.info(f"GPU memory after batch: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ─── Step 4: Forward pass ────────────────────────────────────────────
    logger.info("Running forward pass...")
    model.train()
    t_start = time.time()
    output = model(mouth_crops, waveform, lengths=lengths)
    t_forward = time.time() - t_start

    logger.info(f"Forward pass: {t_forward:.3f}s")
    logger.info(f"  v_embeds: {output.v_embeds.shape}")
    logger.info(f"  a_embeds: {output.a_embeds.shape}")
    logger.info(f"  sync_scores: {output.sync_scores.shape}")
    logger.info(f"  logits: {output.logits.shape}")
    logger.info(f"GPU memory after forward: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Check for NaN
    for name, tensor in [("v_embeds", output.v_embeds), ("a_embeds", output.a_embeds),
                          ("sync_scores", output.sync_scores), ("logits", output.logits)]:
        if torch.isnan(tensor).any():
            logger.error(f"NaN detected in {name}!")
            sys.exit(1)
    logger.info("No NaN in outputs ✓")

    # ─── Step 5: Loss computation ────────────────────────────────────────
    logger.info("Computing combined loss...")
    from src.training.losses import build_finetune_loss

    criterion = build_finetune_loss(config).to(device)

    T = output.v_embeds.shape[1]
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    loss_dict = criterion(
        output.v_embeds, output.a_embeds,
        output.logits, labels,
        mask=mask,
    )

    logger.info(f"  total_loss: {loss_dict['loss'].item():.4f}")
    logger.info(f"  loss_infonce: {loss_dict['loss_infonce'].item():.4f}")
    logger.info(f"  loss_temp: {loss_dict['loss_temp'].item():.4f}")
    logger.info(f"  loss_cls: {loss_dict['loss_cls'].item():.4f}")

    if torch.isnan(loss_dict["loss"]):
        logger.error("NaN in loss!")
        sys.exit(1)
    logger.info("Loss is finite ✓")

    # ─── Step 6: Backward pass ───────────────────────────────────────────
    logger.info("Running backward pass...")
    optimizer = torch.optim.AdamW(
        [p for p in list(model.parameters()) + list(criterion.parameters()) if p.requires_grad],
        lr=5e-5,
    )
    optimizer.zero_grad()
    t_start = time.time()
    loss_dict["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    t_backward = time.time() - t_start

    logger.info(f"Backward + step: {t_backward:.3f}s")
    logger.info(f"GPU memory after backward: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    logger.info(f"GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ─── Step 7: Gradient check ──────────────────────────────────────────
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.isnan(param.grad).any():
                logger.error(f"NaN gradient in {name}!")
                sys.exit(1)
            has_grad = True

    if not has_grad:
        logger.error("No gradients computed!")
        sys.exit(1)
    logger.info("Gradients are finite ✓")

    # ─── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED ✓")
    logger.info(f"  GPU: {gpu_name}")
    logger.info(f"  Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB / {gpu_mem:.1f} GB")
    logger.info(f"  Forward: {t_forward:.3f}s, Backward: {t_backward:.3f}s")
    logger.info(f"  Model: {total_params:,} params ({trainable_params:,} trainable)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
