#!/usr/bin/env python3
"""Debug NaN in pretraining — tests each component with real data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.models.syncguard import build_syncguard
from src.training.losses import build_pretrain_loss
from src.training.dataset import build_dataloaders

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

print("=== Building model ===")
model = build_syncguard(config).to(device)
model.train()

# Check feature extractor mode
fe = model.audio_encoder.wav2vec2.feature_extractor
print(f"Feature extractor training: {fe.training} (should be False)")

print("\n=== Building criterion ===")
criterion = build_pretrain_loss(config).to(device)
print(f"CMP weight: {criterion.cmp_weight}")
print(f"Use CMP: {criterion.use_cross_modal}")

print("\n=== Loading first batch ===")
dataloaders = build_dataloaders(config, phase="pretrain")
batch = next(iter(dataloaders["train"]))

mc = batch.mouth_crops.to(device)
wf = batch.waveforms.to(device)
mask = batch.mask.to(device)

print(f"mouth_crops: {mc.shape}, nan={mc.isnan().any()}, range=[{mc.min():.4f}, {mc.max():.4f}]")
print(f"waveforms: {wf.shape}, nan={wf.isnan().any()}, range=[{wf.min():.6f}, {wf.max():.6f}]")
print(f"all-zero waveforms: {(wf.abs().sum(dim=-1) == 0).sum().item()}/{wf.shape[0]}")

print("\n=== Step 1: Visual encoder ===")
v = model.encode_visual(mc)
print(f"  shape={v.shape}, nan={v.isnan().any()}, inf={v.isinf().any()}")
print(f"  norm: mean={v.norm(dim=-1).mean():.4f}, min={v.norm(dim=-1).min():.6f}")

print("\n=== Step 2: Audio encoder ===")
a = model.encode_audio(wf)
print(f"  shape={a.shape}, nan={a.isnan().any()}, inf={a.isinf().any()}")
print(f"  norm: mean={a.norm(dim=-1).mean():.4f}, min={a.norm(dim=-1).min():.6f}")

print("\n=== Step 3: Align ===")
v, a = model.align_sequences(v, a)
T = v.shape[1]
mask_aligned = mask[:, :T]
print(f"  v={v.shape}, a={a.shape}, T={T}")

print("\n=== Step 4: InfoNCE loss ===")
infonce_loss = criterion.infonce(v, a, mask=mask_aligned)
print(f"  loss={infonce_loss.item():.4f}, finite={infonce_loss.isfinite().item()}")

if criterion.use_cross_modal:
    print("\n=== Step 5: CMP loss ===")
    cmp_out = criterion.cmp(v, a, padding_mask=mask_aligned)
    cmp_loss = cmp_out["loss_cmp"]
    v2a_loss = cmp_out["loss_v2a"]
    a2v_loss = cmp_out["loss_a2v"]
    print(f"  loss_cmp={cmp_loss.item():.6f}, finite={cmp_loss.isfinite().item()}")
    print(f"  loss_v2a={v2a_loss.item():.6f}")
    print(f"  loss_a2v={a2v_loss.item():.6f}")
    print(f"  100 * CMP = {(100 * cmp_loss).item():.4f}")

print("\n=== Step 6: Full combined loss ===")
loss_dict = criterion(v, a, mask=mask_aligned)
for k, val in loss_dict.items():
    if isinstance(val, torch.Tensor):
        print(f"  {k}: {val.item():.6f} finite={val.isfinite().item()}")

print("\n=== Step 7: Backward pass ===")
loss_dict["loss"].backward()
nan_grads = []
for name, p in list(model.named_parameters()) + list(criterion.named_parameters()):
    if p.grad is not None and not p.grad.isfinite().all():
        nan_grads.append(f"{name} (grad_norm={p.grad.norm():.4f})")
if nan_grads:
    print(f"  NaN/Inf gradients in {len(nan_grads)} params:")
    for g in nan_grads[:10]:
        print(f"    {g}")
else:
    print("  All gradients finite!")

print("\n=== DONE ===")
