# SyncGuard Clean Pipeline Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild every SyncGuard artifact lost to the Explorer scratch purge, on a single seeded and fully traceable provenance chain, so research can resume at any pipeline stage.

**Architecture:** Two parts. Part A hardens the codebase locally under TDD — seeding, checkpoint provenance, a silent-failure guard, a resubmit guard, and committed configs — and must be fully committed before any HPC job launches, because the root cause of the loss was recipes living only on scratch. Part B executes the pipeline on Explorer: restore data, preprocess from raw, train five checkpoints in sequence, evaluate, then archive to durable storage and correct the documentation.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, transformers, wandb, mediapipe 0.10.33, retina-face, librosa, SLURM on Northeastern Explorer (H200), conda env `syncguard`.

**Spec:** `docs/superpowers/specs/2026-09-09-clean-pipeline-rebuild-design.md`

## Global Constraints

- Exact numeric replication of April's results is **not** a goal. Acceptance is FakeAVCeleb test AUC **>= 0.90**.
- DFDC AUC in the **0.50–0.60** band is a **success**, not a failure. It reproduces a documented limitation.
- The AV-HuBERT visual encoder stays **randomly initialized**. Do not install fairseq. Do not set `checkpoint_path`.
- Do **not** generate a Wav2Lip adversarial set. Out of scope.
- All configs must be committed to git **before** the run that uses them launches.
- Never write a checkpoint only to `/scratch`. Every checkpoint is archived to `/home/prajapati.aksh/ckpt_archive/`.
- `gpu` and `gpu-short` partitions cap at **8:00:00**. Pretraining needs resume.
- Use **H200 only** (`--gres=gpu:h200:1`). No V100/A100 fallback.
- Wav2Vec 2.0 backbone stays **frozen** during fine-tuning.
- Never `pip install` directly. Add to `requirements.txt`, then `pip install -r requirements.txt`.
- Do not mention LLM assistance in commits, code, or docs.
- Activate the environment with `source ~/.zshrc && conda activate syncguard`.
- Current year is 2026.

## File Structure

**Create:**
- `src/utils/seeding.py` — `seed_everything()`; single source of RNG control.
- `src/utils/provenance.py` — builds the provenance dict embedded in checkpoints.
- `src/utils/checkpoint.py` — one shared `save_checkpoint()`, replacing the duplicate in `pretrain.py` and `finetune.py`.
- `scripts/lib/resubmit_guard.sh` — sourced by SLURM scripts; aborts after 3 consecutive identical failures.
- `scripts/archive_checkpoints.sh` — copies checkpoints to `/home` archive and appends to the manifest.
- `configs/rebuild_pretrain.yaml`, `configs/rebuild_finetune.yaml` — the committed recipes for this rebuild.
- `tests/test_seeding.py`, `tests/test_provenance.py`, `tests/test_checkpoint_provenance.py`.
- `MANIFEST.md` — checkpoint to provenance mapping.

**Modify:**
- `src/training/pretrain.py` — delete local `save_checkpoint` (line 161), import shared one; update call sites at 427 and 436.
- `src/training/finetune.py` — delete local `save_checkpoint` (line 275), import shared one; update call sites at 550 and 560.
- `scripts/train_pretrain.py`, `scripts/train_finetune.py`, `scripts/train_audio_classifier.py` — call `seed_everything()`.
- `src/models/visual_encoder.py:328-336` — warn when `av_hubert` has no `checkpoint_path`.
- `docs/EXECUTION_PLAN.md`, `README.md`, `.claude/CLAUDE.md` — correct the two false claims.

---

# Part A — Codebase hardening (local, before any HPC job)

### Task 1: Archive the surviving April checkpoints

Safety-critical and first. Training writes to `outputs/checkpoints/finetune_best.pt` — the same filename one survivor holds. Any launch before this task destroys the only artifact tied to the April report.

**Files:**
- Create: `demo_assets/checkpoints/april_reference/` (local, not committed — `*.pt` is gitignored)

- [ ] **Step 1: Verify both survivors are intact before touching them**

```bash
cd "/Users/akshayprajapati/Desktop/CVPR Project/SyncGuard/demo_assets/checkpoints"
python3 -c "
import zipfile
for f in ['finetune_best.pt','audio_clf_best.pt']:
    z = zipfile.ZipFile(f)
    assert z.testzip() is None, f
    print(f, 'OK', len(z.namelist()), 'entries')
"
```

Expected: `finetune_best.pt OK 738 entries` and `audio_clf_best.pt OK 239 entries`.

- [ ] **Step 2: Copy (do not move) into the reference archive**

```bash
mkdir -p april_reference
cp -n finetune_best.pt audio_clf_best.pt april_reference/
ls -lh april_reference/
```

`cp -n` refuses to overwrite. Copy rather than move so the demo keeps working.

- [ ] **Step 3: Verify the copies independently**

```bash
python3 -c "
import zipfile
for f in ['april_reference/finetune_best.pt','april_reference/audio_clf_best.pt']:
    assert zipfile.ZipFile(f).testzip() is None, f
    print(f, 'verified')
"
```

- [ ] **Step 4: Make a second copy off this machine**

Copy `april_reference/` to Google Drive or a USB drive. Do not proceed until a copy exists somewhere other than this laptop. These files are single-copy and unreproducible: the config that made them is gone and the run was unseeded.

- [ ] **Step 5: Record it**

```bash
cd "/Users/akshayprajapati/Desktop/CVPR Project/SyncGuard"
echo "- april_reference/: finetune_best.pt, audio_clf_best.pt — April artifacts, verified $(date +%Y-%m-%d), backed up off-machine" >> docs/lab_notebook.md
git add docs/lab_notebook.md && git commit -m "docs: record April checkpoint archive"
```

---

### Task 2: Deterministic seeding

**Files:**
- Create: `src/utils/seeding.py`, `tests/test_seeding.py`
- Modify: `scripts/train_pretrain.py`, `scripts/train_finetune.py`, `scripts/train_audio_classifier.py`

**Interfaces:**
- Produces: `seed_everything(seed: int = 42, deterministic: bool = True) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_seeding.py
"""Tests for deterministic seeding across RNGs."""

import numpy as np
import torch

from src.utils.seeding import seed_everything


def test_torch_rng_is_reproducible():
    seed_everything(1234)
    a = torch.randn(5)
    seed_everything(1234)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_numpy_rng_is_reproducible():
    seed_everything(99)
    a = np.random.rand(5)
    seed_everything(99)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    seed_everything(1)
    a = torch.randn(5)
    seed_everything(2)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_returns_applied_seed():
    assert seed_everything(7) == 7


def test_sets_cudnn_deterministic():
    seed_everything(42, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_seeding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.seeding'`

- [ ] **Step 3: Implement**

```python
# src/utils/seeding.py
"""Deterministic seeding for every training entry point.

April's pretrain and finetune runs were unseeded, which is one reason their
checkpoints cannot be regenerated. Every entry point calls seed_everything()
before constructing models or dataloaders.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all RNGs used by the training pipeline.

    Args:
        seed: Value applied to python, numpy, and torch RNGs.
        deterministic: Force cuDNN into deterministic mode. Costs some
            throughput but makes runs comparable across launches.

    Returns:
        The seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_seeding.py -v`
Expected: 5 passed

- [ ] **Step 5: Wire into the three entry points**

In each of `scripts/train_pretrain.py`, `scripts/train_finetune.py`, and `scripts/train_audio_classifier.py`, add the import near the other `src` imports:

```python
from src.utils.seeding import seed_everything
```

Then, in each script's `main()`, immediately after the config is loaded and before any model, dataset, or dataloader is constructed:

```python
    seed = seed_everything(config.get("seed", 42))
    logger.info(f"Seeded all RNGs with seed={seed}")
```

Do not modify `scripts/train_cross_attention.py` — it already seeds at lines 145-150.

- [ ] **Step 6: Verify the full suite still passes**

Run: `python -m pytest tests/ -q`
Expected: 220 passed, 4 skipped (215 existing + 5 new)

- [ ] **Step 7: Commit**

```bash
git add src/utils/seeding.py tests/test_seeding.py scripts/train_pretrain.py scripts/train_finetune.py scripts/train_audio_classifier.py
git commit -m "feat: deterministic seeding across all training entry points"
```

---

### Task 3: Checkpoint provenance metadata

**Files:**
- Create: `src/utils/provenance.py`, `tests/test_provenance.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `config_sha256(config: dict) -> str`, `build_provenance(config: dict, wandb_run_id: str | None = None, dataset_manifest: str | None = None) -> dict` returning keys `git_sha`, `config_sha256`, `wandb_run_id`, `dataset_manifest`, `timestamp`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_provenance.py
"""Tests for checkpoint provenance metadata."""

import re

from src.utils.provenance import build_provenance, config_sha256

PROVENANCE_KEYS = {
    "git_sha",
    "config_sha256",
    "wandb_run_id",
    "dataset_manifest",
    "timestamp",
}


def test_config_hash_is_order_independent():
    a = {"lr": 1e-4, "epochs": 20}
    b = {"epochs": 20, "lr": 1e-4}
    assert config_sha256(a) == config_sha256(b)


def test_config_hash_changes_with_content():
    assert config_sha256({"lr": 1e-4}) != config_sha256({"lr": 2e-4})


def test_config_hash_handles_nested_and_nonjson():
    from pathlib import Path
    h = config_sha256({"model": {"name": "av_hubert"}, "path": Path("/tmp/x")})
    assert re.fullmatch(r"[0-9a-f]{64}", h)


def test_build_provenance_has_all_keys():
    p = build_provenance({"lr": 1e-4})
    assert set(p) == PROVENANCE_KEYS


def test_git_sha_is_sha_or_unknown():
    sha = build_provenance({})["git_sha"]
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)


def test_optional_fields_default_to_none_string():
    p = build_provenance({})
    assert p["wandb_run_id"] == "none"
    assert p["dataset_manifest"] == "none"


def test_optional_fields_are_recorded():
    p = build_provenance({}, wandb_run_id="abc123", dataset_manifest="deadbeef")
    assert p["wandb_run_id"] == "abc123"
    assert p["dataset_manifest"] == "deadbeef"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.provenance'`

- [ ] **Step 3: Implement**

```python
# src/utils/provenance.py
"""Provenance metadata embedded in every checkpoint.

April's checkpoints carried no link back to the commit or config that
produced them, and the configs themselves lived only on scratch. Embedding
provenance inside the checkpoint makes artifacts self-describing, so the
link survives files being moved, renamed, or restored from backup.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone


def _git_sha() -> str:
    """Current HEAD commit, or "unknown" outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_sha256(config: dict) -> str:
    """Stable hash of a resolved config.

    Sorts keys so logically identical configs hash identically, and coerces
    non-JSON values (such as Path) via str so hashing never raises.
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def build_provenance(
    config: dict,
    wandb_run_id: str | None = None,
    dataset_manifest: str | None = None,
) -> dict:
    """Build the provenance block stored alongside checkpoint weights."""
    return {
        "git_sha": _git_sha(),
        "config_sha256": config_sha256(config),
        "wandb_run_id": wandb_run_id or "none",
        "dataset_manifest": dataset_manifest or "none",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_provenance.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/provenance.py tests/test_provenance.py
git commit -m "feat: checkpoint provenance metadata"
```

---

### Task 4: Shared checkpoint saving with embedded provenance

`save_checkpoint` is currently duplicated verbatim in `src/training/pretrain.py:161` and `src/training/finetune.py:275`. Consolidate into one function that always embeds provenance, so no future stage can save an untraceable checkpoint.

**Files:**
- Create: `src/utils/checkpoint.py`, `tests/test_checkpoint_provenance.py`
- Modify: `src/training/pretrain.py` (delete lines 161-195, update call sites at 427 and 436), `src/training/finetune.py` (delete lines 275-309, update call sites at 550 and 560)

**Interfaces:**
- Consumes: `build_provenance()` from Task 3.
- Produces: `save_checkpoint(model, optimizer, scheduler, criterion, epoch, val_metrics, path, config, wandb_run_id=None) -> Path`. Note `config` is **required** and positional-or-keyword — every checkpoint must carry provenance.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checkpoint_provenance.py
"""Tests for the shared checkpoint saver and its provenance block."""

import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.checkpoint import save_checkpoint


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


def _state():
    model = _Tiny()
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = CosineAnnealingLR(opt, T_max=10)
    crit = torch.nn.BCEWithLogitsLoss()
    return model, opt, sched, crit


def test_saves_all_state_keys():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        save_checkpoint(model, opt, sched, crit, 3, {"val_auc": 0.9}, p,
                        config={"lr": 1e-3})
        ck = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("epoch", "model_state_dict", "optimizer_state_dict",
              "scheduler_state_dict", "criterion_state_dict", "val_metrics"):
        assert k in ck


def test_embeds_provenance_block():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        save_checkpoint(model, opt, sched, crit, 1, {}, p,
                        config={"lr": 1e-3}, wandb_run_id="run42")
        ck = torch.load(p, map_location="cpu", weights_only=False)
    prov = ck["provenance"]
    assert set(prov) == {"git_sha", "config_sha256", "wandb_run_id",
                         "dataset_manifest", "timestamp"}
    assert prov["wandb_run_id"] == "run42"


def test_creates_parent_directory():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nested" / "deeper" / "ckpt.pt"
        save_checkpoint(model, opt, sched, crit, 0, {}, p, config={})
        assert p.exists()


def test_leaves_no_tmp_file():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        save_checkpoint(model, opt, sched, crit, 0, {}, p, config={})
        assert list(Path(d).glob("*.tmp")) == []


def test_returns_path():
    model, opt, sched, crit = _state()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        assert save_checkpoint(model, opt, sched, crit, 0, {}, p,
                               config={}) == p
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_checkpoint_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.checkpoint'`

- [ ] **Step 3: Implement**

```python
# src/utils/checkpoint.py
"""Shared checkpoint saving.

One saver for every training stage. `config` is required so that no stage
can produce a checkpoint without provenance. The save is atomic: write to
a temporary file, then os.replace, which is atomic on POSIX.
"""

import logging
import os
from pathlib import Path

import torch

from src.utils.provenance import build_provenance

logger = logging.getLogger(__name__)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    criterion,
    epoch: int,
    val_metrics: dict,
    path,
    config: dict,
    wandb_run_id: str | None = None,
    dataset_manifest: str | None = None,
) -> Path:
    """Atomically save a training checkpoint with embedded provenance.

    Args:
        model: Model whose state_dict is saved.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        criterion: Loss state (MoCo queue and temperature for pretraining).
        epoch: Current epoch.
        val_metrics: Validation metrics for this epoch.
        path: Destination path.
        config: Resolved run config, hashed into the provenance block.
        wandb_run_id: W&B run id, if a run is active.
        dataset_manifest: Hash of the dataset manifest, if available.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "val_metrics": val_metrics,
            "provenance": build_provenance(config, wandb_run_id, dataset_manifest),
        },
        tmp_path,
    )
    os.replace(str(tmp_path), str(path))  # Atomic on POSIX
    logger.info(f"Checkpoint saved: {path}")
    return path
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_checkpoint_provenance.py -v`
Expected: 5 passed

- [ ] **Step 5: Remove the duplicate in pretrain.py and import the shared one**

Delete the entire `def save_checkpoint(...)` block at `src/training/pretrain.py:161-195`. Add to the imports at the top:

```python
from src.utils.checkpoint import save_checkpoint
```

Both call sites (lines 427 and 436 before deletion) currently end with `path=<...>`. Add the config argument to each:

```python
            save_checkpoint(
                model, optimizer, scheduler, criterion, epoch, val_metrics,
                path=<existing path argument unchanged>,
                config=config,
            )
```

`config` is already in scope in `train()`. If a W&B run is active in that scope, also pass `wandb_run_id=wandb.run.id if wandb.run else None`.

- [ ] **Step 6: Do the same in finetune.py**

Delete `def save_checkpoint(...)` at `src/training/finetune.py:275-309`, add the same import, and add `config=config` to the call sites at lines 550 and 560.

- [ ] **Step 7: Verify no duplicate definition remains**

```bash
grep -rn "def save_checkpoint" src/ | grep -v "src/utils/checkpoint.py"
```

Expected: no output.

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: 232 passed, 4 skipped

- [ ] **Step 9: Commit**

```bash
git add src/utils/checkpoint.py tests/test_checkpoint_provenance.py src/training/pretrain.py src/training/finetune.py
git commit -m "refactor: single checkpoint saver with embedded provenance"
```

---

### Task 5: Warn when AV-HuBERT has no pretrained checkpoint

The encoder silently trained randomly initialized for the entire project because `ve_cfg.get("checkpoint_path")` returned `None` and the load sat behind `if ckpt:`. No exception, no warning. Make the condition audible without changing behaviour.

**Files:**
- Modify: `src/models/visual_encoder.py:328-336`
- Modify: `tests/test_models.py` (append the two tests below)

**Interfaces:**
- Consumes: nothing.
- Produces: no API change. `build_visual_encoder(config)` keeps its signature and still returns a randomly initialized encoder when no path is set.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_av_hubert_warns_without_checkpoint_path(caplog):
    """Silent random init is what hid the missing weights for a whole project."""
    from src.models.visual_encoder import build_visual_encoder

    config = {"model": {"visual_encoder": {
        "name": "av_hubert", "embedding_dim": 256, "freeze_pretrained": False}}}
    with caplog.at_level("WARNING"):
        encoder = build_visual_encoder(config)
    assert encoder is not None
    assert any("randomly initialized" in r.message.lower()
               for r in caplog.records)


def test_resnet18_does_not_warn(caplog):
    from src.models.visual_encoder import build_visual_encoder

    config = {"model": {"visual_encoder": {
        "name": "resnet18", "embedding_dim": 256, "freeze_pretrained": False}}}
    with caplog.at_level("WARNING"):
        build_visual_encoder(config)
    assert not any("randomly initialized" in r.message.lower()
                   for r in caplog.records)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_models.py -k av_hubert_warns -v`
Expected: FAIL — assertion error, no matching warning record

- [ ] **Step 3: Implement**

In `src/models/visual_encoder.py`, replace the `if ckpt:` block inside the `if name == "av_hubert":` branch (around line 333) with:

```python
        ckpt = ve_cfg.get("checkpoint_path")
        if ckpt:
            encoder.load_av_hubert_weights(ckpt)
        else:
            logger.warning(
                "AV-HuBERT selected with no 'checkpoint_path' — the encoder is "
                "RANDOMLY INITIALIZED and uses none of AV-HuBERT's pretrained "
                "lip-reading weights. This is intentional for the 2026-09-09 "
                "rebuild baseline. Set model.visual_encoder.checkpoint_path to "
                "load pretrained weights (requires fairseq)."
            )
        return encoder
```

`logger` is already defined at module scope.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_models.py -k "av_hubert_warns or resnet18_does_not_warn" -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: 234 passed, 4 skipped

- [ ] **Step 6: Commit**

```bash
git add src/models/visual_encoder.py tests/test_models.py
git commit -m "fix: warn when AV-HuBERT runs without pretrained weights"
```

---

### Task 6: Resubmit crash-loop guard, and commit the rebuild configs

On 2026-04-07 an auto-resubmit loop produced roughly 40 job submissions that all died with the same collation error, because the resubmit trap does not distinguish "hit the time limit, resume" from "crashed instantly, will crash again". `scripts/slurm_pretrain.sh:16-27` traps `USR1 TERM INT HUP XCPU` and resubmits with only an in-process `RESUBMITTED` flag, which does not survive across jobs.

**Files:**
- Create: `scripts/lib/resubmit_guard.sh`, `tests/test_resubmit_guard.py`
- Create: `configs/rebuild_pretrain.yaml`, `configs/rebuild_finetune.yaml`
- Modify: `scripts/slurm_pretrain.sh` (resubmit function, lines 16-27)

**Interfaces:**
- Consumes: nothing.
- Produces: shell functions `guard_count <name>`, `guard_record_failure <name>`, `guard_reset <name>`, and `guard_may_resubmit <name> <max>` which exits 0 when resubmission is allowed and 1 when the limit is reached.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resubmit_guard.py
"""Tests for the SLURM resubmit crash-loop guard."""

import subprocess
import tempfile
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1] / "scripts" / "lib" / "resubmit_guard.sh"


def _run(script: str, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f"source {GUARD}\n{script}"],
        cwd=cwd, capture_output=True, text=True,
    )


def test_starts_at_zero():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_count job", d)
        assert r.stdout.strip() == "0"


def test_failures_accumulate():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_record_failure job; guard_record_failure job; guard_count job", d)
        assert r.stdout.strip() == "2"


def test_reset_clears_count():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_record_failure job; guard_reset job; guard_count job", d)
        assert r.stdout.strip() == "0"


def test_may_resubmit_below_limit():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_record_failure job; guard_may_resubmit job 3", d)
        assert r.returncode == 0


def test_may_not_resubmit_at_limit():
    with tempfile.TemporaryDirectory() as d:
        r = _run(
            "guard_record_failure job; guard_record_failure job; "
            "guard_record_failure job; guard_may_resubmit job 3", d)
        assert r.returncode == 1


def test_counts_are_per_job_name():
    with tempfile.TemporaryDirectory() as d:
        r = _run("guard_record_failure a; guard_record_failure a; guard_count b", d)
        assert r.stdout.strip() == "0"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_resubmit_guard.py -v`
Expected: FAIL — the sourced file does not exist, so every case errors

- [ ] **Step 3: Implement the guard**

```bash
# scripts/lib/resubmit_guard.sh
# Consecutive-failure guard for SLURM auto-resubmit loops.
#
# On 2026-04-07 a resubmit trap produced ~40 identical failing jobs because
# the in-process guard flag did not survive across submissions. This keeps
# the count in a file so it persists between jobs.
#
# Reset the counter whenever real progress is made (an epoch completes), so
# a long multi-job run is never throttled — only a genuine crash-loop is.

_guard_file() {
    mkdir -p outputs/logs
    echo "outputs/logs/.resubmit_count_${1}"
}

guard_count() {
    local f
    f="$(_guard_file "$1")"
    if [ -f "$f" ]; then cat "$f"; else echo 0; fi
}

guard_record_failure() {
    local f n
    f="$(_guard_file "$1")"
    n=$(guard_count "$1")
    echo $((n + 1)) > "$f"
}

guard_reset() {
    local f
    f="$(_guard_file "$1")"
    echo 0 > "$f"
}

# guard_may_resubmit <name> <max>  -> exit 0 if allowed, 1 if limit reached
guard_may_resubmit() {
    local n max
    n=$(guard_count "$1")
    max="${2:-3}"
    if [ "$n" -ge "$max" ]; then
        echo "GUARD: ${n} consecutive failures for '$1' (limit ${max}). Not resubmitting." >&2
        return 1
    fi
    return 0
}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_resubmit_guard.py -v`
Expected: 6 passed

- [ ] **Step 5: Wire the guard into slurm_pretrain.sh**

Add after the `#SBATCH` block:

```bash
source scripts/lib/resubmit_guard.sh
GUARD_NAME=syncguard_pretrain
```

Replace the body of `resubmit()` (lines 16-27) with:

```bash
resubmit() {
    if [ $RESUBMITTED -eq 0 ]; then
        RESUBMITTED=1
        if ! guard_may_resubmit "$GUARD_NAME" 3; then
            echo "Refusing to resubmit — see outputs/logs/.resubmit_count_${GUARD_NAME}"
            exit 1
        fi
        guard_record_failure "$GUARD_NAME"
        echo "Signal received — resubmitting... ($(date))"
        LATEST=$(ls -t outputs/checkpoints/pretrain_epoch_*.pt 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            sbatch --export=RESUME_CKPT="$LATEST" scripts/slurm_pretrain.sh
        else
            sbatch scripts/slurm_pretrain.sh
        fi
    fi
}
```

Then, at the very end of the script, after the training command exits successfully, add:

```bash
if [ $? -eq 0 ]; then guard_reset "$GUARD_NAME"; fi
```

- [ ] **Step 6: Author the rebuild configs**

Create `configs/rebuild_pretrain.yaml` and `configs/rebuild_finetune.yaml` by copying the shipped configs and adding the seed. Start from `configs/default.yaml` for pretraining and `configs/finetune_frozen.yaml` for fine-tuning:

```bash
cp configs/default.yaml configs/rebuild_pretrain.yaml
cp configs/finetune_frozen.yaml configs/rebuild_finetune.yaml
```

Add to the top level of **both** files:

```yaml
seed: 42
```

Verify these invariants in both, and fix them if absent — they are Global Constraints:

- `model.visual_encoder.name: av_hubert`
- **no** `model.visual_encoder.checkpoint_path` key (random init is intended)
- Wav2Vec backbone frozen during fine-tuning in `rebuild_finetune.yaml`

```bash
grep -n "seed\|checkpoint_path\|freeze" configs/rebuild_pretrain.yaml configs/rebuild_finetune.yaml
```

Expected: `seed: 42` in both, no `checkpoint_path` line in either.

- [ ] **Step 6b: Point the SLURM scripts at the rebuild configs**

Both launchers currently hardcode `configs/default.yaml`, so without this the
rebuild configs would be committed but never used, and each checkpoint's
`config_sha256` would hash the wrong recipe.

In `scripts/slurm_pretrain.sh:49`, change:

```
    --config configs/default.yaml \
```

to:

```
    --config configs/rebuild_pretrain.yaml \
```

In `scripts/slurm_finetune.sh:53`, change the same line to:

```
    --config configs/rebuild_finetune.yaml \
```

Verify:

```bash
grep -n "config configs/" scripts/slurm_pretrain.sh scripts/slurm_finetune.sh
```

Expected: `rebuild_pretrain.yaml` and `rebuild_finetune.yaml`, no `default.yaml`.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: 240 passed, 4 skipped

- [ ] **Step 8: Commit — this is the gate before any HPC job**

```bash
git add scripts/lib/resubmit_guard.sh tests/test_resubmit_guard.py scripts/slurm_pretrain.sh scripts/slurm_finetune.sh configs/rebuild_pretrain.yaml configs/rebuild_finetune.yaml
git commit -m "feat: resubmit crash-loop guard and committed rebuild configs"
git push
```

The push matters: Part B clones this repo onto scratch. Anything uncommitted now is a recipe that exists only on a laptop, which is the failure this rebuild exists to prevent.

---

# Part B — Pipeline execution on Explorer

All Part B work runs on HPC. Connect with `ssh explorer`. Every stage ends with its verification gate; do not start a stage whose predecessor's gate has not passed.

### Task 7: Environment validation and smoke test (Gate G0)

The `syncguard` conda env survived in `/home` (9.2 GB), so this validates rather than rebuilds.

**Files:** none modified.

- [ ] **Step 1: Clone the repo onto scratch**

```bash
ssh explorer
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone https://github.com/Akshay171124/SyncGuard.git
cd SyncGuard && git log --oneline -1
```

Expected: the resubmit-guard commit from Task 6.

- [ ] **Step 2: Activate and verify the environment**

```bash
module load miniconda3/24.11.1 FFmpeg/7.1.1
conda activate syncguard
python -c "import torch, transformers, wandb, mediapipe, librosa; print(torch.__version__)"
```

Expected: `2.5.1+cu121`, no ImportError.

- [ ] **Step 3: Confirm fairseq is absent — this is intended**

```bash
python -c "import fairseq" 2>&1 | tail -1
```

Expected: `ModuleNotFoundError: No module named 'fairseq'`. Do **not** install it. The rebuild baseline uses a randomly initialized AV-HuBERT; see spec section 12.1.

- [ ] **Step 4: Set the HuggingFace cache to scratch and pre-download Wav2Vec 2.0**

```bash
export HF_HOME=/scratch/$USER/.cache/huggingface
python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')"
```

Expected: downloads about 360 MB, then exits silently.

- [ ] **Step 5: Gate G0 — run the test suite**

```bash
python -m pytest tests/ -q
```

Expected: **240 passed, 4 skipped**. The 4 skips are mediapipe EAR tests. If anything fails, stop and fix before proceeding — no GPU time should be spent on a broken checkout.

- [ ] **Step 6: Verify W&B credentials**

```bash
python -c "import wandb; print(wandb.api.api_key is not None)"
```

Expected: `True`. If `False`, run `wandb login` before any training stage, or the runs will not be logged and provenance will record `wandb_run_id: none`.

---

### Task 8: Restore datasets (Gate G1)

Roughly 77 GB total: about 65 GB from Google Drive plus 12 GB of DFDC Part 0 from Kaggle.

**Files:**
- Create: `data/raw/{fakeavceleb,celebdf,lrs2,avspeech,dfdc}/`, `data/MANIFEST.json`

- [ ] **Step 1: Pull the Drive archives onto scratch**

The lab notebook records rclone as the transfer tool for Drive (`docs/lab_notebook.md:301`).

```bash
cd /scratch/$USER/SyncGuard && mkdir -p data/raw
rclone copy gdrive:SyncGuard_datasets/FakeAVCeleb_v1.2.zip data/raw/ -P
rclone copy "gdrive:SyncGuard_datasets/Celeb DF (v2).zip"  data/raw/ -P
rclone copy gdrive:SyncGuard_datasets/lrs2_v1.tar          data/raw/ -P
rclone copy gdrive:SyncGuard_datasets/avspeech             data/raw/avspeech -P
```

Adjust the remote paths to match your rclone remote. If rclone is not configured, run `rclone config` first and create a `gdrive` remote.

- [ ] **Step 2: Extract**

```bash
cd data/raw
unzip -q FakeAVCeleb_v1.2.zip -d fakeavceleb
unzip -q "Celeb DF (v2).zip"  -d celebdf
tar  -xf lrs2_v1.tar          -C .
```

- [ ] **Step 3: Pin protobuf BEFORE installing the Kaggle CLI**

This ordering is not optional. The Kaggle CLI pulls protobuf 7.x, which breaks mediapipe and cost a debugging cycle in March (`CHANGELOG.md:202`).

```bash
grep -q "protobuf" requirements.txt || echo "protobuf<5" >> requirements.txt
pip install -r requirements.txt
python -c "import google.protobuf as p; print(p.__version__)"
```

Expected: `4.25.8` or another 4.x version. Then install the CLI and re-verify:

```bash
pip install kaggle
python -c "import mediapipe; print('mediapipe OK')"
python -c "import google.protobuf as p; print(p.__version__)"
```

Expected: `mediapipe OK` and still a 4.x protobuf. If protobuf jumped to 7.x, run `pip install 'protobuf<5'` and re-verify mediapipe.

- [ ] **Step 4: Download DFDC Part 0**

```bash
mkdir -p data/raw/dfdc && cd data/raw/dfdc
kaggle competitions download -c deepfake-detection-challenge -f dfdc_train_part_0.zip
unzip -q dfdc_train_part_0.zip
ls *.mp4 | wc -l && ls metadata.json
```

Expected: roughly 1,300-1,400 mp4 files plus `metadata.json`, which supplies the labels. About 12 GB — not the full ~470 GB corpus.

If the transfer drops, resume rather than restart: `rsync --partial --append-verify` for any host-to-host copy.

- [ ] **Step 5: Gate G1 — verify counts**

```bash
cd /scratch/$USER/SyncGuard
for d in fakeavceleb celebdf lrs2 avspeech dfdc; do
  printf "%-14s %s\n" "$d" "$(find data/raw/$d -type f \( -name '*.mp4' -o -name '*.avi' \) 2>/dev/null | wc -l)"
done
```

Expected order of magnitude: fakeavceleb ~21,544; avspeech ~24,760; lrs2 ~96,000; dfdc ~1,343; celebdf several thousand.

Counts need not match exactly — record what you actually get. Exact parity is not an acceptance criterion, but an order-of-magnitude miss means an extraction failed.

- [ ] **Step 6: Write the dataset manifest**

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
m = {}
for d in ["fakeavceleb", "celebdf", "lrs2", "avspeech", "dfdc"]:
    files = sorted(p.name for p in Path(f"data/raw/{d}").rglob("*")
                   if p.suffix in {".mp4", ".avi"})
    m[d] = {"count": len(files),
            "sha256": hashlib.sha256("".join(files).encode()).hexdigest()}
Path("data/MANIFEST.json").write_text(json.dumps(m, indent=2))
print(json.dumps(m, indent=2))
PY
```

This hash goes into every checkpoint's provenance block via `save_checkpoint(..., dataset_manifest=...)`.

- [ ] **Step 7: Commit the manifest**

```bash
git add data/MANIFEST.json requirements.txt
git commit -m "data: record dataset manifest for rebuild"
git push
```

---

### Task 9: Preprocess from raw (Gate G2)

Roughly 6-10 hours of CPU on the `short` partition, which has a 2-day cap, so no resubmit logic is needed. Preprocessing from raw rather than reusing April's `.npy` features is deliberate: those were produced across mixed code versions, including before the `C=1` collation fix.

**Files:**
- Create: `data/processed/<dataset>/…`

- [ ] **Step 1: Submit preprocessing for each dataset**

```bash
cd /scratch/$USER/SyncGuard
sbatch scripts/slurm_preprocess_fakeavceleb.sh
sbatch scripts/slurm_preprocess_avspeech.sh
sbatch scripts/slurm_preprocess_lrs2.sh
sbatch scripts/slurm_preprocess_celebdf.sh
sbatch scripts/slurm_preprocess_dfdc.sh
squeue -u $USER
```

These are CPU jobs and run in parallel.

- [ ] **Step 2: Monitor**

```bash
tail -f outputs/logs/preprocess_*.out
```

Expect per-sample lines of the form `… → T=144 frames audio=16000Hz OK`, with some `SKIPPED (RetinaFace confidence < 0.8 …)`. Skips are normal.

- [ ] **Step 3: Gate G2 — verify output shape on a sample**

```bash
python - <<'PY'
import numpy as np, json
from pathlib import Path
s = next(Path("data/processed/fakeavceleb").rglob("mouth_crops.npy")).parent
crops = np.load(s / "mouth_crops.npy")
mask  = np.load(s / "speech_mask.npy")
meta  = json.loads((s / "metadata.json").read_text())
print("sample:", s)
print("crops:", crops.shape, crops.dtype)
print("mask :", mask.shape)
print("meta :", meta)
assert crops.ndim == 4, "expected (T, C, H, W)"
assert (s / "audio.wav").exists()
print("G2 PASS")
PY
```

Expected: a 4-D crops array, a matching mask, `audio.wav` present, and `G2 PASS`.

- [ ] **Step 4: Record processed counts**

```bash
for d in fakeavceleb avspeech lrs2 celebdf dfdc; do
  printf "%-14s %s\n" "$d" "$(find data/processed/$d -name metadata.json | wc -l)"
done
```

DFDC should land near **1,343** — the corrected-pipeline count, not the pre-fix 1,334.

---

### Task 10: Phase 1 contrastive pretraining (Gate G3)

About 8 hours on H200 against an 8-hour partition cap, so this run depends on per-epoch checkpointing and resume.

- [ ] **Step 1: Launch**

```bash
cd /scratch/$USER/SyncGuard
mkdir -p outputs/logs outputs/checkpoints
sbatch scripts/slurm_pretrain.sh
squeue -u $USER
```

Queue waits of 2-4 hours for H200 are normal.

- [ ] **Step 2: Gate G3 — check the first 100 steps before walking away**

```bash
head -100 outputs/logs/pretrain_*.out | grep -iE "loss|sync"
```

Two failure signatures to watch, both from the README's fresh-run checks:

- **Loss goes NaN** within the first 100 steps → the Wav2Vec backbone is unfrozen. Cancel, fix `freeze_backbone`, resubmit.
- **Sync-score saturates toward 1.0** → representation collapse, again from an unfrozen Wav2Vec during pretraining. Cancel and fix.

Cancel with `scancel <jobid>`. Do not let a run with either signature continue — it will burn eight GPU-hours and produce nothing.

- [ ] **Step 3: Confirm resume works across the cap**

When the job hits the time limit it resubmits itself with `RESUME_CKPT`. After the second job starts:

```bash
grep -i "resum" outputs/logs/pretrain_*.out | tail -5
cat outputs/logs/.resubmit_count_syncguard_pretrain
```

Expected: a resume line naming a `pretrain_epoch_*.pt`, and a guard count of 0 after an epoch completes. A count reaching 3 means a crash-loop was correctly stopped — investigate rather than resubmit.

- [ ] **Step 4: Verify completion and provenance**

```bash
python - <<'PY'
import torch
ck = torch.load("outputs/checkpoints/pretrain_best.pt",
                map_location="cpu", weights_only=False)
print("epoch:", ck["epoch"])
print("val_metrics:", ck["val_metrics"])
print("provenance:", ck["provenance"])
assert ck["provenance"]["git_sha"] != "unknown"
print("G3 PASS")
PY
```

For reference, April reached best metrics near epoch 17 with val InfoNCE 8.06 and sync-score 0.978. Your numbers will differ — the run is seeded differently and AVSpeech plus LRS2 may resolve to different sample counts. That is expected and is not a failure.

- [ ] **Step 5: Archive immediately**

```bash
mkdir -p /home/$USER/ckpt_archive
cp outputs/checkpoints/pretrain_best.pt /home/$USER/ckpt_archive/
ls -lh /home/$USER/ckpt_archive/
```

Do this now, not at the end. An unarchived checkpoint is exactly what was lost in July.

---

### Task 11: Phase 2 fine-tuning (Gate G4)

About 6 hours on H200, which fits inside one 8-hour job.

- [ ] **Step 1: Confirm the pretrained checkpoint is in place**

```bash
cd /scratch/$USER/SyncGuard
ls -lh outputs/checkpoints/pretrain_best.pt
```

- [ ] **Step 2: Launch**

```bash
sbatch scripts/slurm_finetune.sh
squeue -u $USER
```

- [ ] **Step 3: Gate G4 — watch val AUC over the first 3 epochs**

```bash
grep -iE "val_auc|epoch" outputs/logs/finetune_*.out | head -30
```

If val AUC sits at **0.5 across three consecutive epochs**, stop the run. That signature means speaker leakage between the train and val splits — the model is learning speaker identity instead of sync, and the metric is meaningless. Verify the speaker-disjoint split (`tests/test_dataset_loader.py` covers this) before resubmitting.

- [ ] **Step 4: Confirm the backbone stayed frozen**

```bash
grep -i "freeze\|frozen" outputs/logs/finetune_*.out | head
```

Expected: a line confirming the Wav2Vec backbone is frozen. Unfreezing it on a dataset this size causes catastrophic forgetting and is a Global Constraint.

- [ ] **Step 5: Verify and archive**

```bash
python - <<'PY'
import torch
ck = torch.load("outputs/checkpoints/finetune_best.pt",
                map_location="cpu", weights_only=False)
print("epoch:", ck["epoch"], "val:", ck["val_metrics"])
assert "provenance" in ck
print("G4 PASS")
PY
cp outputs/checkpoints/finetune_best.pt /home/$USER/ckpt_archive/
```

April reached val AUC 0.953 around epoch 17. Expect a similar magnitude, not the same number.

---

### Task 12: Audio classifier and cascade stages

Three shorter GPU jobs producing the remaining checkpoints.

- [ ] **Step 1: Train the audio classifier**

```bash
cd /scratch/$USER/SyncGuard
sbatch scripts/slurm_train_audio_clf.sh
```

- [ ] **Step 2: Train both cascade stages**

```bash
sbatch scripts/slurm_train_cross_attention.sh
```

One submission runs both stages: the script invokes `train_cross_attention.py --stage 1` (line 32), then `--stage 2` (line 42) using `CA_STAGE1` as the stage-1 checkpoint. `--stage` is a required argument restricted to `{1, 2}` (`scripts/train_cross_attention.py:329`).

This script already seeds (`scripts/train_cross_attention.py:145-150`), so no change is needed there.

- [ ] **Step 3: Confirm both stage checkpoints landed**

```bash
ls -lh outputs/checkpoints/ca_stage1_best.pt outputs/checkpoints/ca_stage2_best.pt
```

If stage 2 is missing, check the log — stage 2 only starts if stage 1 wrote its checkpoint.

- [ ] **Step 4: Verify all three carry provenance**

```bash
python - <<'PY'
import torch
from pathlib import Path
for name in ["audio_clf_best.pt", "ca_stage1_best.pt", "ca_stage2_best.pt"]:
    p = Path("outputs/checkpoints") / name
    if not p.exists():
        print(f"MISSING: {name}")
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print(name, "provenance:", "provenance" in ck)
PY
```

`scripts/train_audio_classifier.py:298` and `scripts/train_cross_attention.py:286` call `torch.save` directly rather than going through the shared saver. If either checkpoint lacks a provenance block, switch that call to `src.utils.checkpoint.save_checkpoint`, re-run that stage, and commit the change.

- [ ] **Step 5: Archive**

```bash
cp outputs/checkpoints/{audio_clf_best,ca_stage1_best,ca_stage2_best}.pt /home/$USER/ckpt_archive/
ls -lh /home/$USER/ckpt_archive/
```

Expected: five checkpoints.

---

### Task 13: Evaluation suite (Gate G5)

Minutes, not hours.

- [ ] **Step 1: FakeAVCeleb**

```bash
cd /scratch/$USER/SyncGuard
python scripts/evaluate.py --config configs/rebuild_finetune.yaml \
    --checkpoint outputs/checkpoints/finetune_best.pt --test_set fakeavceleb
```

- [ ] **Step 2: Gate G5 — check the acceptance bar**

```bash
python -c "
import json; r = json.load(open('outputs/logs/eval_fakeavceleb.json'))
auc = r.get('auc_roc') or r.get('AUC-ROC')
print('FakeAVCeleb AUC:', auc)
print('G5 PASS' if auc >= 0.90 else 'G5 FAIL — investigate, do not proceed')
"
```

**>= 0.90 passes.** April measured 0.9628. Materially below 0.90 is a genuine regression, not run-to-run variance — stop and diagnose.

- [ ] **Step 3: CelebDF-v2 zero-shot**

```bash
python scripts/evaluate.py --config configs/rebuild_finetune.yaml \
    --checkpoint outputs/checkpoints/finetune_best.pt --test_set celebdf
```

- [ ] **Step 4: DFDC zero-shot**

```bash
python scripts/evaluate.py --config configs/rebuild_finetune.yaml \
    --checkpoint outputs/checkpoints/finetune_best.pt --test_set dfdc
```

**An AUC of 0.50-0.60 here is a PASS.** April measured 0.5263. Near-chance DFDC transfer is a documented finding, not a bug: DFDC face-swaps preserve lip motion, so a sync-based signal has little to detect. Do not "fix" this. If it instead comes out unexpectedly high, that is the surprising result and deserves scrutiny.

- [ ] **Step 5: Generate the plots**

Use the existing helpers in `src/evaluation/visualize.py` — do not write new plotting code. Available functions:

| Function | Figure |
|---|---|
| `plot_roc_curve` | `roc_fakeavceleb.png` |
| `plot_roc_multi_dataset` | `roc_cross_dataset.png` |
| `plot_roc_per_category` | FakeAVCeleb per-category ROC |
| `plot_sync_score_curves` | `sync_score_real_vs_fake.png` (the headline figure) |
| `plot_sync_score_distribution` | `sync_score_distribution.png` |
| `plot_training_curves` | `training_loss_pretrain.png`, `training_loss_finetune.png` |
| `plot_ablation_bar` | the three ablation charts |
| `plot_per_category_auc` | `per_category_auc.png` |

`_save_fig` (line 58) already writes both a 300 DPI PNG and a vector PDF for every figure, matching the Plotting Standards in `.claude/CLAUDE.md`. Colours are set in the module's rcParams block — real `#27AE60`, fake `#E74C3C`.

Drive these from the `eval_*.json` files produced in steps 1-4 and the per-epoch metrics in `outputs/logs/pretrain.json` and `outputs/logs/finetune.json`. Write into `outputs/visualizations/`.

- [ ] **Step 6: Commit results**

Metrics JSON and plots are tracked; checkpoints are not.

```bash
git add outputs/logs/eval_*.json outputs/visualizations/
git commit -m "eval: rebuild results on FakeAVCeleb, CelebDF-v2, DFDC"
git push
```

---

### Task 14: Archive and manifest (Gate G6)

- [ ] **Step 1: Write the archive script**

```bash
# scripts/archive_checkpoints.sh
#!/bin/bash
# Copy every checkpoint to durable /home storage and record its provenance.
# Scratch is purged after 28 days; /home is not.
set -euo pipefail

ARCHIVE="/home/$USER/ckpt_archive"
mkdir -p "$ARCHIVE"

for ckpt in outputs/checkpoints/*_best.pt; do
    [ -e "$ckpt" ] || continue
    cp -v "$ckpt" "$ARCHIVE/"
done

python - <<'PY'
import os, torch
from pathlib import Path
rows = []
for p in sorted(Path(os.path.expanduser("~/ckpt_archive")).glob("*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    pr = ck.get("provenance", {})
    rows.append(f"| `{p.name}` | {p.stat().st_size/1e6:.0f} MB | "
                f"{pr.get('git_sha','unknown')[:8]} | "
                f"{pr.get('config_sha256','unknown')[:8]} | "
                f"{pr.get('wandb_run_id','none')} | {pr.get('timestamp','unknown')} |")
Path("MANIFEST.md").write_text(
    "# Checkpoint Manifest\n\n"
    "Generated by `scripts/archive_checkpoints.sh`. Archive lives at "
    "`/home/$USER/ckpt_archive/`.\n\n"
    "| Checkpoint | Size | Git SHA | Config hash | W&B run | Saved |\n"
    "|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
print(Path("MANIFEST.md").read_text())
PY
```

- [ ] **Step 2: Run it**

```bash
chmod +x scripts/archive_checkpoints.sh
./scripts/archive_checkpoints.sh
```

- [ ] **Step 3: Gate G6 — verify all five are archived with provenance**

```bash
ls -1 /home/$USER/ckpt_archive/*.pt | wc -l
grep -c "^| \`" MANIFEST.md
grep -c "unknown" MANIFEST.md
```

Expected: **5** checkpoints, **5** manifest rows, and **0** occurrences of `unknown`. Any `unknown` means a checkpoint was saved outside the shared saver — fix that stage before considering the rebuild complete.

- [ ] **Step 4: Commit**

```bash
git add scripts/archive_checkpoints.sh MANIFEST.md
git commit -m "feat: checkpoint archive script and manifest"
git push
```

---

### Task 15: Documentation corrections

Two documented claims do not match what the code did. Both must be corrected. Neither is cosmetic — each asserts a capability that was never exercised, and both would survive into a submission.

**Files:**
- Modify: `docs/EXECUTION_PLAN.md:143`, `README.md`, `.claude/CLAUDE.md`, `docs/Final_Project_Proposal.md:70`, `docs/OPERATIONS.md`

- [ ] **Step 1: Correct the AV-HuBERT claim**

`docs/EXECUTION_PLAN.md:143` currently reads "Load AV-HuBERT visual frontend (pretrained lip-reading weights from fairseq)". Replace with:

```
- Build AV-HuBERT visual frontend (architecture only — randomly initialized;
  pretrained fairseq weights are NOT loaded, see
  docs/superpowers/specs/2026-09-09-clean-pipeline-rebuild-design.md §12.1)
```

Apply the same correction wherever `README.md` or `.claude/CLAUDE.md` implies pretrained visual weights are in use.

- [ ] **Step 2: Correct the Wav2Lip claim**

`docs/Final_Project_Proposal.md:70` lists a self-generated ~500-clip Wav2Lip adversarial test set. It was never generated — no logs or results exist. Mark it planned-but-not-built:

```
| **Wav2Lip self-generated** | Adversarial Test (PLANNED — NOT BUILT) | ~500 clips | — | Not generated; see spec §12.2 |
```

Remove or annotate the `--test_set wavlip_adversarial` example in `docs/OPERATIONS.md` so it is not presented as a working evaluation path.

- [ ] **Step 3: Correct the partition time limit**

`.claude/CLAUDE.md` states the `gpu` partition allows 24 hours. `sinfo` reports **8:00:00**, and the lab notebook recorded this on 2026-03-19. Update the GPU Partitions table: `gpu` and `gpu-short` both cap at 8 hours.

- [ ] **Step 4: Append the closing lab notebook entry**

Add a dated entry recording actual dataset counts, the metrics each stage produced, how they compare to April, and any deviations encountered. Follow the existing entry format: `## 2026-XX-XX — Title`, then Owner, Phase, What I Did, Results, Observations, Decision, Artifacts.

- [ ] **Step 5: Update CHANGELOG.md**

Add a version entry summarising the rebuild: seeded runs, provenance-carrying checkpoints, durable archive, and the two documentation corrections.

- [ ] **Step 6: Commit**

```bash
git add docs/ README.md .claude/CLAUDE.md CHANGELOG.md
git commit -m "docs: correct AV-HuBERT and Wav2Lip claims, fix partition limit"
git push
```

---

## Completion criteria

The rebuild is done when all of these hold:

- [ ] Five checkpoints exist in `/home/$USER/ckpt_archive/`, each with a populated provenance block.
- [ ] `MANIFEST.md` has five rows and zero `unknown` values.
- [ ] FakeAVCeleb test AUC >= 0.90.
- [ ] DFDC AUC recorded, with 0.50-0.60 understood as the expected result.
- [ ] Every config used is committed and pushed.
- [ ] `april_reference/` survivors are intact and backed up off-machine.
- [ ] Both documentation corrections are committed.
- [ ] The lab notebook records actual counts and metrics.
