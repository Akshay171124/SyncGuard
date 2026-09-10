# SyncGuard Clean Pipeline Rebuild — Design

**Date:** 2026-09-09
**Owner:** Akshay (solo rebuild)
**Status:** Design approved. Eval scope resolved: full suite including DFDC.
All open questions closed.

## 1. Context

On or around 2026-07-13, `/scratch/prajapati.aksh` was emptied by Explorer's
28-day scratch purge. Scratch held the entire project: raw datasets,
preprocessed features, checkpoints, and configs. `/home` was unaffected but
never held SyncGuard artifacts.

Confirmed lost:

- `pretrain_best.pt`, `ca_stage1_best.pt`, `ca_stage2_best.pt`,
  `finetune_best_run3_audioswap.pt`, all per-epoch checkpoints
- All preprocessed `.npy` features
- `configs/pretrain_frozen.yaml` and `configs/finetune_v4_best.yaml` — the
  recipes the README credits with the headline numbers. Neither was ever
  tracked in git; they existed only on scratch.

Confirmed surviving, on the local Mac only, verified as intact zip archives
with clean CRCs on 2026-09-09:

- `demo_assets/checkpoints/finetune_best.pt` (548 MB)
- `demo_assets/checkpoints/audio_clf_best.pt` (383 MB)

Two compounding irregularities make partial restoration unattractive.

First, the configs that produced the April results are gone.

Second, seeding was incomplete. Correcting an earlier assessment in this
document's drafting: pretrain and finetune were **not** unseeded.
`src/training/pretrain.py:220` and `src/training/finetune.py:337` each carried
a `# CB-5` block seeding python, numpy, and torch from `config.get("seed", 42)`.
But both blocks sit inside `train()`, which receives `train_loader` as an
argument — so the dataloaders, and their shuffle and worker RNG state, were
already constructed before any seed was set. Neither block set
`PYTHONHASHSEED` or cuDNN determinism. The runs were partially and belatedly
seeded, not reproducible.

Either way the conclusion holds: with the configs gone, April's artifacts
cannot be regenerated, only replaced by different ones with no documented
lineage.

**Decision:** full clean rebuild. Exact numeric replication is explicitly not
a goal.

## 2. Goals and non-goals

**Goals**

- Restore the complete artifact set so research can resume at any pipeline stage.
- Establish one coherent provenance chain: every checkpoint traceable to a
  commit, a config, and a W&B run.
- Make recurrence structurally impossible.

**Non-goals**

- Reproducing April's numbers exactly.
- New research directions (CLIP+SBI, closing the DFDC gap). This rebuild is
  the platform those depend on, not the work itself.
- Loading pretrained AV-HuBERT weights. See section 12; deferred to section 13.
- Generating the Wav2Lip adversarial set. It was never built, so there is
  nothing to restore. Deferred to section 13.

The governing principle: **the rebuild restores, it does not change the
science.** Anything that would alter results is quarantined into the research
phase that follows, so that if rebuilt numbers differ from April the cause is
attributable.

## 3. What "clean" means

| Irregularity | Fix |
|---|---|
| April configs never git-tracked | All configs committed before any run launches |
| Seeding ran after dataloader construction and omitted `PYTHONHASHSEED` and cuDNN determinism | One `seed_everything()` called before any model or dataloader is built; `seed` key in every config |
| No link from checkpoint to recipe | Provenance block embedded in each checkpoint state dict |
| Checkpoints lived only on scratch | Auto-archive to `/home/prajapati.aksh/ckpt_archive/` at run end |
| Auto-resubmit burned 40 failed jobs (2026-04-07) | Resubmit loop checks exit status, aborts after 3 consecutive identical failures |
| AV-HuBERT silently ran randomly initialized (section 12) | Warn loudly when `av_hubert` is selected with no `checkpoint_path` |

Every checkpoint gains a provenance block:

```
provenance = {
    "git_sha":          <commit at launch>,
    "config_sha256":    <hash of resolved config>,
    "wandb_run_id":     <run id>,
    "dataset_manifest": <hash of data manifest>,
    "timestamp":        <UTC ISO8601>,
}
```

This makes artifacts self-describing, so provenance survives files being
moved, renamed, or recovered from backup.

## 4. Storage architecture

| Content | Location | Rationale |
|---|---|---|
| Code | GitHub, cloned to `/scratch/$USER/SyncGuard` | Fast local reads; git is the durable copy |
| Datasets (~77 GB) | `/scratch/$USER/SyncGuard/data` | Large, purgeable; re-downloadable from Drive and Kaggle |
| Checkpoints | Written to scratch, auto-copied to `/home/prajapati.aksh/ckpt_archive/` | Survives purge |
| April survivors | `april_reference/`, plus off-machine backup | Only artifacts tied to the April report |
| Configs, logs, metrics JSON | Git-tracked | Recipes must never live only on scratch again |
| `MANIFEST.md` | Git-tracked | Maps each checkpoint to its provenance |

`/home` holds 59 GB with no enforced per-user quota. Six checkpoints is
roughly 3 GB, comfortably affordable. Datasets stay on scratch.

`/projects/cvpr/` exists but belongs to another group (`dino.k`, group `cvpr`);
this account is in `users` and `Forge` only, so it is unavailable.

**Ordering constraint:** archive the two survivors under `april_reference/`
*before* any training job launches. Training writes to
`outputs/checkpoints/finetune_best.pt` by default — the same name the
survivor holds — so an early launch would silently destroy the only copy of
the April result.

## 5. Artifacts produced

Checkpoints: `pretrain_best.pt`, `finetune_best.pt`, `audio_clf_best.pt`,
`ca_stage1_best.pt`, `ca_stage2_best.pt`.

Evaluation: `eval_fakeavceleb.json`, `eval_celebdf.json`, `eval_dfdc.json`.

Plots: the ten figures listed under Plotting Standards in `.claude/CLAUDE.md`.

## 6. Pipeline stages

| # | Stage | Compute | Estimate |
|---|---|---|---|
| 0 | Environment validation, smoke test | none | 15 min |
| 1 | Restore data: Drive (~65 GB) + DFDC from Kaggle (12 GB) | none | 2–3 h |
| 2 | Preprocess from raw | CPU (`short`, 2-day cap) | 6–10 h |
| 3 | Phase 1 contrastive pretraining | H200 | ~8 h, resume required |
| 4 | Phase 2 fine-tuning | H200 | ~6 h |
| 5 | Audio classifier | H200 | short |
| 6 | Cascade stages 1–2 | H200 | short |
| 7 | Evaluation suite and plots | H200 | ~15 min |
| 8 | Archive, manifest, doc corrections | none | 45 min |

Expect 2–4 days wall-clock including H200 queue waits of 2–4 hours per job.
Executed solo; no stage depends on another person.

**The `syncguard` conda environment survived.** It lives in `/home`, not
scratch: 9.2 GB, with torch 2.5.1+cu121, torchaudio, torchvision, transformers,
wandb, mediapipe 0.10.33, retina-face, librosa, soundfile, and opencv present.
Stage 0 validates rather than rebuilds it. `fairseq` is absent — deliberately
left so; see section 12.

**Preprocess from raw rather than reusing April's features.** The cost is CPU
time on an uncapped partition. The April `.npy` features were produced across
mixed code versions: the `C=1` collation bug was not fixed until v3.3.0, and
DFDC was separately reprocessed into `dfdc_pre_fix_backup`. Reusing them would
reimport the irregularity this rebuild exists to remove.

**The 8-hour partition cap is the scheduling constraint.** `sinfo` confirms
both `gpu` and `gpu-short` cap at `8:00:00`, and pretraining takes about eight
hours. The lab notebook recorded this on 2026-03-19; `.claude/CLAUDE.md` still
claims 24 h and must be corrected as part of this work. Mitigation is
per-epoch checkpointing with auto-resubmit and resume, guarded by the exit
status check from section 3.

## 7. Verification gates

Each gate must pass before the next stage begins, so failures surface in
minutes rather than after eight GPU-hours.

| Gate | Check | Pass condition |
|---|---|---|
| G0 | `pytest tests/` | 215 passed, 4 skipped |
| G1 | Dataset counts | FakeAVCeleb 21,544; AVSpeech 24,760; LRS2 ~96K; DFDC 1,343 |
| G2 | Preprocessing output | Sample has `T` frames, 16 kHz audio, `speech_mask` present |
| G3 | Pretrain, first 100 steps | Loss not NaN; sync-score not saturating toward 1.0 |
| G4 | Fine-tune, first 3 epochs | Val AUC not pinned at 0.5 |
| G5 | Final evaluation | FakeAVCeleb test AUC >= 0.90 |
| G6 | Each checkpoint | Archived to `/home`, provenance block present |

G3 and G4 are taken from the README's own fresh-run sanity checks. A saturating
sync-score indicates an unfrozen Wav2Vec backbone; a val AUC stuck at 0.5
indicates speaker leakage across the train/val split.

## 8. Acceptance criteria

- FakeAVCeleb test AUC >= 0.90. April reached 0.9628; materially below 0.90 is
  a regression to investigate, not run-to-run variance.
- **DFDC in the 0.50–0.60 band is a success, not a failure.** April measured
  0.5263, and the README documents near-chance DFDC transfer as a genuine
  finding: sync-based detection loses to face-swap generators that preserve lip
  motion. A rebuild landing near chance has reproduced the known limitation
  correctly and must not be read as broken.
- All five checkpoints archived to `/home` with populated provenance blocks.
- Any stage can be relaunched from a committed config plus a git SHA.

## 9. Deviations from April, recorded deliberately

- Configs are authored fresh. The originals are unrecoverable, so this rebuild
  defines a new documented baseline rather than claiming continuity.
- Seeding is complete and correctly ordered. April seeded partially and after
  dataloader construction (see section 1).
- Pretraining corpus is AVSpeech plus LRS2, both confirmed present on Drive.
- The AV-HuBERT visual encoder remains randomly initialized, matching what
  April actually ran rather than what its documentation claimed. This is a
  deliberate choice to keep the rebuild attributable; see sections 12 and 13.
- Evaluation covers FakeAVCeleb, CelebDF-v2, and DFDC. DFDC is re-downloaded
  from Kaggle; see section 11.
- **Wav2Vec is frozen during pretraining.** `configs/rebuild_pretrain.yaml` sets
  `audio_encoder.freeze_pretrained: true`. The config was initially copied from
  `default.yaml`, which leaves it unfrozen — the state section 7's G3 gate
  defines as a failure (representation collapse, saturating sync-score). The
  lost April recipe was named `pretrain_frozen.yaml`, so frozen is also the
  better guess at what April actually ran. Decided by the project owner
  2026-09-10.
- **W&B runs offline.** All four training launchers export
  `WANDB_MODE=offline`. `wandb.init()` is unguarded in all four training stages
  (`src/training/pretrain.py:184`, `src/training/finetune.py:347`,
  `scripts/train_audio_classifier.py:207`,
  `scripts/train_cross_attention.py:198`), so an invalid account would crash
  every stage at authentication and burn H200 queue cycles. No gate depends on
  W&B: per-epoch metrics are written independently to
  `outputs/logs/pretrain.json` and `outputs/logs/finetune.json`, which is what
  G3 and G4 read. Offline runs still receive run IDs, so checkpoint provenance
  stays populated, and `wandb sync` can upload retroactively.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Pretraining (~8 h) meets an 8 h cap | Per-epoch checkpointing, resume, guarded auto-resubmit |
| H200 queue waits of 2–4 h | Batch submissions; treat wall-clock as days, not hours |
| fairseq / AV-HuBERT version conflict (pitfall #1) | Verify import before submitting GPU jobs |
| Resubmit crash-loop | Abort after 3 consecutive identical failures |
| RetinaFace skip rate shifts dataset size vs April | Record actual counts in the manifest; do not assume parity |
| DFDC re-download / protobuf-mediapipe conflict | Pin `protobuf<5` before installing Kaggle CLI; see section 11 |

## 11. DFDC acquisition

DFDC was never on Google Drive and never resident on HPC. It was downloaded
from Kaggle to a local machine and rsynced to Explorer
(`docs/lab_notebook.md:726`, `docs/EXECUTION_PLAN.md:133`). Its absence from
the 2026-09-09 Drive listing is therefore expected, not a second loss.

Re-acquisition path:

1. Pin `protobuf<5` (4.25.8) **before** installing the Kaggle CLI. The CLI
   pulls protobuf 7.x, which breaks mediapipe (`CHANGELOG.md:202`).
2. Download DFDC **Part 0** only — about 12 GB, not the full ~470 GB corpus.
3. Transfer to scratch with `rsync --partial --append-verify`. The original
   transfer needed several retries after connection drops.
4. Preprocess with the corrected pipeline (fps fix, label fix, resolution
   normalization).

**Expected count: 1,343 samples.** The earlier figure of 1,334 is the
pre-fix March run; the corrected pipeline recovered nine clips that had
previously failed (`docs/EXECUTION_PLAN.md:51`). 1,343 is the count behind the
0.5263 headline and is the number G1 should assert.

These are Part 0 **training** clips labeled from `metadata.json`, not DFDC's
official test partition. The README describes them as a test set, which
reflects their role rather than their provenance. Zero-shot evaluation remains
valid because the model never trains on DFDC, but the distinction should be
stated wherever the result is reported.

Because DFDC is re-obtainable at modest cost, the full evaluation suite —
FakeAVCeleb, CelebDF-v2, and DFDC — is in scope. Section 9's conditional is
resolved.

## 12. Documentation corrections required

Two claims in the current documentation do not match what the code did. Both
must be corrected as part of stage 8. Neither is cosmetic: each asserts a
capability that was never exercised.

### 12.1 AV-HuBERT ran randomly initialized

`docs/EXECUTION_PLAN.md:143` states the pipeline loads "AV-HuBERT visual
frontend (pretrained lip-reading weights from fairseq)." It did not. Two
independent confirmations:

- `src/models/visual_encoder.py:333` reads
  `ckpt = ve_cfg.get("checkpoint_path")` and guards the load behind `if ckpt:`.
  No config defines `checkpoint_path`, so the value is `None` and
  `load_av_hubert_weights()` never executes.
- `fairseq>=0.12.0` is declared in `requirements.txt` but is not installed in
  the surviving `syncguard` environment. Had a path been set, the
  `import fairseq` inside the loader would have raised `ImportError`.

With `freeze_pretrained: false`, the encoder used the AV-HuBERT architecture
and none of its weights. The failure was silent because `.get()` returning
`None` into a truthiness check produces no exception, no warning, and no log
line — the pipeline behaves identically whether pretrained weights were
configured or forgotten.

Actions:

1. Correct `docs/EXECUTION_PLAN.md`, `README.md`, and `.claude/CLAUDE.md` to
   state that the AV-HuBERT *architecture* is used without pretrained weights.
2. Emit a loud warning when `name == "av_hubert"` and no `checkpoint_path` is
   set, so the condition cannot recur silently.
3. Keep random initialization for this rebuild. Loading real weights is a
   research change and belongs in section 13.

### 12.2 The Wav2Lip adversarial set was never generated

`docs/Final_Project_Proposal.md:70`, `.claude/CLAUDE.md`, and
`docs/OPERATIONS.md` (`--test_set wavlip_adversarial`) present a ~500-clip
self-generated Wav2Lip adversarial test set as part of the evaluation. No
generation log, result, or metric for it appears anywhere in
`docs/lab_notebook.md` or `CHANGELOG.md`. It was planned
(`docs/EXECUTION_PLAN.md:252`, assigned 2026-03-19 to 03-21) and not delivered.

Action: correct the documentation to describe it as planned and not built.
Presenting an adversarial test set that was never generated would not survive
review.

## 13. Deferred follow-ups

Ranked, to begin once the baseline is restored and verified.

1. **Load pretrained AV-HuBERT weights.** Requires installing `fairseq`
   (pitfall #1: 0.12.x conflicts with recent transformers and does not build
   cleanly against torch 2.5.1) and obtaining a checkpoint. Highest expected
   value: a from-scratch visual encoder is likely underperforming, so this is
   better motivated than the CLIP+SBI direction it would replace. Also
   reframes the DFDC result — a randomly initialized visual stream is a
   weaker explanation for near-chance transfer than the architectural claim.
2. **Generate the Wav2Lip adversarial set** (~500 clips). Directly tests the
   project's central claim about sync-optimized fakes. Requires the Wav2Lip
   repository, its checkpoints, GPU time, and a new evaluation loader.
3. **DFDC generalization work.** The 0.72 target; depends on 1 and 2 for a
   clean read on whether the limitation is architectural or an artifact of
   the under-initialized encoder.
