# Multi-Agent Review Pipeline for SyncGuard

**Date:** 2026-03-28
**Author:** Akshay Prajapati + Claude
**Status:** Approved

## Goal

Comprehensive project review of SyncGuard before the final push to the April 13, 2026 deadline. Three objectives:

1. **Quality gate** — Find bugs, architectural issues, or experiment flaws that could derail remaining work (Phase 1 v3 CMP pretraining, Phase 2 v2 fine-tuning with EAR, DFDC generalization)
2. **Cross-dataset generalization** — Critically examine the DFDC gap (0.5712 → target 0.72) and validate/challenge the planned fix (CMP + EAR), propose alternatives
3. **Educational audit** — Full review for learning purposes across code, architecture, experiment design, statistics, and reproducibility

## Constraints

- Code-review only (no HPC access from this machine)
- Generate ready-to-run verification scripts for anything requiring GPU/data
- Run lightweight local checks where possible (smoke tests, config validation)

## Architecture: Parallel Specialist Swarm

Two phases: 7 parallel specialist agents, then 1 synthesis agent.

### Phase 1: Parallel Review (7 Agents)

All 7 agents launch simultaneously in a single message.

#### Agent 1: Code Quality Reviewer
- **Plugin:** `pr-review-toolkit:code-reviewer`
- **Scope:** All 38 Python source files in `src/` and `scripts/`
- **Focus:**
  - Training loops (`pretrain.py`, `finetune.py`) — gradient flow, loss computation, checkpoint save/load
  - Loss functions (`losses.py`) — numerical stability, InfoNCE with learnable tau, MoCo queue updates, temperature clamping
  - Data pipeline (`dataset.py`, `pipeline.py`) — speaker leakage, collation with variable-length sequences, hard negative mining
  - Model forward passes (`syncguard.py`, `visual_encoder.py`, `audio_encoder.py`) — shape mismatches, device consistency, frozen/unfrozen parameters
- **Output:** Prioritized bug list (critical / warning / style)

#### Agent 2: Architecture Analyzer
- **Plugin:** `feature-dev:code-explorer`
- **Scope:** Full execution path from raw data to evaluation output
- **Focus:**
  - Temporal alignment consistency (25fps video → 49Hz audio) across all stages
  - Model composition in `syncguard.py` — projection head dimensions, encoder output compatibility
  - Config propagation — does `default.yaml` reach every component, or are there hardcoded divergences?
  - Modularity — can you swap encoders/classifiers without breaking the pipeline?
- **Output:** Architecture dependency map + consistency issues

#### Agent 3: Experiment Design Auditor
- **Plugin:** `general-purpose`
- **Scope:** Experiment methodology, the 15-experiment plan, completed runs, planned runs
- **Focus:**
  - Verify speaker-disjoint train/val/test splits in code (not just docs)
  - Evaluation protocol fairness — same preprocessing for all datasets?
  - Ablation control — one variable at a time?
  - Phase 1 → Phase 2 transfer — encoder loading, layer freezing correctness
  - Missing baselines — random classifier, majority class, raw Wav2Vec cosine similarity
- **Files:** `src/training/dataset.py`, `src/preprocessing/dataset_loader.py`, `scripts/train_pretrain.py`, `scripts/train_finetune.py`, `configs/default.yaml`, `docs/EXECUTION_PLAN.md`, `outputs/logs/experiment_summary.md`
- **Output:** Methodology issues + missing experiments + plan corrections

#### Agent 4: Results & Statistics Validator
- **Plugin:** `general-purpose`
- **Scope:** All reported metrics and their computation
- **Focus:**
  - AUC/EER/pAUC implementation correctness in `metrics.py` (edge cases: all-same-class, tiny test sets)
  - DFDC sample size (n=1,334) — are confidence intervals reported? Is n sufficient for reliable AUC?
  - Per-category sample sizes — is FV-FA 0.9902 AUC on enough samples?
  - Cascade fusion implementation correctness in `evaluate_cascade.py`
  - Data leakage — could preprocessing leak label information?
- **Files:** `src/evaluation/metrics.py`, `scripts/evaluate_cascade.py`, `src/evaluation/evaluate.py`, `outputs/logs/eval_*.json`
- **Output:** Statistical validity report + corrected metrics if needed + verification scripts

#### Agent 5a: DFDC Preprocessing Parity Checker
- **Plugin:** `feature-dev:code-explorer`
- **Scope:** Preprocessing code paths for DFDC vs FakeAVCeleb
- **Focus:**
  - Trace both datasets through `dataset_loader.py` → `pipeline.py` → `face_detector.py` / `audio_extractor.py`
  - Compare: face detector thresholds, crop sizes, audio extraction params, VAD settings, temporal alignment
  - Check if DFDC videos have different properties (resolution, fps, codec) that could cause silent preprocessing differences
  - Verify metadata.json structure matches between datasets
- **Output:** Concrete preprocessing discrepancies between DFDC and FakeAVCeleb

#### Agent 5b: DFDC Gap Hypothesis Challenger
- **Plugin:** `general-purpose`
- **Scope:** The DFDC generalization failure and planned fix
- **Focus:**
  - **Challenge CMP hypothesis:** DFDC face-swaps preserve lip-sync — will CMP pretraining help if the signal isn't in AV correspondence?
  - **Challenge EAR hypothesis:** Do DFDC face-swaps produce detectable blink artifacts, or are they high-quality enough to evade EAR?
  - **Distribution shift:** What are the fundamental differences between DFDC and FakeAVCeleb fakes?
  - **Alternative approaches:** Domain adaptation, test-time augmentation, feature normalization, threshold calibration, ensemble methods
  - **Literature review:** Recent DFDC benchmark results and methods that achieved >0.72 AUC
  - **Feasibility filter:** Rank alternatives by what's achievable in 16 days
- **Output:** Hypothesis critique + ranked alternative strategies + suggested experiments + verification scripts

#### Agent 6: Silent Failure Hunter
- **Plugin:** `pr-review-toolkit:silent-failure-hunter`
- **Scope:** All training, evaluation, and data loading code
- **Focus:**
  - NaN loss propagation in training loops
  - Corrupted sample handling in DataLoader
  - Partial checkpoint saves (SLURM timeout mid-write) and resume behavior
  - Wav2Vec frozen mode group normalization NaN fix completeness
  - MoCo queue device/staleness issues
  - Silero VAD failure modes (returns empty speech mask → what happens downstream?)
- **Output:** Silent failure catalog with reproduction conditions and fixes

### Phase 2: Synthesis (1 Agent)

Launches after all 7 Phase 1 agents complete.

#### Agent 7: Synthesis & Action Plan Generator
- **Plugin:** `general-purpose`
- **Input:** All 7 agent reports
- **Output:**
  1. **Critical Blockers** — Must fix before Phase 1 v3
  2. **High-Priority Fixes** — Affect result validity
  3. **DFDC Strategy Recommendation** — Concrete plan for 0.57 → 0.72
  4. **Updated Experiment Plan** — Revised sequence for remaining 16 days
  5. **HPC Verification Scripts** — Ready-to-run Python/SLURM scripts
  6. **Educational Audit Report** — Full findings by category
- **Written to:** `docs/superpowers/specs/review-findings.md`

## Execution

1. Launch all 7 Phase 1 agents in a single message (true parallel)
2. Collect findings as each completes
3. Launch synthesis agent with all 7 reports
4. Write final output to `docs/superpowers/specs/review-findings.md`

## Scope

- ~38 Python source files
- ~30 scripts
- 6 documentation files
- 15 experiments (completed + planned)
- 5 datasets
- 1 config file
- DFDC hypothesis stress-tested from code and research angles
