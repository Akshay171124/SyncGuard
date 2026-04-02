# SyncGuard Multi-Agent Review Synthesis

**Date:** 2026-03-28
**Reviewers:** 7 specialist agents (code quality, architecture, experiment design, statistics, DFDC preprocessing, DFDC hypothesis, silent failures)
**Status:** In-domain AUC 0.9458 (FakeAVCeleb), cross-dataset AUC 0.5712 (DFDC). Target: >= 0.72 DFDC AUC.
**Deadline:** April 13, 2026 (16 days remaining)
**Team:** Akshay (visual encoder, integration lead), Ritik (audio encoder, pretraining), Atharva (classifier, assessment)

---

## 1. Critical Blockers -- Must Fix Before Phase 1 v3 Pretraining

These bugs **invalidate current pretraining results** and must be resolved before launching any new pretraining run.

### CB-1: InfoNCE in-batch fallback uses wrong labels
- **Files:** `losses.py:151-158`
- **Impact:** When MoCo queue is empty (first batch, after resume), positive is assigned to column 0 for all rows. Only row 0 learns correctly; all other rows receive wrong gradients.
- **Fix:** Replace `labels = torch.zeros(N)` with `labels = torch.arange(logits.shape[0], device=logits.device)`
- **Owner:** Ritik
- **Time:** 15 minutes

### CB-2: MoCo queue not saved/restored in checkpoints
- **Files:** `losses.py:10-73`
- **Impact:** MoCoQueue is a plain Python class, not `nn.Module`. Queue, pointer, and `full` flag are not in `state_dict()`. Every resume reinitializes to random noise AND triggers the broken in-batch fallback (CB-1). This means **every resumed pretraining run has been corrupted**.
- **Fix:** Convert MoCoQueue to `nn.Module` using `register_buffer` for queue, ptr, and full flag.
- **Owner:** Ritik
- **Time:** 30 minutes

### CB-3: Validation pollutes MoCo queue with validation embeddings
- **Files:** `pretrain.py:115-135`
- **Impact:** `validate()` calls `criterion()` which enqueues validation embeddings. Training resumes with mixed train/validation embeddings in the queue, leaking validation signal into training negatives.
- **Fix:** Add `update_queue=False` parameter to `criterion.forward()` and pass it during validation.
- **Owner:** Ritik
- **Time:** 20 minutes

### CB-4: EAR features not passed during model assessment
- **Files:** `evaluate.py:70`
- **Impact:** BiLSTM is trained on 2-channel input (sync-scores + EAR) but assessed on 1-channel. **All reported EAR results are invalid** -- the model silently degrades to sync-score-only at inference time.
- **Fix:** Pass `ear_features` in `run_inference()` matching the training call signature.
- **Owner:** Atharva
- **Time:** 15 minutes

### CB-5: No random seeds set anywhere
- **Files:** `pretrain.py`, `finetune.py`
- **Impact:** No `torch.manual_seed`, `np.random.seed`, `random.seed`. Results are not reproducible. For a CS 5330 submission, this is a basic requirement.
- **Fix:** Add seed-setting function called at the start of every training script. Use config value `seed: 42`.
- **Owner:** Akshay
- **Time:** 15 minutes

### CB-6: NaN loss propagation -- no guard
- **Files:** `pretrain.py`, `finetune.py`
- **Impact:** If loss goes NaN (possible via all-false speech mask -> zero-vector -> F.normalize), training continues silently with garbage gradients, corrupting all weights.
- **Fix:** Add `if torch.isnan(loss): log warning, skip step, continue`. Add gradient NaN check after backward.
- **Owner:** Akshay
- **Time:** 20 minutes

**Total time for all critical blockers: ~2 hours of coding.**
All six must be fixed and verified before launching Phase 1 v3 pretraining.

---

## 2. High-Priority Fixes -- Affect Result Validity

These do not block pretraining launch but affect whether results can be trusted and must be fixed within the first week.

### HP-1: DFDC label fallback defaults unknowns to REAL
- **Files:** `dataset_loader.py:239`
- **Impact:** If DFDC `metadata.json` has filename mismatches, all unmatched videos become label=0 (REAL). Could partially explain DFDC AUC near chance.
- **Fix:** Default to `label=None` and skip unmatched files with a warning log.
- **Owner:** Atharva
- **Priority day:** Day 1-2

### HP-2: 30fps temporal drift for DFDC
- **Files:** `io.py:20-21`, `audio_extractor.py:62`
- **Impact:** DFDC is 30fps. Pipeline assumes 25fps for duration calculation: 300 frames / 25fps = 12s (actual: 10s). Creates 20% AV alignment drift, directly corrupting sync-score computation.
- **Fix:** Read actual fps from video metadata; compute duration as `n_frames / actual_fps`.
- **Owner:** Akshay
- **Priority day:** Day 1-2

### HP-3: RetinaFace resolution not normalized for DFDC
- **Files:** `face_detector.py:79`
- **Impact:** DFDC at 1920x1080: face occupies smaller fraction, confidence drops below 0.8 threshold, high zero-crop rate.
- **Fix:** Downscale input to max 720p before detection, or lower confidence threshold for high-res inputs.
- **Owner:** Akshay
- **Priority day:** Day 2-3

### HP-4: VAD parameters tuned for FakeAVCeleb, fail on DFDC
- **Files:** `vad.py:56-76`
- **Impact:** `min_speech_duration_ms=250` and `threshold=0.5` tuned for short FakeAVCeleb clips. DFDC's compressed audio causes low speech_ratio.
- **Fix:** Lower threshold to 0.3 for DFDC; increase min_speech_duration tolerance. Make VAD params configurable per dataset.
- **Owner:** Ritik
- **Priority day:** Day 2-3

### HP-5: Finetune grad clipping excludes criterion parameters
- **Files:** `finetune.py:443`
- **Impact:** Learnable `log_temperature` in criterion not gradient-clipped, unlike pretrain which clips both. Temperature can explode during finetuning.
- **Fix:** `all_params = list(model.parameters()) + list(criterion.parameters())`
- **Owner:** Ritik
- **Priority day:** Day 1

### HP-6: strict=False with no key logging on checkpoint load
- **Files:** `finetune.py:336`
- **Impact:** No verification pretrained weights actually loaded. Could silently fall back to random init. AV-HuBERT loading catches ALL exceptions and falls back silently too.
- **Fix:** Log missing/unexpected keys. Assert no missing keys in required modules.
- **Owner:** Akshay
- **Priority day:** Day 1

### HP-7: Pretraining split not speaker-disjoint for LRS2
- **Files:** `dataset.py:451-456`
- **Impact:** Random shuffle split. Safe for AVSpeech (unique speaker per clip) but LRS2 shares speaker_ids across clips. Same speaker in train+val inflates val metrics.
- **Fix:** Group by `speaker_id` before splitting when dataset has shared speakers.
- **Owner:** Ritik
- **Priority day:** Day 1

### HP-8: Non-atomic checkpoint save
- **Files:** Checkpoint saving logic
- **Impact:** SLURM preemption mid-write corrupts the best model file.
- **Fix:** Write to temp file, then `os.rename()` (atomic on same filesystem).
- **Owner:** Akshay
- **Priority day:** Day 1

### HP-9: Visual never upsampled to 49Hz (stated in docs)
- **Files:** `upsample_visual_to_audio()` defined but never called
- **Impact:** Model operates at 25fps visual, truncating ~half of Wav2Vec output via `align_sequences`. Wastes half the audio encoder's temporal resolution.
- **Decision needed:** Either call the upsample function or update documentation. For v3 pretraining, document current behavior; consider upsampling for v4.
- **Owner:** Akshay
- **Priority day:** Day 3 (decision), Day 8+ (implementation if chosen)

### HP-10: Classifier lengths not clamped in forward pass
- **Files:** `syncguard.py:145-157`
- **Impact:** After `align_sequences` truncation, lengths can exceed T, causing potential index errors. `evaluate.py` clamps (line 81) but main forward does not.
- **Fix:** `lengths = lengths.clamp(max=T)` in `forward()`.
- **Owner:** Atharva
- **Priority day:** Day 1

---

## 3. DFDC Strategy Recommendation -- Concrete Plan for 0.57 -> 0.72

### Diagnosis

The current DFDC AUC of 0.5712 (95% CI: [0.534, 0.609]) is essentially at chance. Three factors compound:

1. **Preprocessing bugs** (HP-1 through HP-4): Label fallback, 30fps drift, resolution issues, and VAD killing clips. These alone could account for the majority of the gap. The model may never have seen correctly-aligned DFDC data.

2. **Fundamental signal mismatch**: DFDC face-swaps PRESERVE lip-sync. SyncGuard's CMP signal detects AV desynchronization, which is intact in DFDC fakes. Raw sync-score AUC on DFDC is 0.4378 -- inverted. The core signal does not just fail; it is anti-correlated.

3. **Sync-score bottleneck**: Compressing rich AV embeddings into a scalar cosine similarity discards identity-mismatch information that IS present in the embeddings but lost through the bottleneck.

### Recommended Strategy (3 tiers, executed in order)

**Tier 1: Fix preprocessing + BN adaptation (Days 1-4, expected +0.05-0.12)**
- Fix all DFDC preprocessing bugs (HP-1 through HP-4)
- Re-preprocess DFDC Part 0 with corrected pipeline
- Run batch normalization adaptation: forward pass DFDC through model with `model.train()` on BN layers only, `model.eval()` on everything else. This adapts running statistics to DFDC distribution.
- Recalibrate classification threshold on a small DFDC held-out set
- Re-assess. If AUC >= 0.65, Tier 1 may be sufficient with Tier 2.

**Tier 2: Embedding-level features bypass bottleneck (Days 4-9, expected +0.05-0.10)**
- Instead of reducing AV embeddings to a scalar sync-score, concatenate the full frame-level AV embedding difference vectors (or use cross-attention) as input to the classifier.
- This preserves identity-mismatch information that CMP's scalar bottleneck discards.
- Train a lightweight classifier head on these richer features.
- This is architecturally aligned with AVFF (CVPR 2024, 86.2% on DFDC).

**Tier 3: DCT frequency features (Days 7-11, expected +0.05-0.08)**
- Extract DCT coefficients from face crops (proven signal for face-swap detection).
- Fuse as additional channel to the classifier alongside sync features.
- Low-risk, well-established in literature.

### Expected Outcome

| Tier | Intervention | Expected DFDC AUC | Cumulative |
|------|-------------|-------------------|------------|
| Baseline | Current | 0.57 | 0.57 |
| Tier 1 | Preprocessing fixes + BN adapt | 0.62-0.69 | 0.62-0.69 |
| Tier 2 | Embedding bypass | 0.67-0.75 | 0.67-0.75 |
| Tier 3 | DCT features | 0.72-0.80 | 0.72-0.80 |

Conservative estimate with Tiers 1+2: **0.70-0.73 AUC**. Adding Tier 3 provides margin.

### What NOT to pursue (given 16-day deadline)
- Full visual-only ensemble (4-6 days, too risky for timeline)
- Visual upsampling to 49Hz (significant architectural change, insufficient time to validate)
- Expanding DFDC to all parts (storage and preprocessing time prohibitive)
- EAR as DFDC signal (no evidence DFDC fakes show abnormal blinks; modern face-swaps preserve blink patterns)

---

## 4. Updated 16-Day Experiment Plan

### Phase 0: Bug Fixes + Infrastructure (Days 1-2: Mar 28-29)

| Day | Owner | Task | Deliverable |
|-----|-------|------|-------------|
| 1 (Mar 28) | Ritik | Fix CB-1 (InfoNCE labels), CB-2 (MoCo as nn.Module), CB-3 (validation queue pollution), HP-5 (grad clip), HP-7 (speaker-disjoint split) | PR with unit tests for each fix |
| 1 (Mar 28) | Akshay | Fix CB-5 (seeds), CB-6 (NaN guard), HP-6 (strict load logging), HP-8 (atomic checkpoints), HP-2 (30fps drift) | PR with unit tests |
| 1 (Mar 28) | Atharva | Fix CB-4 (EAR in inference), HP-1 (DFDC label fallback), HP-10 (length clamping) | PR with unit tests |
| 2 (Mar 29) | Akshay | Fix HP-3 (RetinaFace resolution) | Tested on DFDC sample |
| 2 (Mar 29) | Ritik | Fix HP-4 (VAD params for DFDC) | Tested on DFDC sample |
| 2 (Mar 29) | Atharva | Run verification scripts on existing results. Enable bootstrap CIs. Document which prior results are invalidated. | `outputs/logs/result_validity_audit.md` |

**Gate check (end of Day 2):** All critical blockers fixed. Verification scripts pass. Team reviews which results to discard.

### Phase 1: Pretraining v3 + DFDC Reprocessing (Days 3-6: Mar 30 - Apr 2)

| Day | Owner | Task | Deliverable |
|-----|-------|------|-------------|
| 3 (Mar 30) | Ritik | Launch Phase 1 v3 pretraining (AVSpeech + LRS2, CMP objective, all fixes applied). H200 GPU. | SLURM job running, monitoring dashboard |
| 3 (Mar 30) | Akshay | Re-preprocess DFDC Part 0 with corrected pipeline (fps-aware, resolution-normalized, fixed labels) | Preprocessed DFDC data on HPC |
| 3 (Mar 30) | Atharva | Implement BN adaptation script. Prepare DFDC assessment with corrected preprocessing. | `scripts/bn_adapt_dfdc.py` |
| 4 (Apr 1) | Ritik | Monitor pretraining. Begin implementing Tier 2 embedding bypass in classifier. | Architecture design doc |
| 4 (Apr 1) | Akshay | Verify DFDC reprocessing quality (spot-check alignments, face crops, labels). Run framerate validation. | Verification report |
| 4 (Apr 1) | Atharva | Run BN adaptation + threshold recalibration on re-preprocessed DFDC. Assess results. | Tier 1 DFDC AUC number |
| 5 (Apr 2) | ALL | **Tier 1 checkpoint.** Review DFDC AUC after preprocessing fixes + BN adapt. Decide Tier 2 scope. | Decision: proceed with Tier 2 as planned or adjust |
| 6 (Apr 3) | Ritik | Pretraining v3 completes (~72h on H200). Validate pretrain metrics. | Pretrained checkpoint |

**Gate check (end of Day 6):** v3 pretraining done. Tier 1 DFDC AUC known. Go/no-go on Tier 2/3.

### Phase 2: Finetuning + DFDC Interventions (Days 7-11: Apr 3 - Apr 7)

| Day | Owner | Task | Deliverable |
|-----|-------|------|-------------|
| 7 (Apr 3) | Ritik | Launch finetuning on FakeAVCeleb with v3 pretrained weights | SLURM job |
| 7 (Apr 3) | Akshay | Implement Tier 2: embedding-level classifier bypass (concatenate AV embedding diffs instead of scalar sync-score) | New classifier variant |
| 7 (Apr 3) | Atharva | Begin DCT feature extraction pipeline (Tier 3) | `scripts/extract_dct_features.py` |
| 8 (Apr 4) | Ritik | Finetuning completes. Assess FakeAVCeleb AUC (target: maintain >= 0.94) | In-domain AUC |
| 8 (Apr 4) | Akshay | Train Tier 2 embedding classifier on FakeAVCeleb | Embedding classifier checkpoint |
| 8 (Apr 4) | Atharva | DCT extraction on FakeAVCeleb + DFDC | DCT features ready |
| 9 (Apr 5) | ALL | Cross-dataset scoring: standard model + Tier 2 embedding model on DFDC | DFDC AUC for both approaches |
| 10 (Apr 6) | Akshay | Integrate DCT features (Tier 3) with best-performing model from Day 9 | Fused model |
| 10 (Apr 6) | Ritik | Ablation runs: visual-only baseline, sync-score-only baseline, linear probe | Ablation table |
| 10 (Apr 6) | Atharva | Per-category DFDC analysis. Compute bootstrap CIs on all results. | Category-level results with CIs |
| 11 (Apr 7) | ALL | **Tier 2+3 checkpoint.** Full scoring: FakeAVCeleb + DFDC, all model variants. | Complete results table |

**Gate check (end of Day 11):** Target 0.72 DFDC AUC achieved or final strategy adjustment.

### Phase 3: Paper + Polish (Days 12-16: Apr 8 - Apr 13)

| Day | Owner | Task | Deliverable |
|-----|-------|------|-------------|
| 12 (Apr 8) | ALL | Finalize model selection. Run 3x seed variance runs on best config. | Mean +/- std for all metrics |
| 13 (Apr 9) | Atharva | Generate all tables, figures, confusion matrices, ROC curves | `outputs/figures/` |
| 13 (Apr 9) | Akshay | Write methodology + experiments sections | Draft sections |
| 13 (Apr 9) | Ritik | Write related work + introduction | Draft sections |
| 14 (Apr 10) | ALL | Full paper draft assembly. Internal review. | Complete draft |
| 15 (Apr 11) | ALL | Revisions based on internal review. Supplementary materials. | Revised draft |
| 16 (Apr 12) | ALL | Final proofread. Code cleanup. Package submission. | Submission-ready |
| -- (Apr 13) | ALL | **SUBMIT** | Done |

### Contingency Plan
- If Tier 1 alone reaches 0.72 (Day 5): Skip Tier 3, invest Days 7-11 in ablations and paper writing.
- If Tier 2 does not improve over Tier 1 (Day 9): Pivot fully to DCT features + cascade fusion.
- If all tiers combined reach only 0.65-0.71: Report honestly with analysis of why DFDC is harder (preserved lip-sync in face-swaps). This is still a valid scientific finding.

---

## 5. HPC Verification Scripts

### Already Created (from review agents)
| Script | Location | Purpose |
|--------|----------|---------|
| `scripts/verify_metrics.py` | Exists | Validate AUC/EER computation correctness |
| `scripts/slurm_verify_metrics.sh` | Exists | SLURM wrapper for metric verification |
| `outputs/logs/statistical_validity_report.md` | Exists | Statistical validity analysis |

### New Scripts Needed

**VS-1: `scripts/verify_moco_queue.py`** -- Verify MoCo queue fix
- Load checkpoint, confirm queue/ptr/full in state_dict
- Resume training for 5 steps, verify queue contents are not random
- Run validation, verify queue is NOT modified
- Print queue statistics before/after validation
- Owner: Ritik, Day 2

**VS-2: `scripts/verify_dfdc_preprocessing.py`** -- Verify DFDC pipeline fixes
- Sample 50 DFDC videos
- Verify fps detection (should report 30, not assume 25)
- Verify face detection rate at native resolution vs downscaled
- Verify label assignment matches metadata.json ground truth
- Verify audio-visual alignment (duration match within 0.1s)
- Report VAD speech_ratio distribution
- Owner: Akshay, Day 2-3

**VS-3: `scripts/verify_infonce_labels.py`** -- Verify InfoNCE fix
- Create synthetic batch of known embeddings
- Verify loss is minimized when positives are on diagonal
- Verify gradients flow correctly to all rows (not just row 0)
- Test with empty queue and full queue
- Owner: Ritik, Day 1

**VS-4: `scripts/verify_inference_ear.py`** -- Verify EAR passed at inference time
- Load finetuned model
- Run inference on 10 samples with and without ear_features
- Verify outputs differ (confirming EAR is actually used)
- Verify output shape matches expected
- Owner: Atharva, Day 1

**VS-5: `scripts/verify_checkpoint_integrity.py`** -- Verify atomic checkpoint save
- Save checkpoint using new atomic method
- Simulate interrupt during save (kill mid-write)
- Verify original checkpoint is not corrupted
- Verify strict=True loading logs all keys
- Owner: Akshay, Day 2

**VS-6: `scripts/slurm_verify_all.sh`** -- Master verification SLURM script
- Runs VS-1 through VS-5 sequentially
- Outputs pass/fail summary
- Must pass before any training job is submitted
- Owner: Akshay, Day 2

---

## 6. Educational Audit Report -- Full Findings by Category

### Category A: Loss Function and Contrastive Learning

| ID | Severity | Finding | Source Agent(s) |
|----|----------|---------|-----------------|
| CB-1 | CRITICAL | InfoNCE in-batch fallback uses `torch.zeros(N)` instead of `torch.arange(N)` for labels | Agent 1, Agent 6 |
| CB-2 | CRITICAL | MoCo queue not persisted in checkpoints (plain class, not nn.Module) | Agent 1, Agent 6 |
| CB-3 | CRITICAL | Validation enqueues embeddings into MoCo queue, polluting training negatives | Agent 1, Agent 6 |
| HP-5 | HIGH | Finetune gradient clipping excludes learnable temperature in criterion | Agent 1 |
| W-1 | WARNING | Temperature clamping range mismatch between code default and config | Agent 1 |

**Summary:** The MoCo queue has three interacting bugs (wrong labels on empty, not persisted, validation pollution). After any training interruption, the queue reinitializes to random AND uses wrong labels. This is the most critical cluster of bugs. Every resumed pretraining run has been operating with a corrupted contrastive objective.

### Category B: Data Pipeline and Preprocessing

| ID | Severity | Finding | Source Agent(s) |
|----|----------|---------|-----------------|
| HP-1 | HIGH | DFDC label fallback defaults unknowns to REAL (label=0) | Agent 5a |
| HP-2 | HIGH | 30fps temporal drift: pipeline assumes 25fps, DFDC is 30fps, 20% alignment error | Agent 5a |
| HP-3 | HIGH | RetinaFace confidence drops at 1080p, high zero-crop rate on DFDC | Agent 5a |
| HP-4 | HIGH | VAD parameters tuned for FakeAVCeleb, kill most DFDC frames | Agent 5a |
| HP-7 | HIGH | Pretraining split not speaker-disjoint (problem for LRS2) | Agent 1 |
| W-2 | WARNING | Audio-swap augmentation discards original fake artifacts, replaces with real content | Agent 1 |
| W-3 | WARNING | EAR features loaded from wrong directory after audio-swap | Agent 1 |
| W-4 | WARNING | Silent audio produces degenerate constant embeddings | Agent 6 |
| W-5 | WARNING | Audio codec failures silently skip DFDC videos, biasing the subset used | Agent 5a |
| W-6 | WARNING | No clip duration cap: DFDC 10s (300 frames) vs FakeAVCeleb ~100 frames | Agent 5a |

**Summary:** Six DFDC-specific preprocessing issues were found. Together, these likely explain a large portion of the 0.57 AUC. The model has never seen correctly-aligned DFDC data. Fixing preprocessing is the highest-ROI intervention.

### Category C: Model Architecture

| ID | Severity | Finding | Source Agent(s) |
|----|----------|---------|-----------------|
| CB-4 | CRITICAL | EAR features not passed during inference (train=2ch, inference=1ch) | Agent 2 |
| HP-9 | HIGH | Visual never upsampled to 49Hz despite documentation claims | Agent 2 |
| HP-10 | HIGH | Classifier lengths not clamped after align_sequences truncation | Agent 1 |
| W-7 | WARNING | StatisticalClassifier and CNN1DClassifier ignore padding in statistics | Agent 1 |
| W-8 | WARNING | Duplicate ProjectionHead in visual_encoder.py and audio_encoder.py | Agent 1 |
| W-9 | WARNING | hard_negative_ratio stored but never used (dead code) | Agent 1 |
| W-10 | WARNING | Hardcoded dimensions (max_frames=150, 96x96, BiLSTM 256) not from config | Agent 2 |
| W-11 | WARNING | Config propagation gap: audio dim assumed equal to visual dim by convention | Agent 2 |
| D-1 | DESIGN | Sync-score bottleneck discards identity-mismatch information needed for DFDC | Agent 5b |
| D-2 | DESIGN | VAD frame stride mismatch: 326 vs Wav2Vec actual 320 (latent, not used in training) | Agent 2 |
| D-3 | DESIGN | Stated "49Hz" audio rate is actually 50Hz (16000/320) | Agent 2 |

**Summary:** The EAR inference bug means all reported EAR results are invalid. The visual upsample omission means the model discards half of audio temporal resolution. The sync-score bottleneck is the fundamental architectural limitation for DFDC generalization.

### Category D: Training Robustness and Infrastructure

| ID | Severity | Finding | Source Agent(s) |
|----|----------|---------|-----------------|
| CB-5 | CRITICAL | No random seeds set anywhere | Agent 3 |
| CB-6 | CRITICAL | NaN loss propagation: no check, training continues with garbage weights | Agent 6 |
| HP-6 | HIGH | strict=False with no key logging; AV-HuBERT catches ALL exceptions | Agent 3, Agent 6 |
| HP-8 | HIGH | Non-atomic checkpoint save: SLURM kill mid-write corrupts best model | Agent 6 |
| W-12 | WARNING | model.train() re-enables frozen Wav2Vec group norm, potential NaN path | Agent 6 |
| W-13 | WARNING | model.train() after validation re-enables frozen batch norms | Agent 6 |
| W-14 | WARNING | Corrupt .npy files crash DataLoader workers with opaque errors | Agent 6 |
| W-15 | WARNING | All-false speech mask produces zero-vector embeddings, NaN via F.normalize | Agent 6 |
| W-16 | WARNING | No framerate validation in pipeline | Agent 6 |
| W-17 | WARNING | AUC=0.5 for single-class sets treated as valid improvement | Agent 6 |
| W-18 | WARNING | No config saved with checkpoints | Agent 3 |

**Summary:** Training infrastructure lacks basic robustness. Any interruption risks data corruption or silent degradation. Seeds, NaN guards, and atomic checkpoints are table stakes.

### Category E: Experimental Methodology and Statistics

| ID | Severity | Finding | Source Agent(s) |
|----|----------|---------|-----------------|
| E-1 | HIGH | Confidence intervals never computed (bootstrap implemented but never called) | Agent 3, Agent 4 |
| E-2 | HIGH | Missing baselines: no visual-only, no random, no linear probe | Agent 3 |
| E-3 | HIGH | Ablation design confounds: CMP + LRS2 tested simultaneously | Agent 3 |
| E-4 | HIGH | Config/doc mismatch on frozen Wav2Vec (unclear what actually ran) | Agent 3 |
| E-5 | HIGH | DFDC sample n=1,334 gives 95% CI [0.534, 0.609]; too small for reliable conclusions | Agent 4 |
| E-6 | MEDIUM | pAUC normalization non-standard (area/max_fpr, not McClish) | Agent 4 |
| E-7 | MEDIUM | RV-FA sync-only category AUC 0.507, CI crosses 0.5 | Agent 4 |
| E-8 | MEDIUM | FV-FA category AUC 0.990 imprecise due to ~75 real samples | Agent 4 |
| E-9 | MEDIUM | Lexicographic speaker ordering in split may introduce bias | Agent 3 |
| E-10 | LOW | CelebDF-v2 still listed as target despite being dropped | Agent 3 |

**Summary:** The experimental methodology has gaps that would be flagged in peer review. Bootstrap CIs, proper baselines, and controlled ablations are needed for a credible submission.

### Category F: DFDC-Specific Strategic Assessment

| ID | Severity | Finding | Source Agent(s) |
|----|----------|---------|-----------------|
| F-1 | CRITICAL | Raw sync-score AUC on DFDC is 0.4378 (inverted): core signal is anti-correlated | Agent 5b |
| F-2 | CRITICAL | DFDC face-swaps preserve lip-sync; CMP detects desync which is intact in DFDC fakes | Agent 5b |
| F-3 | HIGH | No evidence DFDC fakes show abnormal blink patterns (EAR signal unvalidated) | Agent 5b |
| F-4 | HIGH | CMP + EAR hypothesis viability rated 4/10 for DFDC by hypothesis challenger | Agent 5b |
| F-5 | MEDIUM | MediaPipe landmark reliability on swapped faces is unvalidated | Agent 5b |

**Summary:** The core architectural hypothesis (AV desync detection) is fundamentally mismatched with DFDC's face-swap attack type. Preprocessing fixes will recover some AUC, but reaching 0.72 likely requires bypassing the sync-score bottleneck to access identity-mismatch information in the raw embeddings (Tier 2 strategy).

---

## Summary of Finding Counts

| Severity | Count |
|----------|-------|
| CRITICAL (blockers) | 6 |
| CRITICAL (strategic) | 2 |
| HIGH | 15 |
| WARNING | 18 |
| MEDIUM | 5 |
| DESIGN | 3 |
| LOW | 1 |
| **Total** | **50** |

## Key Takeaway

The path to 0.72 DFDC AUC is: **fix preprocessing bugs (Tier 1) + bypass sync-score bottleneck (Tier 2) + optionally add DCT features (Tier 3)**. The critical code blockers (MoCo queue cluster, EAR inference bug, seeds, NaN guards) must be fixed first as they invalidate any new experiments. The 16-day timeline is tight but achievable if the team parallelizes effectively and the Day 5 gate check shows Tier 1 reaching at least 0.62.
