# SyncGuard — Final Project Video Script

**Target length:** 12 minutes (Phase 5 window is 10–15 min).
**Presenters:** Akshay Prajapati, Ritik Mahyavanshi, Atharva Dhumal — each speaks for a clearly identifiable segment.
**Recording format suggestion:** One person per segment on screen (webcam) with slide / terminal / plot overlay; smooth handoffs named on-camera ("Over to Ritik for the audio side…").

---

## 0. Title slide (≤ 10 s) — Akshay

> "Hi, we're team SyncGuard — Akshay, Ritik, and Atharva — and this is our final project for CS 5330 at Northeastern."

**On screen:** Title card with project name, team, course, date (April 21, 2026), logo/architecture thumbnail.

---

## 1. Problem & motivation (1:00) — Akshay

**Slide:** "Why audio-visual deepfake detection?"

**Script:**
> "Generative AI has made photorealistic deepfakes trivially cheap to produce. Existing detectors are trained on visual artifacts — compression, blending, frequency patterns — and they generalize poorly: drops of 20 to 30 AUC points when you move across datasets.
>
> We took a different angle. In genuine human speech, there's a tight biomechanical coupling between lip motion and the acoustic signal: lip closure for /p/ precedes the acoustic burst by 10–30 ms; lip rounding for /o/ co-occurs with specific formants. This coupling is **generator-agnostic** — it comes from physics, not from any particular model's artifacts. Our hypothesis: a model that learns this coupling should detect deepfakes across generators, not just the ones it trained on."

**On screen:** Side-by-side sync-score curves of real vs. fake clips (`outputs/visualizations/sync_scores/test_sync_curves.pdf`).

---

## 2. System architecture (1:30) — Akshay

**Slide:** Architecture diagram (`outputs/visualizations/architecture_diagram.png`).

**Script:**
> "Our system is a two-stream contrastive model. We pair an AV-HuBERT visual encoder — pretrained for lip-reading — with a frozen Wav2Vec 2.0 audio encoder, extracting hidden states from layer 9 where phonemic content peaks. Both streams project to 256-dim L2-normalized embeddings.
>
> Frame by frame we compute a sync-score — cosine similarity between the visual and audio embeddings. That scalar sequence plus an eye-aspect-ratio blink signal feeds a Bi-LSTM classifier.
>
> On top of that we added a novel cross-modal attention path: bidirectional V-to-A and A-to-V attention with residual connections, operating on the full embeddings. A learnable fusion weight combines the two classification heads."

**Pointer moment:** Mouse over the fusion weight on the diagram when saying "learnable fusion weight."

---

## 3. Visual encoder & preprocessing pipeline (2:00) — Akshay

**Slide:** Preprocessing diagram + mouth-crop montage.

**Script:**
> "On the visual side, our preprocessing pipeline takes raw video through RetinaFace detection at 0.8 confidence, MediaPipe for 468 facial landmarks, and then we crop a 96×96 grayscale mouth region per frame.
>
> Temporal alignment was non-obvious: visual features are at 25 fps, Wav2Vec outputs at 49 Hz. We linearly upsample visual features to match — and we assert the two sequence lengths align before every forward pass, because off-by-one bugs here destroyed our early experiments.
>
> On top of AV-HuBERT we also wired ResNet-18 and SyncNet as ablation baselines — AV-HuBERT wins by +4 AUC points, confirming that lip-reading pretraining beats generic ImageNet features.
>
> On the infrastructure side we set up the full SyncGuard integration, the HPC pipeline — SLURM scripts, W&B tracking, auto-resume on preemption — and the public repo with reproducibility docs."

**On screen:** Short video of the mouth-crop pipeline running on a test clip, OR a static montage if recording is constrained.

---

## 4. Audio encoder & contrastive pretraining (2:00) — Ritik

**Slide:** Audio pipeline + pretraining loss curves (`outputs/visualizations/training_curves/test_pretrain.pdf`).

**Script:**
> "On the audio side, we extract Wav2Vec 2.0 layer-9 hidden states and freeze the backbone. Unfreezing Wav2Vec at our data scale — 21K fine-tuning samples — caused catastrophic forgetting, dropping AUC from 0.58 to 0.47. Freeze-first became our default.
>
> Our pretraining objective is contrastive: InfoNCE with a MoCo queue of 4096 negatives plus a cross-modal prediction head that masks 30% of frames and predicts across modalities. 20 epochs on 121K real speech clips from AVSpeech and LRS2 got us to a validation InfoNCE of 8.06 and a sync-score of 0.978.
>
> The interesting debugging moment was NaN losses at batch zero. We tracked this down to two separate bugs — SF-3, where the projection-head epsilon was too small for normalized embeddings, and SF-6, where Wav2Vec's group norm needed its entire backbone kept in inference mode, not just the feature extractor. Both are documented in our lab notebook and covered by tests now."

**On screen:** Terminal clip of the fixed training log showing "no NaN, loss descending."

---

## 5. Temporal classifier, cross-attention, and evaluation (2:00) — Atharva

**Slide:** Per-category bar chart (`outputs/visualizations/ablation_charts/test_per_category.pdf`) + ROC curves (`test_roc.pdf`).

**Script:**
> "On the classifier side, our Bi-LSTM over sync-scores works well on face-swaps — FV-RA hits 0.94 AUC — but voice-clone-only attacks are harder, because high-quality voice cloning preserves phoneme timing well enough to keep sync-scores plausible. Our sync-only RV-FA AUC was 0.667, basically random.
>
> Our cross-attention module fixes that by looking at the full embedding space, not just the scalar sync-score. Bidirectional V-to-A and A-to-V attention with 2 heads, 394K parameters. With fusion enabled, RV-FA jumps from 0.667 to 0.895 — a 22.8 percentage point gain.
>
> We also built out our evaluation framework: AUC-ROC, EER, partial AUC, bootstrap confidence intervals, per-category breakdown, and a cascade evaluator that combines the sync head with a standalone audio classifier for additional recall. And a 219-test pytest suite: every metric, every loss, every model — all runnable in 12 seconds on CPU."

**On screen:** Terminal showing `pytest tests/ -v` running and ending with `215 passed, 4 skipped in 11.8s`.

---

## 6. Demo (1:30) — Akshay (or whoever is most comfortable live)

**Option A (safer):** pre-recorded terminal clip, narrated live.
**Option B (more engaging):** live run of the evaluation script on a held-out sample.

**Script:**
> "Here's the model running end-to-end on a test clip. [show terminal]
>
> We load the v4+CA checkpoint, pull a clip from FakeAVCeleb, run preprocessing, and print the sync-score curve alongside the final logit.
>
> [Scroll through output] — AUC 0.9628 on 2,950 test samples, EER 9.3%. The sync-score curve dips exactly at the frames where the generator's lip motion lags the audio."

**On screen:** Live terminal with `python scripts/evaluate.py …` producing the final metrics; optionally a matplotlib window showing the s(t) curve overlaid on the waveform.

---

## 7. Results (1:00) — Atharva

**Slide:** Headline numbers + comparison table.

**Script:**
> "Headline results: on FakeAVCeleb we hit 96.3 AUC, surpassing AVoiD-DF at 89.2 and MRDF-CE at 92.4. Cross-attention gives us the +22.8 lift on voice-clone specifically.
>
> The zero-shot DFDC number is 52.6 — near chance. We're showing this honestly because it's the most interesting scientific finding: DFDC's face-swap generators preserve the original lip motion, so audio-visual correspondence can't distinguish them from real clips. This isn't a tuning problem — it's a signal problem. We tried BN adaptation, threshold recalibration, and preprocessing parity fixes; none raised AUC above 0.55."

**On screen:** Comparison table + DFDC ROC curve side-by-side.

---

## 8. Challenges & lessons learned (1:00) — one person (suggest Akshay or rotate)

**Slide:** Bullet list — 3 points max.

**Script (condensed, pick any 2–3):**
> "Three lessons that will stick with us past this class.
>
> One — freeze pretrained backbones by default on small downstream datasets. At 21K samples, fine-tuning 94M speech parameters destroys more than it refines.
>
> Two — when a method fails to generalize, check whether the signal you depend on is even present in the new distribution before blaming domain shift. We spent a month chasing distributional fixes before accepting the signal just isn't there.
>
> Three — auto-resubmit on SLURM without error gating burned 40 jobs in one afternoon when our collation bug was deterministic. Fail fast on deterministic errors, only retry on infrastructure failures."

---

## 9. Closing (20 s) — Akshay

**Script:**
> "The full codebase, reports, and this recording are on GitHub at Akshay171124/SyncGuard. Thanks for watching."

**On screen:** GitHub URL + QR code (optional).

---

## Production checklist

- [ ] Everyone's webcam on for at least one speaking segment (course requires identifiable individual contributions).
- [ ] Sync slides timing to script so transitions feel natural.
- [ ] Record with the terminal zoomed to ~125% — text must be readable.
- [ ] One rehearsal pass end-to-end before the final take.
- [ ] Upload unlisted YouTube / Google Drive / Vimeo; **only the link gets submitted** (per Phase 5 instructions).
- [ ] Verify the link is accessible from a browser not logged into your account before submitting.
- [ ] Final length target: 11:00–13:00. Trim ruthlessly if you overshoot 14:30.

---

## Figure references (for slide prep)

| Slide topic | File to use |
|---|---|
| Sync-score curves real vs fake | `outputs/visualizations/sync_scores/test_sync_curves.pdf` |
| Sync-score distribution | `outputs/visualizations/sync_scores/test_sync_dist.pdf` |
| Architecture diagram | `outputs/visualizations/architecture_diagram.png` (or `Architecture_diagram_updated.png`) |
| Pretraining loss curves | `outputs/visualizations/training_curves/test_pretrain.pdf` |
| ROC on FakeAVCeleb | `outputs/visualizations/roc_curves/test_roc.pdf` |
| ROC cross-dataset | `outputs/visualizations/roc_curves/test_cross_dataset.pdf` |
| Per-category bars | `outputs/visualizations/ablation_charts/test_per_category.pdf` |
| Visual encoder ablation | `outputs/visualizations/ablation_charts/test_visual_encoder.pdf` |
