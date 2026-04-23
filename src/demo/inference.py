"""Demo-facing inference wrapper around SyncGuard v4+CA."""
import time
from pathlib import Path

import numpy as np
import torch

from src.demo.explain import (
    AnalysisResult,
    SYNC_THRESHOLD,
    MIN_DIP_DURATION_S,
)
from src.models.syncguard import build_syncguard, SyncGuardOutput
from src.models.audio_classifier import build_standalone_audio_classifier
from src.utils.config import load_config, get_device


def _find_dip_segments(sync_curve, fps, threshold=SYNC_THRESHOLD):
    """Contiguous (start_s, end_s) segments where s(t) < threshold."""
    below = sync_curve < threshold
    segments = []
    i = 0
    while i < len(below):
        if below[i]:
            start = i
            while i < len(below) and below[i]:
                i += 1
            segments.append((start / fps, i / fps))
        else:
            i += 1
    return segments


def _ear_anomaly_score(ear_curve):
    """Multiple-of-baseline EAR variance. Baseline = first 10-frame variance."""
    if ear_curve is None or len(ear_curve) < 10:
        return 0.0
    baseline = float(np.var(ear_curve[:10]) + 1e-6)
    overall_var = float(np.var(ear_curve))
    return overall_var / baseline


def _normalize_mouth_crops(crops):
    """Convert mouth crops to (T, 1, 96, 96) float32 in [0, 1].

    Accepts (T, H, W, 3) RGB, (T, H, W) grayscale, or (T, 1, H, W) grayscale.
    Matches the conversion done in src/training/dataset.py._load_mouth_crops.
    """
    crops = np.asarray(crops)
    if crops.ndim == 4 and crops.shape[-1] == 3:
        # (T, H, W, 3) RGB -> (T, H, W) grayscale
        crops = np.mean(crops, axis=-1)
    if crops.ndim == 3:
        # (T, H, W) -> (T, 1, H, W)
        crops = crops[:, np.newaxis, :, :]
    crops = crops.astype(np.float32)
    if crops.max() > 1.0:
        crops = crops / 255.0
    return crops


def _load_checkpoint(path, device):
    """Load a SyncGuard checkpoint. Tries weights_only=True first (safer);
    falls back to weights_only=False for legacy checkpoints that embed
    non-tensor state (optimizer, scheduler) at the top level."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


class DemoInference:
    """Loads SyncGuard v4+CA once; .analyze() runs one forward pass."""

    def __init__(self, config_path, checkpoint_path, audio_checkpoint_path=None,
                 device=None):
        self.config = load_config(config_path)

        # v4+CA checkpoint includes cross-attention + DCT extractor heads.
        # The shipped default.yaml has these disabled (training config had
        # them enabled via CLI overrides or a separate experiment config),
        # so we enable them explicitly here so the model architecture matches
        # the checkpoint. Without this, load_state_dict(strict=False) silently
        # drops ~20% of the trained weights and the sync head degenerates
        # to a bias-dominated "always real" output (sigmoid ~ 0.001).
        model_cfg = self.config["model"]
        model_cfg.setdefault("cross_attention", {})
        model_cfg["cross_attention"]["enabled"] = True
        model_cfg["cross_attention"].setdefault("num_heads", 2)
        model_cfg["cross_attention"].setdefault("dropout", 0.1)
        model_cfg["cross_attention"].setdefault("embed_classifier_hidden", 256)
        model_cfg["cross_attention"].setdefault("fusion_init", 0.0)
        model_cfg.setdefault("dct_extractor", {})
        model_cfg["dct_extractor"]["enabled"] = True
        model_cfg["dct_extractor"]["output_dim"] = 16  # pool_dim=(2*256+dct)*2=1024+2*dct=1056 -> dct=16

        self.device = device or get_device(self.config)
        self.model = build_syncguard(self.config).to(self.device)
        ckpt = _load_checkpoint(checkpoint_path, self.device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            import logging
            logging.getLogger(__name__).warning(
                "Checkpoint missing keys (first 5): %s", list(missing)[:5]
            )
        if unexpected:
            import logging
            logging.getLogger(__name__).warning(
                "Checkpoint unexpected keys (first 5): %s", list(unexpected)[:5]
            )
        self.model.train(False)
        self.fps_sync = self.config["preprocessing"]["audio"]["target_fps"]

        # Optional cascade audio classifier (boosts FV-RA/RV-FA AUC)
        self.audio_model = None
        if audio_checkpoint_path is not None:
            self.audio_model = build_standalone_audio_classifier(
                self.config
            ).to(self.device)
            ackpt = _load_checkpoint(audio_checkpoint_path, self.device)
            astate = ackpt["model_state_dict"] if "model_state_dict" in ackpt else ackpt
            self.audio_model.load_state_dict(astate, strict=False)
            self.audio_model.train(False)

    @torch.no_grad()
    def analyze(self, mouth_crops, audio_waveform, ear_features=None,
                clip_duration_s=None, use_audio_head=True):
        """Run one forward pass and build an AnalysisResult.

        Args:
            mouth_crops: np.ndarray (T, 1, 96, 96) float32 in [0,1]
            audio_waveform: np.ndarray (N,) float32 at 16kHz
            ear_features: np.ndarray (T,) or None
            clip_duration_s: float or None (inferred from waveform if None)
            use_audio_head: if False, skip the cascade audio classifier in
                            the verdict (useful for out-of-distribution
                            inputs like live webcam audio). The audio head
                            still runs and its score is returned for
                            transparency.

        Returns:
            AnalysisResult
        """
        timings = {}
        t0 = time.perf_counter()

        mc_np = _normalize_mouth_crops(mouth_crops)
        mc = torch.from_numpy(mc_np).unsqueeze(0).to(self.device)
        wf = torch.from_numpy(audio_waveform).unsqueeze(0).to(self.device)
        ear = None
        if ear_features is not None:
            ear = torch.from_numpy(ear_features).unsqueeze(0).to(self.device)
        lengths = torch.tensor([mc.shape[1]], device=self.device)

        t_pre = time.perf_counter()
        timings["prep"] = t_pre - t0

        out: SyncGuardOutput = self.model(
            mouth_crops=mc, waveform=wf, lengths=lengths, ear_features=ear,
        )
        timings["forward"] = time.perf_counter() - t_pre

        sync_prob = float(torch.sigmoid(out.logits.squeeze()).item())

        audio_prob = 0.0
        if self.audio_model is not None:
            audio_logit = self.audio_model(wf).squeeze()
            audio_prob = float(torch.sigmoid(audio_logit).item())
            timings["audio_forward"] = time.perf_counter() - t_pre - timings["forward"]

        if self.audio_model is not None and use_audio_head:
            fake_prob = max(sync_prob, audio_prob)
            real_prob = max(1.0 - sync_prob, 1.0 - audio_prob)
        else:
            fake_prob = sync_prob
            real_prob = 1.0 - sync_prob

        # Verdict uses max-fusion (any head screams fake -> fake).
        # Confidence uses asymmetric fusion:
        #  - for fake verdict: max(sync_p, audio_p) — the "loudest" fake signal
        #  - for real verdict: max(1-sync_p, 1-audio_p) — the strongest real signal
        # This avoids the miscalibrated audio head pulling real confidence
        # down to 54% when the sync head is 99.9% sure it's real.
        verdict = "fake" if fake_prob >= 0.5 else "real"
        display_conf = fake_prob if verdict == "fake" else real_prob

        sync_curve = out.sync_scores.squeeze(0).cpu().numpy()
        mean_sync = float(np.mean(sync_curve))
        dip_segments = _find_dip_segments(sync_curve, self.fps_sync)

        ear_curve_np = (ear_features if ear_features is not None
                        else np.zeros(len(sync_curve), dtype=np.float32))
        ear_anomaly = _ear_anomaly_score(ear_curve_np)

        if clip_duration_s is None:
            clip_duration_s = float(
                len(audio_waveform) /
                self.config["preprocessing"]["audio"]["sample_rate"]
            )

        audio_prob_out = audio_prob

        return AnalysisResult(
            verdict=verdict,
            confidence=display_conf,
            sync_curve=sync_curve,
            ear_curve=ear_curve_np,
            sync_dip_segments=dip_segments,
            ear_anomaly_score=ear_anomaly,
            clip_duration_s=clip_duration_s,
            mean_sync=mean_sync,
            sync_prob=sync_prob,
            audio_prob=audio_prob_out,
            timings=timings,
        )
