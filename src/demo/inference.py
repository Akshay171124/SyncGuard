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

    def __init__(self, config_path, checkpoint_path, device=None):
        self.config = load_config(config_path)
        self.device = device or get_device()
        self.model = build_syncguard(self.config).to(self.device)
        ckpt = _load_checkpoint(checkpoint_path, self.device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.train(False)   # inference mode (equivalent to .eval())
        self.fps_sync = self.config["preprocessing"]["audio"]["target_fps"]

    @torch.no_grad()
    def analyze(self, mouth_crops, audio_waveform, ear_features=None,
                clip_duration_s=None):
        """Run one forward pass and build an AnalysisResult.

        Args:
            mouth_crops: np.ndarray (T, 1, 96, 96) float32 in [0,1]
            audio_waveform: np.ndarray (N,) float32 at 16kHz
            ear_features: np.ndarray (T,) or None
            clip_duration_s: float or None (inferred from waveform if None)

        Returns:
            AnalysisResult
        """
        timings = {}
        t0 = time.perf_counter()

        mc = torch.from_numpy(mouth_crops).unsqueeze(0).to(self.device)
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

        logit = out.logits.squeeze().item()
        confidence = float(torch.sigmoid(torch.tensor(logit)).item())
        verdict = "fake" if confidence >= 0.5 else "real"
        display_conf = confidence if verdict == "fake" else (1.0 - confidence)

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

        return AnalysisResult(
            verdict=verdict,
            confidence=display_conf,
            sync_curve=sync_curve,
            ear_curve=ear_curve_np,
            sync_dip_segments=dip_segments,
            ear_anomaly_score=ear_anomaly,
            clip_duration_s=clip_duration_s,
            mean_sync=mean_sync,
            timings=timings,
        )
