"""Rule-based natural-language explanations for SyncGuard demo verdicts.

Pure function — no I/O, no model, no external state. Fully unit-testable.
"""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


SYNC_THRESHOLD = 0.55
CONF_HIGH = 0.75
CONF_LOW = 0.55
EAR_ANOMALY_THRESHOLD = 2.0
MIN_DIP_DURATION_S = 0.15
BORDERLINE_BAND = 0.05
SHORT_CLIP_S = 2.0


@dataclass
class AnalysisResult:
    """Output of DemoInference.analyze. Feeds the explainer and the UI plot."""
    verdict: Literal["real", "fake"]
    confidence: float
    sync_curve: np.ndarray
    ear_curve: np.ndarray
    sync_dip_segments: list[tuple[float, float]]
    ear_anomaly_score: float
    clip_duration_s: float
    mean_sync: float
    timings: dict[str, float] = field(default_factory=dict)


def _count_significant_dips(segments):
    return [(s, e) for s, e in segments if (e - s) >= MIN_DIP_DURATION_S]


def _total_dip_duration(segments):
    return sum(e - s for s, e in segments)


def _peak_dip_time(segments):
    if not segments:
        return 0.0
    longest = max(segments, key=lambda seg: seg[1] - seg[0])
    return (longest[0] + longest[1]) / 2


def _pct(conf):
    return f"{int(round(conf * 100))}%"


def _is_borderline(confidence):
    return abs(confidence - 0.5) < BORDERLINE_BAND


def generate_explanation(result: AnalysisResult) -> str:
    """Produce a 1-2 sentence human-readable explanation from model outputs."""
    dips = _count_significant_dips(result.sync_dip_segments)
    num_dips = len(dips)
    total_dip_s = _total_dip_duration(dips)
    peak_time = _peak_dip_time(dips)
    ear_anomaly = result.ear_anomaly_score
    conf = result.confidence
    mean_sync = result.mean_sync
    prefix = ""
    if result.clip_duration_s < SHORT_CLIP_S:
        prefix = "Limited temporal evidence — "

    if result.verdict == "real":
        if _is_borderline(conf) or conf < CONF_LOW:
            body = (
                f"Likely real — {_pct(conf)} confidence (borderline). "
                f"Sync mostly stable with {num_dips} brief dip(s). "
                f"May be natural speech pauses; recommend human review."
            )
        else:
            body = (
                f"Real clip — {_pct(conf)} confidence. "
                f"Audio-visual alignment stable throughout "
                f"(mean sync {mean_sync:.2f}, threshold {SYNC_THRESHOLD:.2f}). "
                f"No blink anomalies detected."
            )
        return prefix + body

    if _is_borderline(conf):
        body = (
            f"Likely fake — {_pct(conf)} confidence (borderline). "
            f"Detected {num_dips} sync dip(s); recommend human review."
        )
        return prefix + body

    if num_dips >= 2 and ear_anomaly > EAR_ANOMALY_THRESHOLD:
        body = (
            f"Fake detected — {_pct(conf)} confidence. "
            f"Lip-sync drops below {SYNC_THRESHOLD:.2f} in {num_dips} "
            f"segments ({total_dip_s:.1f}s total), and blink pattern shows "
            f"{ear_anomaly:.1f}× baseline variance. "
            f"Consistent with face-swap combined with audio manipulation."
        )
    elif num_dips >= 1 and ear_anomaly <= EAR_ANOMALY_THRESHOLD:
        body = (
            f"Fake detected — {_pct(conf)} confidence. "
            f"Audio-visual sync drops below threshold in {num_dips} "
            f"segments totaling {total_dip_s:.1f}s "
            f"(most prominent at {peak_time:.1f}s). "
            f"Face appears natural — pattern consistent with "
            f"audio dubbing or lip-sync generation."
        )
    elif num_dips == 0 and ear_anomaly > EAR_ANOMALY_THRESHOLD:
        body = (
            f"Fake detected — {_pct(conf)} confidence. "
            f"Sync scores remain stable (mean {mean_sync:.2f}), "
            f"but blink pattern anomaly detected "
            f"(EAR variance {ear_anomaly:.1f}× baseline). "
            f"Consistent with face-swap that preserved original audio alignment."
        )
    else:
        body = (
            f"Fake detected — {_pct(conf)} confidence. "
            f"No strong sync or blink anomalies, but learned representation "
            f"indicates manipulation. May be a well-executed fake or an "
            f"out-of-distribution sample."
        )
    return prefix + body
