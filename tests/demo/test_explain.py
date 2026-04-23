"""Unit tests for rule-based explanation generation."""
from dataclasses import dataclass, field
import numpy as np
import pytest

from src.demo.explain import (
    generate_explanation,
    AnalysisResult,
    SYNC_THRESHOLD,
)


def _mk(verdict="real", confidence=0.9, sync_curve=None, ear_curve=None,
        sync_dip_segments=None, ear_anomaly_score=1.0,
        clip_duration_s=4.0, mean_sync=0.7):
    T = 100
    return AnalysisResult(
        verdict=verdict,
        confidence=confidence,
        sync_curve=sync_curve if sync_curve is not None else np.full(T, mean_sync),
        ear_curve=ear_curve if ear_curve is not None else np.full(T, 0.25),
        sync_dip_segments=sync_dip_segments or [],
        ear_anomaly_score=ear_anomaly_score,
        clip_duration_s=clip_duration_s,
        mean_sync=mean_sync,
        timings={},
    )


class TestRealVerdicts:
    def test_confident_real(self):
        result = _mk(verdict="real", confidence=0.92, mean_sync=0.71)
        text = generate_explanation(result)
        assert "Real" in text and "92%" in text and "stable" in text.lower()

    def test_borderline_real(self):
        result = _mk(verdict="real", confidence=0.52,
                     sync_dip_segments=[(2.0, 2.1)])
        text = generate_explanation(result)
        assert "borderline" in text.lower() or "review" in text.lower()


class TestFakeCategories:
    def test_fv_fa_pattern_sync_and_ear(self):
        result = _mk(verdict="fake", confidence=0.94,
                     sync_dip_segments=[(1.0, 1.4), (2.5, 2.9)],
                     ear_anomaly_score=120.0, mean_sync=0.42)
        text = generate_explanation(result)
        assert "face-swap" in text.lower() and "audio" in text.lower()

    def test_rv_fa_pattern_sync_only(self):
        result = _mk(verdict="fake", confidence=0.81,
                     sync_dip_segments=[(1.0, 1.4), (2.1, 2.6), (3.0, 3.3)],
                     ear_anomaly_score=1.1, mean_sync=0.38)
        text = generate_explanation(result)
        assert ("desync" in text.lower() or "dub" in text.lower() or
                "lip-sync" in text.lower())

    def test_fv_ra_pattern_ear_only(self):
        result = _mk(verdict="fake", confidence=0.76,
                     sync_dip_segments=[],
                     ear_anomaly_score=100.0, mean_sync=0.62)
        text = generate_explanation(result)
        assert "blink" in text.lower() or "face-swap" in text.lower()

    def test_subtle_fake(self):
        result = _mk(verdict="fake", confidence=0.70,
                     sync_dip_segments=[], ear_anomaly_score=1.0)
        text = generate_explanation(result)
        assert "fake" in text.lower()


class TestEdgeCases:
    def test_short_clip_prepends_limitation(self):
        result = _mk(verdict="real", confidence=0.85, clip_duration_s=1.5)
        text = generate_explanation(result)
        assert "limited" in text.lower()

    def test_missing_ear_does_not_crash(self):
        result = _mk(verdict="fake", confidence=0.8,
                     sync_dip_segments=[(1.0, 1.5)],
                     ear_curve=np.zeros(100), ear_anomaly_score=0.0)
        text = generate_explanation(result)
        assert "fake" in text.lower()

    def test_near_threshold_uses_borderline(self):
        result = _mk(verdict="fake", confidence=0.51,
                     sync_dip_segments=[(1.0, 1.1)])
        text = generate_explanation(result)
        assert "borderline" in text.lower() or "review" in text.lower()


class TestConfidencePercent:
    @pytest.mark.parametrize("conf, pct", [(0.5, "50%"), (0.87, "87%"), (0.999, "100%")])
    def test_percent_format(self, conf, pct):
        result = _mk(verdict="fake", confidence=conf,
                     sync_dip_segments=[(1.0, 1.2)])
        text = generate_explanation(result)
        assert pct in text
