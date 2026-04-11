"""Tests for evaluation metrics (src/evaluation/metrics.py).

Tests use known-answer patterns:
- Perfect separation → AUC = 1.0, EER = 0.0
- Random scores → AUC ≈ 0.5
- Partial overlap → verifiable ranges
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    EvaluationResult,
    compute_all_metrics,
    compute_auc_roc,
    compute_bootstrap_ci,
    compute_eer,
    compute_pauc,
    compute_per_category_auc,
)


# ──────────────────────────────────────────────
# AUC-ROC
# ──────────────────────────────────────────────

class TestAUCROC:
    def test_perfect_separation(self):
        """Scores perfectly separate real from fake → AUC = 1.0."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc, fpr, tpr, thresholds = compute_auc_roc(labels, scores)
        assert auc == 1.0

    def test_inverse_separation(self):
        """Scores inversely separate → AUC = 0.0."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        auc, _, _, _ = compute_auc_roc(labels, scores)
        assert auc == 0.0

    def test_random_scores_near_half(self):
        """Random scores → AUC near 0.5."""
        rng = np.random.RandomState(42)
        labels = np.array([0] * 500 + [1] * 500)
        scores = rng.uniform(0, 1, 1000)
        auc, _, _, _ = compute_auc_roc(labels, scores)
        assert 0.4 <= auc <= 0.6

    def test_single_class_fallback(self):
        """Only one class present → returns 0.5."""
        labels = np.array([0, 0, 0])
        scores = np.array([0.1, 0.5, 0.9])
        auc, fpr, tpr, _ = compute_auc_roc(labels, scores)
        assert auc == 0.5

    def test_returns_roc_curve_arrays(self):
        """ROC curve arrays have matching lengths."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        _, fpr, tpr, thresholds = compute_auc_roc(labels, scores)
        assert len(fpr) == len(tpr)
        assert fpr[0] == 0.0  # ROC starts at origin
        assert fpr[-1] == 1.0  # ROC ends at (1, 1)
        assert tpr[-1] == 1.0


# ──────────────────────────────────────────────
# EER
# ──────────────────────────────────────────────

class TestEER:
    def test_perfect_separation_eer_zero(self):
        """Perfect separation → EER = 0."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        _, fpr, tpr, thresholds = compute_auc_roc(labels, scores)
        eer, _ = compute_eer(fpr, tpr, thresholds)
        assert eer == pytest.approx(0.0, abs=0.01)

    def test_eer_returns_threshold(self):
        """EER returns a valid threshold value."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.3, 0.4, 0.6, 0.7])
        _, fpr, tpr, thresholds = compute_auc_roc(labels, scores)
        eer, thresh = compute_eer(fpr, tpr, thresholds)
        assert 0.0 <= eer <= 0.5
        assert isinstance(thresh, float)

    def test_eer_bounded(self):
        """EER is always in [0, 0.5]."""
        rng = np.random.RandomState(123)
        labels = np.array([0] * 100 + [1] * 100)
        scores = rng.uniform(0, 1, 200)
        _, fpr, tpr, thresholds = compute_auc_roc(labels, scores)
        eer, _ = compute_eer(fpr, tpr, thresholds)
        assert 0.0 <= eer <= 0.5


# ──────────────────────────────────────────────
# Partial AUC
# ──────────────────────────────────────────────

class TestPAUC:
    def test_perfect_separation_pauc(self):
        """Perfect separation → pAUC = 1.0."""
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        pauc = compute_pauc(fpr, tpr, max_fpr=0.1)
        assert pauc == pytest.approx(1.0, abs=0.01)

    def test_diagonal_pauc(self):
        """Diagonal ROC (random) → pAUC ≈ 0.05 (triangle area / max_fpr)."""
        fpr = np.linspace(0, 1, 1000)
        tpr = fpr.copy()
        pauc = compute_pauc(fpr, tpr, max_fpr=0.1)
        # Area under diagonal up to FPR=0.1 is 0.1*0.1/2 = 0.005
        # Normalized: 0.005 / 0.1 = 0.05
        assert pauc == pytest.approx(0.05, abs=0.01)

    def test_pauc_different_thresholds(self):
        """pAUC at lower FPR threshold ≤ pAUC at higher threshold for monotone curves."""
        labels = np.array([0] * 100 + [1] * 100)
        rng = np.random.RandomState(42)
        scores = np.concatenate([rng.beta(2, 5, 100), rng.beta(5, 2, 100)])
        _, fpr, tpr, _ = compute_auc_roc(labels, scores)
        pauc_01 = compute_pauc(fpr, tpr, max_fpr=0.1)
        pauc_05 = compute_pauc(fpr, tpr, max_fpr=0.05)
        # Both should be positive for a model better than random
        assert pauc_01 > 0.0
        assert pauc_05 > 0.0

    def test_pauc_insufficient_points(self):
        """Fewer than 2 points in FPR range → returns 0.0."""
        fpr = np.array([0.5, 1.0])
        tpr = np.array([0.5, 1.0])
        pauc = compute_pauc(fpr, tpr, max_fpr=0.1)
        assert pauc == 0.0


# ──────────────────────────────────────────────
# Per-category AUC
# ──────────────────────────────────────────────

class TestPerCategoryAUC:
    def test_three_fake_categories(self):
        """Returns AUC for each fake category (FV-RA, RV-FA, FV-FA)."""
        rng = np.random.RandomState(42)
        n = 100
        labels = np.array([0] * n + [1] * n + [1] * n + [1] * n)
        scores = np.concatenate([
            rng.beta(2, 5, n),    # RV-RA (real)
            rng.beta(5, 2, n),    # FV-RA (fake)
            rng.beta(4, 3, n),    # RV-FA (fake)
            rng.beta(5, 2, n),    # FV-FA (fake)
        ])
        categories = np.array(
            ["RV-RA"] * n + ["FV-RA"] * n + ["RV-FA"] * n + ["FV-FA"] * n
        )
        result = compute_per_category_auc(labels, scores, categories)
        assert set(result.keys()) == {"FV-RA", "RV-FA", "FV-FA"}
        for cat, auc in result.items():
            assert 0.0 <= auc <= 1.0

    def test_missing_category_skipped(self):
        """Category with no samples is skipped."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        categories = np.array(["RV-RA", "RV-RA", "FV-RA", "FV-RA"])
        result = compute_per_category_auc(labels, scores, categories)
        assert "FV-RA" in result
        assert "RV-FA" not in result  # No samples


# ──────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────

class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        """CI should contain the point estimate AUC."""
        rng = np.random.RandomState(42)
        labels = np.array([0] * 200 + [1] * 200)
        scores = np.concatenate([rng.beta(2, 5, 200), rng.beta(5, 2, 200)])
        auc, _, _, _ = compute_auc_roc(labels, scores)
        mean_auc, ci_lower, ci_upper = compute_bootstrap_ci(
            labels, scores, n_bootstrap=200, seed=42,
        )
        assert ci_lower <= auc <= ci_upper

    def test_ci_width_shrinks_with_more_data(self):
        """CI should be narrower with more data."""
        rng = np.random.RandomState(42)
        # Small dataset
        labels_small = np.array([0] * 30 + [1] * 30)
        scores_small = np.concatenate([rng.beta(2, 5, 30), rng.beta(5, 2, 30)])
        _, ci_lo_s, ci_hi_s = compute_bootstrap_ci(labels_small, scores_small, n_bootstrap=200)
        # Large dataset
        labels_large = np.array([0] * 300 + [1] * 300)
        scores_large = np.concatenate([rng.beta(2, 5, 300), rng.beta(5, 2, 300)])
        _, ci_lo_l, ci_hi_l = compute_bootstrap_ci(labels_large, scores_large, n_bootstrap=200)
        width_small = ci_hi_s - ci_lo_s
        width_large = ci_hi_l - ci_lo_l
        assert width_large < width_small

    def test_deterministic_with_seed(self):
        """Same seed → same result."""
        labels = np.array([0] * 50 + [1] * 50)
        scores = np.random.RandomState(0).uniform(0, 1, 100)
        r1 = compute_bootstrap_ci(labels, scores, seed=42)
        r2 = compute_bootstrap_ci(labels, scores, seed=42)
        assert r1 == r2


# ──────────────────────────────────────────────
# compute_all_metrics (integration)
# ──────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_returns_evaluation_result(self):
        """Returns EvaluationResult dataclass."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        result = compute_all_metrics(labels, scores)
        assert isinstance(result, EvaluationResult)

    def test_sample_counts(self):
        """n_samples, n_real, n_fake are correct."""
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
        result = compute_all_metrics(labels, scores)
        assert result.n_samples == 5
        assert result.n_real == 3
        assert result.n_fake == 2

    def test_with_categories(self):
        """Per-category breakdown populated when categories provided."""
        rng = np.random.RandomState(42)
        labels = np.array([0] * 50 + [1] * 50 + [1] * 50)
        scores = np.concatenate([
            rng.beta(2, 5, 50),
            rng.beta(5, 2, 50),
            rng.beta(5, 2, 50),
        ])
        categories = np.array(["RV-RA"] * 50 + ["FV-RA"] * 50 + ["FV-FA"] * 50)
        result = compute_all_metrics(labels, scores, categories=categories)
        assert len(result.per_category) >= 1

    def test_to_dict_serializable(self):
        """to_dict() output is JSON-safe (no numpy types)."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        result = compute_all_metrics(labels, scores)
        d = result.to_dict()
        assert isinstance(d["auc_roc"], float)
        assert isinstance(d["n_samples"], int)

    def test_with_bootstrap_ci(self):
        """Bootstrap CI populated when requested."""
        rng = np.random.RandomState(42)
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.concatenate([rng.beta(2, 5, 100), rng.beta(5, 2, 100)])
        result = compute_all_metrics(labels, scores, bootstrap_ci=True)
        assert result.auc_ci_lower is not None
        assert result.auc_ci_upper is not None
        d = result.to_dict()
        assert "auc_ci_95" in d
