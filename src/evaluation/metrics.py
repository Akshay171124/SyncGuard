"""Evaluation metrics for SyncGuard deepfake detection.

Provides AUC-ROC, EER, pAUC (at low FPR), per-category breakdown for
FakeAVCeleb, and confidence intervals via bootstrapping.

Usage:
    from src.evaluation.metrics import compute_all_metrics
    results = compute_all_metrics(labels, scores, categories=categories)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


# FakeAVCeleb manipulation categories
FAKEAVCELEB_CATEGORIES = {
    "RV-RA": "Real Video + Real Audio",
    "FV-RA": "Fake Video + Real Audio",
    "RV-FA": "Real Video + Fake Audio",
    "FV-FA": "Fake Video + Fake Audio",
}


@dataclass
class EvaluationResult:
    """Container for evaluation metrics.

    Attributes:
        auc_roc: Area under the ROC curve.
        eer: Equal Error Rate.
        eer_threshold: Threshold at EER.
        pauc_fpr01: Partial AUC at FPR < 0.1.
        pauc_fpr05: Partial AUC at FPR < 0.05.
        fpr: False positive rates for ROC curve.
        tpr: True positive rates for ROC curve.
        thresholds: Thresholds for ROC curve.
        per_category: Per-category AUC breakdown (FakeAVCeleb).
        n_samples: Total number of samples evaluated.
        n_real: Number of real samples.
        n_fake: Number of fake samples.
    """
    auc_roc: float = 0.0
    eer: float = 0.0
    eer_threshold: float = 0.0
    pauc_fpr01: float = 0.0
    pauc_fpr05: float = 0.0
    fpr: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr: np.ndarray = field(default_factory=lambda: np.array([]))
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    per_category: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_real: int = 0
    n_fake: int = 0
    auc_ci_lower: float | None = None
    auc_ci_upper: float | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (excludes large arrays)."""
        d = {
            "auc_roc": round(self.auc_roc, 4),
            "eer": round(self.eer, 4),
            "eer_threshold": round(self.eer_threshold, 4),
            "pauc_fpr01": round(self.pauc_fpr01, 4),
            "pauc_fpr05": round(self.pauc_fpr05, 4),
            "per_category": {k: round(v, 4) for k, v in self.per_category.items()},
            "n_samples": self.n_samples,
            "n_real": self.n_real,
            "n_fake": self.n_fake,
        }
        if self.auc_ci_lower is not None:
            d["auc_ci_95"] = [round(self.auc_ci_lower, 4), round(self.auc_ci_upper, 4)]
        return d


def compute_auc_roc(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute AUC-ROC with full ROC curve points.

    Args:
        labels: Binary ground truth (0=real, 1=fake).
        scores: Prediction scores (higher = more likely fake).

    Returns:
        Tuple of (auc, fpr, tpr, thresholds).
    """
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present — AUC undefined, returning 0.5")
        return 0.5, np.array([0, 1]), np.array([0, 1]), np.array([1, 0])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    return auc, fpr, tpr, thresholds


def compute_eer(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    """Compute Equal Error Rate from ROC curve.

    EER is the point where FPR == FNR (i.e., FPR == 1 - TPR).

    Args:
        fpr: False positive rates from roc_curve.
        tpr: True positive rates from roc_curve.
        thresholds: Thresholds from roc_curve.

    Returns:
        Tuple of (eer, threshold_at_eer).
    """
    fnr = 1 - tpr
    # Find the point where FPR and FNR cross
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_pauc(
    fpr: np.ndarray, tpr: np.ndarray, max_fpr: float = 0.1
) -> float:
    """Compute partial AUC up to a given FPR threshold.

    Normalized to [0, 1] range (divided by max_fpr so perfect = 1.0).

    Args:
        fpr: False positive rates from roc_curve.
        tpr: True positive rates from roc_curve.
        max_fpr: Maximum FPR to include.

    Returns:
        Normalized partial AUC.
    """
    # Clip to the FPR range of interest
    mask = fpr <= max_fpr
    if mask.sum() < 2:
        return 0.0

    fpr_clipped = fpr[mask]
    tpr_clipped = tpr[mask]

    # Add the endpoint at max_fpr via interpolation
    if fpr_clipped[-1] < max_fpr:
        tpr_at_max = np.interp(max_fpr, fpr, tpr)
        fpr_clipped = np.append(fpr_clipped, max_fpr)
        tpr_clipped = np.append(tpr_clipped, tpr_at_max)

    # Trapezoidal integration, normalized
    pauc = np.trapezoid(tpr_clipped, fpr_clipped) / max_fpr
    return float(pauc)


def compute_per_category_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray,
) -> dict[str, float]:
    """Compute AUC-ROC per FakeAVCeleb manipulation category.

    Each category is evaluated as a binary problem: real (RV-RA) vs that category.

    Args:
        labels: Binary ground truth (0=real, 1=fake).
        scores: Prediction scores.
        categories: Category string per sample (e.g., "RV-RA", "FV-FA").

    Returns:
        Dict mapping category name to AUC-ROC.
    """
    results = {}
    real_mask = categories == "RV-RA"
    real_labels = labels[real_mask]
    real_scores = scores[real_mask]

    for cat in FAKEAVCELEB_CATEGORIES:
        if cat == "RV-RA":
            continue  # Skip real-vs-real

        cat_mask = categories == cat
        if cat_mask.sum() == 0:
            logger.warning(f"No samples for category {cat}")
            continue

        cat_labels = np.concatenate([real_labels, labels[cat_mask]])
        cat_scores = np.concatenate([real_scores, scores[cat_mask]])

        if len(np.unique(cat_labels)) < 2:
            results[cat] = 0.5
            continue

        results[cat] = float(roc_auc_score(cat_labels, cat_scores))

    return results


def compute_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrapped confidence interval for AUC-ROC.

    Args:
        labels: Binary ground truth.
        scores: Prediction scores.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level (default 95%).
        seed: Random seed.

    Returns:
        Tuple of (mean_auc, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    aucs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = labels[idx]
        boot_scores = scores[idx]

        if len(np.unique(boot_labels)) < 2:
            continue

        aucs.append(roc_auc_score(boot_labels, boot_scores))

    if not aucs:
        return 0.5, 0.5, 0.5

    aucs = np.array(aucs)
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(aucs, 100 * alpha))
    ci_upper = float(np.percentile(aucs, 100 * (1 - alpha)))
    return float(np.mean(aucs)), ci_lower, ci_upper


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray = None,
    bootstrap_ci: bool = False,
) -> EvaluationResult:
    """Compute all evaluation metrics.

    Args:
        labels: Binary ground truth (0=real, 1=fake).
        scores: Prediction scores (higher = more likely fake).
        categories: Optional per-sample category labels (for FakeAVCeleb).
        bootstrap_ci: Whether to compute bootstrap confidence intervals.

    Returns:
        EvaluationResult with all metrics.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    auc, fpr, tpr, thresholds = compute_auc_roc(labels, scores)
    eer, eer_thresh = compute_eer(fpr, tpr, thresholds)
    pauc_01 = compute_pauc(fpr, tpr, max_fpr=0.1)
    pauc_05 = compute_pauc(fpr, tpr, max_fpr=0.05)

    per_category = {}
    if categories is not None:
        categories = np.asarray(categories)
        per_category = compute_per_category_auc(labels, scores, categories)

    auc_ci_lower, auc_ci_upper = None, None
    if bootstrap_ci:
        _, auc_ci_lower, auc_ci_upper = compute_bootstrap_ci(labels, scores)

    result = EvaluationResult(
        auc_roc=auc,
        eer=eer,
        eer_threshold=eer_thresh,
        pauc_fpr01=pauc_01,
        pauc_fpr05=pauc_05,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        per_category=per_category,
        n_samples=len(labels),
        n_real=int((labels == 0).sum()),
        n_fake=int((labels == 1).sum()),
        auc_ci_lower=auc_ci_lower,
        auc_ci_upper=auc_ci_upper,
    )

    logger.info(
        f"Evaluation: AUC={auc:.4f} EER={eer:.4f} "
        f"pAUC@0.1={pauc_01:.4f} pAUC@0.05={pauc_05:.4f} "
        f"({result.n_real} real, {result.n_fake} fake)"
    )
    if per_category:
        for cat, cat_auc in per_category.items():
            logger.info(f"  {cat}: AUC={cat_auc:.4f}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic test
    rng = np.random.RandomState(42)
    n = 500
    labels = np.array([0] * 250 + [1] * 250)
    # Real clips get lower scores, fake clips get higher scores
    scores = np.concatenate([
        rng.beta(2, 5, 250),  # Real: skewed low
        rng.beta(5, 2, 250),  # Fake: skewed high
    ])

    categories = np.array(
        ["RV-RA"] * 250
        + ["FV-RA"] * 83
        + ["RV-FA"] * 83
        + ["FV-FA"] * 84
    )

    result = compute_all_metrics(labels, scores, categories=categories)
    print(f"\nResults: {result.to_dict()}")

    # Verify metrics are reasonable
    assert result.auc_roc > 0.8, f"AUC too low: {result.auc_roc}"
    assert result.eer < 0.3, f"EER too high: {result.eer}"
    assert result.pauc_fpr01 > 0.0, "pAUC@0.1 should be positive"
    assert len(result.per_category) == 3, "Should have 3 fake categories"
    print("All metrics tests passed.")
