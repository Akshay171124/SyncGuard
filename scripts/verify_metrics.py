"""Recalculate all SyncGuard metrics from saved .npz predictions.

Recomputes AUC-ROC, EER, pAUC, per-category breakdown, and bootstrap
confidence intervals from the raw predictions saved during evaluation.
Serves as an independent verification of reported numbers.

Usage (HPC):
    python scripts/verify_metrics.py \
        --predictions_dir outputs/logs \
        --n_bootstrap 5000

Usage (local, if .npz files are available):
    python scripts/verify_metrics.py --predictions_dir outputs/logs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple:
    """Compute EER from ROC curve (independent reimplementation)."""
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_pauc_custom(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    """Compute partial AUC with custom normalization (area / max_fpr)."""
    mask = fpr <= max_fpr
    if mask.sum() < 2:
        return 0.0
    fpr_c = fpr[mask]
    tpr_c = tpr[mask]
    if fpr_c[-1] < max_fpr:
        tpr_at_max = np.interp(max_fpr, fpr, tpr)
        fpr_c = np.append(fpr_c, max_fpr)
        tpr_c = np.append(tpr_c, tpr_at_max)
    _integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_integrate(tpr_c, fpr_c) / max_fpr)


def compute_pauc_sklearn(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Compute partial AUC using sklearn (McClish standardization)."""
    try:
        return float(roc_auc_score(labels, scores, max_fpr=max_fpr))
    except ValueError:
        return 0.5


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray,
    n_bootstrap: int = 5000, confidence: float = 0.95, seed: int = 42,
) -> dict:
    """Compute bootstrapped CI for AUC-ROC."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bl, bs = labels[idx], scores[idx]
        if len(np.unique(bl)) < 2:
            continue
        aucs.append(roc_auc_score(bl, bs))

    if not aucs:
        return {"mean": 0.5, "ci_lower": 0.5, "ci_upper": 0.5, "n_valid": 0}

    aucs = np.array(aucs)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.mean(aucs)),
        "std": float(np.std(aucs)),
        "ci_lower": float(np.percentile(aucs, 100 * alpha)),
        "ci_upper": float(np.percentile(aucs, 100 * (1 - alpha))),
        "n_valid": len(aucs),
        "n_skipped": n_bootstrap - len(aucs),
    }


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil SE approximation for AUC.

    Args:
        auc: Observed AUC-ROC.
        n_pos: Number of positive (fake) samples.
        n_neg: Number of negative (real) samples.

    Returns:
        Standard error of the AUC estimate.
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se2 = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    ) / (n_pos * n_neg)
    return float(np.sqrt(max(se2, 0)))


def verify_dataset(
    name: str,
    npz_path: Path,
    n_bootstrap: int = 5000,
) -> dict:
    """Verify all metrics for one dataset's predictions.

    Args:
        name: Dataset/strategy identifier.
        npz_path: Path to .npz file with saved predictions.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with recomputed metrics, CIs, and significance tests.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Verifying: {name}")
    logger.info(f"Loading: {npz_path}")

    # np.load is used here to read .npz files produced by our own evaluation
    # pipeline (numpy arrays saved via np.savez). These files contain only
    # numeric arrays and string arrays, not arbitrary objects.
    data = np.load(npz_path, allow_pickle=True)

    # Determine score keys
    has_cascade = "sync_scores" in data and "audio_scores" in data
    labels = data["labels"]
    categories = data.get("categories", None)

    n_total = len(labels)
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    logger.info(f"Samples: {n_total} total, {n_real} real, {n_fake} fake")

    # Category distribution
    if categories is not None:
        unique_cats, cat_counts = np.unique(categories, return_counts=True)
        logger.info("Category distribution:")
        for cat, count in zip(unique_cats, cat_counts):
            logger.info(f"  {cat}: {count}")

    results = {"name": name, "n_total": n_total, "n_real": n_real, "n_fake": n_fake}

    # Build strategy -> scores mapping
    strategies = {}
    if has_cascade:
        strategies["sync_only"] = data["sync_scores"]
        strategies["audio_only"] = data["audio_scores"]
        strategies["max_fusion"] = np.maximum(data["sync_scores"], data["audio_scores"])
        strategies["avg_fusion"] = (data["sync_scores"] + data["audio_scores"]) / 2
    elif "scores" in data:
        strategies["single_model"] = data["scores"]
    elif "max_scores" in data:
        strategies["max_fusion"] = data["max_scores"]

    results["strategies"] = {}

    for strat_name, scores in strategies.items():
        logger.info(f"\n--- Strategy: {strat_name} ---")

        if len(np.unique(labels)) < 2:
            logger.warning("Single class in labels -- skipping")
            continue

        # Core metrics
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        eer, eer_thresh = compute_eer(fpr, tpr, thresholds)
        pauc_01_custom = compute_pauc_custom(fpr, tpr, 0.1)
        pauc_05_custom = compute_pauc_custom(fpr, tpr, 0.05)
        pauc_01_sklearn = compute_pauc_sklearn(labels, scores, 0.1)
        pauc_05_sklearn = compute_pauc_sklearn(labels, scores, 0.05)

        logger.info(f"  AUC-ROC:          {auc:.4f}")
        logger.info(f"  EER:              {eer:.4f} (threshold={eer_thresh:.4f})")
        logger.info(f"  pAUC@0.1 custom:  {pauc_01_custom:.4f}")
        logger.info(f"  pAUC@0.1 sklearn: {pauc_01_sklearn:.4f}")
        logger.info(f"  pAUC@0.05 custom: {pauc_05_custom:.4f}")
        logger.info(f"  pAUC@0.05 sklearn:{pauc_05_sklearn:.4f}")

        strat_result = {
            "auc_roc": round(auc, 4),
            "eer": round(eer, 4),
            "eer_threshold": round(eer_thresh, 4),
            "pauc_01_custom_norm": round(pauc_01_custom, 4),
            "pauc_01_sklearn_mcclish": round(pauc_01_sklearn, 4),
            "pauc_05_custom_norm": round(pauc_05_custom, 4),
            "pauc_05_sklearn_mcclish": round(pauc_05_sklearn, 4),
        }

        # Hanley-McNeil SE
        se = hanley_mcneil_se(auc, n_fake, n_real)
        hm_lower = max(0, auc - 1.96 * se)
        hm_upper = min(1, auc + 1.96 * se)
        logger.info(f"  Hanley-McNeil SE: {se:.4f}")
        logger.info(f"  95% CI (analytic): [{hm_lower:.4f}, {hm_upper:.4f}]")
        strat_result["hanley_mcneil_se"] = round(se, 4)
        strat_result["ci_95_analytic"] = [round(hm_lower, 4), round(hm_upper, 4)]

        # Test against random (AUC = 0.5)
        z = (auc - 0.5) / se if se > 0 else 0
        strat_result["z_vs_random"] = round(z, 2)
        strat_result["significant_vs_random"] = bool(abs(z) > 1.96)
        logger.info(
            f"  z vs random:       {z:.2f} "
            f"({'significant' if abs(z) > 1.96 else 'NOT significant'})"
        )

        # Bootstrap CI
        logger.info(f"  Computing bootstrap CI ({n_bootstrap} iterations)...")
        boot = bootstrap_auc(labels, scores, n_bootstrap=n_bootstrap)
        logger.info(f"  Bootstrap AUC:    {boot['mean']:.4f} +/- {boot['std']:.4f}")
        logger.info(
            f"  95% CI (bootstrap): [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]"
        )
        logger.info(f"  Valid iterations: {boot['n_valid']}/{n_bootstrap}")
        strat_result["bootstrap"] = {
            "mean": round(boot["mean"], 4),
            "std": round(boot["std"], 4),
            "ci_lower": round(boot["ci_lower"], 4),
            "ci_upper": round(boot["ci_upper"], 4),
            "n_valid": boot["n_valid"],
        }

        # Per-category AUC (if categories available)
        if categories is not None:
            real_mask = categories == "RV-RA"
            real_labels_cat = labels[real_mask]
            real_scores_cat = scores[real_mask]
            n_real_cat = len(real_labels_cat)
            logger.info(f"\n  Per-category (n_real_baseline={n_real_cat}):")

            cat_results = {}
            for cat in ["FV-RA", "RV-FA", "FV-FA"]:
                cat_mask = categories == cat
                n_cat = int(cat_mask.sum())
                if n_cat == 0:
                    logger.info(f"    {cat}: no samples")
                    continue

                cat_labels = np.concatenate([real_labels_cat, labels[cat_mask]])
                cat_scores = np.concatenate([real_scores_cat, scores[cat_mask]])

                if len(np.unique(cat_labels)) < 2:
                    logger.info(f"    {cat}: single class, AUC undefined")
                    cat_results[cat] = {
                        "auc": 0.5,
                        "n_samples": n_cat + n_real_cat,
                        "n_fake": n_cat,
                        "n_real": n_real_cat,
                    }
                    continue

                cat_auc = roc_auc_score(cat_labels, cat_scores)
                cat_se = hanley_mcneil_se(cat_auc, n_cat, n_real_cat)
                cat_ci_lo = max(0, cat_auc - 1.96 * cat_se)
                cat_ci_hi = min(1, cat_auc + 1.96 * cat_se)
                cat_z = (cat_auc - 0.5) / cat_se if cat_se > 0 else 0

                # Bootstrap for this category
                cat_boot = bootstrap_auc(
                    cat_labels, cat_scores, n_bootstrap=n_bootstrap
                )

                logger.info(
                    f"    {cat}: AUC={cat_auc:.4f} "
                    f"[{cat_ci_lo:.4f}, {cat_ci_hi:.4f}] "
                    f"(n_real={n_real_cat}, n_fake={n_cat}, z={cat_z:.2f})"
                )
                logger.info(
                    f"           Bootstrap: {cat_boot['mean']:.4f} "
                    f"[{cat_boot['ci_lower']:.4f}, {cat_boot['ci_upper']:.4f}]"
                )

                cat_results[cat] = {
                    "auc": round(cat_auc, 4),
                    "n_samples": n_cat + n_real_cat,
                    "n_fake": n_cat,
                    "n_real": n_real_cat,
                    "se": round(cat_se, 4),
                    "ci_95_analytic": [round(cat_ci_lo, 4), round(cat_ci_hi, 4)],
                    "z_vs_random": round(cat_z, 2),
                    "significant_vs_random": bool(abs(cat_z) > 1.96),
                    "bootstrap_ci": [
                        round(cat_boot["ci_lower"], 4),
                        round(cat_boot["ci_upper"], 4),
                    ],
                }

            strat_result["per_category"] = cat_results

        results["strategies"][strat_name] = strat_result

    return results


def main():
    """CLI entry point for metric verification."""
    parser = argparse.ArgumentParser(
        description="Verify SyncGuard metrics from saved predictions"
    )
    parser.add_argument(
        "--predictions_dir", type=str, default="outputs/logs",
        help="Directory containing .npz prediction files",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=5000,
        help="Number of bootstrap iterations for CIs",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: predictions_dir/verification_results.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pred_dir = Path(args.predictions_dir)
    if not pred_dir.exists():
        logger.error(f"Predictions directory not found: {pred_dir}")
        sys.exit(1)

    # Find all .npz files
    npz_files = sorted(pred_dir.glob("predictions_*.npz"))
    if not npz_files:
        logger.error(f"No predictions_*.npz files found in {pred_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} prediction files:")
    for f in npz_files:
        logger.info(f"  {f.name}")

    all_results = {}
    for npz_path in npz_files:
        # Extract name from filename: predictions_cascade_fakeavceleb.npz -> cascade_fakeavceleb
        name = npz_path.stem.replace("predictions_", "")
        result = verify_dataset(name, npz_path, n_bootstrap=args.n_bootstrap)
        all_results[name] = result

    # Save results
    output_path = (
        Path(args.output) if args.output
        else pred_dir / "verification_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nVerification results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    for name, result in all_results.items():
        print(
            f"\n{name} (n={result['n_total']}, "
            f"{result['n_real']} real, {result['n_fake']} fake)"
        )
        for strat, metrics in result["strategies"].items():
            boot = metrics.get("bootstrap", {})
            ci_lo = boot.get("ci_lower", "?")
            ci_hi = boot.get("ci_upper", "?")
            sig = "SIG" if metrics.get("significant_vs_random") else "n.s."
            print(
                f"  {strat:15s}: AUC={metrics['auc_roc']:.4f} "
                f"[{ci_lo}, {ci_hi}] "
                f"EER={metrics['eer']:.4f} "
                f"z={metrics['z_vs_random']:.1f} ({sig})"
            )
            if "per_category" in metrics:
                for cat, cm in metrics["per_category"].items():
                    cat_sig = "SIG" if cm.get("significant_vs_random") else "n.s."
                    cat_ci = cm.get(
                        "bootstrap_ci", cm.get("ci_95_analytic", ["?", "?"])
                    )
                    print(
                        f"    {cat:8s}: AUC={cm['auc']:.4f} "
                        f"[{cat_ci[0]}, {cat_ci[1]}] "
                        f"(n_real={cm['n_real']}, n_fake={cm['n_fake']}) "
                        f"z={cm['z_vs_random']:.1f} ({cat_sig})"
                    )


if __name__ == "__main__":
    main()
