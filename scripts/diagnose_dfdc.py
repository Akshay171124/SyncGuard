#!/usr/bin/env python3
"""DFDC diagnostic suite — run after DFDC reprocessing and Phase 2 evaluation.

Produces:
1. Sync-score distribution comparison (FakeAVCeleb vs DFDC, real vs fake)
2. EAR distribution analysis on DFDC (tests whether blink artifacts are detectable)
3. Preprocessing quality report (detection rates, frame counts, speech ratios)

Usage:
    python scripts/diagnose_dfdc.py --config configs/default.yaml
    python scripts/diagnose_dfdc.py --config configs/default.yaml --predictions_dir outputs/logs
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def analyze_preprocessing_quality(processed_dir: Path, dataset_name: str):
    """Check detection rates, frame counts, and speech ratios."""
    print(f"\n{'='*60}")
    print(f"  Preprocessing Quality: {dataset_name}")
    print(f"{'='*60}")

    sample_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    if not sample_dirs:
        print(f"  No processed samples found in {processed_dir}")
        return

    frame_counts = []
    detection_rates = []
    speech_ratios = []
    labels = []

    for sd in sample_dirs:
        meta_path = sd / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        labels.append(meta.get("label", -1))

        crops_path = sd / "mouth_crops.npy"
        if crops_path.exists():
            crops = np.load(crops_path)
            frame_counts.append(crops.shape[0])

        if "detection_rate" in meta:
            detection_rates.append(meta["detection_rate"])

        speech_path = sd / "speech_mask.npy"
        if speech_path.exists():
            mask = np.load(speech_path)
            speech_ratios.append(mask.mean())

    print(f"  Samples: {len(labels)}")
    print(f"  Label distribution: {dict(Counter(labels))}")

    if frame_counts:
        fc = np.array(frame_counts)
        print(f"\n  Frame counts:")
        print(f"    Mean: {fc.mean():.0f}, Std: {fc.std():.0f}")
        print(f"    Min: {fc.min()}, Max: {fc.max()}")
        print(f"    Expected for 10s@25fps: ~250")

    if detection_rates:
        dr = np.array(detection_rates)
        print(f"\n  Face detection rates:")
        print(f"    Mean: {dr.mean():.3f}, Std: {dr.std():.3f}")
        print(f"    Below 0.5: {(dr < 0.5).sum()} ({(dr < 0.5).mean():.1%})")
        print(f"    Below 0.8: {(dr < 0.8).sum()} ({(dr < 0.8).mean():.1%})")

    if speech_ratios:
        sr = np.array(speech_ratios)
        print(f"\n  Speech ratios (VAD):")
        print(f"    Mean: {sr.mean():.3f}, Std: {sr.std():.3f}")
        print(f"    Below 0.1: {(sr < 0.1).sum()} ({(sr < 0.1).mean():.1%})")
        print(f"    Below 0.3: {(sr < 0.3).sum()} ({(sr < 0.3).mean():.1%})")


def analyze_ear_distributions(processed_dir: Path, metadata_path: Path = None):
    """Test whether EAR differs between real and fake DFDC clips."""
    print(f"\n{'='*60}")
    print(f"  EAR Distribution Analysis (DFDC)")
    print(f"{'='*60}")

    real_stats = []
    fake_stats = []

    for sd in sorted(processed_dir.iterdir()):
        if not sd.is_dir():
            continue

        ear_path = sd / "ear_features.npy"
        meta_path = sd / "metadata.json"

        if not ear_path.exists() or not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        ear = np.load(ear_path)
        if len(ear) == 0:
            continue

        stats = {
            "mean": float(np.mean(ear)),
            "std": float(np.std(ear)),
            "jitter": float(np.mean(np.abs(np.diff(ear)))) if len(ear) > 1 else 0.0,
            "blinks": int(np.sum(np.diff((ear < 0.2).astype(int)) == 1)) if len(ear) > 1 else 0,
        }

        if meta.get("label", 0) == 1:
            fake_stats.append(stats)
        else:
            real_stats.append(stats)

    if not real_stats or not fake_stats:
        print(f"  Insufficient data: {len(real_stats)} real, {len(fake_stats)} fake")
        return

    print(f"  Real clips: {len(real_stats)}, Fake clips: {len(fake_stats)}")

    from scipy import stats as sp_stats

    for metric in ["mean", "std", "jitter", "blinks"]:
        real_vals = [s[metric] for s in real_stats]
        fake_vals = [s[metric] for s in fake_stats]
        t_stat, p_val = sp_stats.ttest_ind(real_vals, fake_vals)

        sig = "YES ***" if p_val < 0.01 else "YES *" if p_val < 0.05 else "NO"
        print(f"\n  EAR {metric}:")
        print(f"    Real: {np.mean(real_vals):.4f} +/- {np.std(real_vals):.4f}")
        print(f"    Fake: {np.mean(fake_vals):.4f} +/- {np.std(fake_vals):.4f}")
        print(f"    t={t_stat:.3f}, p={p_val:.4f} → Significant: {sig}")

    print(f"\n  Verdict:")
    # Check if any metric has p < 0.05
    has_signal = any(
        sp_stats.ttest_ind(
            [s[m] for s in real_stats],
            [s[m] for s in fake_stats]
        ).pvalue < 0.05
        for m in ["mean", "std", "jitter", "blinks"]
    )
    if has_signal:
        print(f"    EAR shows statistically significant differences — worth including.")
    else:
        print(f"    EAR shows NO significant differences — unlikely to help on DFDC.")
        print(f"    Consider dropping EAR for DFDC and focusing on Tier 2/3 interventions.")


def analyze_sync_score_distributions(predictions_dir: Path):
    """Compare sync-score distributions between FakeAVCeleb and DFDC."""
    print(f"\n{'='*60}")
    print(f"  Sync-Score Distribution Comparison")
    print(f"{'='*60}")

    for dataset in ["fakeavceleb", "dfdc"]:
        # Try multiple naming conventions
        for prefix in ["predictions_", "predictions_cascade_"]:
            npz_path = predictions_dir / f"{prefix}{dataset}.npz"
            if npz_path.exists():
                break
        else:
            print(f"\n  {dataset}: No predictions file found, skipping.")
            print(f"    (Run evaluation first: python scripts/evaluate.py --test_sets {dataset})")
            continue

        data = np.load(npz_path)
        scores = data.get("sync_means", data.get("sync_scores", None))
        labels = data["labels"]

        if scores is None:
            print(f"\n  {dataset}: No sync_means/sync_scores in predictions file.")
            continue

        real_scores = scores[labels == 0]
        fake_scores = scores[labels == 1]

        print(f"\n  {dataset} (n={len(labels)}):")
        print(f"    Real: mean={real_scores.mean():.4f}, std={real_scores.std():.4f}, n={len(real_scores)}")
        print(f"    Fake: mean={fake_scores.mean():.4f}, std={fake_scores.std():.4f}, n={len(fake_scores)}")
        gap = real_scores.mean() - fake_scores.mean()
        print(f"    Gap (real - fake): {gap:.4f} {'(correct direction)' if gap > 0 else '(INVERTED!)'}")

        # Save plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(real_scores, bins=50, alpha=0.6, label="Real", color="#27AE60", density=True)
            ax.hist(fake_scores, bins=50, alpha=0.6, label="Fake", color="#E74C3C", density=True)
            ax.set_title(f"{dataset}: Sync-Score Distribution")
            ax.set_xlabel("Mean Sync Score")
            ax.set_ylabel("Density")
            ax.legend()

            out_path = Path("outputs/visualizations") / f"sync_dist_{dataset}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    Plot saved: {out_path}")
        except ImportError:
            print(f"    (matplotlib not available for plots)")


def main():
    parser = argparse.ArgumentParser(description="DFDC diagnostic suite")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--predictions_dir", type=str, default="outputs/logs")
    parser.add_argument("--skip_ear", action="store_true", help="Skip EAR analysis (requires scipy)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.utils.config import load_config
    config = load_config(args.config)

    predictions_dir = Path(args.predictions_dir)

    # 1. Preprocessing quality for both datasets
    fav_processed = Path(config["data"]["processed_dir"]) / "fakeavceleb"
    dfdc_processed = Path(config["data"]["processed_dir"]) / "dfdc"

    if fav_processed.exists():
        analyze_preprocessing_quality(fav_processed, "FakeAVCeleb")
    if dfdc_processed.exists():
        analyze_preprocessing_quality(dfdc_processed, "DFDC")

    # 2. EAR analysis on DFDC
    if not args.skip_ear and dfdc_processed.exists():
        try:
            analyze_ear_distributions(dfdc_processed)
        except ImportError:
            print("\n  scipy not available — skipping EAR analysis")

    # 3. Sync-score distributions (requires evaluation to have run)
    analyze_sync_score_distributions(predictions_dir)

    print(f"\n{'='*60}")
    print(f"  Diagnostic complete. Review findings above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
