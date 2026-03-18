"""Visualization tools for SyncGuard evaluation.

Generates publication-quality plots following the project's plotting standards:
- Format: PNG (300 DPI) + PDF
- Font: Arial/Helvetica, 12pt body, 14pt titles
- Consistent color palette per CLAUDE.md

Usage:
    from src.evaluation.visualize import plot_roc_curves, plot_sync_scores
    plot_roc_curves(results, output_dir="outputs/visualizations/roc_curves/")
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

# ─── Color Palette (from CLAUDE.md plotting standards) ─────────────────────
COLORS = {
    "real": "#27AE60",
    "fake": "#E74C3C",
    "fakeavceleb": "#1A5276",
    "celebdf": "#F39C12",
    "dfdc": "#8E44AD",
    "wavlip": "#E67E22",
    "syncguard": "#3498DB",
    "baseline": "#95A5A6",
    # Per-category (FakeAVCeleb)
    "FV-RA": "#E74C3C",
    "RV-FA": "#F39C12",
    "FV-FA": "#8E44AD",
}

# ─── Plot Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save_fig(fig: plt.Figure, path: Path):
    """Save figure as PNG + PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path.with_suffix('.png')} + .pdf")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    dataset_name: str,
    output_path: str | Path,
    eer: float = None,
):
    """Plot a single ROC curve.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc: AUC-ROC value.
        dataset_name: Name for title.
        output_path: Save path (without extension).
        eer: Optional EER to mark on plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color=COLORS["syncguard"], lw=2,
            label=f"SyncGuard (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

    if eer is not None:
        ax.plot(eer, 1 - eer, "o", color=COLORS["fake"], markersize=8,
                label=f"EER = {eer:.3f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {dataset_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    _save_fig(fig, output_path)


def plot_roc_multi_dataset(
    results: dict[str, dict],
    output_path: str | Path,
):
    """Plot ROC curves for multiple datasets on one figure.

    Args:
        results: Dict mapping dataset name to dict with fpr, tpr, auc keys.
        output_path: Save path.
    """
    dataset_colors = {
        "fakeavceleb": COLORS["fakeavceleb"],
        "celebdf": COLORS["celebdf"],
        "dfdc": COLORS["dfdc"],
        "wavlip_adversarial": COLORS["wavlip"],
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, data in results.items():
        color = dataset_colors.get(name, COLORS["baseline"])
        label = f"{name} (AUC = {data['auc']:.3f})"
        ax.plot(data["fpr"], data["tpr"], color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-Dataset ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    _save_fig(fig, output_path)


def plot_roc_per_category(
    fpr_tpr_per_cat: dict[str, dict],
    output_path: str | Path,
):
    """Plot per-category ROC curves for FakeAVCeleb.

    Args:
        fpr_tpr_per_cat: Dict mapping category to {fpr, tpr, auc}.
        output_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for cat, data in fpr_tpr_per_cat.items():
        color = COLORS.get(cat, COLORS["baseline"])
        label = f"{cat} (AUC = {data['auc']:.3f})"
        ax.plot(data["fpr"], data["tpr"], color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("FakeAVCeleb — Per-Category ROC")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    _save_fig(fig, output_path)


def plot_sync_score_curves(
    real_sync: np.ndarray,
    fake_sync: np.ndarray,
    output_path: str | Path,
    n_examples: int = 5,
):
    """Plot sync-score s(t) curves for real vs fake clips.

    Args:
        real_sync: (N_real, T) sync-score sequences for real clips.
        fake_sync: (N_fake, T) sync-score sequences for fake clips.
        output_path: Save path.
        n_examples: Number of example curves to plot per class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Real clips
    ax = axes[0]
    for i in range(min(n_examples, len(real_sync))):
        ax.plot(real_sync[i], color=COLORS["real"], alpha=0.5, lw=1)
    if len(real_sync) > 0:
        mean = np.mean(real_sync[:n_examples], axis=0)
        ax.plot(mean, color=COLORS["real"], lw=2.5, label="Mean")
    ax.set_title("Real Clips")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Sync-Score s(t)")
    ax.set_ylim([-1.05, 1.05])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.legend()

    # Fake clips
    ax = axes[1]
    for i in range(min(n_examples, len(fake_sync))):
        ax.plot(fake_sync[i], color=COLORS["fake"], alpha=0.5, lw=1)
    if len(fake_sync) > 0:
        mean = np.mean(fake_sync[:n_examples], axis=0)
        ax.plot(mean, color=COLORS["fake"], lw=2.5, label="Mean")
    ax.set_title("Fake Clips")
    ax.set_xlabel("Frame")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.legend()

    fig.suptitle("Sync-Score Temporal Profiles: Real vs Fake", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_sync_score_distribution(
    real_means: np.ndarray,
    fake_means: np.ndarray,
    output_path: str | Path,
):
    """Plot histogram of mean sync-scores for real vs fake.

    Args:
        real_means: (N_real,) mean sync-score per real clip.
        fake_means: (N_fake,) mean sync-score per fake clip.
        output_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(-1, 1, 50)
    ax.hist(real_means, bins=bins, alpha=0.6, color=COLORS["real"],
            label=f"Real (μ={np.mean(real_means):.3f})", density=True)
    ax.hist(fake_means, bins=bins, alpha=0.6, color=COLORS["fake"],
            label=f"Fake (μ={np.mean(fake_means):.3f})", density=True)

    ax.set_xlabel("Mean Sync-Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Mean Sync-Scores")
    ax.legend()

    _save_fig(fig, output_path)


def plot_training_curves(
    history: list[dict],
    phase: str,
    output_path: str | Path,
):
    """Plot training loss and metric curves.

    Args:
        history: List of per-epoch metric dicts (from pretrain.json or finetune.json).
        phase: "pretrain" or "finetune".
        output_path: Save path.
    """
    epochs = [h["epoch"] for h in history]

    if phase == "pretrain":
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Loss
        ax = axes[0]
        ax.plot(epochs, [h["train_loss"] for h in history],
                color=COLORS["syncguard"], lw=2, label="Train")
        ax.plot(epochs, [h["val_loss"] for h in history],
                color=COLORS["fake"], lw=2, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("InfoNCE Loss")
        ax.set_title("Pretraining Loss")
        ax.legend()

        # Sync-score
        ax = axes[1]
        ax.plot(epochs, [h.get("avg_sync_score", 0) for h in history],
                color=COLORS["real"], lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Sync-Score")
        ax.set_title("Sync-Score (Real Clips)")

        # Temperature + LR
        ax = axes[2]
        ax.plot(epochs, [h.get("temperature", 0.07) for h in history],
                color=COLORS["dfdc"], lw=2, label="τ")
        ax2 = ax.twinx()
        ax2.plot(epochs, [h.get("lr", 0) for h in history],
                 color=COLORS["baseline"], lw=2, label="LR")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Temperature τ")
        ax2.set_ylabel("Learning Rate")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.set_title("Temperature & Learning Rate")

    else:  # finetune
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Loss components
        ax = axes[0]
        ax.plot(epochs, [h["train_loss"] for h in history],
                color=COLORS["syncguard"], lw=2, label="Total")
        ax.plot(epochs, [h.get("train_loss_infonce", 0) for h in history],
                color=COLORS["celebdf"], lw=1.5, ls="--", label="InfoNCE")
        ax.plot(epochs, [h.get("train_loss_temp", 0) for h in history],
                color=COLORS["dfdc"], lw=1.5, ls="--", label="Temporal")
        ax.plot(epochs, [h.get("train_loss_cls", 0) for h in history],
                color=COLORS["fake"], lw=1.5, ls="--", label="BCE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Fine-tuning Loss Components")
        ax.legend()

        # Val AUC + EER
        ax = axes[1]
        ax.plot(epochs, [h.get("val_auc", 0.5) for h in history],
                color=COLORS["real"], lw=2, label="AUC-ROC")
        ax.plot(epochs, [h.get("val_eer", 0.5) for h in history],
                color=COLORS["fake"], lw=2, label="EER")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title("Validation AUC & EER")
        ax.legend()

        # Hard negative ratio + LR
        ax = axes[2]
        ax.plot(epochs, [h.get("hard_negative_ratio", 0) for h in history],
                color=COLORS["wavlip"], lw=2, label="HN Ratio")
        ax2 = ax.twinx()
        ax2.plot(epochs, [h.get("lr", 0) for h in history],
                 color=COLORS["baseline"], lw=2, label="LR")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hard Negative Ratio")
        ax2.set_ylabel("Learning Rate")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.set_title("Hard Negatives & Learning Rate")

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_ablation_bar(
    ablation_results: dict[str, float],
    title: str,
    ylabel: str,
    output_path: str | Path,
    highlight: str = None,
):
    """Plot ablation comparison bar chart.

    Args:
        ablation_results: Dict mapping variant name to metric value.
        title: Plot title.
        ylabel: Y-axis label (e.g., "AUC-ROC").
        output_path: Save path.
        highlight: Variant name to highlight (our primary choice).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(ablation_results.keys())
    values = list(ablation_results.values())

    colors = []
    for name in names:
        if name == highlight:
            colors.append(COLORS["syncguard"])
        else:
            colors.append(COLORS["baseline"])

    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim([min(values) * 0.9, min(max(values) * 1.08, 1.02)])

    # Rotate x labels if needed
    if max(len(n) for n in names) > 10:
        plt.xticks(rotation=30, ha="right")

    _save_fig(fig, output_path)


def plot_per_category_auc(
    per_category: dict[str, float],
    output_path: str | Path,
):
    """Plot FakeAVCeleb per-category AUC bar chart.

    Args:
        per_category: Dict mapping category to AUC.
        output_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    cats = list(per_category.keys())
    aucs = list(per_category.values())
    colors = [COLORS.get(c, COLORS["baseline"]) for c in cats]

    bars = ax.bar(cats, aucs, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11
        )

    ax.set_ylabel("AUC-ROC")
    ax.set_title("FakeAVCeleb — Per-Category Detection AUC")
    ax.set_ylim([0.5, 1.02])

    _save_fig(fig, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_dir = Path("outputs/visualizations")

    rng = np.random.RandomState(42)

    # --- Test ROC curve ---
    fpr = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0])
    tpr = np.array([0, 0.4, 0.65, 0.82, 0.91, 0.96, 1.0])
    plot_roc_curve(fpr, tpr, auc=0.89, dataset_name="FakeAVCeleb",
                   output_path=output_dir / "roc_curves" / "test_roc",
                   eer=0.15)

    # --- Test multi-dataset ROC ---
    results = {
        "fakeavceleb": {"fpr": fpr, "tpr": tpr, "auc": 0.89},
        "celebdf": {"fpr": fpr, "tpr": tpr * 0.92, "auc": 0.81},
        "dfdc": {"fpr": fpr, "tpr": tpr * 0.85, "auc": 0.74},
    }
    plot_roc_multi_dataset(results, output_dir / "roc_curves" / "test_cross_dataset")

    # --- Test sync-score curves ---
    T = 100
    real_sync = rng.normal(0.7, 0.1, (10, T)).clip(-1, 1)
    fake_sync = rng.normal(0.2, 0.15, (10, T)).clip(-1, 1)
    plot_sync_score_curves(real_sync, fake_sync,
                           output_dir / "sync_scores" / "test_sync_curves")

    # --- Test distribution ---
    plot_sync_score_distribution(
        real_sync.mean(axis=1), fake_sync.mean(axis=1),
        output_dir / "sync_scores" / "test_sync_dist",
    )

    # --- Test training curves ---
    history = [
        {"epoch": i, "train_loss": 5.0 - i * 0.15, "val_loss": 5.1 - i * 0.12,
         "avg_sync_score": 0.1 + i * 0.04, "temperature": 0.07 + i * 0.002,
         "lr": 1e-4 * (1 - i / 20)}
        for i in range(20)
    ]
    plot_training_curves(history, "pretrain",
                         output_dir / "training_curves" / "test_pretrain")

    # --- Test ablation bar ---
    plot_ablation_bar(
        {"AV-HuBERT": 0.89, "ResNet-18": 0.76, "SyncNet": 0.72},
        title="Visual Encoder Ablation",
        ylabel="AUC-ROC",
        output_path=output_dir / "ablation_charts" / "test_visual_encoder",
        highlight="AV-HuBERT",
    )

    # --- Test per-category ---
    plot_per_category_auc(
        {"FV-RA": 0.91, "RV-FA": 0.78, "FV-FA": 0.95},
        output_dir / "ablation_charts" / "test_per_category",
    )

    print("All visualization tests passed.")
