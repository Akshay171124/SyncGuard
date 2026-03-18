"""SyncGuard evaluation framework.

Provides metrics (AUC-ROC, EER, pAUC), inference runner, and visualization tools.
"""

from src.evaluation.metrics import (
    EvaluationResult,
    compute_all_metrics,
    compute_auc_roc,
    compute_eer,
    compute_pauc,
    compute_per_category_auc,
    compute_bootstrap_ci,
    FAKEAVCELEB_CATEGORIES,
)
from src.evaluation.evaluate import (
    run_inference,
    evaluate_test_set,
    evaluate,
    load_checkpoint,
)
from src.evaluation.visualize import (
    plot_roc_curve,
    plot_roc_multi_dataset,
    plot_roc_per_category,
    plot_sync_score_curves,
    plot_sync_score_distribution,
    plot_training_curves,
    plot_ablation_bar,
    plot_per_category_auc,
)

__all__ = [
    # Metrics
    "EvaluationResult",
    "compute_all_metrics",
    "compute_auc_roc",
    "compute_eer",
    "compute_pauc",
    "compute_per_category_auc",
    "compute_bootstrap_ci",
    "FAKEAVCELEB_CATEGORIES",
    # Evaluation runner
    "run_inference",
    "evaluate_test_set",
    "evaluate",
    "load_checkpoint",
    # Visualization
    "plot_roc_curve",
    "plot_roc_multi_dataset",
    "plot_roc_per_category",
    "plot_sync_score_curves",
    "plot_sync_score_distribution",
    "plot_training_curves",
    "plot_ablation_bar",
    "plot_per_category_auc",
]
