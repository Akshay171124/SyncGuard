"""CLI script to preprocess a dataset for SyncGuard.

Usage:
    python scripts/preprocess_dataset.py --dataset fakeavceleb --data_dir data/raw/FakeAVCeleb
    python scripts/preprocess_dataset.py --dataset celebdf --data_dir data/raw/CelebDF-v2
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.preprocessing.dataset_loader import get_dataset_loader
from src.preprocessing.pipeline import PreprocessingPipeline


def summarize_results(results: list[dict]) -> tuple[int, int, dict]:
    """Count usable samples and tally the failure modes.

    A sample counts as successful only if it carries no error key at all.
    The previous check looked at "error" and "error_video" but not
    "error_audio", so a run with ffmpeg missing reported 100% success while
    producing samples with no audio — useless for audio-visual sync training,
    and invisible until training failed much later.

    Args:
        results: Per-sample result dicts from the preprocessing pipeline.

    Returns:
        Tuple of (successes, failures, {error_key: count}).
    """
    from collections import Counter

    def errors(r: dict) -> list[str]:
        return sorted(k for k in r if k.startswith("error"))

    n_success = sum(1 for r in results if not errors(r))
    breakdown = Counter(k for r in results for k in errors(r))
    return n_success, len(results) - n_success, dict(breakdown)


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for SyncGuard")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fakeavceleb", "celebdf", "dfdc", "avspeech", "lrs2"],
        help="Dataset name",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to raw dataset root directory (defaults to config value)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to process (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for preprocessing",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load config
    config = load_config(args.config)

    # Resolve data directory from args or config
    data_dir = args.data_dir
    if data_dir is None:
        dir_key = f"{args.dataset}_dir"
        data_dir = config["data"].get(dir_key)
        if not data_dir:
            logger.error(f"No --data_dir provided and no '{dir_key}' in config")
            sys.exit(1)

    # Load dataset
    logger.info(f"Loading {args.dataset} from {data_dir}")
    loader = get_dataset_loader(args.dataset, data_dir)
    samples = loader.load_samples()
    logger.info(f"Found {len(samples)} samples")

    if args.max_samples:
        samples = samples[: args.max_samples]
        logger.info(f"Limiting to {len(samples)} samples")

    # Log category distribution
    from collections import Counter
    cat_counts = Counter(s.category for s in samples)
    for cat, count in sorted(cat_counts.items()):
        logger.info(f"  {cat}: {count} samples")

    # Run preprocessing
    pipeline = PreprocessingPipeline(config)
    results = pipeline.process_dataset(samples, max_workers=args.workers)
    pipeline.close()

    # Summary
    n_success, n_fail, breakdown = summarize_results(results)
    logger.info(f"Done. Success: {n_success}, Failed: {n_fail}")
    for key, count in sorted(breakdown.items()):
        logger.warning(f"  {key}: {count} samples")


if __name__ == "__main__":
    main()
