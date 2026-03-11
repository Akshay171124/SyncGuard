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


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for SyncGuard")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fakeavceleb", "celebdf"],
        help="Dataset name",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to raw dataset root directory",
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

    # Load dataset
    logger.info(f"Loading {args.dataset} from {args.data_dir}")
    loader = get_dataset_loader(args.dataset, args.data_dir)
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
    results = pipeline.process_dataset(samples)
    pipeline.close()

    # Summary
    n_success = sum(1 for r in results if "error" not in r and "error_video" not in r)
    n_fail = len(results) - n_success
    logger.info(f"Done. Success: {n_success}, Failed: {n_fail}")


if __name__ == "__main__":
    main()
