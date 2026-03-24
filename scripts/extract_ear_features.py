"""Extract EAR (Eye Aspect Ratio) features for already-preprocessed datasets.

Reads original video files, runs MediaPipe FaceMesh to get eye landmarks,
computes per-frame EAR, and saves ear_features.npy alongside existing
preprocessed data. Does NOT redo mouth crops, audio, or VAD.

Usage:
    python scripts/extract_ear_features.py --dataset fakeavceleb \
        --config configs/default.yaml

    python scripts/extract_ear_features.py --dataset dfdc \
        --config configs/default.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from src.utils.config import load_config
from src.utils.io import read_video_frames

# Eye landmark indices for MediaPipe FaceMesh (468-point model).
# Defined here to avoid importing face_detector (which pulls in
# retinaface → tensorflow, unnecessary for EAR extraction).
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

logger = logging.getLogger(__name__)


def compute_ear_from_landmarks(landmarks, h: int, w: int) -> float:
    """Compute average EAR from MediaPipe face landmarks."""
    def _eye_ear(indices):
        pts = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            pts.append(np.array([lm.x * w, lm.y * h]))
        p1, p2, p3, p4, p5, p6 = pts
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-6:
            return 0.0
        return (vertical1 + vertical2) / (2.0 * horizontal)

    left_ear = _eye_ear(LEFT_EYE_IDX)
    right_ear = _eye_ear(RIGHT_EYE_IDX)
    return (left_ear + right_ear) / 2.0


def extract_ear_for_video(
    video_path: str, target_fps: int, face_mesh
) -> np.ndarray | None:
    """Extract per-frame EAR values from a video.

    Args:
        video_path: Path to original video file.
        target_fps: Target FPS for frame extraction.
        face_mesh: MediaPipe FaceMesh instance.

    Returns:
        (T,) float32 array of EAR values, or None on failure.
    """
    try:
        frames, _ = read_video_frames(video_path, target_fps)
    except Exception as e:
        logger.warning(f"Failed to read video {video_path}: {e}")
        return None

    ear_values = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            ear = compute_ear_from_landmarks(results.multi_face_landmarks[0], h, w)
        else:
            ear = 0.0
        ear_values.append(ear)

    return np.array(ear_values, dtype=np.float32)


def find_processed_dirs(processed_root: Path, dataset: str) -> list[Path]:
    """Find all preprocessed sample directories for a dataset."""
    dataset_dir = processed_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {dataset_dir}")

    dirs = []
    for meta_file in sorted(dataset_dir.rglob("metadata.json")):
        dirs.append(meta_file.parent)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Extract EAR features for preprocessed datasets"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (fakeavceleb, dfdc, lrs2, avspeech)"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing ear_features.npy files"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    processed_dir = Path(config["data"]["processed_dir"])
    video_fps = config["preprocessing"]["video"]["fps"]

    # Find all preprocessed sample directories
    sample_dirs = find_processed_dirs(processed_dir, args.dataset)
    logger.info(f"Found {len(sample_dirs)} preprocessed samples for {args.dataset}")

    # Initialize MediaPipe FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    processed = 0
    skipped = 0
    failed = 0

    for i, sample_dir in enumerate(sample_dirs):
        ear_path = sample_dir / "ear_features.npy"

        # Skip if already exists
        if ear_path.exists() and not args.overwrite:
            skipped += 1
            continue

        # Load metadata to get original video path
        meta_path = sample_dir / "metadata.json"
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            logger.warning(f"Cannot read metadata: {meta_path}")
            failed += 1
            continue

        video_path = meta.get("video_path")
        if not video_path or not Path(video_path).exists():
            logger.warning(f"Video not found: {video_path}")
            failed += 1
            continue

        # Extract EAR
        ear_values = extract_ear_for_video(video_path, video_fps, face_mesh)
        if ear_values is None:
            failed += 1
            continue

        # Save
        np.save(str(ear_path), ear_values)

        # Update metadata
        meta["ear_features_path"] = str(ear_path)
        meta["mean_ear"] = float(
            ear_values[ear_values > 0].mean() if (ear_values > 0).any() else 0.0
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        processed += 1

        if (i + 1) % 100 == 0:
            logger.info(
                f"Progress: {i+1}/{len(sample_dirs)} "
                f"(processed={processed}, skipped={skipped}, failed={failed})"
            )

    face_mesh.close()

    logger.info(
        f"Done. processed={processed}, skipped={skipped}, failed={failed}, "
        f"total={len(sample_dirs)}"
    )


if __name__ == "__main__":
    main()
