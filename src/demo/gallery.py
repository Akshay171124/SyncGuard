"""Gallery manifest + cached-tensor loader."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GalleryClip:
    clip_id: str
    category: str
    ground_truth: str
    hpc_confidence: float
    video_path: Path
    mouth_crops: np.ndarray
    audio_waveform: np.ndarray
    ear_features: np.ndarray
    duration_s: float


def _load_clip(gallery_dir, entry):
    cid = entry["clip_id"]
    npz = np.load(gallery_dir / f"{cid}.npz")
    return GalleryClip(
        clip_id=cid,
        category=entry["category"],
        ground_truth=entry["ground_truth"],
        hpc_confidence=float(entry["hpc_confidence"]),
        video_path=gallery_dir / f"{cid}.mp4",
        mouth_crops=npz["mouth_crops"],
        audio_waveform=npz["audio_waveform"],
        ear_features=npz["ear_features"],
        duration_s=float(npz["duration_s"]),
    )


def load_gallery(gallery_dir):
    """Load manifest + all .npz caches into memory."""
    gallery_dir = Path(gallery_dir)
    with open(gallery_dir / "manifest.json") as f:
        manifest = json.load(f)
    return [_load_clip(gallery_dir, entry) for entry in manifest["clips"]]


def load_gallery_lazy(gallery_dir):
    """Yield clips one at a time (lower memory)."""
    gallery_dir = Path(gallery_dir)
    with open(gallery_dir / "manifest.json") as f:
        manifest = json.load(f)
    for entry in manifest["clips"]:
        yield _load_clip(gallery_dir, entry)
