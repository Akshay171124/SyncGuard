import csv
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoSample:
    """Represents a single video sample with metadata."""
    video_path: str
    label: int          # 0 = real, 1 = fake
    category: str       # e.g., "RV-RA", "FV-RA", "RV-FA", "FV-FA", "real"
    dataset: str        # e.g., "fakeavceleb", "celebdf", "dfdc"
    speaker_id: str = ""
    language: str = ""


class FakeAVCelebLoader:
    """Loader for the FakeAVCeleb dataset.

    Expected directory structure:
        FakeAVCeleb/
        ├── RealVideo-RealAudio/
        │   ├── <speaker_id>/
        │   │   ├── *.mp4
        ├── FakeVideo-RealAudio/
        │   ├── <speaker_id>/
        │   │   ├── *.mp4
        ├── RealVideo-FakeAudio/
        │   ├── ...
        ├── FakeVideo-FakeAudio/
        │   ├── ...
        └── meta_data.csv  (if available)
    """

    CATEGORY_MAP = {
        "RealVideo-RealAudio": ("RV-RA", 0),
        "FakeVideo-RealAudio": ("FV-RA", 1),
        "RealVideo-FakeAudio": ("RV-FA", 1),
        "FakeVideo-FakeAudio": ("FV-FA", 1),
    }

    # Alternative folder naming conventions
    ALT_CATEGORY_MAP = {
        "RV-RA": ("RV-RA", 0),
        "FV-RA": ("FV-RA", 1),
        "RV-FA": ("RV-FA", 1),
        "FV-FA": ("FV-FA", 1),
    }

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"FakeAVCeleb root not found: {self.root}")

    def _find_category_dirs(self) -> dict[str, tuple[str, int]]:
        """Auto-detect category folder names."""
        found = {}
        for dirname, mapping in {**self.CATEGORY_MAP, **self.ALT_CATEGORY_MAP}.items():
            dirpath = self.root / dirname
            if dirpath.is_dir():
                found[dirname] = mapping
        return found

    def load_samples(self) -> list[VideoSample]:
        """Scan the dataset directory and return all video samples."""
        samples = []
        category_dirs = self._find_category_dirs()

        if not category_dirs:
            raise FileNotFoundError(
                f"No recognized category folders found in {self.root}. "
                f"Expected: {list(self.CATEGORY_MAP.keys())}"
            )

        for dirname, (category, label) in category_dirs.items():
            dir_path = self.root / dirname
            for video_file in sorted(dir_path.rglob("*.mp4")):
                # Try to extract speaker ID from path
                # Typical: FakeAVCeleb/RealVideo-RealAudio/<speaker_id>/video.mp4
                parts = video_file.relative_to(dir_path).parts
                speaker_id = parts[0] if len(parts) > 1 else ""

                samples.append(VideoSample(
                    video_path=str(video_file),
                    label=label,
                    category=category,
                    dataset="fakeavceleb",
                    speaker_id=speaker_id,
                ))

        return samples

    def split_by_speaker(
        self, samples: list[VideoSample], train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> tuple[list[VideoSample], list[VideoSample], list[VideoSample]]:
        """Split samples by speaker ID to prevent identity leakage."""
        speakers = sorted(set(s.speaker_id for s in samples if s.speaker_id))
        n = len(speakers)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_speakers = set(speakers[:n_train])
        val_speakers = set(speakers[n_train : n_train + n_val])
        test_speakers = set(speakers[n_train + n_val :])

        train = [s for s in samples if s.speaker_id in train_speakers]
        val = [s for s in samples if s.speaker_id in val_speakers]
        test = [s for s in samples if s.speaker_id in test_speakers]

        return train, val, test


class CelebDFLoader:
    """Loader for CelebDF-v2 dataset.

    Expected directory structure:
        Celeb-DF-v2/
        ├── Celeb-real/
        │   ├── *.mp4
        ├── Celeb-synthesis/
        │   ├── *.mp4
        ├── YouTube-real/
        │   ├── *.mp4
        └── List_of_testing_videos.txt
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"CelebDF-v2 root not found: {self.root}")

    def load_samples(self) -> list[VideoSample]:
        """Scan the dataset directory and return all video samples."""
        samples = []

        folder_label_map = {
            "Celeb-real": ("real", 0),
            "Celeb-synthesis": ("fake", 1),
            "YouTube-real": ("real", 0),
        }

        for folder_name, (category, label) in folder_label_map.items():
            folder_path = self.root / folder_name
            if not folder_path.is_dir():
                continue
            for video_file in sorted(folder_path.rglob("*.mp4")):
                samples.append(VideoSample(
                    video_path=str(video_file),
                    label=label,
                    category=category,
                    dataset="celebdf",
                ))

        return samples

    def load_test_list(self) -> list[str]:
        """Load the official test split file if available."""
        test_file = self.root / "List_of_testing_videos.txt"
        if not test_file.exists():
            return []
        with open(test_file) as f:
            return [line.strip().split()[-1] for line in f if line.strip()]


def get_dataset_loader(dataset_name: str, root_dir: str):
    """Factory function to get the appropriate dataset loader."""
    loaders = {
        "fakeavceleb": FakeAVCelebLoader,
        "celebdf": CelebDFLoader,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    return loaders[dataset_name](root_dir)
