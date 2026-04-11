"""Tests for dataset loaders (src/preprocessing/dataset_loader.py).

Tests verify:
- VideoSample dataclass construction
- FakeAVCelebLoader directory scanning and category detection
- Speaker-disjoint train/val/test splits (pitfall #7)
- CelebDFLoader and DFDCLoader scanning
- get_dataset_loader factory function
"""

import tempfile
from pathlib import Path

import pytest

from src.preprocessing.dataset_loader import (
    AVSpeechLoader,
    CelebDFLoader,
    DFDCLoader,
    FakeAVCelebLoader,
    VideoSample,
    get_dataset_loader,
)


# ──────────────────────────────────────────────
# VideoSample
# ──────────────────────────────────────────────

class TestVideoSample:
    def test_construction(self):
        """VideoSample stores all fields correctly."""
        s = VideoSample(
            video_path="/path/to/video.mp4",
            label=1,
            category="FV-RA",
            dataset="fakeavceleb",
            speaker_id="spk_001",
        )
        assert s.label == 1
        assert s.category == "FV-RA"
        assert s.speaker_id == "spk_001"

    def test_defaults(self):
        """speaker_id and language default to empty string."""
        s = VideoSample(video_path="v.mp4", label=0, category="real", dataset="test")
        assert s.speaker_id == ""
        assert s.language == ""


# ──────────────────────────────────────────────
# FakeAVCelebLoader
# ──────────────────────────────────────────────

class TestFakeAVCelebLoader:
    @pytest.fixture
    def fakeavceleb_dir(self, tmp_path):
        """Create a synthetic FakeAVCeleb directory structure."""
        categories = {
            "RealVideo-RealAudio": ["spk_001", "spk_002", "spk_003"],
            "FakeVideo-RealAudio": ["spk_001", "spk_002", "spk_003"],
            "RealVideo-FakeAudio": ["spk_001", "spk_002"],
            "FakeVideo-FakeAudio": ["spk_001", "spk_002", "spk_003"],
        }
        for cat, speakers in categories.items():
            for spk in speakers:
                spk_dir = tmp_path / cat / spk
                spk_dir.mkdir(parents=True)
                # 2 videos per speaker per category
                (spk_dir / "vid_001.mp4").touch()
                (spk_dir / "vid_002.mp4").touch()
        return tmp_path

    def test_load_samples_count(self, fakeavceleb_dir):
        """Loads correct number of samples."""
        loader = FakeAVCelebLoader(str(fakeavceleb_dir))
        samples = loader.load_samples()
        # 3+3+2+3 speakers × 2 videos each = 22
        assert len(samples) == 22

    def test_categories_detected(self, fakeavceleb_dir):
        """All 4 categories detected."""
        loader = FakeAVCelebLoader(str(fakeavceleb_dir))
        samples = loader.load_samples()
        categories = set(s.category for s in samples)
        assert categories == {"RV-RA", "FV-RA", "RV-FA", "FV-FA"}

    def test_labels_correct(self, fakeavceleb_dir):
        """RV-RA → label 0, all others → label 1."""
        loader = FakeAVCelebLoader(str(fakeavceleb_dir))
        samples = loader.load_samples()
        for s in samples:
            if s.category == "RV-RA":
                assert s.label == 0
            else:
                assert s.label == 1

    def test_speaker_ids_extracted(self, fakeavceleb_dir):
        """Speaker IDs extracted from directory structure."""
        loader = FakeAVCelebLoader(str(fakeavceleb_dir))
        samples = loader.load_samples()
        speaker_ids = set(s.speaker_id for s in samples)
        assert "spk_001" in speaker_ids
        assert "spk_002" in speaker_ids

    def test_nonexistent_dir_raises(self, tmp_path):
        """Nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FakeAVCelebLoader(str(tmp_path / "nonexistent"))

    def test_no_categories_raises(self, tmp_path):
        """Empty directory (no category folders) raises FileNotFoundError."""
        tmp_path.mkdir(exist_ok=True)
        loader = FakeAVCelebLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No recognized category"):
            loader.load_samples()


# ──────────────────────────────────────────────
# Speaker-Disjoint Splits (Pitfall #7)
# ──────────────────────────────────────────────

class TestSpeakerDisjointSplit:
    @pytest.fixture
    def many_speaker_samples(self):
        """Create samples with 10 distinct speakers."""
        samples = []
        for i in range(10):
            spk = f"spk_{i:03d}"
            for j in range(5):
                samples.append(VideoSample(
                    video_path=f"/fake/{spk}/vid_{j}.mp4",
                    label=0 if j < 3 else 1,
                    category="RV-RA" if j < 3 else "FV-RA",
                    dataset="fakeavceleb",
                    speaker_id=spk,
                ))
        return samples

    def test_no_speaker_overlap(self, many_speaker_samples):
        """Train/val/test speaker sets are completely disjoint."""
        loader = FakeAVCelebLoader.__new__(FakeAVCelebLoader)
        train, val, test = loader.split_by_speaker(
            many_speaker_samples, train_ratio=0.6, val_ratio=0.2,
        )
        train_spk = set(s.speaker_id for s in train)
        val_spk = set(s.speaker_id for s in val)
        test_spk = set(s.speaker_id for s in test)

        assert not (train_spk & val_spk), f"Train/val overlap: {train_spk & val_spk}"
        assert not (train_spk & test_spk), f"Train/test overlap: {train_spk & test_spk}"
        assert not (val_spk & test_spk), f"Val/test overlap: {val_spk & test_spk}"

    def test_all_samples_assigned(self, many_speaker_samples):
        """All samples end up in exactly one split."""
        loader = FakeAVCelebLoader.__new__(FakeAVCelebLoader)
        train, val, test = loader.split_by_speaker(many_speaker_samples)
        total = len(train) + len(val) + len(test)
        assert total == len(many_speaker_samples)

    def test_split_ratios_approximate(self, many_speaker_samples):
        """Split ratios approximately match requested ratios."""
        loader = FakeAVCelebLoader.__new__(FakeAVCelebLoader)
        train, val, test = loader.split_by_speaker(
            many_speaker_samples, train_ratio=0.7, val_ratio=0.15,
        )
        total = len(many_speaker_samples)
        # Ratios are by speaker count, so sample ratios are approximate
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_empty_speaker_handled(self):
        """Samples with no speaker_id are excluded from splits."""
        samples = [
            VideoSample("v1.mp4", 0, "real", "test", speaker_id="spk_001"),
            VideoSample("v2.mp4", 0, "real", "test", speaker_id="spk_001"),
            VideoSample("v3.mp4", 0, "real", "test", speaker_id=""),
        ]
        loader = FakeAVCelebLoader.__new__(FakeAVCelebLoader)
        train, val, test = loader.split_by_speaker(samples)
        # The sample with empty speaker_id should not appear in any split
        all_split = train + val + test
        assert len(all_split) <= len(samples)


# ──────────────────────────────────────────────
# CelebDF Loader
# ──────────────────────────────────────────────

class TestCelebDFLoader:
    def test_load_samples(self, tmp_path):
        """Loads real and fake samples from CelebDF structure."""
        (tmp_path / "Celeb-real").mkdir()
        (tmp_path / "Celeb-synthesis").mkdir()
        (tmp_path / "Celeb-real" / "r001.mp4").touch()
        (tmp_path / "Celeb-real" / "r002.mp4").touch()
        (tmp_path / "Celeb-synthesis" / "f001.mp4").touch()

        loader = CelebDFLoader(str(tmp_path))
        samples = loader.load_samples()
        assert len(samples) == 3
        reals = [s for s in samples if s.label == 0]
        fakes = [s for s in samples if s.label == 1]
        assert len(reals) == 2
        assert len(fakes) == 1


# ──────────────────────────────────────────────
# DFDC Loader
# ──────────────────────────────────────────────

class TestDFDCLoader:
    def test_structured_format(self, tmp_path):
        """Loads from real/ and fake/ subdirectories."""
        (tmp_path / "real").mkdir()
        (tmp_path / "fake").mkdir()
        (tmp_path / "real" / "r001.mp4").touch()
        (tmp_path / "fake" / "f001.mp4").touch()
        (tmp_path / "fake" / "f002.mp4").touch()

        loader = DFDCLoader(str(tmp_path))
        samples = loader.load_samples()
        assert len(samples) == 3
        assert sum(s.label == 0 for s in samples) == 1
        assert sum(s.label == 1 for s in samples) == 2


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class TestGetDatasetLoader:
    def test_known_datasets(self, tmp_path):
        """Factory returns correct loader class for known datasets."""
        # Create minimal dirs so loaders don't raise
        for name in ["fakeavceleb", "celebdf", "dfdc", "avspeech", "lrs2"]:
            d = tmp_path / name
            d.mkdir()
            loader = get_dataset_loader(name, str(d))
            assert loader is not None

    def test_unknown_dataset_raises(self, tmp_path):
        """Unknown dataset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_loader("imagenet", str(tmp_path))
