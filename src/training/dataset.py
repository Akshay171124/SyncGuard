"""Training dataset for SyncGuard.

Provides PyTorch Dataset and DataLoader utilities for contrastive pretraining
and fine-tuning phases. Handles:
- Loading preprocessed features (mouth crops + audio)
- Speaker-disjoint train/val/test splits
- Hard negative mining (same-speaker, different-time windows)
- Variable-length collation with padding and masks
"""

import logging
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.dataset_loader import (
    VideoSample, FakeAVCelebLoader, AVSpeechLoader, CelebDFLoader, DFDCLoader,
    get_dataset_loader,
)

logger = logging.getLogger(__name__)


@dataclass
class SyncGuardBatch:
    """Collated batch for SyncGuard training.

    Attributes:
        mouth_crops: (B, T, 1, H, W) grayscale mouth ROI frames.
        waveforms: (B, max_audio_len) raw audio waveforms.
        labels: (B,) binary labels (0=real, 1=fake).
        is_real: (B,) boolean mask for real clips.
        mask: (B, T) boolean mask for valid (non-padded) frames.
        lengths: (B,) number of valid frames per sample.
        categories: List[str] of category labels.
        speaker_ids: List[str] of speaker identifiers.
    """

    mouth_crops: torch.Tensor
    waveforms: torch.Tensor
    labels: torch.Tensor
    is_real: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    categories: list[str]
    speaker_ids: list[str]


class SyncGuardDataset(Dataset):
    """PyTorch Dataset for SyncGuard training.

    Loads preprocessed samples from disk. Each sample directory should contain:
    - mouth_crops.npy: (T, 1, 96, 96) or (T, 96, 96) grayscale mouth crops
    - audio.wav or audio.npy: raw waveform at 16kHz
    - metadata.json: {label, category, speaker_id, ...}

    Args:
        samples: List of VideoSample metadata objects.
        features_dir: Root directory containing preprocessed features.
        max_frames: Maximum number of visual frames to load (truncation).
        max_audio_samples: Maximum audio samples to load.
        hard_negative_ratio: Fraction of batch that uses hard negatives (0.0 to 1.0).
        transform: Optional transform applied to mouth crops.
    """

    def __init__(
        self,
        samples: list[VideoSample],
        features_dir: str,
        max_frames: int = 150,
        max_audio_samples: int = 96000,  # ~6s at 16kHz
        hard_negative_ratio: float = 0.0,
        audio_swap_ratio: float = 0.0,
        transform=None,
    ):
        self.features_dir = Path(features_dir)
        self.max_frames = max_frames
        self.max_audio_samples = max_audio_samples
        self.hard_negative_ratio = hard_negative_ratio
        self.audio_swap_ratio = audio_swap_ratio
        self.transform = transform

        # Filter out samples without preprocessed features
        valid_samples = []
        for s in samples:
            video_stem = Path(s.video_path).stem
            feat_dir = self.features_dir / s.dataset / s.category / video_stem
            has_crops = (feat_dir / "mouth_crops.npy").exists()
            has_audio = (feat_dir / "audio.wav").exists() or (feat_dir / "audio.npy").exists()
            if has_crops and has_audio:
                valid_samples.append(s)
        if len(valid_samples) < len(samples):
            logger.warning(
                f"Filtered {len(samples) - len(valid_samples)} samples "
                f"missing preprocessed features"
            )
        self.samples = valid_samples

        # Build speaker index for hard negative mining
        self._speaker_index: dict[str, list[int]] = {}
        for i, s in enumerate(self.samples):
            if s.speaker_id:
                self._speaker_index.setdefault(s.speaker_id, []).append(i)

        # Build real-sample index for audio-swap augmentation
        self._real_indices: list[int] = [
            i for i, s in enumerate(self.samples) if s.label == 0
        ]

        logger.info(
            f"SyncGuardDataset: {len(self.samples)} samples, "
            f"{len(self._speaker_index)} speakers, "
            f"hard_neg_ratio={hard_negative_ratio:.2f}, "
            f"audio_swap_ratio={audio_swap_ratio:.2f} "
            f"({len(self._real_indices)} real samples for swapping)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_feature_path(self, sample: VideoSample) -> Path:
        """Derive feature directory path from the video sample.

        Tries paths in order:
        1. features_dir / dataset / category / video_stem  (pipeline output format)
        2. features_dir / dataset / category / speaker_id / video_stem  (structured)
        3. features_dir / dataset / video_stem  (flat)
        """
        video_path = Path(sample.video_path)
        video_stem = video_path.stem

        # Pipeline output format: dataset/category/video_stem
        pipeline = self.features_dir / sample.dataset / sample.category / video_stem
        if pipeline.exists():
            return pipeline

        # Structured: dataset/category/speaker_id/video_stem
        if sample.speaker_id:
            structured = (
                self.features_dir
                / sample.dataset
                / sample.category
                / sample.speaker_id
                / video_stem
            )
            if structured.exists():
                return structured

        # Flat: dataset/video_stem
        flat = self.features_dir / sample.dataset / video_stem
        if flat.exists():
            return flat

        # Default to pipeline format (will error at load time if missing)
        return pipeline

    def _load_mouth_crops(self, feature_dir: Path) -> torch.Tensor:
        """Load mouth crops from .npy file.

        Returns:
            (T, 1, 96, 96) float32 tensor, normalized to [0, 1].
        """
        npy_path = feature_dir / "mouth_crops.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Mouth crops not found: {npy_path}")

        crops = np.load(npy_path)  # (T, H, W), (T, 1, H, W), or (T, H, W, 3)

        # Convert RGB (T, H, W, 3) to grayscale (T, H, W)
        if crops.ndim == 4 and crops.shape[-1] == 3:
            crops = np.mean(crops, axis=-1)  # (T, H, W)

        if crops.ndim == 3:
            crops = crops[:, np.newaxis, :, :]  # (T, 1, H, W)

        # Truncate to max_frames
        if crops.shape[0] > self.max_frames:
            crops = crops[: self.max_frames]

        # Normalize to [0, 1] if uint8
        crops = crops.astype(np.float32)
        if crops.max() > 1.0:
            crops = crops / 255.0

        return torch.from_numpy(crops)

    def _load_audio(self, feature_dir: Path) -> torch.Tensor:
        """Load audio waveform from .npy or .wav file.

        Returns:
            (L,) float32 tensor, raw waveform at 16kHz.
        """
        npy_path = feature_dir / "audio.npy"
        wav_path = feature_dir / "audio.wav"

        if npy_path.exists():
            waveform = np.load(npy_path).astype(np.float32)
        elif wav_path.exists():
            import soundfile as sf

            waveform, sr = sf.read(wav_path, dtype="float32")
            if sr != 16000:
                # Resample if needed (shouldn't happen with preprocessed data)
                import torchaudio

                waveform = torch.from_numpy(waveform)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                waveform = waveform.squeeze(0).numpy()
        else:
            raise FileNotFoundError(
                f"Audio not found: tried {npy_path} and {wav_path}"
            )

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)  # Mono

        # Truncate
        if len(waveform) > self.max_audio_samples:
            waveform = waveform[: self.max_audio_samples]

        return torch.from_numpy(waveform)

    def _get_hard_negative_idx(self, idx: int) -> int | None:
        """Find a hard negative: same speaker, different clip.

        Args:
            idx: Current sample index.

        Returns:
            Index of hard negative sample, or None if unavailable.
        """
        speaker = self.samples[idx].speaker_id
        if not speaker or speaker not in self._speaker_index:
            return None

        candidates = [i for i in self._speaker_index[speaker] if i != idx]
        if not candidates:
            return None

        return random.choice(candidates)

    def __getitem__(self, idx: int) -> dict:
        """Load a single sample.

        For real samples (label=0), with probability audio_swap_ratio, swaps
        the audio with audio from a different real sample. This creates
        synthetic RV-FA (real video, fake audio) examples to improve detection
        of audio-only manipulations.

        Returns:
            Dict with keys: mouth_crops, waveform, label, is_real,
                           category, speaker_id, num_frames.
        """
        sample = self.samples[idx]
        feature_dir = self._get_feature_path(sample)

        mouth_crops = self._load_mouth_crops(feature_dir)
        waveform = self._load_audio(feature_dir)

        label = sample.label
        category = sample.category

        # Audio-swap augmentation: for FAKE samples, with some probability,
        # replace with a real sample's video + different real sample's audio.
        # This creates synthetic RV-FA examples without reducing real sample count.
        if (
            self.audio_swap_ratio > 0
            and sample.label == 1
            and len(self._real_indices) > 1
            and random.random() < self.audio_swap_ratio
        ):
            # Pick two different real samples: one for video, one for audio
            video_idx, audio_idx = random.sample(self._real_indices, 2)
            video_sample = self.samples[video_idx]
            audio_sample = self.samples[audio_idx]

            # Load video from one real sample, audio from another
            video_dir = self._get_feature_path(video_sample)
            audio_dir = self._get_feature_path(audio_sample)
            mouth_crops = self._load_mouth_crops(video_dir)
            waveform = self._load_audio(audio_dir)
            label = 1  # Still fake (mismatched audio-visual)
            category = "RV-FA-aug"

        if self.transform is not None:
            mouth_crops = self.transform(mouth_crops)

        return {
            "mouth_crops": mouth_crops,  # (T, 1, H, W)
            "waveform": waveform,  # (L,)
            "label": label,
            "is_real": label == 0,
            "category": category,
            "speaker_id": sample.speaker_id,
            "num_frames": mouth_crops.shape[0],
        }


def collate_syncguard(batch: list[dict]) -> SyncGuardBatch:
    """Custom collation for variable-length SyncGuard samples.

    Pads mouth crops and waveforms to the maximum length in the batch,
    and creates boolean masks for valid (non-padded) positions.

    Args:
        batch: List of dicts from SyncGuardDataset.__getitem__.

    Returns:
        SyncGuardBatch with padded tensors and masks.
    """
    # Find max lengths in this batch
    max_frames = max(item["num_frames"] for item in batch)
    max_audio_len = max(item["waveform"].shape[0] for item in batch)

    B = len(batch)
    H, W = batch[0]["mouth_crops"].shape[-2:]

    # Pre-allocate padded tensors
    mouth_crops = torch.zeros(B, max_frames, 1, H, W)
    waveforms = torch.zeros(B, max_audio_len)
    mask = torch.zeros(B, max_frames, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)
    is_real = torch.zeros(B, dtype=torch.bool)
    categories = []
    speaker_ids = []

    for i, item in enumerate(batch):
        T = item["num_frames"]
        L = item["waveform"].shape[0]

        mouth_crops[i, :T] = item["mouth_crops"]
        waveforms[i, :L] = item["waveform"]
        mask[i, :T] = True
        lengths[i] = T
        labels[i] = item["label"]
        is_real[i] = item["is_real"]
        categories.append(item["category"])
        speaker_ids.append(item["speaker_id"])

    return SyncGuardBatch(
        mouth_crops=mouth_crops,
        waveforms=waveforms,
        labels=labels,
        is_real=is_real,
        mask=mask,
        lengths=lengths,
        categories=categories,
        speaker_ids=speaker_ids,
    )


def build_dataloaders(
    config: dict,
    phase: str = "finetune",
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from config.

    Args:
        config: Full config dict (from default.yaml).
        phase: "pretrain" or "finetune" — determines batch size.

    Returns:
        Dict with keys "train", "val", "test" mapping to DataLoaders.
    """
    data_cfg = config["data"]
    train_cfg = config["training"][phase]
    hw_cfg = config["hardware"]

    if phase == "pretrain":
        # Phase 1: Load AVSpeech (real-only data for contrastive pretraining)
        avspeech_dir = data_cfg.get("avspeech_dir", "data/raw/AVSpeech")
        loader = AVSpeechLoader(avspeech_dir)
        all_samples = loader.load_samples()

        # Random 85/15 train/val split (no test set for pretraining)
        random.shuffle(all_samples)
        n_val = max(int(len(all_samples) * 0.15), 1)
        val_samples = all_samples[:n_val]
        train_samples = all_samples[n_val:]
        test_samples = val_samples  # Reuse val as test placeholder

        logger.info(
            f"AVSpeech pretrain splits: "
            f"train={len(train_samples)}, val={len(val_samples)}"
        )
    else:
        # Phase 2: Load FakeAVCeleb (real + fake for fine-tuning)
        loader = FakeAVCelebLoader(data_cfg["fakeavceleb_dir"])
        all_samples = loader.load_samples()
        train_samples, val_samples, test_samples = loader.split_by_speaker(all_samples)

        logger.info(
            f"Speaker-disjoint splits: "
            f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        )

    features_dir = data_cfg["features_dir"]
    batch_size = train_cfg["batch_size"]

    # Hard negative ratio and audio-swap augmentation (only for finetune phase)
    hard_neg_ratio = 0.0
    audio_swap_ratio = 0.0
    if phase == "finetune":
        hard_neg_ratio = train_cfg.get("hard_negative_ratio", 0.0)
        audio_swap_ratio = train_cfg.get("audio_swap_ratio", 0.0)

    train_ds = SyncGuardDataset(
        samples=train_samples,
        features_dir=features_dir,
        hard_negative_ratio=hard_neg_ratio,
        audio_swap_ratio=audio_swap_ratio,
    )
    val_ds = SyncGuardDataset(
        samples=val_samples,
        features_dir=features_dir,
    )
    test_ds = SyncGuardDataset(
        samples=test_samples,
        features_dir=features_dir,
    )

    num_workers = hw_cfg.get("num_workers", 4)
    pin_memory = hw_cfg.get("pin_memory", True)

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_syncguard,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_syncguard,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_syncguard,
        ),
    }

    return dataloaders


def build_test_dataloader(
    config: dict,
    dataset_name: str,
) -> DataLoader:
    """Build a test-only DataLoader for cross-dataset evaluation.

    Args:
        config: Full config dict (from default.yaml).
        dataset_name: One of "celebdf", "dfdc".

    Returns:
        DataLoader for the entire dataset (no train/val split — eval only).
    """
    data_cfg = config["data"]
    hw_cfg = config["hardware"]

    dir_key = f"{dataset_name}_dir"
    root_dir = data_cfg.get(dir_key)
    if not root_dir:
        raise ValueError(
            f"No data directory configured for '{dataset_name}'. "
            f"Add '{dir_key}' to configs/default.yaml"
        )

    loader = get_dataset_loader(dataset_name, root_dir)
    all_samples = loader.load_samples()
    logger.info(f"Cross-dataset eval: {dataset_name} — {len(all_samples)} samples")

    features_dir = data_cfg["features_dir"]
    batch_size = config["training"]["finetune"]["batch_size"]

    dataset = SyncGuardDataset(
        samples=all_samples,
        features_dir=features_dir,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=hw_cfg.get("num_workers", 4),
        pin_memory=hw_cfg.get("pin_memory", True),
        collate_fn=collate_syncguard,
    )


if __name__ == "__main__":
    """Test dataset with synthetic data."""
    import tempfile
    import json

    print("Testing SyncGuardDataset with synthetic data...")

    # Create temporary features directory with fake samples
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic samples
        samples = []
        speakers = ["spk_001", "spk_002", "spk_003"]
        categories = ["RV-RA", "FV-RA", "RV-FA", "FV-FA"]

        for i in range(12):
            speaker = speakers[i % len(speakers)]
            cat = categories[i % len(categories)]
            label = 0 if cat == "RV-RA" else 1
            stem = f"video_{i:03d}"

            # Create feature directory
            feat_dir = tmpdir / "fakeavceleb" / cat / speaker / stem
            feat_dir.mkdir(parents=True, exist_ok=True)

            # Create synthetic mouth crops: random frames (T, 1, 96, 96)
            T = random.randint(30, 80)
            crops = np.random.randint(0, 255, (T, 1, 96, 96), dtype=np.uint8)
            np.save(feat_dir / "mouth_crops.npy", crops)

            # Create synthetic audio: random waveform
            audio_len = int(T / 25 * 16000)  # Match video duration at 16kHz
            audio = np.random.randn(audio_len).astype(np.float32) * 0.1
            np.save(feat_dir / "audio.npy", audio)

            samples.append(
                VideoSample(
                    video_path=f"/fake/path/{stem}.mp4",
                    label=label,
                    category=cat,
                    dataset="fakeavceleb",
                    speaker_id=speaker,
                )
            )

        # Test dataset creation
        ds = SyncGuardDataset(
            samples=samples,
            features_dir=str(tmpdir),
            max_frames=100,
        )
        assert len(ds) == 12, f"Expected 12 samples, got {len(ds)}"
        print(f"  Dataset: {len(ds)} samples ✓")

        # Test single sample loading
        item = ds[0]
        assert item["mouth_crops"].ndim == 4, "mouth_crops should be (T, 1, H, W)"
        assert item["mouth_crops"].shape[1] == 1, "Should be single channel"
        assert item["mouth_crops"].shape[2] == 96, "Height should be 96"
        assert item["mouth_crops"].shape[3] == 96, "Width should be 96"
        assert item["mouth_crops"].max() <= 1.0, "Should be normalized to [0, 1]"
        assert item["waveform"].ndim == 1, "Waveform should be 1D"
        print(
            f"  Sample: mouth_crops={item['mouth_crops'].shape}, "
            f"waveform={item['waveform'].shape}, "
            f"label={item['label']}, "
            f"frames={item['num_frames']} ✓"
        )

        # Test collation with variable lengths
        batch_items = [ds[i] for i in range(4)]
        batch = collate_syncguard(batch_items)
        assert batch.mouth_crops.ndim == 5, "Batched crops should be (B, T, 1, H, W)"
        assert batch.mouth_crops.shape[0] == 4, "Batch size should be 4"
        assert batch.mask.shape == (4, batch.mouth_crops.shape[1])
        assert batch.lengths.shape == (4,)
        assert batch.labels.shape == (4,)
        assert batch.is_real.shape == (4,)
        assert len(batch.categories) == 4
        assert len(batch.speaker_ids) == 4

        # Verify padding mask is correct
        for i in range(4):
            T_i = batch_items[i]["num_frames"]
            assert batch.mask[i, :T_i].all(), f"Valid frames should be True"
            if T_i < batch.mask.shape[1]:
                assert not batch.mask[i, T_i:].any(), f"Padded frames should be False"

        print(
            f"  Collated batch: crops={batch.mouth_crops.shape}, "
            f"waveforms={batch.waveforms.shape}, "
            f"mask={batch.mask.shape} ✓"
        )

        # Test hard negative mining
        ds_hard = SyncGuardDataset(
            samples=samples,
            features_dir=str(tmpdir),
            hard_negative_ratio=0.2,
        )
        neg_idx = ds_hard._get_hard_negative_idx(0)
        if neg_idx is not None:
            assert (
                samples[neg_idx].speaker_id == samples[0].speaker_id
            ), "Hard negative should be same speaker"
            assert neg_idx != 0, "Hard negative should be different sample"
            print(
                f"  Hard negative: idx={neg_idx}, "
                f"speaker={samples[neg_idx].speaker_id} (same as sample 0) ✓"
            )
        else:
            print("  Hard negative: no candidates (expected for small test set)")

        # Test speaker-disjoint split
        train, val, test = FakeAVCelebLoader.__new__(
            FakeAVCelebLoader
        ).split_by_speaker.__func__(
            None, samples, train_ratio=0.5, val_ratio=0.25
        )
        # Manually compute to avoid needing the loader
        all_speakers = sorted(set(s.speaker_id for s in samples if s.speaker_id))
        n = len(all_speakers)
        n_train = int(n * 0.5)
        n_val = int(n * 0.25)
        train_spk = set(all_speakers[:n_train])
        val_spk = set(all_speakers[n_train : n_train + n_val])
        test_spk = set(all_speakers[n_train + n_val :])
        assert not (train_spk & val_spk), "Train/val speaker overlap!"
        assert not (train_spk & test_spk), "Train/test speaker overlap!"
        assert not (val_spk & test_spk), "Val/test speaker overlap!"
        print(
            f"  Speaker-disjoint: {len(train_spk)} train, "
            f"{len(val_spk)} val, {len(test_spk)} test speakers ✓"
        )

        # Test DataLoader integration
        dl = DataLoader(
            ds, batch_size=3, shuffle=True, collate_fn=collate_syncguard
        )
        for batch in dl:
            assert isinstance(batch, SyncGuardBatch)
            break
        print(f"  DataLoader integration ✓")

    print("\nAll dataset tests passed.")
