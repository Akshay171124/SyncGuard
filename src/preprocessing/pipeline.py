import json
import logging
from pathlib import Path

import numpy as np

from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.audio_extractor import AudioExtractor
from src.preprocessing.vad import VoiceActivityDetector
from src.preprocessing.dataset_loader import VideoSample
from src.utils.io import read_video_frames

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """End-to-end preprocessing: video → mouth crops + audio + VAD mask."""

    def __init__(self, config: dict):
        pp_cfg = config["preprocessing"]

        # Video settings
        self.video_fps = pp_cfg["video"]["fps"]
        self.crop_size = pp_cfg["video"]["mouth_crop_size"]
        self.face_confidence = pp_cfg["video"]["face_detection_confidence"]

        # Audio settings
        self.sample_rate = pp_cfg["audio"]["sample_rate"]
        self.target_audio_fps = pp_cfg["audio"]["target_fps"]

        # VAD settings
        vad_cfg = pp_cfg["vad"]
        self.vad_threshold = vad_cfg["threshold"]

        # Output dirs
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.features_dir = Path(config["data"]["features_dir"])

        # Initialize components
        self.face_detector = FaceDetector(
            crop_size=self.crop_size,
            confidence_threshold=self.face_confidence,
        )
        self.audio_extractor = AudioExtractor(
            sample_rate=self.sample_rate,
            target_visual_fps=self.target_audio_fps,
        )
        self.vad = None  # Lazy init (downloads model on first use)

    def _ensure_vad(self):
        if self.vad is None:
            self.vad = VoiceActivityDetector(
                threshold=self.vad_threshold,
                sample_rate=self.sample_rate,
            )

    def process_single_video(
        self, sample: VideoSample, output_dir: Path | None = None
    ) -> dict:
        """Process a single video sample.

        Returns:
            Dict with paths to saved outputs and metadata.
        """
        video_path = Path(sample.video_path)
        video_id = video_path.stem

        if output_dir is None:
            output_dir = self.processed_dir / sample.dataset / sample.category / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "video_id": video_id,
            "video_path": str(video_path),
            "label": sample.label,
            "category": sample.category,
            "dataset": sample.dataset,
            "speaker_id": sample.speaker_id,
        }

        # 1. Extract video frames and mouth crops
        try:
            frames, original_fps = read_video_frames(str(video_path), self.video_fps)
            mouth_crops, valid_mask = self.face_detector.process_video_frames(frames)

            crops_path = output_dir / "mouth_crops.npy"
            mask_path = output_dir / "valid_mask.npy"
            np.save(str(crops_path), mouth_crops)
            np.save(str(mask_path), valid_mask)

            result["mouth_crops_path"] = str(crops_path)
            result["valid_mask_path"] = str(mask_path)
            result["num_frames"] = len(frames)
            result["num_valid_frames"] = int(valid_mask.sum())
            result["detection_rate"] = float(valid_mask.mean())

        except Exception as e:
            logger.warning(f"Video processing failed for {video_path}: {e}")
            result["error_video"] = str(e)
            return result

        # 2. Extract audio
        try:
            wav_path = output_dir / "audio.wav"
            waveform = self.audio_extractor.extract_from_video(
                str(video_path), str(wav_path)
            )
            result["audio_path"] = str(wav_path)
            result["audio_duration_s"] = len(waveform) / self.sample_rate

        except Exception as e:
            logger.warning(f"Audio extraction failed for {video_path}: {e}")
            result["error_audio"] = str(e)
            return result

        # 3. Voice Activity Detection
        try:
            self._ensure_vad()
            # Estimate number of Wav2Vec output frames
            num_audio_frames = len(waveform) // (self.sample_rate // self.target_audio_fps)
            speech_mask = self.vad.get_speech_mask(
                waveform, num_frames=num_audio_frames, fps=self.target_audio_fps
            )

            speech_mask_path = output_dir / "speech_mask.npy"
            np.save(str(speech_mask_path), speech_mask)
            result["speech_mask_path"] = str(speech_mask_path)
            result["speech_ratio"] = float(speech_mask.mean())

        except Exception as e:
            logger.warning(f"VAD failed for {video_path}: {e}")
            result["error_vad"] = str(e)

        # 4. Save metadata
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def process_dataset(
        self, samples: list[VideoSample], max_workers: int = 1
    ) -> list[dict]:
        """Process all samples in a dataset.

        Args:
            samples: List of VideoSample objects
            max_workers: Number of parallel workers (1 = sequential)

        Returns:
            List of result dicts
        """
        results = []
        total = len(samples)

        for i, sample in enumerate(samples):
            logger.info(f"Processing [{i+1}/{total}]: {sample.video_path}")
            try:
                result = self.process_single_video(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {sample.video_path}: {e}")
                results.append({
                    "video_path": sample.video_path,
                    "error": str(e),
                })

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i+1}/{total} videos processed")

        # Save manifest
        manifest_path = self.processed_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Manifest saved to {manifest_path}")

        return results

    def close(self):
        self.face_detector.close()
