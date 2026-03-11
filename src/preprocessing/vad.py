import numpy as np
import torch


class VoiceActivityDetector:
    """Silero-VAD wrapper for detecting speech segments in audio."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate

        # Load Silero VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (
            self.get_speech_timestamps,
            _,
            self.read_audio,
            *_,
        ) = self.utils

    def detect_speech(self, waveform: np.ndarray) -> list[dict]:
        """Detect speech segments in a waveform.

        Args:
            waveform: (num_samples,) float32 array at self.sample_rate

        Returns:
            List of dicts with 'start' and 'end' sample indices.
        """
        wav_tensor = torch.from_numpy(waveform).float()
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.squeeze()

        speech_timestamps = self.get_speech_timestamps(
            wav_tensor,
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            sampling_rate=self.sample_rate,
        )
        return speech_timestamps

    def get_speech_mask(self, waveform: np.ndarray, num_frames: int, fps: int = 49) -> np.ndarray:
        """Create a per-frame binary mask indicating speech activity.

        Args:
            waveform: (num_samples,) float32 array
            num_frames: number of output frames (e.g., Wav2Vec frames)
            fps: frame rate of the output mask

        Returns:
            (num_frames,) bool array — True where speech is active
        """
        speech_segments = self.detect_speech(waveform)
        mask = np.zeros(num_frames, dtype=bool)

        for seg in speech_segments:
            start_frame = int(seg["start"] / self.sample_rate * fps)
            end_frame = int(seg["end"] / self.sample_rate * fps)
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)
            mask[start_frame:end_frame] = True

        return mask

    def filter_non_speech_frames(
        self,
        features: np.ndarray,
        speech_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Zero out features for non-speech frames.

        Returns:
            filtered_features: same shape, non-speech frames zeroed
            speech_mask: the boolean mask used
        """
        filtered = features.copy()
        filtered[~speech_mask] = 0.0
        return filtered, speech_mask
