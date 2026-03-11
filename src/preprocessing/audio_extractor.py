import numpy as np
import torch

from src.utils.io import extract_audio, load_audio


class AudioExtractor:
    """Extracts audio from video and prepares features for Wav2Vec 2.0."""

    def __init__(self, sample_rate: int = 16000, target_visual_fps: int = 49):
        self.sample_rate = sample_rate
        self.target_visual_fps = target_visual_fps

    def extract_from_video(self, video_path: str, output_wav_path: str) -> np.ndarray:
        """Extract audio from video, save as .wav, return waveform."""
        extract_audio(video_path, output_wav_path, self.sample_rate)
        return load_audio(output_wav_path, self.sample_rate)

    def load_waveform(self, audio_path: str) -> np.ndarray:
        """Load existing audio file."""
        return load_audio(audio_path, self.sample_rate)

    def waveform_to_tensor(self, waveform: np.ndarray) -> torch.Tensor:
        """Convert numpy waveform to torch tensor for Wav2Vec input.

        Returns:
            (1, num_samples) float32 tensor
        """
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        return torch.from_numpy(waveform).float()

    def compute_frame_timestamps(
        self, waveform_length: int, wav2vec_fps: int = 49
    ) -> np.ndarray:
        """Compute the sample index corresponding to each Wav2Vec output frame.

        Wav2Vec 2.0 outputs features at ~49 Hz (stride of 320 samples at 16kHz).

        Returns:
            (num_frames,) array of center sample indices
        """
        stride = self.sample_rate // wav2vec_fps  # 16000 // 49 ≈ 326
        num_frames = waveform_length // stride
        return np.arange(num_frames) * stride + stride // 2

    @staticmethod
    def upsample_visual_to_audio(
        visual_features: np.ndarray, visual_fps: int = 25, audio_fps: int = 49
    ) -> np.ndarray:
        """Upsample visual features from video FPS to Wav2Vec FPS via linear interpolation.

        Args:
            visual_features: (T_v, D) array at visual_fps
            visual_fps: source FPS (typically 25)
            audio_fps: target FPS (Wav2Vec native, 49)

        Returns:
            (T_a, D) array at audio_fps
        """
        T_v, D = visual_features.shape
        duration = T_v / visual_fps
        T_a = int(duration * audio_fps)

        # Source and target time axes
        t_src = np.linspace(0, duration, T_v, endpoint=False)
        t_tgt = np.linspace(0, duration, T_a, endpoint=False)

        # Linear interpolation per dimension
        upsampled = np.zeros((T_a, D), dtype=visual_features.dtype)
        for d in range(D):
            upsampled[:, d] = np.interp(t_tgt, t_src, visual_features[:, d])

        return upsampled
