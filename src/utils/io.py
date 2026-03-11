import subprocess
from pathlib import Path

import cv2
import numpy as np


def read_video_frames(video_path: str, fps: int = 25) -> tuple[np.ndarray, int]:
    """Read video frames at a target FPS.

    Returns:
        frames: (T, H, W, 3) uint8 BGR array
        original_fps: original video FPS
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(original_fps / fps))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames read from: {video_path}")
    return np.stack(frames), int(original_fps)


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """Extract audio from video using ffmpeg.

    Returns:
        Path to the extracted .wav file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return str(output_path)


def load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load a .wav file as a float32 numpy array."""
    import soundfile as sf

    waveform, sr = sf.read(str(audio_path), dtype="float32")
    if sr != sample_rate:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
    return waveform
