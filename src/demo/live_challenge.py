"""Live challenge: webcam record -> audio swap -> inference pipeline.

The demo records a 3-5s webcam clip, swaps the visitor's audio with a
random sentence from demo_assets/live_audio_pool/, then preprocesses the
swapped clip (MediaPipe-only, no RetinaFace/TF) and hands it to the cascade
inference model.

Only imported when SYNCGUARD_LIVE=1.
"""
import logging
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from src.preprocessing.face_detector import FaceDetector

logger = logging.getLogger(__name__)


AUDIO_POOL_DIR = Path("demo_assets/live_audio_pool")
TARGET_SR = 16000
TARGET_FPS = 25


def _audio_signature(path):
    """First 10 samples of 16kHz audio, for debug/verification."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(path),
             "-ar", "16000", "-ac", "1",
             "-f", "s16le", "-t", "0.01", "-"],
            check=True, capture_output=True,
        )
        samples = np.frombuffer(proc.stdout, dtype=np.int16)[:10]
        return samples.tolist()
    except Exception:
        return None


def swap_audio(input_video, output_video, pool_dir=AUDIO_POOL_DIR):
    """Replace the audio track of input_video with a random clip from the pool.

    Args:
        input_video: Path to the visitor's recorded MP4 (with their audio).
        output_video: Destination path for the audio-swapped MP4.
        pool_dir: Directory containing .wav files to sample from.

    Returns:
        Name of the pool file used for swapping.
    """
    pool = sorted(Path(pool_dir).glob("*.wav"))
    if not pool:
        raise RuntimeError(f"No .wav files in {pool_dir}")
    chosen = random.choice(pool)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_video),
        "-i", str(chosen),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Debug: log audio signatures so we can verify the swap actually happened
    orig_sig = _audio_signature(input_video)
    swap_sig = _audio_signature(output_video)
    pool_sig = _audio_signature(chosen)
    logger.info(f"swap_audio: pool_file={chosen.name}")
    logger.info(f"  original video audio (first 10 samples): {orig_sig}")
    logger.info(f"  swapped video audio  (first 10 samples): {swap_sig}")
    logger.info(f"  pool file audio      (first 10 samples): {pool_sig}")
    if orig_sig is not None and swap_sig is not None and orig_sig == swap_sig:
        logger.warning("swap_audio: OUTPUT AUDIO MATCHES INPUT — swap did not take effect!")
    return chosen.name


def _extract_audio(video_path, target_sr=TARGET_SR):
    """Use ffmpeg to extract audio at target_sr mono as float32 numpy."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video_path),
         "-ar", str(target_sr), "-ac", "1",
         "-f", "s16le", "-"],
        check=True, capture_output=True,
    )
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    return audio


def _read_video_frames(video_path, target_fps=TARGET_FPS):
    """Read a video as BGR frames at the target fps via OpenCV.

    Returns:
        (T, H, W, 3) uint8 array.
    """
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / target_fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames), src_fps / stride


def process_live_clip(video_path, face_detector):
    """Preprocess a recorded/swapped MP4 into cascade-model inputs.

    Uses MediaPipe-only detection (no RetinaFace/TF). All CPU; takes
    ~3-5s for a 5-second clip on a modern Mac.

    Args:
        video_path: Path to the swapped MP4.
        face_detector: Pre-initialized FaceDetector instance
                      (create once, reuse across requests).

    Returns:
        Dict with mouth_crops, audio_waveform, ear_features, duration_s.
    """
    frames, eff_fps = _read_video_frames(video_path)
    logger.info(f"Decoded {len(frames)} frames at {eff_fps:.1f} fps")

    crops, valid, ears = face_detector.process_video_frames_with_ear(
        frames, skip_failed=False,
    )
    detection_rate = float(valid.mean())
    logger.info(f"Face detection: {detection_rate*100:.1f}% of frames")

    audio = _extract_audio(video_path)
    duration_s = len(audio) / TARGET_SR

    return {
        "mouth_crops": crops,                    # (T, 96, 96, 3) uint8
        "audio_waveform": audio.astype(np.float32),
        "ear_features": ears.astype(np.float32),
        "duration_s": float(duration_s),
        "detection_rate": detection_rate,
        "num_frames": int(len(frames)),
    }
