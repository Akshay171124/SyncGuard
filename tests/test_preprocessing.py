"""Tests for preprocessing helpers.

Covers:
- AudioExtractor: waveform_to_tensor, frame timestamps, upsampling
- EAR (Eye Aspect Ratio) computation formula
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.preprocessing.audio_extractor import AudioExtractor


# ──────────────────────────────────────────────
# Waveform to Tensor
# ──────────────────────────────────────────────

class TestWaveformToTensor:
    def test_1d_input(self):
        """1D waveform → (1, N) tensor."""
        ext = AudioExtractor()
        waveform = np.random.randn(16000).astype(np.float32)
        t = ext.waveform_to_tensor(waveform)
        assert t.shape == (1, 16000)
        assert t.dtype == torch.float32

    def test_2d_input_passthrough(self):
        """2D waveform (1, N) → (1, N) tensor unchanged."""
        ext = AudioExtractor()
        waveform = np.random.randn(1, 16000).astype(np.float32)
        t = ext.waveform_to_tensor(waveform)
        assert t.shape == (1, 16000)

    def test_preserves_values(self):
        """Conversion preserves waveform values."""
        ext = AudioExtractor()
        waveform = np.array([0.1, -0.5, 0.3], dtype=np.float32)
        t = ext.waveform_to_tensor(waveform)
        np.testing.assert_allclose(t.numpy()[0], waveform, atol=1e-7)


# ──────────────────────────────────────────────
# Frame Timestamps
# ──────────────────────────────────────────────

class TestFrameTimestamps:
    def test_stride_matches_wav2vec(self):
        """Stride ≈ 326 samples (16000 / 49)."""
        ext = AudioExtractor(sample_rate=16000)
        timestamps = ext.compute_frame_timestamps(16000, wav2vec_fps=49)
        # For 1 second of audio at 16kHz, should get ~49 frames
        assert len(timestamps) == 49

    def test_center_aligned(self):
        """Timestamps are center-aligned (start at stride/2)."""
        ext = AudioExtractor(sample_rate=16000)
        stride = 16000 // 49
        timestamps = ext.compute_frame_timestamps(32000, wav2vec_fps=49)
        assert timestamps[0] == stride // 2
        assert timestamps[1] == stride + stride // 2

    def test_short_audio(self):
        """Audio shorter than one stride → 0 frames."""
        ext = AudioExtractor(sample_rate=16000)
        timestamps = ext.compute_frame_timestamps(100, wav2vec_fps=49)
        assert len(timestamps) == 0

    def test_frame_count_scales_with_length(self):
        """2x audio length → ~2x frames."""
        ext = AudioExtractor(sample_rate=16000)
        t1 = ext.compute_frame_timestamps(16000)
        t2 = ext.compute_frame_timestamps(32000)
        assert len(t2) == pytest.approx(2 * len(t1), abs=1)


# ──────────────────────────────────────────────
# Visual-to-Audio Upsampling
# ──────────────────────────────────────────────

class TestUpsampleVisualToAudio:
    def test_output_length(self):
        """25fps → 49Hz: T_a ≈ T_v * 49/25."""
        T_v, D = 50, 256  # 2 seconds at 25fps
        visual = np.random.randn(T_v, D).astype(np.float32)
        upsampled = AudioExtractor.upsample_visual_to_audio(visual, 25, 49)
        expected_T_a = int((T_v / 25) * 49)  # 98
        assert upsampled.shape == (expected_T_a, D)

    def test_preserves_dimensionality(self):
        """Feature dimension D is preserved."""
        visual = np.random.randn(100, 128).astype(np.float32)
        upsampled = AudioExtractor.upsample_visual_to_audio(visual, 25, 49)
        assert upsampled.shape[1] == 128

    def test_no_nan_or_inf(self):
        """Upsampled output has no NaN or Inf."""
        visual = np.random.randn(50, 64).astype(np.float32)
        upsampled = AudioExtractor.upsample_visual_to_audio(visual, 25, 49)
        assert np.all(np.isfinite(upsampled))

    def test_single_frame(self):
        """Single visual frame → constant across all audio frames."""
        visual = np.array([[1.0, 2.0, 3.0]])  # (1, 3)
        upsampled = AudioExtractor.upsample_visual_to_audio(visual, 25, 49)
        # All frames should be [1, 2, 3] (constant interpolation from single point)
        for i in range(upsampled.shape[0]):
            np.testing.assert_allclose(upsampled[i], [1.0, 2.0, 3.0], atol=1e-5)

    def test_same_fps_identity(self):
        """Same source and target FPS → same length output."""
        visual = np.random.randn(100, 64).astype(np.float32)
        upsampled = AudioExtractor.upsample_visual_to_audio(visual, 25, 25)
        assert upsampled.shape[0] == 100

    def test_dtype_preserved(self):
        """Output dtype matches input."""
        visual = np.random.randn(50, 32).astype(np.float32)
        upsampled = AudioExtractor.upsample_visual_to_audio(visual, 25, 49)
        assert upsampled.dtype == np.float32


# ──────────────────────────────────────────────
# EAR (Eye Aspect Ratio) Computation
# ──────────────────────────────────────────────

def _make_landmark(x, y, z=0.0):
    """Create a mock NormalizedLandmark with x, y attributes."""
    return SimpleNamespace(x=x, y=y, z=z)


def _make_landmarks_dict(left_eye_pts, right_eye_pts):
    """Build a sparse landmark list with eye landmarks placed at correct indices.

    left_eye_pts: list of 6 (x, y) tuples for LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    right_eye_pts: list of 6 (x, y) tuples for RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    """
    from src.preprocessing.face_detector import LEFT_EYE_IDX, RIGHT_EYE_IDX

    # Create 478 dummy landmarks
    landmarks = [_make_landmark(0.5, 0.5) for _ in range(478)]

    for idx, (x, y) in zip(LEFT_EYE_IDX, left_eye_pts):
        landmarks[idx] = _make_landmark(x, y)
    for idx, (x, y) in zip(RIGHT_EYE_IDX, right_eye_pts):
        landmarks[idx] = _make_landmark(x, y)

    return landmarks


try:
    import mediapipe  # noqa: F401
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False


@pytest.mark.skipif(not _HAS_MEDIAPIPE, reason="mediapipe not installed")
class TestEARComputation:
    """Test the EAR formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""

    def test_open_eyes_ear(self):
        """Open eyes should produce EAR ~0.3 (typical range 0.2-0.4)."""
        from src.preprocessing.face_detector import FaceDetector

        # Simulate open eye landmarks (normalized coords)
        # p1 (outer), p2 (upper-outer), p3 (upper-inner), p4 (inner), p5 (lower-inner), p6 (lower-outer)
        h, w = 480, 640
        open_eye = [
            (0.2, 0.4),    # p1 outer corner
            (0.25, 0.35),  # p2 upper-outer
            (0.30, 0.35),  # p3 upper-inner
            (0.35, 0.4),   # p4 inner corner
            (0.30, 0.45),  # p5 lower-inner
            (0.25, 0.45),  # p6 lower-outer
        ]
        landmarks = _make_landmarks_dict(open_eye, open_eye)

        # Compute EAR without instantiating full FaceDetector (avoid model download)
        detector = FaceDetector.__new__(FaceDetector)
        ear = detector.compute_ear(landmarks, h, w)
        assert 0.15 < ear < 0.6, f"Open eye EAR={ear}, expected 0.15-0.6"

    def test_closed_eyes_ear(self):
        """Closed eyes (vertical distance ~0) → EAR near 0."""
        from src.preprocessing.face_detector import FaceDetector

        h, w = 480, 640
        # Closed eye: upper and lower at same y
        closed_eye = [
            (0.2, 0.4),   # p1
            (0.25, 0.4),  # p2 same y as p6
            (0.30, 0.4),  # p3 same y as p5
            (0.35, 0.4),  # p4
            (0.30, 0.4),  # p5 same y as p3
            (0.25, 0.4),  # p6 same y as p2
        ]
        landmarks = _make_landmarks_dict(closed_eye, closed_eye)

        detector = FaceDetector.__new__(FaceDetector)
        ear = detector.compute_ear(landmarks, h, w)
        assert ear < 0.05, f"Closed eye EAR={ear}, expected <0.05"

    def test_ear_is_nonnegative(self):
        """EAR should always be >= 0."""
        from src.preprocessing.face_detector import FaceDetector

        h, w = 480, 640
        eye_pts = [
            (0.2, 0.4), (0.25, 0.35), (0.30, 0.36),
            (0.35, 0.4), (0.30, 0.44), (0.25, 0.45),
        ]
        landmarks = _make_landmarks_dict(eye_pts, eye_pts)

        detector = FaceDetector.__new__(FaceDetector)
        ear = detector.compute_ear(landmarks, h, w)
        assert ear >= 0.0

    def test_degenerate_horizontal_returns_zero(self):
        """p1 == p4 (zero horizontal distance) → returns 0.0."""
        from src.preprocessing.face_detector import FaceDetector

        h, w = 480, 640
        # p1 and p4 at same position → horizontal = 0 → guarded division
        degenerate = [
            (0.3, 0.4),  # p1
            (0.3, 0.35), # p2
            (0.3, 0.35), # p3
            (0.3, 0.4),  # p4 = same as p1
            (0.3, 0.45), # p5
            (0.3, 0.45), # p6
        ]
        landmarks = _make_landmarks_dict(degenerate, degenerate)

        detector = FaceDetector.__new__(FaceDetector)
        ear = detector.compute_ear(landmarks, h, w)
        assert ear == 0.0
