import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from retinaface import RetinaFace
import os
import urllib.request
import logging

logger = logging.getLogger(__name__)

# MediaPipe lip landmark indices (from 478-point FaceLandmarker)
# Outer lip: defines the bounding box for mouth ROI
LIP_OUTER_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

# MediaPipe eye landmark indices for EAR (Eye Aspect Ratio) computation.
# Each eye uses 6 landmarks: p1 (outer corner), p2 (upper-outer), p3 (upper-inner),
# p4 (inner corner), p5 (lower-inner), p6 (lower-outer).
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# Left eye landmarks (from subject's perspective)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # p1..p6
# Right eye landmarks
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # p1..p6

# FaceLandmarker model URL and local path
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mediapipe")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _ensure_model():
    """Download face_landmarker.task if not present."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    logger.info("Downloading face_landmarker.task model...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    logger.info("Model saved to %s", _MODEL_PATH)
    return _MODEL_PATH


class FaceDetector:
    """Detects faces via RetinaFace and extracts mouth-ROI crops via MediaPipe."""

    def __init__(self, crop_size: int = 96, confidence_threshold: float = 0.8):
        self.crop_size = crop_size
        self.confidence_threshold = confidence_threshold

        # Initialize MediaPipe FaceLandmarker (new Tasks API, CPU-only)
        model_path = _ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=BaseOptions.Delegate.CPU,
            ),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_face(self, frame: np.ndarray) -> dict | None:
        """Detect the primary face in a frame using RetinaFace.

        HP-3: Downscales high-res frames before detection to maintain
        consistent confidence scores across resolutions.

        Returns:
            Dict with 'bbox' (x1, y1, x2, y2) and 'confidence', or None.
        """
        h, w = frame.shape[:2]
        max_dim = 720
        scale = 1.0

        # HP-3: Normalize resolution for consistent detection across datasets
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            detect_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            detect_frame = frame

        detections = RetinaFace.detect_faces(detect_frame)
        if not isinstance(detections, dict) or len(detections) == 0:
            return None

        # Pick the detection with highest confidence
        best_key = max(detections, key=lambda k: detections[k]["score"])
        det = detections[best_key]

        if det["score"] < self.confidence_threshold:
            return None

        # Scale bbox back to original resolution
        bbox = det["facial_area"]  # [x1, y1, x2, y2]
        if scale != 1.0:
            bbox = [int(c / scale) for c in bbox]

        return {
            "bbox": bbox,
            "confidence": det["score"],
        }

    def _get_landmarks(self, frame: np.ndarray):
        """Run FaceLandmarker on a BGR frame, return landmark list or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]  # list of NormalizedLandmark

    def compute_ear(self, landmarks, h: int, w: int) -> float:
        """Compute average Eye Aspect Ratio (EAR) from MediaPipe landmarks.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||) averaged over both eyes.
        Returns 0.0 if landmarks are invalid.

        Args:
            landmarks: List of NormalizedLandmark from FaceLandmarker.
            h: Frame height.
            w: Frame width.

        Returns:
            Average EAR value (typically 0.2-0.4 for open eyes, <0.15 for closed).
        """
        def _eye_ear(indices):
            pts = []
            for idx in indices:
                lm = landmarks[idx]
                pts.append(np.array([lm.x * w, lm.y * h]))
            p1, p2, p3, p4, p5, p6 = pts
            vertical1 = np.linalg.norm(p2 - p6)
            vertical2 = np.linalg.norm(p3 - p5)
            horizontal = np.linalg.norm(p1 - p4)
            if horizontal < 1e-6:
                return 0.0
            return (vertical1 + vertical2) / (2.0 * horizontal)

        left_ear = _eye_ear(LEFT_EYE_IDX)
        right_ear = _eye_ear(RIGHT_EYE_IDX)
        return (left_ear + right_ear) / 2.0

    def _extract_mouth_crop(self, landmarks, frame: np.ndarray) -> np.ndarray | None:
        """Extract mouth-ROI crop from landmarks and frame."""
        h, w = frame.shape[:2]

        lip_points = []
        for idx in LIP_OUTER_IDX:
            lm = landmarks[idx]
            lip_points.append((int(lm.x * w), int(lm.y * h)))
        lip_points = np.array(lip_points)

        x_min, y_min = lip_points.min(axis=0)
        x_max, y_max = lip_points.max(axis=0)

        lip_w = x_max - x_min
        lip_h = y_max - y_min
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # Expand to square with 1.5x margin
        side = int(max(lip_w, lip_h) * 1.5)
        half = side // 2

        # Clamp to frame boundaries
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        return crop

    def extract_mouth_roi(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract a mouth-ROI crop from a single BGR frame.

        Returns:
            (crop_size, crop_size, 3) uint8 array, or None if detection fails.
        """
        landmarks = self._get_landmarks(frame)
        if landmarks is None:
            return None
        return self._extract_mouth_crop(landmarks, frame)

    def extract_mouth_roi_and_ear(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """Extract mouth-ROI crop and EAR from a single BGR frame in one pass.

        Args:
            frame: (H, W, 3) uint8 BGR frame.

        Returns:
            Tuple of (crop, ear_value). crop is (crop_size, crop_size, 3) or None.
            ear_value is the average EAR (0.0 if detection failed).
        """
        landmarks = self._get_landmarks(frame)
        if landmarks is None:
            return None, 0.0

        h, w = frame.shape[:2]
        ear = self.compute_ear(landmarks, h, w)
        crop = self._extract_mouth_crop(landmarks, frame)
        return crop, ear

    def process_video_frames(
        self, frames: np.ndarray, skip_failed: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract mouth-ROI crops from all video frames.

        Args:
            frames: (T, H, W, 3) uint8 BGR
            skip_failed: If True, skip frames where detection fails.
                         If False, use zero-filled crops for failed frames.

        Returns:
            mouth_crops: (T', crop_size, crop_size, 3) uint8
            valid_mask: (T,) bool array indicating successful detections
        """
        crops = []
        valid = []

        for i in range(len(frames)):
            crop = self.extract_mouth_roi(frames[i])
            if crop is not None:
                crops.append(crop)
                valid.append(True)
            elif not skip_failed:
                crops.append(np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8))
                valid.append(False)
            else:
                valid.append(False)

        if len(crops) == 0:
            raise ValueError("No valid mouth crops extracted from any frame")

        return np.stack(crops), np.array(valid)

    def process_video_frames_with_ear(
        self, frames: np.ndarray, skip_failed: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract mouth-ROI crops and EAR values from all video frames.

        Args:
            frames: (T, H, W, 3) uint8 BGR
            skip_failed: If True, skip frames where detection fails.

        Returns:
            mouth_crops: (T', crop_size, crop_size, 3) uint8
            valid_mask: (T,) bool array
            ear_values: (T,) float32 array of per-frame EAR values
        """
        crops = []
        valid = []
        ears = []

        for i in range(len(frames)):
            crop, ear = self.extract_mouth_roi_and_ear(frames[i])
            ears.append(ear)
            if crop is not None:
                crops.append(crop)
                valid.append(True)
            elif not skip_failed:
                crops.append(
                    np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                )
                valid.append(False)
            else:
                valid.append(False)

        if len(crops) == 0:
            raise ValueError("No valid mouth crops extracted from any frame")

        return np.stack(crops), np.array(valid), np.array(ears, dtype=np.float32)

    def close(self):
        self.landmarker.close()

    def __del__(self):
        self.close()
