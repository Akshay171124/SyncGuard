import cv2
import numpy as np
import mediapipe as mp
from retinaface import RetinaFace


# MediaPipe lip landmark indices (from 468-point FaceMesh)
# Outer lip: defines the bounding box for mouth ROI
LIP_OUTER_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]


class FaceDetector:
    """Detects faces via RetinaFace and extracts mouth-ROI crops via MediaPipe."""

    def __init__(self, crop_size: int = 96, confidence_threshold: float = 0.8):
        self.crop_size = crop_size
        self.confidence_threshold = confidence_threshold

        # Initialize MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect_face(self, frame: np.ndarray) -> dict | None:
        """Detect the primary face in a frame using RetinaFace.

        Returns:
            Dict with 'bbox' (x1, y1, x2, y2) and 'confidence', or None.
        """
        detections = RetinaFace.detect_faces(frame)
        if not isinstance(detections, dict) or len(detections) == 0:
            return None

        # Pick the detection with highest confidence
        best_key = max(detections, key=lambda k: detections[k]["score"])
        det = detections[best_key]

        if det["score"] < self.confidence_threshold:
            return None

        return {
            "bbox": det["facial_area"],  # [x1, y1, x2, y2]
            "confidence": det["score"],
        }

    def extract_mouth_roi(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract a mouth-ROI crop from a single BGR frame.

        Returns:
            (crop_size, crop_size, 3) uint8 array, or None if detection fails.
        """
        h, w = frame.shape[:2]

        # Get lip landmarks via MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        # Extract lip landmark coordinates
        lip_points = []
        for idx in LIP_OUTER_IDX:
            lm = landmarks.landmark[idx]
            lip_points.append((int(lm.x * w), int(lm.y * h)))
        lip_points = np.array(lip_points)

        # Compute bounding box around lips with margin
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

    def close(self):
        self.face_mesh.close()

    def __del__(self):
        self.close()
