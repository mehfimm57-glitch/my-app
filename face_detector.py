"""
Face Detection module for MoodTune AI.
Uses OpenCV Haar Cascade classifier to locate face bounding boxes in webcam frames.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class FaceDetector:
    """OpenCV Haar Cascade based face detector with support for single/multiple faces,

    primary face selection, and bounding box drawing.
    """

    def __init__(self, cascade_path: Optional[str] = None):
        """Initialize OpenCV Haar Cascade Face Detector.

        Args:
            cascade_path (str, optional): Custom path to XML cascade file.
                Defaults to cv2 built-in haarcascade_frontalface_default.xml.
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load OpenCV face cascade classifier from: {cascade_path}"
            )

    def detect_faces(
        self,
        frame: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> List[Tuple[int, int, int, int]]:
        """Detect all face bounding boxes in a frame.

        Args:
            frame (np.ndarray): Input image/webcam frame (BGR or RGB).
            scale_factor (float): Image scale reduction per scale level. Defaults to 1.1.
            min_neighbors (int): Retained rectangle neighbor parameter. Defaults to 5.
            min_size (Tuple[int, int]): Minimum face size threshold. Defaults to (30, 30).

        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes as (x, y, w, h).
        """
        if frame is None or frame.size == 0:
            return []

        # Convert to grayscale for detection efficiency
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Equalize histogram to handle varying light conditions
        gray_eq = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        return [tuple(map(int, bbox)) for bbox in faces]

    def get_primary_face(
        self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract the primary (largest bounding area) face crop from frame.

        Args:
            frame (np.ndarray): Input video frame.
            faces (List[Tuple[int, int, int, int]]): Detected bounding boxes.

        Returns:
            Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
                (cropped_face_array, (x, y, w, h)) or None if no faces detected.
        """
        if not faces or frame is None:
            return None

        # Select face with maximum area (w * h)
        primary_bbox = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = primary_bbox

        # Ensure coordinates are within image boundaries
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        return face_crop, (x1, y1, x2 - x1, y2 - y1)

    def draw_bounding_boxes(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        primary_bbox: Optional[Tuple[int, int, int, int]] = None,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw aesthetic bounding boxes and emotion labels on frame.

        Args:
            frame (np.ndarray): Original image frame.
            faces (List[Tuple[int, int, int, int]]): Detected face bounding boxes.
            primary_bbox (Tuple[int, int, int, int], optional): Primary face box.
            label (str, optional): Text label to overlay above primary box.

        Returns:
            np.ndarray: Annotated video frame.
        """
        annotated = frame.copy()

        for (x, y, w, h) in faces:
            is_primary = primary_bbox and (x, y, w, h) == primary_bbox
            # Primary face highlighted in Cyan/Magenta, secondary in Gray
            color = (255, 191, 0) if is_primary else (128, 128, 128)
            thickness = 2 if is_primary else 1

            # Draw corner accents for modern aesthetic
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

            if is_primary and label:
                # Text header box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_size, _ = cv2.getTextSize(label, font, font_scale, 2)
                text_w, text_h = text_size

                # Label background rectangle above face box
                lbl_y1 = max(0, y - text_h - 12)
                cv2.rectangle(
                    annotated,
                    (x, lbl_y1),
                    (x + text_w + 12, y),
                    (20, 20, 30),
                    -1,
                )
                cv2.rectangle(
                    annotated,
                    (x, lbl_y1),
                    (x + text_w + 12, y),
                    color,
                    1,
                )
                # Label text
                cv2.putText(
                    annotated,
                    label,
                    (x + 6, y - 6),
                    font,
                    font_scale,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return annotated
