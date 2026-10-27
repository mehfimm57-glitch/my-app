"""
Preprocessing module for MoodTune AI.
Provides image preprocessing utilities matching CNN training specs (48x48 grayscale normalized).
"""

from typing import List, Tuple, Union
import cv2
import numpy as np


def preprocess_face(
    face_img: np.ndarray, target_size: Tuple[int, int] = (48, 48)
) -> np.ndarray:
    """Preprocess a face crop image for CNN model inference.

    Steps:
    1. Validate input frame/crop.
    2. Handle dimensional variations (2D grayscale, 3D BGR/RGB/BGRA, 1-channel).
    3. Ensure safe uint8 representation before OpenCV operations.
    4. Resize to target dimensions (default 48x48).
    5. Normalize pixel intensity values from [0, 255] to [0.0, 1.0].
    6. Reshape to batch tensor format: (1, height, width, 1) float32.

    Args:
        face_img (np.ndarray): Image array of cropped face region.
        target_size (Tuple[int, int]): Target image size (width, height). Defaults to (48, 48).

    Returns:
        np.ndarray: Preprocessed 4D tensor ready for TensorFlow/Keras prediction.
    """
    if face_img is None or not isinstance(face_img, np.ndarray) or face_img.size == 0:
        raise ValueError("Invalid face image input provided for preprocessing.")

    img = face_img

    # Handle 4D input if a single-element batch was passed
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # Convert to 2D grayscale based on channel count
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            gray = img[:, :, 0]
        else:
            gray = img[:, :, 0]
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(f"Unsupported face image shape: {img.shape}")

    # Ensure valid uint8 dtype for OpenCV interpolation
    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating) and gray.max() <= 1.0:
            gray = (gray * 255.0).clip(0, 255).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Resize to target size (width, height)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel intensity to [0.0, 1.0] range as float32
    normalized = resized.astype(np.float32) / 255.0

    # Reshape tensor to (1, height, width, 1) for Keras batch input
    tensor = normalized.reshape(1, target_size[1], target_size[0], 1)

    return tensor


def preprocess_batch(
    face_images: List[np.ndarray], target_size: Tuple[int, int] = (48, 48)
) -> np.ndarray:
    """Preprocess a batch of cropped face images for CNN batch inference.

    Args:
        face_images (List[np.ndarray]): List of face image crops.
        target_size (Tuple[int, int]): Target dimensions (width, height).

    Returns:
        np.ndarray: 4D batch tensor of shape (N, height, width, 1).
    """
    if not face_images:
        return np.empty((0, target_size[1], target_size[0], 1), dtype=np.float32)

    tensors = [preprocess_face(img, target_size=target_size) for img in face_images]
    return np.concatenate(tensors, axis=0)

