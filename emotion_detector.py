"""
Emotion Detector module for MoodTune AI.
Loads trained Keras CNN model and maps 48x48 input tensor to 7-class facial expression probabilities.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


class EmotionDetector:
    """Emotion classification engine using trained TensorFlow/Keras CNN model."""

    DEFAULT_CLASSES = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    ]

    def __init__(
        self,
        model_path: str = "model/emotion_cnn.keras",
        class_mapping_path: str = "model/class_names.json",
    ):
        """Initialize EmotionDetector.

        Args:
            model_path (str): Path to saved Keras model (.keras).
            class_mapping_path (str): Path to JSON mapping class indices to names.
        """
        self.model_path = Path(model_path)
        self.class_mapping_path = Path(class_mapping_path)
        repo_root = Path(__file__).resolve().parent.parent

        if not self.model_path.exists():
            fallback_model = repo_root / model_path
            if fallback_model.exists():
                self.model_path = fallback_model

        if not self.class_mapping_path.exists():
            fallback_json = repo_root / class_mapping_path
            if fallback_json.exists():
                self.class_mapping_path = fallback_json

        self.model = None
        self.class_names: List[str] = self.DEFAULT_CLASSES

        self._load_class_names()
        self._load_model()

    def _load_class_names(self) -> None:
        """Load class labels from JSON if available, otherwise use default mapping."""
        if self.class_mapping_path.exists():
            try:
                with open(self.class_mapping_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                    # Convert dict keys (str or int) to sorted class list
                    sorted_keys = sorted(mapping.keys(), key=lambda k: int(k))
                    self.class_names = [mapping[k] for k in sorted_keys]
            except Exception as e:
                print(f"Warning: Could not parse {self.class_mapping_path}: {e}")
                self.class_names = self.DEFAULT_CLASSES

    def _load_model(self) -> None:
        """Load Keras CNN model."""
        if not self.model_path.exists():
            print(
                f"Notice: Keras model file not found at {self.model_path}. "
                "Call create_default_model.py or train_model.py to generate it."
            )
            return

        try:
            import tensorflow as tf

            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading Keras model from {self.model_path}: {e}")
            self.model = None

    def is_model_loaded(self) -> bool:
        """Check if CNN model is loaded and ready for inference."""
        return self.model is not None

    def predict_emotion(
        self, processed_face_tensor: np.ndarray
    ) -> Dict[str, Any]:
        """Perform CNN forward pass to calculate emotion probabilities.

        Args:
            processed_face_tensor (np.ndarray): Shape (1, 48, 48, 1) normalized image tensor.

        Returns:
            Dict[str, Any]: Structured dictionary with top predicted emotion,
                            confidence score, and all 7 class probabilities.
        """
        if not self.is_model_loaded():
            # Baseline placeholder distribution if model not yet loaded
            probs = {name: (1.0 / len(self.class_names)) for name in self.class_names}
            return {
                "emotion": "Neutral",
                "confidence": 0.0,
                "probabilities": probs,
                "disclaimer": "Predicted visible facial expression (Model pending training)",
            }

        # Keras model prediction
        predictions = self.model.predict(processed_face_tensor, verbose=0)[0]

        # Calculate softmax probabilities (ensure float type for JSON serialization)
        probabilities_dict = {}
        for idx, class_name in enumerate(self.class_names):
            probabilities_dict[class_name] = float(predictions[idx])

        # Find max probability emotion
        top_idx = int(np.argmax(predictions))
        top_emotion = self.class_names[top_idx]
        confidence = float(predictions[top_idx])

        return {
            "emotion": top_emotion,
            "confidence": round(confidence, 4),
            "probabilities": {
                k: round(v, 4) for k, v in probabilities_dict.items()
            },
            "disclaimer": "Predicted visible facial expression",
        }
