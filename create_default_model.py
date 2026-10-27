"""
Default CNN Model Generator for MoodTune AI.
Initializes and saves the 4-block Conv2D emotion classification architecture
and class mapping JSON so that app.py is immediately testable and functional.
"""

import json
from pathlib import Path
from typing import Tuple, Any


def build_emotion_cnn(input_shape: Tuple[int, int, int] = (48, 48, 1), num_classes: int = 7) -> Any:
    """Build custom 4-block Convolutional Neural Network (CNN) architecture.

    Architecture:
    - Block 1: Conv2D(32) -> BatchNorm -> ReLU -> MaxPooling2D(2,2)
    - Block 2: Conv2D(64) -> BatchNorm -> ReLU -> MaxPooling2D(2,2)
    - Block 3: Conv2D(128) -> BatchNorm -> ReLU -> MaxPooling2D(2,2)
    - Block 4: Conv2D(256) -> BatchNorm -> ReLU -> MaxPooling2D(2,2)
    - Classifier: Flatten -> Dense(128) -> BatchNorm -> ReLU -> Dropout(0.5) -> Dense(7, Softmax)

    Args:
        input_shape (Tuple[int, int, int]): Input tensor shape (48, 48, 1).
        num_classes (int): Number of facial expression classes (7).

    Returns:
        models.Sequential: Compiled Keras Sequential CNN model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            # Block 1
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 4
            layers.Conv2D(256, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Dense Classifier Head
            layers.Flatten(),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="MoodTune_Emotion_CNN",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    repo_root = Path(__file__).resolve().parent
    model_dir = repo_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "emotion_cnn.keras"
    json_path = model_dir / "class_names.json"

    # Class indexing matching FER-2013 convention
    class_mapping = {
        "0": "Angry",
        "1": "Disgust",
        "2": "Fear",
        "3": "Happy",
        "4": "Sad",
        "5": "Surprise",
        "6": "Neutral",
    }

    # Save class names JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4)
    print(f"[+] Saved class names mapping to {json_path}")

    try:
        print("Building MoodTune CNN model architecture...")
        model = build_emotion_cnn()
        model.summary()

        # Save compiled baseline model
        model.save(model_path)
        print(f"[+] Saved initialized baseline CNN model to {model_path}")
    except ImportError as e:
        print(f"[!] TensorFlow not installed: {e}")
        print("    Install dependencies via: pip install -r requirements.txt")
        print(f"    (class_names.json has been generated successfully)")


if __name__ == "__main__":
    main()
