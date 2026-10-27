"""
Model Training Module for MoodTune AI.
Trains custom CNN model on FER-2013 facial expression dataset with data augmentation,
callbacks (Early Stopping & Model Checkpoint), evaluation, and loss/accuracy plots.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from create_default_model import build_emotion_cnn


def train_emotion_cnn(
    dataset_dir: str = "dataset/fer2013",
    batch_size: int = 64,
    epochs: int = 30,
    img_size: int = 48,
):
    """Train custom CNN on FER-2013 dataset.

    Args:
        dataset_dir (str): Path to dataset directory.
        batch_size (int): Training batch size. Defaults to 64.
        epochs (int): Max training epochs. Defaults to 30.
        img_size (int): Target square image dimension (48).
    """
    repo_root = Path(__file__).resolve().parent
    data_path = Path(dataset_dir)
    if not data_path.exists():
        fallback_data = repo_root / dataset_dir
        if fallback_data.exists():
            data_path = fallback_data

    train_dir = data_path / "train"
    test_dir = data_path / "test"

    if not train_dir.exists() or not test_dir.exists():
        print(f"\n[!] Dataset directory not found at '{dataset_dir}'.")
        print("Please follow instructions in dataset/README.md to setup FER-2013:")
        print("Expected layout:")
        print("  dataset/fer2013/train/<emotion>/")
        print("  dataset/fer2013/test/<emotion>/")
        print("\nGenerating baseline default model instead via create_default_model.py...")
        from create_default_model import main as generate_default

        generate_default()
        return

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
    except ImportError:
        print("[!] TensorFlow not installed. Install via: pip install -r requirements.txt")
        return

    print(f"\n[+] Loading FER-2013 datasets from {data_path}...")

    # Load training dataset split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        color_mode="grayscale",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        validation_split=0.2,
        subset="training",
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        color_mode="grayscale",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        color_mode="grayscale",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    # Save class names mapping
    class_names = train_ds.class_names
    class_mapping = {str(i): name.capitalize() for i, name in enumerate(class_names)}

    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4)

    # Normalization & Data Augmentation Pipeline
    data_augmentation = models.Sequential(
        [
            layers.Rescaling(1.0 / 255.0),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ]
    )

    rescale_only = models.Sequential([layers.Rescaling(1.0 / 255.0)])

    # Apply preprocessing to dataset pipelines
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (rescale_only(x, training=False), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (rescale_only(x, training=False), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    print("\n[+] Building CNN Architecture...")
    model = build_emotion_cnn(
        input_shape=(img_size, img_size, 1), num_classes=len(class_names)
    )
    model.summary()

    # Callbacks
    best_model_path = model_dir / "emotion_cnn.keras"
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    cb_earlystop = callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )

    print(f"\n[+] Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cb_checkpoint, cb_earlystop, cb_reduce_lr],
    )

    # Evaluate on held-out test dataset
    print("\n[+] Evaluating best model on test set...")
    best_model = tf.keras.models.load_model(best_model_path)
    test_loss, test_acc = best_model.evaluate(test_ds)

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    print("\n" + "=" * 50)
    print(f"  Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  Test Accuracy:       {test_acc * 100:.2f}%")
    print(f"  Test Loss:           {test_loss:.4f}")
    print("=" * 50 + "\n")

    # Generate training curves
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
        linestyle="--",
    )
    plt.title("MoodTune CNN - Training vs Validation Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(assets_dir / "training_accuracy.png", dpi=200)
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2,
        linestyle="--",
    )
    plt.title("MoodTune CNN - Training vs Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(assets_dir / "training_loss.png", dpi=200)
    plt.close()

    print(f"[+] Training plots saved under {assets_dir}/")


if __name__ == "__main__":
    train_emotion_cnn()
