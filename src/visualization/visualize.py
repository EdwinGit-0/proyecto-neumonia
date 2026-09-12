"""Visualization helpers for the pneumonia chest X-ray project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.preprocessing import image_to_array, load_image
from src.data.splitting import load_split_manifest
from src.utils.paths import DATA_DIR, FIGURES_DIR, PROJECT_ROOT, ensure_directory


IMAGE_SIZE = (224, 224)
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "stratified_split_70_15_15.csv"


def _select_train_image() -> Path | None:
    """Return one real image from the experimental training split."""
    if MANIFEST_PATH.exists():
        manifest = load_split_manifest(MANIFEST_PATH)
        train_records = manifest[manifest["split"] == "train"]
        train_records = train_records[train_records["label"] == "NORMAL"]
        if not train_records.empty:
            return Path(train_records.iloc[0]["path"])
    train_normal = Path(DATA_DIR) / "train" / "NORMAL"
    if train_normal.is_dir():
        images = sorted(train_normal.iterdir())
        if images:
            return images[0]
    return None


def _apply_layer(image_array: np.ndarray, layer: tf.keras.layers.Layer) -> np.ndarray:
    """Apply a single augmentation layer to a normalized image."""
    batch = layer(image_array[np.newaxis, ...], training=True)
    return np.asarray(batch[0])


def build_augmentation_variants(image_array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Apply the project's existing augmentation transforms to a single image.

    Only RandomFlip("horizontal"), RandomRotation(0.05) and RandomZoom(0.05)
    are used, matching the training pipeline exactly.
    """
    flip = lambda: tf.keras.layers.RandomFlip("horizontal")
    rotation = lambda: tf.keras.layers.RandomRotation(0.05)
    zoom = lambda: tf.keras.layers.RandomZoom(0.05)

    variants = [
        ("Original", None),
        ("Volteo horizontal", [flip()]),
        ("Rotación", [rotation()]),
        ("Zoom", [zoom()]),
        ("Volteo + Rotación", [flip(), rotation()]),
        ("Volteo + Zoom", [flip(), zoom()]),
        ("Rotación + Zoom", [rotation(), zoom()]),
        ("Volteo + Rotación + Zoom", [flip(), rotation(), zoom()]),
        ("Volteo + Rotación + Zoom (2.ª muestra)", [flip(), rotation(), zoom()]),
    ]

    generated: list[tuple[str, np.ndarray]] = []
    original = image_array.copy()
    for label, layers in variants:
        if layers is None:
            generated.append((label, original))
            continue
        array = original
        for layer in layers:
            before = array
            array = _apply_layer(array, layer)
            if isinstance(layer, tf.keras.layers.RandomFlip) and np.array_equal(array, before):
                seed = 0
                while np.array_equal(array, before) and seed < 64:
                    array = _apply_layer(before, tf.keras.layers.RandomFlip("horizontal", seed=seed))
                    seed += 1
        generated.append((label, array))
    return generated


def plot_data_augmentation_examples(
    output_path: str | Path = FIGURES_DIR / "data_augmentation_examples.png",
) -> Path:
    """Create a 3x3 figure showing data augmentation on one training image."""
    train_image = _select_train_image()
    if train_image is None:
        raise FileNotFoundError("No training images found to illustrate data augmentation.")

    source = load_image(train_image)
    image_array = image_to_array(source, IMAGE_SIZE)
    variants = build_augmentation_variants(image_array)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(
        "Data augmentation aplicada al conjunto de entrenamiento (imágenes de 224 x 224 px)",
        fontsize=13,
    )
    for axis, (label, array) in zip(axes.ravel(), variants):
        axis.imshow(array, cmap="gray")
        axis.set_title(label, fontsize=10)
        axis.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = Path(output_path)
    ensure_directory(output.parent)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    result = plot_data_augmentation_examples()
    print(result)