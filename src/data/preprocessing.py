"""Image preprocessing utilities for the chest X-ray dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk and return a PIL Image object."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def resize_image(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize an image while preserving the requested dimensions."""
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError("target_size must contain positive values")
    return image.resize(target_size, Image.Resampling.BILINEAR)


def normalize_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a float array normalized to the [0, 1] range."""
    array = np.asarray(image, dtype=np.float32)
    return array / 255.0


def compute_image_statistics(image: Image.Image) -> dict[str, float | int]:
    """Compute basic image statistics for a loaded image."""
    array = np.asarray(image)
    mean_value = float(array.mean())
    std_value = float(array.std())
    return {
        "width": int(image.width),
        "height": int(image.height),
        "channels": int(array.shape[2]) if array.ndim == 3 else 1,
        "mean": mean_value,
        "std": std_value,
    }


def build_class_distribution(records: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Summarize the dataset distribution by split and class."""
    distribution: dict[str, dict[str, int]] = {}
    for split_name in sorted(records["split"].dropna().unique()):
        split_records = records[records["split"] == split_name]
        counts = {}
        for label in sorted(split_records["label"].dropna().unique()):
            counts[label] = int((split_records["label"] == label).sum())
        distribution[str(split_name)] = counts
    return distribution


def image_to_array(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """Return a normalized image tensor ready for model input."""
    resized = resize_image(image, target_size)
    normalized = normalize_image(resized)
    return normalized.astype(np.float32)


def build_dataset_dataframe(dataset_dir: Path | str) -> pd.DataFrame:
    """Build a dataframe with one row per image and the corresponding target label."""
    dataset_path = Path(dataset_dir)
    rows: list[dict[str, Any]] = []

    for split_name in ["train", "val", "test"]:
        split_dir = dataset_path / split_name
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file():
                    rows.append(
                        {
                            "split": split_name,
                            "label": label,
                            "path": str(image_path),
                            "target": 1 if label == "PNEUMONIA" else 0,
                        }
                    )
    return pd.DataFrame(rows)
