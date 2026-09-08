"""Dataset utilities for the pneumonia chest X-ray project.

This module provides reusable helpers for inspecting the raw image dataset,
collecting metadata, and producing EDA assets for the current project stage.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"


def _hash_file(path: Path) -> str:
    """Compute the SHA256 hash of the file content."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_image_records(dataset_dir: Path | str) -> pd.DataFrame:
    """Collect one row per image with metadata needed for the EDA."""
    data_dir = Path(dataset_dir)
    rows: list[dict[str, object]] = []

    if not data_dir.exists():
        return pd.DataFrame(
            columns=[
                "split",
                "label",
                "path",
                "filename",
                "extension",
                "file_size_bytes",
                "width",
                "height",
                "mode",
            ]
        )

    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue

        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            for image_path in sorted(label_dir.iterdir()):
                if not image_path.is_file():
                    continue

                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        mode = image.mode
                except (UnidentifiedImageError, OSError):
                    width, height, mode = None, None, None

                rows.append(
                    {
                        "split": split_dir.name,
                        "label": label_dir.name,
                        "path": str(image_path),
                        "filename": image_path.name,
                        "extension": image_path.suffix.lower(),
                        "file_size_bytes": image_path.stat().st_size,
                        "width": width,
                        "height": height,
                        "mode": mode,
                    }
                )

    return pd.DataFrame(rows)


def get_dataset_summary(dataset_dir: Path | str) -> dict[str, object]:
    """Build a summary of the image dataset for EDA and validation."""
    data_dir = Path(dataset_dir)
    records = collect_image_records(data_dir)

    counts_by_split_and_class: dict[str, dict[str, int]] = defaultdict(dict)
    splits = sorted({value for value in records["split"].dropna().tolist() if value})
    for split_name in splits:
        split_records = records[records["split"] == split_name]
        labels = sorted({value for value in split_records["label"].dropna().tolist() if value})
        for label_name in labels:
            counts_by_split_and_class[split_name][label_name] = int(
                (split_records["label"] == label_name).sum()
            )

    corrupt_images: list[Path] = []
    for _, row in records.iterrows():
        path = Path(row["path"])
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt_images.append(path)

    file_hashes: dict[str, list[Path]] = defaultdict(list)
    for _, row in records.iterrows():
        path = Path(row["path"])
        try:
            file_hashes[_hash_file(path)].append(path)
        except OSError:
            continue

    duplicate_paths = [
        path
        for paths in file_hashes.values()
        for path in paths[1:]
    ]

    summary = {
        "dataset_dir": str(data_dir),
        "total_images": int(len(records)),
        "counts_by_split_and_class": dict(counts_by_split_and_class),
        "corrupt_images": corrupt_images,
        "duplicate_paths": duplicate_paths,
        "records": records,
    }

    return summary


def save_eda_figures(records: pd.DataFrame, output_dir: Path | str) -> dict[str, Path]:
    """Create the principal EDA figures for distribution and image metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Path] = {}

    distribution = records.groupby(["split", "label"]).size().unstack(fill_value=0)
    distribution_path = output_path / "dataset_distribution.png"
    distribution.plot(kind="bar", figsize=(10, 6), title="Dataset distribution by split and class")
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=200)
    plt.close()
    figures["distribution"] = distribution_path

    sizes = records["file_size_bytes"].dropna()
    sizes_path = output_path / "file_size_distribution.png"
    plt.figure(figsize=(10, 5))
    plt.hist(sizes, bins=30, color="steelblue", edgecolor="black")
    plt.title("Distribution of image file sizes")
    plt.xlabel("File size (bytes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(sizes_path, dpi=200)
    plt.close()
    figures["sizes"] = sizes_path

    valid_dims = records.dropna(subset=["width", "height"]).copy()
    dimensions_path = output_path / "image_dimensions.png"
    plt.figure(figsize=(8, 8))
    plt.scatter(valid_dims["width"], valid_dims["height"], alpha=0.6)
    plt.title("Image dimensions")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.tight_layout()
    plt.savefig(dimensions_path, dpi=200)
    plt.close()
    figures["dimensions"] = dimensions_path

    return figures


def plot_sample_images(records: pd.DataFrame, output_dir: Path | str, max_per_class: int = 3) -> dict[str, Path]:
    """Save representative images from each class for visual inspection."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_paths: dict[str, Path] = {}
    for label in ["NORMAL", "PNEUMONIA"]:
        label_records = records[records["label"] == label].head(max_per_class)
        if label_records.empty:
            continue

        fig, axes = plt.subplots(1, min(len(label_records), max_per_class), figsize=(12, 4))
        fig.suptitle(f"Examples: {label}")

        if len(label_records) == 1:
            axes = [axes]

        for axis, (_, row) in zip(axes, label_records.iterrows()):
            image_path = Path(row["path"])
            try:
                with Image.open(image_path) as image:
                    axis.imshow(image)
            except Exception:
                axis.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
                axis.set_title("Unreadable image")
            axis.axis("off")

        fig.tight_layout()
        sample_path = output_path / f"sample_{label.lower()}.png"
        fig.savefig(sample_path, dpi=200)
        plt.close(fig)
        sample_paths[label] = sample_path

    return sample_paths


def validate_dataset_structure(dataset_dir: Path | str) -> dict[str, object]:
    """Validate that the expected train/val/test folders and class labels exist."""
    data_dir = Path(dataset_dir)
    expected = {
        "train": ["NORMAL", "PNEUMONIA"],
        "val": ["NORMAL", "PNEUMONIA"],
        "test": ["NORMAL", "PNEUMONIA"],
    }

    validation = {}
    for split_name, labels in expected.items():
        split_dir = data_dir / split_name
        validation[split_name] = {
            "exists": split_dir.exists(),
            "labels": sorted([path.name for path in split_dir.iterdir() if path.is_dir()])
            if split_dir.exists()
            else [],
            "missing_labels": [label for label in labels if not (split_dir / label).exists()],
        }

    return validation


def run_eda(dataset_dir: Path | str = DATA_DIR) -> dict[str, object]:
    """Run the dataset-level EDA and return summary plus generated figures."""
    data_dir = Path(dataset_dir)
    validation = validate_dataset_structure(data_dir)
    records = collect_image_records(data_dir)
    summary = get_dataset_summary(data_dir)
    figures_dir = PROJECT_ROOT / "reports" / "figures"

    figures = save_eda_figures(records, figures_dir)
    sample_figures = plot_sample_images(records, figures_dir)

    return {
        "validation": validation,
        "summary": summary,
        "figures": {**figures, **sample_figures},
    }


if __name__ == "__main__":
    summary = run_eda(DATA_DIR)
    print(summary["summary"]["total_images"])
    print(summary["summary"]["counts_by_split_and_class"])
