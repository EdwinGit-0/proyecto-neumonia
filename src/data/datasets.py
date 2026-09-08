"""Dataset pipeline builders for the chest X-ray project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from src.data.preprocessing import build_dataset_dataframe, image_to_array, load_image


def _to_tensor_dataset(df: Any, image_size: tuple[int, int], batch_size: int = 32) -> tf.data.Dataset:
    """Create a TensorFlow dataset from a dataframe of image paths."""
    def generator():
        for _, row in df.iterrows():
            image = load_image(row["path"])
            feature = image_to_array(image, image_size)
            label = float(row["target"])
            yield feature, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    return dataset.shuffle(buffer_size=max(1000, len(df))).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_data_pipelines(
    dataset_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: str = "val",
    test_split: str = "test",
) -> dict[str, tf.data.Dataset]:
    """Create the training, validation and test datasets from the project folders."""
    dataset_path = Path(dataset_dir)
    dataframe = build_dataset_dataframe(dataset_path)

    datasets: dict[str, tf.data.Dataset] = {}
    for split_name in ["train", validation_split, test_split]:
        split_df = dataframe[dataframe["split"] == split_name].copy()
        if split_df.empty:
            continue
        datasets[split_name] = _to_tensor_dataset(split_df, image_size=image_size, batch_size=batch_size)

    return datasets
