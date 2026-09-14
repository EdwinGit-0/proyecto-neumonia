"""Constructores de pipelines de dataset para el proyecto de radiografías de tórax."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from src.data.preprocessing import construir_dataframe_dataset, imagen_a_arreglo, cargar_imagen
from src.data.splitting import cargar_manifiesto_division


def _a_dataset_tensor(df: Any, image_size: tuple[int, int], batch_size: int = 32) -> tf.data.Dataset:
    """Crear un dataset de TensorFlow a partir de un DataFrame de rutas de imágenes."""
    def generator():
        for _, row in df.iterrows():
            image = cargar_imagen(row["path"])
            feature = imagen_a_arreglo(image, image_size)
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


def construir_pipelines_datos(
    directorio_dataset: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: str = "val",
    test_split: str = "test",
    manifiesto_division: str | Path | None = None,
) -> dict[str, tf.data.Dataset]:
    """Crear los datasets de entrenamiento, validación y prueba desde carpetas o un manifiesto."""
    dataset_path = Path(directorio_dataset)
    dataframe = cargar_manifiesto_division(manifiesto_division) if manifiesto_division else construir_dataframe_dataset(dataset_path)

    datasets: dict[str, tf.data.Dataset] = {}
    for split_name in ["train", validation_split, test_split]:
        split_df = dataframe[dataframe["split"] == split_name].copy()
        if split_df.empty:
            continue
        datasets[split_name] = _a_dataset_tensor(split_df, image_size=image_size, batch_size=batch_size)

    return datasets
