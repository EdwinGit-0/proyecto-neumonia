"""Tratamiento del desbalance de clases para el flujo de optimización.

Este módulo aporta al flujo vigente únicamente las dos piezas del tratamiento del
desbalance que ya existían implementadas:

- ``class_weight``: pesos por clase equilibrados calculados sobre ``train``.
- ``oversampling``: duplicación de la clase minoritaria NORMAL solo dentro de ``train``.

La etapa que las entrenaba y evaluaba por separado (``entrenar_y_evaluar_desbalance``,
basada en el MobileNetV2 ajustado a ``lr=3e-4``) quedó superada por
:mod:`src.training.optimizacion_mobilenetv2`, que parte del ganador de la
sensibilidad (``lr=0.001``) y compara ambas estrategias sobre ``validation`` junto
al fine-tuning. Ese código se retiró del flujo activo; sus artefactos históricos
permanecen en ``models/mobilenetv2_desbalance_results.json``.

Las funciones conservadas se usan con ``seed=SEMILLA`` en el ``shuffle`` para que el
orden de los ejemplos sea reproducible.

El nombre del módulo se conserva porque los guiones históricos de
``experiments/`` lo importan por nombre; cambiarlo rompería su reproducción.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data.preprocessing import cargar_imagen, imagen_a_arreglo
from src.data.splitting import cargar_manifiesto_division
from src.utils.reproducibility import SEMILLA


def construir_dataset_tensor(df: pd.DataFrame, image_size: tuple[int, int], batch_size: int = 32) -> tf.data.Dataset:
    """Réplica exacta del constructor interno de ``src.data.datasets`` (mismo shuffle/batch/prefetch)."""

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
    return (
        dataset.shuffle(buffer_size=max(1000, len(df)), seed=SEMILLA, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def construir_train_oversampling(
    ruta_manifiesto: Path, image_size: tuple[int, int], batch_size: int
) -> tuple[tf.data.Dataset, dict[str, Any]]:
    """Aumentar la clase minoritaria NORMAL dentro de TRAIN hasta equiparar recuentos.

    La selección de los duplicados y el orden del dataframe usan ``random_state=SEMILLA``
    y el ``shuffle`` del dataset también, de modo que la composición del train
    oversampleado es reproducible.
    """
    dataframe = cargar_manifiesto_division(ruta_manifiesto)
    df_train = dataframe[dataframe["split"] == "train"].copy()
    df_normal = df_train[df_train["target"] == 0]
    df_pneumonia = df_train[df_train["target"] == 1]

    faltantes = len(df_pneumonia) - len(df_normal)
    df_normal_extra = df_normal.sample(n=faltantes, replace=True, random_state=SEMILLA)
    df_train_oversampled = pd.concat([df_train, df_normal_extra], ignore_index=True)
    df_train_oversampled = df_train_oversampled.sample(frac=1.0, random_state=SEMILLA).reset_index(drop=True)

    return construir_dataset_tensor(df_train_oversampled, image_size=image_size, batch_size=batch_size), {
        "NORMAL_original": int(len(df_normal)),
        "PNEUMONIA_original": int(len(df_pneumonia)),
        "NORMAL_oversampled": int(df_train_oversampled["target"].value_counts().get(0)),
        "total_train_oversampled": int(len(df_train_oversampled)),
        "duplicadas_NORMAL": int(faltantes),
    }


def calcular_pesos_clase(ruta_manifiesto: Path) -> dict[int, float]:
    """Pesos balanceados por clase calculados sobre TRAIN."""
    dataframe = cargar_manifiesto_division(ruta_manifiesto)
    y_train = (dataframe[dataframe["split"] == "train"]["target"]).to_numpy().astype(int)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    return {int(clase): float(peso) for clase, peso in zip([0, 1], weights)}
