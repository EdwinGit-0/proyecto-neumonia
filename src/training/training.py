"""Utilidades de entrenamiento y checkpoints para el pipeline de clasificación de neumonía."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.models.architectures import construir_modelo_transferencia
from src.utils.paths import asegurar_directorio, DIRECTORIO_MODELOS


def establecer_semilla(seed: int = 42) -> None:
    """Establecer una semilla aleatoria reproducible en las librerías relevantes."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def construir_callbacks_entrenamiento(directorio_modelo: Path | str) -> list[Any]:
    """Crear los callbacks estándar para el entrenamiento de modelos."""
    model_path = Path(directorio_modelo)
    asegurar_directorio(model_path)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path / "best_model.keras"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    return [checkpoint, early_stopping, reduce_lr]


def entrenar_modelo(
    model_name: str,
    datos_entrenamiento: Any,
    datos_validacion: Any,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 42,
    directorio_salida: Path | str | None = None,
) -> tuple[Any, tf.keras.callbacks.History]:
    """Entrenar un modelo con transferencia de aprendizaje sobre los datasets proporcionados."""
    establecer_semilla(seed)
    model = construir_modelo_transferencia(
        model_name=model_name,
        input_shape=input_shape,
        classes=2,
        weights="imagenet",
    )
    model.optimizer.learning_rate.assign(learning_rate)

    directorio_modelo = Path(directorio_salida) if directorio_salida is not None else DIRECTORIO_MODELOS / model_name.lower()
    asegurar_directorio(directorio_modelo)
    history = model.fit(
        datos_entrenamiento,
        validation_data=datos_validacion,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=construir_callbacks_entrenamiento(directorio_modelo),
        verbose=1,
    )
    return model, history
