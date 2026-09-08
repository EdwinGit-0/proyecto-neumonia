"""Training and checkpoint utilities for the Pneumonia classification pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.models.architectures import build_transfer_model
from src.utils.paths import ensure_directory, MODELS_DIR


def set_seed(seed: int = 42) -> None:
    """Set a reproducible random seed across the relevant libraries."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_training_callbacks(model_dir: Path | str) -> list[Any]:
    """Create the standard callbacks for model training."""
    model_path = Path(model_dir)
    ensure_directory(model_path)

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


def train_model(
    model_name: str,
    train_data: Any,
    validation_data: Any,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 42,
    output_dir: Path | str | None = None,
) -> tuple[Any, tf.keras.callbacks.History]:
    """Train a transfer-learning model on the provided datasets."""
    set_seed(seed)
    model = build_transfer_model(
        model_name=model_name,
        input_shape=input_shape,
        classes=2,
        weights="imagenet",
    )
    model.optimizer.learning_rate.assign(learning_rate)

    model_dir = Path(output_dir) if output_dir is not None else MODELS_DIR / model_name.lower()
    ensure_directory(model_dir)
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=build_training_callbacks(model_dir),
        verbose=1,
    )
    return model, history
