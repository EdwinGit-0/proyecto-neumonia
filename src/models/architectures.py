"""Transfer-learning model builders for the pneumonia classification task."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.models import Model


MODEL_NAMES = {"VGG16", "ResNet50", "MobileNetV2"}


def build_transfer_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    classes: int = 2,
    weights: str | None = "imagenet",
) -> Model:
    """Create a transfer-learning model for chest X-ray classification."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {sorted(MODEL_NAMES)}")

    if input_shape[0] < 32 or input_shape[1] < 32:
        raise ValueError("Input shape must be at least 32x32 pixels.")

    if model_name == "VGG16":
        base_model = VGG16(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    elif model_name == "ResNet50":
        base_model = ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    else:
        base_model = MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )

    base_model.trainable = False
    inputs = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model_registry() -> dict[str, Any]:
    """Return the available model builders."""
    return {
        "VGG16": lambda **kwargs: build_transfer_model("VGG16", **kwargs),
        "ResNet50": lambda **kwargs: build_transfer_model("ResNet50", **kwargs),
        "MobileNetV2": lambda **kwargs: build_transfer_model("MobileNetV2", **kwargs),
    }
