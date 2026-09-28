"""Constructores de modelos con transferencia de aprendizaje para la clasificación de neumonía."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.models import Model


NOMBRES_MODELOS = {"VGG16", "ResNet50", "MobileNetV2"}

DROPOUT_BASE = 0.3
LEARNING_RATE_BASE = 1e-4

_CONSTRUCTORES_BASE = {"VGG16": VGG16, "ResNet50": ResNet50, "MobileNetV2": MobileNetV2}


def construir_modelo_proyecto(
    model_name: str,
    dropout: float = DROPOUT_BASE,
    learning_rate: float = LEARNING_RATE_BASE,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str | None = "imagenet",
) -> Model:
    """Crear la arquitectura del proyecto con ``dropout`` y ``learning_rate`` configurables.

    Es el único punto donde se construye una arquitectura del proyecto: base ImageNet
    congelada y cabeza GAP -> Dense(128, relu) -> Dropout -> Dense(1, sigmoid) compilada
    con Adam y ``binary_crossentropy``. Lo usan tanto la etapa de sensibilidad como el
    flujo final de estrategias, de modo que la única diferencia entre corridas sea el
    valor del hiperparámetro que se esté variando.
    """
    if model_name not in NOMBRES_MODELOS:
        raise ValueError(f"Modelo no compatible: {model_name}. Modelos disponibles: {sorted(NOMBRES_MODELOS)}")

    if input_shape[0] < 32 or input_shape[1] < 32:
        raise ValueError("La forma de entrada debe tener al menos 32x32 píxeles.")

    base_model = _CONSTRUCTORES_BASE[model_name](
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def construir_modelo_transferencia(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    classes: int = 2,
    weights: str | None = "imagenet",
) -> Model:
    """Crear un modelo con transferencia de aprendizaje para la clasificación de radiografías."""
    return construir_modelo_proyecto(
        model_name=model_name,
        input_shape=input_shape,
        weights=weights,
    )


def obtener_registro_modelos() -> dict[str, Any]:
    """Devolver los constructores de modelos disponibles.

    Cada constructor acepta la firma completa de :func:`construir_modelo_proyecto`, de modo
    que pueda invocarse tanto como ``registro[nombre](nombre)`` como con ``dropout``,
    ``learning_rate``, ``input_shape`` o ``weights``.
    """
    return {
        model_name: (lambda nombre, **kwargs: construir_modelo_proyecto(nombre, **kwargs))
        for model_name in sorted(NOMBRES_MODELOS)
    }
