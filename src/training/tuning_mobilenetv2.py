"""Fine-tuning de MobileNetV2 para el flujo de optimización.

Aporta al flujo vigente el constructor de MobileNetV2 con backbone congelado
o descongelado a partir de un índice del extractor, más el criterio de
selección y la tolerancia de recall que comparte con
:mod:`src.training.optimizacion_mobilenetv2`.

La etapa de tuning que variaba ``learning_rate``, ``dropout`` y ``epochs``
(``entrenar_y_evaluar_tuning``) quedó superada por el análisis de sensibilidad de
21 pruebas, que ya evaluó esos tres factores y fijó MobileNetV2 / ``lr=0.001`` /
``dropout=0.3`` / ``3`` épocas como punto de partida. Ese código se retiró del flujo
activo; su artefacto histórico permanece en ``models/mobilenetv2_tuning_results.json``.

El nombre del módulo se conserva porque los guiones históricos de
``experiments/`` lo importan por nombre.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model


CRITERIO_METRICAS = ("balanced_accuracy", "roc_auc", "f1", "accuracy")


def construir_modelo_mobilenetv2_tunable(
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    fine_tune_from: int | None = None,
) -> Model:
    """Crear MobileNetV2 con la misma cabeza que el baseline pero afinable.

    Réplica fiel de ``src/models/architectures.py`` (GAP, Dense 128 relu,
    Dropout, Dense 1 sigmoid). Si ``fine_tune_from`` no es ``None`` se descongelan
    las capas del extractor desde ese índice (sin tocar las cabezas).

    Los pesos de ImageNet se cargan de la caché local, así que la única
    inicialización aleatoria es la cabeza densa. El flujo de optimización llama a
    :func:`src.utils.reproducibility.reiniciar_semilla` justo antes de construir
    el modelo, de modo que esa inicialización también es reproducible.
    """
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    if fine_tune_from is not None:
        if fine_tune_from <= 0 or fine_tune_from >= len(base_model.layers):
            raise ValueError(f"fine_tune_from inválido: {fine_tune_from}")
        for layer in base_model.layers[fine_tune_from:]:
            layer.trainable = True

    inputs = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
