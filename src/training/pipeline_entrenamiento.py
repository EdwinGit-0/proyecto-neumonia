"""Piezas compartidas de entrenamiento y evaluación del proyecto.

Este módulo agrupa lo que comparten la etapa de sensibilidad y el flujo final de
estrategias: configuración del entorno, augmentación, entrenamiento de una corrida,
recolección de probabilidades y figuras de evaluación.

No contiene ningún pipeline que cargue el split ``test``: la única evaluación sobre
test del proyecto es la evaluación final del modelo definitivo, en
:mod:`src.training.flujo_final`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve

from src.utils.paths import DIRECTORIO_FIGURAS, asegurar_directorio
from src.utils.reproducibility import SEMILLA, configurar_reproducibilidad


TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 16
UMBRAL_BASE = 0.5

CLASES = ("NORMAL", "PNEUMONIA")


def configurar_ejecucion() -> None:
    """Configurar TensorFlow para un entrenamiento reproducible en este entorno."""
    configurar_reproducibilidad(SEMILLA)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)


def crear_capas_augmentation() -> list:
    """Devolver las capas de data augmentation usadas en el entrenamiento.

    Se descarta el volteo horizontal porque las radiografías de tórax pueden
    contener información de lateralidad anatómica y marcadores L/R; un volteo
    horizontal podría invertir artificialmente esa información.

    Cada capa recibe ``seed=SEMILLA``: sin ella la augmentación usa el estado
    global del generador en el momento de crearse, y dos ejecuciones del mismo
    código entrenaban sobre imágenes distintas. La magnitud de la augmentation no
    cambia; solo pasa a ser reproducible.
    """
    return [
        tf.keras.layers.RandomRotation(0.05, seed=SEMILLA),
        tf.keras.layers.RandomZoom(0.05, seed=SEMILLA),
    ]


def construir_pipeline_augmentation(train_dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Aplicar augmentation ligera únicamente al conjunto de entrenamiento."""
    augmentation = tf.keras.Sequential(crear_capas_augmentation(), name="augmentation")

    def augment_examples(features: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return augmentation(features, training=True), labels

    # deterministic=True mantiene el orden de salida aunque el map se ejecute en
    # paralelo; con las capas ya sembradas el resultado es bit a bit reproducible.
    return train_dataset.map(
        augment_examples, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )


def construir_callbacks_entrenamiento(
    ruta_checkpoint: Path,
    validar: bool = True,
    paciencia: int = 3,
) -> list[tf.keras.callbacks.Callback]:
    """Devolver los callbacks del proyecto para una corrida de entrenamiento.

    Con ``validar=True`` el checkpoint se elige por ``val_loss`` y ``EarlyStopping``
    vigila la validación, que es el modo usado para las corridas que deciden el
    modelo. Con ``validar=False`` no hay conjunto de validación sobre el que elegir:
    el modelo definitivo se entrena con el número de épocas ya decidido y el
    checkpoint guarda los pesos del último epoch.
    """
    if not validar:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ruta_checkpoint),
                monitor="loss",
                mode="min",
                save_best_only=True,
            )
        ]

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ruta_checkpoint),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]


def entrenar_corrida(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset | None,
    epochs: int,
    ruta_checkpoint: Path,
    class_weight: dict[int, float] | None = None,
    verbose: int = 2,
) -> dict[str, Any]:
    """Entrenar una corrida y devolver tiempos e historial.

    ``val_dataset=None`` entrena el número exacto de épocas pedido sin callbacks que
    dependan de la validación, que es el caso del modelo definitivo entrenado con
    ``train + val``.
    """
    asegurar_directorio(Path(ruta_checkpoint).parent)
    if Path(ruta_checkpoint).exists():
        Path(ruta_checkpoint).unlink()

    inicio = time.time()
    historial = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=int(epochs),
        batch_size=TAMANO_LOTE,
        class_weight=class_weight,
        callbacks=construir_callbacks_entrenamiento(Path(ruta_checkpoint), validar=val_dataset is not None),
        verbose=verbose,
    )
    segundos = time.time() - inicio

    return {
        "segundos_entrenamiento": float(segundos),
        "epochs_ejecutadas": int(len(historial.history["loss"])),
        "early_stopping": bool(len(historial.history["loss"]) < int(epochs)),
        "historial": {clave: [float(valor) for valor in valores] for clave, valores in historial.history.items()},
    }


def predecir_probabilidades(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """Devolver etiquetas reales y probabilidades sigmoides de un dataset."""
    etiquetas: list[np.ndarray] = []
    probabilidades: list[np.ndarray] = []
    for features, labels in dataset:
        predicciones = model.predict(features, verbose=0)
        etiquetas.append(labels.numpy().astype(int))
        probabilidades.append(predicciones.ravel())
    return np.concatenate(etiquetas), np.concatenate(probabilidades)


def guardar_matriz_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_name: str,
    directorio: Path | None = None,
) -> Path:
    """Guardar la gráfica de la matriz de confusión normalizada para un modelo."""
    directorio = Path(directorio) if directorio is not None else DIRECTORIO_FIGURAS
    asegurar_directorio(directorio)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm = cm.astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    imagen = ax.imshow(cm_norm, cmap="Blues")
    ax.set_title(f"Matriz de confusión - {model_name}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASES)
    ax.set_yticklabels(CLASES)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    fig.colorbar(imagen, ax=ax)
    plt.tight_layout()
    ruta_salida = directorio / f"confusion_matrix_{split_name.lower()}_{model_name.lower()}.png"
    plt.savefig(ruta_salida, dpi=200)
    plt.close(fig)
    return ruta_salida


def guardar_curva_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    split_name: str,
    directorio: Path | None = None,
    etiqueta_adicional: str | None = None,
) -> Path:
    """Guardar la curva ROC para un modelo."""
    directorio = Path(directorio) if directorio is not None else DIRECTORIO_FIGURAS
    asegurar_directorio(directorio)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=etiqueta_adicional or f"{model_name}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"Curva ROC - {model_name}")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend()
    plt.tight_layout()
    ruta_salida = directorio / f"roc_curve_{split_name.lower()}_{model_name.lower()}.png"
    plt.savefig(ruta_salida, dpi=200)
    plt.close(fig)
    return ruta_salida
