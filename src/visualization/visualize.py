"""Funciones de visualización para el proyecto de neumonía en radiografías de tórax."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.preprocessing import imagen_a_arreglo, cargar_imagen
from src.data.splitting import cargar_manifiesto_division
from src.models.architectures import NOMBRES_MODELOS
from src.utils.paths import DIRECTORIO_DATOS, DIRECTORIO_FIGURAS, RAIZ_PROYECTO, asegurar_directorio


TAMANO_IMAGEN = (224, 224)
RUTA_MANIFIESTO = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"


def _seleccionar_imagen_entrenamiento() -> Path | None:
    """Devolver una imagen real del conjunto de entrenamiento experimental."""
    if RUTA_MANIFIESTO.exists():
        manifest = cargar_manifiesto_division(RUTA_MANIFIESTO)
        train_records = manifest[manifest["split"] == "train"]
        train_records = train_records[train_records["label"] == "NORMAL"]
        if not train_records.empty:
            return Path(train_records.iloc[0]["path"])
    train_normal = Path(DIRECTORIO_DATOS) / "train" / "NORMAL"
    if train_normal.is_dir():
        images = sorted(train_normal.iterdir())
        if images:
            return images[0]
    return None


def _aplicar_capa(image_array: np.ndarray, layer: tf.keras.layers.Layer) -> np.ndarray:
    """Aplicar una capa de augmentation a una imagen normalizada."""
    batch = layer(image_array[np.newaxis, ...], training=True)
    return np.asarray(batch[0])


def construir_variantes_augmentation(image_array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Aplicar las transformaciones de augmentation existentes del proyecto a una sola imagen.

    Solo se utilizan RandomRotation(0.05) y RandomZoom(0.05), reproduciendo
    exactamente el pipeline de entrenamiento. El volteo horizontal fue descartado
    porque las radiografías de tórax pueden contener información de lateralidad
    anatómica (marcadores L/R) que un volteo podría invertir artificialmente.
    """
    rotation = lambda: tf.keras.layers.RandomRotation(0.05)
    zoom = lambda: tf.keras.layers.RandomZoom(0.05)

    variants = [
        ("Original", None),
        ("Rotación", [rotation()]),
        ("Zoom", [zoom()]),
        ("Rotación + Zoom", [rotation(), zoom()]),
    ]

    generated: list[tuple[str, np.ndarray]] = []
    original = image_array.copy()
    for label, layers in variants:
        if layers is None:
            generated.append((label, original))
            continue
        array = original
        for layer in layers:
            array = _aplicar_capa(array, layer)
        generated.append((label, array))
    return generated


def graficar_ejemplos_augmentation(
    ruta_salida: str | Path = DIRECTORIO_FIGURAS / "data_augmentation_examples.png",
) -> Path:
    """Crear una figura 2x2 que muestra la augmentation de datos sobre una imagen de entrenamiento."""
    train_image = _seleccionar_imagen_entrenamiento()
    if train_image is None:
        raise FileNotFoundError("No se encontraron imágenes de entrenamiento para ilustrar la augmentation de datos.")

    source = cargar_imagen(train_image)
    image_array = imagen_a_arreglo(source, TAMANO_IMAGEN)
    variants = construir_variantes_augmentation(image_array)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(
        "Data augmentation aplicada al conjunto de entrenamiento (imágenes de 224 x 224 px)",
        fontsize=13,
    )
    for axis, (label, array) in zip(axes.ravel(), variants):
        axis.imshow(array, cmap="gray")
        axis.set_title(label, fontsize=10)
        axis.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = Path(ruta_salida)
    asegurar_directorio(output.parent)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def graficar_sensibilidad(
    payload: dict[str, Any] | None = None,
    ruta_salida: str | Path = DIRECTORIO_FIGURAS / "sensitivity_validation.png",
) -> Path:
    """Crear la figura de sensibilidad de hiperparámetros por arquitectura.

    Muestra la Balanced Accuracy de validación de las 7 configuraciones evaluadas
    (un factor por vez) para cada arquitectura. La etapa de sensibilidad solo
    utiliza TRAIN y VALIDATION, por lo que la figura no incluye TEST.
    """
    if payload is None:
        from src.training.sensitivity import cargar_resultados_sensibilidad

        payload = cargar_resultados_sensibilidad()

    arquitecturas = [nombre for nombre in NOMBRES_MODELOS if nombre in payload["mejor_por_arquitectura"]]
    if not arquitecturas:
        raise ValueError("El artefacto de sensibilidad no contiene resultados por arquitectura.")

    fig, axes = plt.subplots(1, len(arquitecturas), figsize=(5.2 * len(arquitecturas), 4.6), sharey=True)
    if len(arquitecturas) == 1:
        axes = [axes]

    for axis, model_name in zip(axes, arquitecturas):
        info = payload["mejor_por_arquitectura"][model_name]
        etiquetados = info["tabla_ordenada"]
        etiquetas = [f"{fila['config_id']}\nlr={fila['config']['learning_rate']:g}, do={fila['config']['dropout']:g}, ep={fila['config']['epochs']}" for fila in etiquetados]
        valores = [float(fila["balanced_accuracy"]) for fila in etiquetados]
        colores = []
        for fila in etiquetados:
            colores.append("#c44e52" if fila["config_id"] == info["config_id"] else "#4c72b0")
        eje = np.arange(len(etiquetados))
        axis.bar(eje, valores, color=colores)
        axis.set_xticks(eje)
        axis.set_xticklabels(etiquetas, fontsize=7, rotation=30, ha="right")
        axis.set_ylim(0.4, 1.0)
        axis.set_title(f"{model_name}: mejor {info['config_id']} (BA={info['validation']['balanced_accuracy']:.4f})", fontsize=10)
        axis.grid(axis="y", linestyle=":", alpha=0.6)
        if info.get("referencia_original"):
            referencia = float(info["referencia_original"]["validation"]["balanced_accuracy"])
            axis.axhline(referencia, color="#555555", linestyle="--", linewidth=1)
            axis.text(
                0.02,
                referencia + 0.01,
                f"base={referencia:.4f}",
                transform=axis.get_yaxis_transform(),
                fontsize=7,
                color="#555555",
            )
        for posicion, valor in zip(eje, valores):
            axis.text(posicion, valor + 0.005, f"{valor:.3f}", ha="center", fontsize=7)

    axes[0].set_ylabel("Balanced Accuracy (validation)")
    fig.suptitle(
        "Sensibilidad de hiperparámetros por arquitectura (solo TRAIN y VALIDATION; el test no interviene)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(ruta_salida)
    asegurar_directorio(output.parent)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    result = graficar_ejemplos_augmentation()
    print(result)