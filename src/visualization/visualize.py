"""Funciones de visualización para el proyecto de neumonía en radiografías de tórax."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.preprocessing import imagen_a_arreglo, cargar_imagen
from src.data.splitting import cargar_manifiesto_division
from src.utils.paths import DIRECTORIO_DATOS, DIRECTORIO_FIGURAS, RAIZ_PROYECTO, asegurar_directorio


TAMANO_IMAGEN = (224, 224)
RUTA_MANIFIESTO = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_70_15_15.csv"


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

    Solo se utilizan RandomFlip("horizontal"), RandomRotation(0.05) y RandomZoom(0.05),
    reproduciendo exactamente el pipeline de entrenamiento.
    """
    flip = lambda: tf.keras.layers.RandomFlip("horizontal")
    rotation = lambda: tf.keras.layers.RandomRotation(0.05)
    zoom = lambda: tf.keras.layers.RandomZoom(0.05)

    variants = [
        ("Original", None),
        ("Volteo horizontal", [flip()]),
        ("Rotación", [rotation()]),
        ("Zoom", [zoom()]),
        ("Volteo + Rotación", [flip(), rotation()]),
        ("Volteo + Zoom", [flip(), zoom()]),
        ("Rotación + Zoom", [rotation(), zoom()]),
        ("Volteo + Rotación + Zoom", [flip(), rotation(), zoom()]),
        ("Volteo + Rotación + Zoom (2.ª muestra)", [flip(), rotation(), zoom()]),
    ]

    generated: list[tuple[str, np.ndarray]] = []
    original = image_array.copy()
    for label, layers in variants:
        if layers is None:
            generated.append((label, original))
            continue
        array = original
        for layer in layers:
            before = array
            array = _aplicar_capa(array, layer)
            if isinstance(layer, tf.keras.layers.RandomFlip) and np.array_equal(array, before):
                seed = 0
                while np.array_equal(array, before) and seed < 64:
                    array = _aplicar_capa(before, tf.keras.layers.RandomFlip("horizontal", seed=seed))
                    seed += 1
        generated.append((label, array))
    return generated


def graficar_ejemplos_augmentation(
    ruta_salida: str | Path = DIRECTORIO_FIGURAS / "data_augmentation_examples.png",
) -> Path:
    """Crear una figura de 3x3 que muestra la augmentation de datos sobre una imagen de entrenamiento."""
    train_image = _seleccionar_imagen_entrenamiento()
    if train_image is None:
        raise FileNotFoundError("No se encontraron imágenes de entrenamiento para ilustrar la augmentation de datos.")

    source = cargar_imagen(train_image)
    image_array = imagen_a_arreglo(source, TAMANO_IMAGEN)
    variants = construir_variantes_augmentation(image_array)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
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


if __name__ == "__main__":
    result = graficar_ejemplos_augmentation()
    print(result)