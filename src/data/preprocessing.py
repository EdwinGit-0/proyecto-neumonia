"""Utilidades de preprocesamiento de imágenes para el dataset de radiografías de tórax."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

def cargar_imagen(ruta_imagen: str | Path) -> Image.Image:
    """Cargar una imagen desde disco y devolver un objeto PIL Image."""

    path = Path(ruta_imagen)
    if not path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {path}")
    return Image.open(path).convert("RGB")


def redimensionar_imagen(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Redimensionar una imagen conservando las dimensiones solicitadas."""
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError("target_size debe contener valores positivos")
    return image.resize(target_size, Image.Resampling.BILINEAR)


def normalizar_imagen(image: Image.Image) -> np.ndarray:
    """Convertir una imagen PIL a un arreglo flotante normalizado al rango [0, 1]."""
    array = np.asarray(image, dtype=np.float32)
    return array / 255.0


def calcular_estadisticas_imagen(image: Image.Image) -> dict[str, float | int]:
    """Calcular estadísticas básicas de una imagen cargada."""
    array = np.asarray(image)
    mean_value = float(array.mean())
    std_value = float(array.std())
    return {
        "width": int(image.width),
        "height": int(image.height),
        "channels": int(array.shape[2]) if array.ndim == 3 else 1,
        "mean": mean_value,
        "std": std_value,
    }


def construir_distribucion_clases(records: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Resumir la distribución del dataset por conjunto y clase."""
    distribution: dict[str, dict[str, int]] = {}
    for split_name in sorted(records["split"].dropna().unique()):
        split_records = records[records["split"] == split_name]
        counts = {}
        for label in sorted(split_records["label"].dropna().unique()):
            counts[label] = int((split_records["label"] == label).sum())
        distribution[str(split_name)] = counts
    return distribution


def imagen_a_arreglo(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """Devolver un tensor de imagen normalizado y listo para la entrada del modelo."""
    resized = redimensionar_imagen(image, target_size)
    normalized = normalizar_imagen(resized)
    return normalized.astype(np.float32)


def construir_dataframe_dataset(directorio_dataset: Path | str) -> pd.DataFrame:
    """Construir un DataFrame con una fila por imagen y su etiqueta objetivo."""
    dataset_path = Path(directorio_dataset)
    rows: list[dict[str, Any]] = []

    for split_name in ["train", "val", "test"]:
        split_dir = dataset_path / split_name
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for ruta_imagen in sorted(class_dir.iterdir()):
                if ruta_imagen.is_file():
                    rows.append(
                        {
                            "split": split_name,
                            "label": label,
                            "path": str(ruta_imagen),
                            "target": 1 if label == "PNEUMONIA" else 0,
                        }
                    )
    return pd.DataFrame(rows)
