"""División estratificada reproducible para el dataset de radiografías de tórax.

Estrategia de división:
- El train original (5.216 imágenes) se divide en 80% train y 20% validation.
- El test original (624 imágenes) permanece completamente intacto.
- Las 16 imágenes del val original del dataset crudo no se utilizan.
- Agrupación por hash SHA-256 para evitar duplicados entre train y validation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.preprocessing import construir_dataframe_dataset


NOMBRES_DIVISION_POR_DEFECTO = ("train", "val", "test")
RATIOS_POR_DEFECTO = (0.80, 0.20)


def _hash_archivo(path: str | Path) -> str:
    """Devolver el resumen SHA-256 de un archivo."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asignar_conteos_grupos(group_sizes: Iterable[int], targets: list[int], rng: np.random.Generator) -> list[int]:
    """Asignar grupos duplicados a los conjuntos respetando los objetivos por clase."""
    sizes = list(group_sizes)
    order = rng.permutation(len(sizes))
    assigned = [0] * len(sizes)
    current = [0] * len(targets)
    for group_index in order:
        size = sizes[group_index]
        candidates = [index for index, target in enumerate(targets) if current[index] + size <= target]
        if not candidates:
            candidates = list(range(len(targets)))
        split_index = min(candidates, key=lambda index: (current[index] / max(targets[index], 1), current[index]))
        assigned[group_index] = split_index
        current[split_index] += size
    return assigned


def _calcular_objetivos_division(total: int, ratios: tuple[float, float]) -> list[int]:
    """Calcular objetivos enteros de división cuya suma sea igual al total de la clase."""
    raw = np.asarray(ratios) * total
    targets = np.floor(raw).astype(int)
    remainder = total - int(targets.sum())
    for index in np.argsort(-(raw - targets))[:remainder]:
        targets[index] += 1
    return targets.tolist()


def crear_manifiesto_division_estratificada(
    directorio_dataset: str | Path,
    ruta_salida: str | Path,
    random_state: int = 42,
    ratios: tuple[float, float] = RATIOS_POR_DEFECTO,
) -> pd.DataFrame:
    """Crear un manifiesto estratificado 80/20 reproducible sin duplicados por hash.

    El train original se divide en train (80%) y validation (20%).
    El test original (624 imágenes) permanece intacto y no participa en la división.
    Las 16 imágenes del val original del dataset crudo no se utilizan.
    """
    if len(ratios) != 2 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios debe contener dos valores que sumen 1.0")

    dataframe = construir_dataframe_dataset(directorio_dataset).copy()
    if dataframe.empty:
        raise ValueError("El dataset no contiene ninguna imagen.")

    dataframe["content_hash"] = dataframe["path"].map(_hash_archivo)
    rng = np.random.default_rng(random_state)
    assignments: dict[str, str] = {}

    df_train_original = dataframe[dataframe["split"] == "train"].copy()

    for label, label_records in df_train_original.groupby("label", sort=True):
        groups = label_records.groupby("content_hash", sort=True).size()
        targets = _calcular_objetivos_division(len(label_records), ratios)
        group_assignments = _asignar_conteos_grupos(groups.tolist(), targets, rng)
        for content_hash, split_index in zip(groups.index, group_assignments):
            split_name = NOMBRES_DIVISION_POR_DEFECTO[split_index]
            assignments[str(content_hash)] = split_name

    df_test_original = dataframe[dataframe["split"] == "test"].copy()
    for content_hash in df_test_original["content_hash"].unique():
        assignments[str(content_hash)] = "test"

    dataframe["split"] = dataframe["content_hash"].map(assignments)
    dataframe = dataframe.drop(columns=["content_hash"])

    dataframe = dataframe[dataframe["split"].isin(["train", "val", "test"])].copy()

    output = Path(ruta_salida)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    return dataframe


def cargar_manifiesto_division(ruta_manifiesto: str | Path) -> pd.DataFrame:
    """Cargar un manifiesto de división generado previamente."""
    manifest = pd.read_csv(ruta_manifiesto)
    required = {"split", "label", "path", "target"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"El manifiesto de división no tiene las columnas: {sorted(missing)}")
    return manifest
