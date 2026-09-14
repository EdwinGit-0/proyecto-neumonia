"""Utilidades de dataset para el proyecto de neumonía en radiografías de tórax.

Este módulo proporciona funciones reutilizables para inspeccionar el dataset de
imágenes crudas, recopilar metadatos y generar los artefactos EDA para la etapa
actual del proyecto.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
DIRECTORIO_DATOS = RAIZ_PROYECTO / "data" / "raw" / "chest_xray"


def _calcular_hash_archivo(path: Path) -> str:
    """Calcular el hash SHA256 del contenido de un archivo."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recopilar_registros_imagenes(directorio_dataset: Path | str) -> pd.DataFrame:
    """Recopilar una fila por imagen con la metadata necesaria para el EDA."""
    data_dir = Path(directorio_dataset)
    rows: list[dict[str, object]] = []

    if not data_dir.exists():
        return pd.DataFrame(
            columns=[
                "split",
                "label",
                "path",
                "filename",
                "extension",
                "file_size_bytes",
                "width",
                "height",
                "mode",
            ]
        )

    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue

        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            for ruta_imagen in sorted(label_dir.iterdir()):
                if not ruta_imagen.is_file():
                    continue

                try:
                    with Image.open(ruta_imagen) as image:
                        width, height = image.size
                        mode = image.mode
                except (UnidentifiedImageError, OSError):
                    width, height, mode = None, None, None

                rows.append(
                    {
                        "split": split_dir.name,
                        "label": label_dir.name,
                        "path": str(ruta_imagen),
                        "filename": ruta_imagen.name,
                        "extension": ruta_imagen.suffix.lower(),
                        "file_size_bytes": ruta_imagen.stat().st_size,
                        "width": width,
                        "height": height,
                        "mode": mode,
                    }
                )

    return pd.DataFrame(rows)


def construir_resumen_dataset(directorio_dataset: Path | str) -> dict[str, object]:
    """Construir un resumen del dataset de imágenes para EDA y validación."""
    data_dir = Path(directorio_dataset)
    records = recopilar_registros_imagenes(data_dir)

    counts_by_split_and_class: dict[str, dict[str, int]] = defaultdict(dict)
    splits = sorted({value for value in records["split"].dropna().tolist() if value})
    for split_name in splits:
        split_records = records[records["split"] == split_name]
        labels = sorted({value for value in split_records["label"].dropna().tolist() if value})
        for label_name in labels:
            counts_by_split_and_class[split_name][label_name] = int(
                (split_records["label"] == label_name).sum()
            )

    corrupt_images: list[Path] = []
    for _, row in records.iterrows():
        path = Path(row["path"])
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt_images.append(path)

    file_hashes: dict[str, list[Path]] = defaultdict(list)
    for _, row in records.iterrows():
        path = Path(row["path"])
        try:
            file_hashes[_calcular_hash_archivo(path)].append(path)
        except OSError:
            continue

    duplicate_paths = [
        path
        for paths in file_hashes.values()
        for path in paths[1:]
    ]

    summary = {
        "dataset_dir": str(data_dir),
        "total_images": int(len(records)),
        "counts_by_split_and_class": dict(counts_by_split_and_class),
        "corrupt_images": corrupt_images,
        "duplicate_paths": duplicate_paths,
        "records": records,
    }

    return summary


def guardar_figuras_eda(records: pd.DataFrame, directorio_salida: Path | str) -> dict[str, Path]:
    """Crear las principales figuras EDA para distribución y metadata de imágenes."""
    ruta_salida = Path(directorio_salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Path] = {}

    distribution = records.groupby(["split", "label"]).size().unstack(fill_value=0)
    distribution_path = ruta_salida / "dataset_distribution.png"
    distribution.plot(kind="bar", figsize=(10, 6), title="Distribución del dataset por conjunto y clase")
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=200)
    plt.close()
    figures["distribution"] = distribution_path

    sizes = records["file_size_bytes"].dropna()
    sizes_path = ruta_salida / "file_size_distribution.png"
    plt.figure(figsize=(10, 5))
    plt.hist(sizes, bins=30, color="steelblue", edgecolor="black")
    plt.title("Distribución del tamaño de los archivos de imagen")
    plt.xlabel("Tamaño de archivo (bytes)")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(sizes_path, dpi=200)
    plt.close()
    figures["sizes"] = sizes_path

    valid_dims = records.dropna(subset=["width", "height"]).copy()
    dimensions_path = ruta_salida / "image_dimensions.png"
    plt.figure(figsize=(8, 8))
    plt.scatter(valid_dims["width"], valid_dims["height"], alpha=0.6)
    plt.title("Dimensiones de las imágenes")
    plt.xlabel("Ancho (píxeles)")
    plt.ylabel("Alto (píxeles)")
    plt.tight_layout()
    plt.savefig(dimensions_path, dpi=200)
    plt.close()
    figures["dimensions"] = dimensions_path

    return figures


def graficar_imagenes_muestra(records: pd.DataFrame, directorio_salida: Path | str, max_por_clase: int = 3) -> dict[str, Path]:
    """Guardar imágenes representativas de cada clase para su inspección visual."""
    ruta_salida = Path(directorio_salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    sample_paths: dict[str, Path] = {}
    for label in ["NORMAL", "PNEUMONIA"]:
        label_records = records[records["label"] == label].head(max_por_clase)
        if label_records.empty:
            continue

        fig, axes = plt.subplots(1, min(len(label_records), max_por_clase), figsize=(12, 4))
        fig.suptitle(f"Ejemplos: {label}")

        if len(label_records) == 1:
            axes = [axes]

        for axis, (_, row) in zip(axes, label_records.iterrows()):
            ruta_imagen = Path(row["path"])
            try:
                with Image.open(ruta_imagen) as image:
                    axis.imshow(image)
            except Exception:
                axis.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
                axis.set_title("Imagen ilegible")
            axis.axis("off")

        fig.tight_layout()
        sample_path = ruta_salida / f"sample_{label.lower()}.png"
        fig.savefig(sample_path, dpi=200)
        plt.close(fig)
        sample_paths[label] = sample_path

    return sample_paths


def validar_estructura_dataset(directorio_dataset: Path | str) -> dict[str, object]:
    """Validar que existan las carpetas train/val/test esperadas y las etiquetas de clase."""
    data_dir = Path(directorio_dataset)
    expected = {
        "train": ["NORMAL", "PNEUMONIA"],
        "val": ["NORMAL", "PNEUMONIA"],
        "test": ["NORMAL", "PNEUMONIA"],
    }

    validation = {}
    for split_name, labels in expected.items():
        split_dir = data_dir / split_name
        validation[split_name] = {
            "exists": split_dir.exists(),
            "labels": sorted([path.name for path in split_dir.iterdir() if path.is_dir()])
            if split_dir.exists()
            else [],
            "missing_labels": [label for label in labels if not (split_dir / label).exists()],
        }

    return validation


def ejecutar_eda(directorio_dataset: Path | str = DIRECTORIO_DATOS) -> dict[str, object]:
    """Ejecutar el EDA a nivel de dataset y devolver el resumen más las figuras generadas."""
    data_dir = Path(directorio_dataset)
    validation = validar_estructura_dataset(data_dir)
    records = recopilar_registros_imagenes(data_dir)
    summary = construir_resumen_dataset(data_dir)
    figures_dir = RAIZ_PROYECTO / "reports" / "figures"

    figures = guardar_figuras_eda(records, figures_dir)
    sample_figures = graficar_imagenes_muestra(records, figures_dir)

    return {
        "validation": validation,
        "summary": summary,
        "figures": {**figures, **sample_figures},
    }


if __name__ == "__main__":
    summary = ejecutar_eda(DIRECTORIO_DATOS)
    print(summary["summary"]["total_images"])
    print(summary["summary"]["counts_by_split_and_class"])
