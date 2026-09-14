"""Configuración de rutas del proyecto."""

from __future__ import annotations

from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
DIRECTORIO_DATOS = RAIZ_PROYECTO / "data" / "raw" / "chest_xray"
DIRECTORIO_REPORTES = RAIZ_PROYECTO / "reports"
DIRECTORIO_FIGURAS = DIRECTORIO_REPORTES / "figures"
DIRECTORIO_MODELOS = RAIZ_PROYECTO / "models"
DIRECTORIO_NOTEBOOKS = RAIZ_PROYECTO / "notebooks"


def asegurar_directorio(path: Path | str) -> Path:
    """Crear un directorio si no existe."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
