"""Utilidades de comparación para la selección de modelos y los reportes."""

from __future__ import annotations

from typing import Any

from src.models.evaluation import seleccionar_mejor_modelo


def construir_tabla_resultados(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Devolver los resultados ordenados por exactitud balanceada y desempates estándar."""
    return sorted(
        results,
        key=lambda item: (
            -float(item.get("balanced_accuracy", 0.0)),
            -float(item.get("roc_auc", 0.0)),
            -float(item.get("f1", 0.0)),
            -float(item.get("accuracy", 0.0)),
        ),
    )


def resumir_comparacion_modelos(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Crear un resumen con el modelo ganador y la tabla ordenada."""
    if not results:
        raise ValueError("results no debe estar vacío.")
    ordered = construir_tabla_resultados(results)
    winner = seleccionar_mejor_modelo(ordered)
    return {"winner": winner, "results": ordered}
