"""Utilidades de evaluación y lógica de comparación para el desempeño de los modelos."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def _a_predicciones_binarias(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Convertir probabilidades o logits a etiquetas binarias para la evaluación."""
    true_values = np.asarray(y_true).ravel().astype(int)
    pred_values = np.asarray(y_pred).ravel()

    if pred_values.size == 0:
        raise ValueError("y_pred no debe estar vacío.")

    if pred_values.dtype.kind in {"f", "i", "u"} and pred_values.size and pred_values.max() <= 1.0 and pred_values.min() >= 0.0 and not np.all(np.isin(pred_values, [0, 1])):
        prob_values = pred_values.astype(float)
        binary_pred = (prob_values >= threshold).astype(int)
        return true_values, binary_pred, prob_values

    binary_pred = pred_values.astype(int).ravel()
    return true_values, binary_pred, binary_pred.astype(float)


def calcular_matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Calcular la matriz de confusión para la clasificación binaria."""
    true_labels, binary_pred, _ = _a_predicciones_binarias(y_true, y_pred, threshold=threshold)
    if true_labels.size == 0 or binary_pred.size == 0:
        raise ValueError("y_true y y_pred no deben estar vacíos.")
    return confusion_matrix(true_labels, binary_pred, labels=[0, 1])


def calcular_reporte_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Devolver las métricas de evaluación solicitadas para la tarea de clasificación."""
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true y y_pred no deben estar vacíos.")

    true_labels, binary_pred, probability_values = _a_predicciones_binarias(y_true, y_pred, threshold=threshold)
    cm = confusion_matrix(true_labels, binary_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = recall_score(true_labels, binary_pred, zero_division=0)

    metrics = {
        "accuracy": float(accuracy_score(true_labels, binary_pred)),
        "precision": float(precision_score(true_labels, binary_pred, zero_division=0)),
        "recall": float(recall),
        "f1": float(f1_score(true_labels, binary_pred, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
        "roc_auc": float(roc_auc_score(true_labels, probability_values)),
    }
    return metrics


def seleccionar_mejor_modelo(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Seleccionar el modelo con mejor exactitud balanceada y desempates estándar."""
    if not results:
        raise ValueError("results no debe estar vacío.")

    def sorting_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(item.get("balanced_accuracy", 0.0)),
            float(item.get("roc_auc", 0.0)),
            float(item.get("f1", 0.0)),
            float(item.get("accuracy", 0.0)),
        )

    return max(results, key=sorting_key)


def comparar_resultados_modelos(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordenar la tabla de resultados para un reporte de comparación de modelos."""
    return sorted(
        results,
        key=lambda item: (
            -float(item.get("balanced_accuracy", 0.0)),
            -float(item.get("roc_auc", 0.0)),
            -float(item.get("f1", 0.0)),
            -float(item.get("accuracy", 0.0)),
        ),
    )
