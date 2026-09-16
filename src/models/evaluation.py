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


def evaluar_baseline_mayoritaria(y_true: np.ndarray) -> dict[str, float | str]:
    """Evaluar un baseline sencillo que siempre predice la clase mayoritaria.

    Devolver sus métricas para comparar contra el modelo seleccionado.
    """
    true_labels = np.asarray(y_true).ravel().astype(int)
    if true_labels.size == 0:
        raise ValueError("y_true no debe estar vacío.")

    majority_class = int(np.bincount(true_labels).argmax())
    baseline_pred = np.full(true_labels.shape, majority_class)
    metrics = calcular_reporte_metricas(true_labels, baseline_pred)
    return {
        "baseline": "clase mayoritaria",
        "majority_class": "PNEUMONIA" if majority_class == 1 else "NORMAL",
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
    }


def evaluar_criterio_exito(metrics_test: dict[str, Any], metrics_baseline_test: dict[str, Any]) -> dict[str, Any]:
    """Comprobar el criterio de éxito sobre el test original independiente.

    El modelo seleccionado debe:
    1. superar al baseline de clase mayoritaria en balanced_accuracy;
    2. mostrar sensibilidad y especificidad por encima del nivel de azar (0.5),
       es decir, un equilibrio adecuado entre ambas clases.
    """
    balanced_accuracy = float(metrics_test.get("balanced_accuracy", 0.0))
    recall = float(metrics_test.get("recall", 0.0))
    specificity = float(metrics_test.get("specificity", 0.0))
    baseline_balanced_accuracy = float(metrics_baseline_test.get("balanced_accuracy", 0.0))

    supera_baseline = balanced_accuracy > baseline_balanced_accuracy
    sensibilidad_sobre_azar = recall > 0.5
    especificidad_sobre_azar = specificity > 0.5
    equilibrio_adecuado = sensibilidad_sobre_azar and especificidad_sobre_azar

    return {
        "cumple": bool(supera_baseline and equilibrio_adecuado),
        "supera_baseline": bool(supera_baseline),
        "sensibilidad_sobre_azar": bool(sensibilidad_sobre_azar),
        "especificidad_sobre_azar": bool(especificidad_sobre_azar),
        "equilibrio_adecuado": bool(equilibrio_adecuado),
        "balanced_accuracy_modelo": balanced_accuracy,
        "balanced_accuracy_baseline": baseline_balanced_accuracy,
    }


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
