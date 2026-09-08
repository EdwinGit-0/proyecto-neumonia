"""Comparison utilities for model selection and reporting."""

from __future__ import annotations

from typing import Any

from src.models.evaluation import select_best_model


def build_results_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return results ordered by balanced accuracy and standard tie-breakers."""
    return sorted(
        results,
        key=lambda item: (
            -float(item.get("balanced_accuracy", 0.0)),
            -float(item.get("roc_auc", 0.0)),
            -float(item.get("f1", 0.0)),
            -float(item.get("accuracy", 0.0)),
        ),
    )


def summarize_model_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a summary with the winning model and the ordered table."""
    if not results:
        raise ValueError("results must not be empty.")
    ordered = build_results_table(results)
    winner = select_best_model(ordered)
    return {"winner": winner, "results": ordered}
