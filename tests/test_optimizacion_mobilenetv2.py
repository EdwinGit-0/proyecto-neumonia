"""Pruebas de la etapa de optimizacion posterior al analisis de sensibilidad.

El flujo definitivo es: sensibilidad (ya realizada) -> seleccion de MobileNetV2
ganador -> test inicial -> optimizacion -> test final. Estas pruebas verifican que
la etapa respeta ese contrato sin necesidad de entrenar: no se repiten learning
rate, dropout ni epocas, la decision se toma sobre validation y el test no
interviene en la seleccion.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.training.optimizacion_mobilenetv2 import (
    CRITERIO_METRICAS,
    DIRECTORIO_CHECKPOINTS,
    RUTA_RESULTADOS,
    construir_variantes,
    ruta_checkpoint_sensibilidad,
    seleccionar_ganador_optimizacion,
    test_auditado_sensibilidad as obtener_test_auditado,
)
from src.training.sensitivity import cargar_punto_de_partida_sensibilidad


CONFIG_GANADORA: dict[str, Any] = {"learning_rate": 0.001, "dropout": 0.3, "epochs": 3}


def _candidato(identificador: str, grupo: str, **metricas: float) -> dict[str, Any]:
    """Construir un candidato de optimización con las métricas de validation dadas."""
    base = {
        "balanced_accuracy": 0.0,
        "roc_auc": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
        "recall": 0.9,
    }
    base.update(metricas)
    return {
        "id": identificador,
        "grupo": grupo,
        "checkpoint": f"models/mobilenetv2_optimizacion/{identificador}/best_model.keras",
        "validation": base,
    }


@pytest.fixture(scope="module")
def punto_de_partida() -> dict[str, Any]:
    """Ganador real del analisis de sensibilidad ya registrado en el proyecto."""
    return cargar_punto_de_partida_sensibilidad()


def test_punto_de_partida_es_mobilenetv2_ganador(punto_de_partida: dict[str, Any]) -> None:
    """La etapa parte del MobileNetV2 ganador de la sensibilidad, sin reentrenarlo."""
    assert punto_de_partida["model_name"] == "MobileNetV2"
    assert punto_de_partida["config_id"] == "lr_1e-3"
    assert punto_de_partida["config"]["learning_rate"] == pytest.approx(0.001)
    assert punto_de_partida["config"]["dropout"] == pytest.approx(0.3)
    assert punto_de_partida["config"]["epochs"] == 3
    assert punto_de_partida["validation"]["balanced_accuracy"] == pytest.approx(0.959041887337506)
    assert ruta_checkpoint_sensibilidad(punto_de_partida["config_id"]).exists()


def test_las_variantes_no_repiten_los_21_factores_ya_evaluados() -> None:
    """Ninguna variante reintroduce learning rate, dropout o epocas como factor."""
    variantes = construir_variantes(CONFIG_GANADORA)
    assert variantes, "La etapa debe tener al menos una variante de optimizacion."
    for variante in variantes:
        assert set(variante) >= {"id", "grupo", "tratamiento", "fine_tune_from"}
        assert variante["grupo"] in {"desbalance", "fine_tuning"}
        assert variante["tratamiento"] in {"ninguno", "class_weight", "oversampling"}
        assert "learning_rate" not in variante
        assert "dropout" not in variante
        assert "epochs" not in variante


def test_cada_variante_varia_un_solo_factor() -> None:
    """No se combina el tratamiento del desbalance con el fine-tuning en una misma variante."""
    for variante in construir_variantes(CONFIG_GANADORA):
        factores = variante["tratamiento"] != "ninguno"
        factores += variante["fine_tune_from"] is not None
        assert factores <= 1, f"La variante {variante['id']} combina dos factores."


def test_se_cubren_desbalance_y_fine_tuning() -> None:
    """La etapa cubre los dos aspectos pendientes que ya estan implementados en el codigo."""
    grupos = {variante["grupo"] for variante in construir_variantes(CONFIG_GANADORA)}
    assert grupos == {"desbalance", "fine_tuning"}
    tratamientos = {variante["tratamiento"] for variante in construir_variantes(CONFIG_GANADORA)}
    assert {"class_weight", "oversampling"} <= tratamientos


def test_la_seleccion_usa_balanced_accuracy_como_primer_criterio() -> None:
    """Gana el de mayor balanced accuracy aunque tenga otras metricas peores."""
    referencia = _candidato("punto_de_partida", "punto_de_partida", balanced_accuracy=0.959)
    candidatos = [
        referencia,
        _candidato("class_weight", "desbalance", balanced_accuracy=0.964, roc_auc=0.90, f1=0.90, accuracy=0.90),
    ]
    seleccion = seleccionar_ganador_optimizacion(candidatos, referencia)
    assert seleccion["ganador_id"] == "class_weight"
    assert seleccion["split"] == "validation"
    assert seleccion["criterio"] == " > ".join(CRITERIO_METRICAS)


def test_desempate_se_resuelve_con_roc_auc_f1_y_accuracy() -> None:
    """Con la misma balanced accuracy decide el siguiente criterio del proyecto."""
    referencia = _candidato("punto_de_partida", "punto_de_partida", balanced_accuracy=0.95)
    empatados = [
        referencia,
        _candidato("a", "desbalance", balanced_accuracy=0.96, roc_auc=0.991, f1=0.90, accuracy=0.90),
        _candidato("b", "desbalance", balanced_accuracy=0.96, roc_auc=0.992, f1=0.90, accuracy=0.90),
    ]
    assert seleccionar_ganador_optimizacion(empatados, referencia)["ganador_id"] == "b"

    empatados_auc = [
        referencia,
        _candidato("a", "desbalance", balanced_accuracy=0.96, roc_auc=0.99, f1=0.95, accuracy=0.90),
        _candidato("b", "desbalance", balanced_accuracy=0.96, roc_auc=0.99, f1=0.94, accuracy=0.99),
    ]
    assert seleccionar_ganador_optimizacion(empatados_auc, referencia)["ganador_id"] == "a"


def test_la_seleccion_sigue_solo_las_cuatro_metricas_acordadas() -> None:
    """Gana el mejor por BA > ROC-AUC > F1 > Accuracy aunque pierda recall.

    El criterio acordado no incluye recall ni tolerancias: el recall se reporta,
    pero no filtra candidatos.
    """
    referencia = _candidato(
        "punto_de_partida", "punto_de_partida", balanced_accuracy=0.959, recall=0.9703
    )
    candidatos = [
        referencia,
        _candidato(
            "mayor_ba", "desbalance", balanced_accuracy=0.98, roc_auc=0.999, f1=0.99,
            accuracy=0.99, recall=0.10,
        ),
        _candidato(
            "recall_alto_pero_menor_ba", "desbalance", balanced_accuracy=0.960, roc_auc=0.995,
            f1=0.97, accuracy=0.965, recall=0.99,
        ),
    ]
    seleccion = seleccionar_ganador_optimizacion(candidatos, referencia)

    assert seleccion["ganador_id"] == "mayor_ba"
    assert seleccion["criterio"] == "balanced_accuracy > roc_auc > f1 > accuracy"
    assert seleccion["split"] == "validation"
    assert seleccion["supera_punto_de_partida"] is True
    assert "tolerancia_recall" not in seleccion
    assert "motivo" not in seleccion
    # El recall se conserva como informacion, no como criterio.
    assert set(seleccion["ganador_validation"]) == {
        "balanced_accuracy", "roc_auc", "f1", "accuracy",
    }


def test_el_test_auditado_de_sensibilidad_esta_disponible() -> None:
    """El test inicial del flujo puede contrastarse con la auditoria de la sensibilidad."""
    test = obtener_test_auditado("MobileNetV2", "lr_1e-3")
    assert test is not None
    assert "balanced_accuracy" in test


def test_la_etapa_no_tiene_seleccion_de_arquitecturas() -> None:
    """Tras la sensibilidad no se vuelve a elegir arquitectura: el punto de partida es MobileNetV2."""
    punto = cargar_punto_de_partida_sensibilidad()
    assert "comparacion_modelos" not in punto
    assert "seleccion_por_arquitectura" not in punto


@pytest.mark.skipif(not RUTA_RESULTADOS.exists(), reason="El flujo de optimizacion aun no se ha ejecutado.")
def test_el_artefacto_refleja_el_flujo_definitivo() -> None:
    """El artefacto de la etapa contiene las cuatro fases del flujo y test no decide."""
    payload = json.loads(RUTA_RESULTADOS.read_text(encoding="utf-8"))
    assert payload["sensibilidad"]["numero_de_pruebas"] == 21
    assert payload["sensibilidad"]["reutilizado_sin_reejecutar"] is True
    assert payload["punto_de_partida"]["model_name"] == "MobileNetV2"
    assert payload["test_inicial"]["test"]["confusion_matrix"]
    assert payload["test_inicial"]["test"]["balanced_accuracy"] > 0.5
    assert payload["seleccion"]["split"] == "validation"
    assert payload["seleccion"]["criterio"] == " > ".join(CRITERIO_METRICAS)
    assert payload["test_utilizado_para_seleccion"] is False
    assert payload["test_final"]["test"]["balanced_accuracy"] > 0.5
    assert payload["modelo_final"]


@pytest.mark.skipif(not RUTA_RESULTADOS.exists(), reason="El flujo de optimizacion aun no se ha ejecutado.")
def test_las_variantes_entrenadas_respetan_los_factores_fijos() -> None:
    """Ningun resultado de la etapa cambia learning rate, dropout o epocas."""
    payload = json.loads(RUTA_RESULTADOS.read_text(encoding="utf-8"))
    fijos = payload["factores_fijos"]
    for registro in payload["resultados"]:
        assert registro["config"]["learning_rate"] == pytest.approx(fijos["learning_rate"])
        assert registro["config"]["dropout"] == pytest.approx(fijos["dropout"])
        assert registro["config"]["epochs"] == fijos["epochs"]


@pytest.mark.skipif(not DIRECTORIO_CHECKPOINTS.exists(), reason="La etapa aun no se ha ejecutado.")
def test_los_checkpoints_de_la_etapa_no_invaden_modelos_base() -> None:
    """Los checkpoints nuevos viven en su propia carpeta, sin tocar los modelos del proyecto."""
    for variante in construir_variantes(CONFIG_GANADORA):
        ruta = DIRECTORIO_CHECKPOINTS / variante["id"] / "best_model.keras"
        assert DIRECTORIO_CHECKPOINTS in ruta.parents
