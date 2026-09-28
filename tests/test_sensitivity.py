"""Pruebas de la etapa de sensibilidad de hiperparámetros previa a la selección de arquitecturas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.models.architectures import NOMBRES_MODELOS
from src.training.sensitivity import (
    CRITERIO_METRICAS,
    DROPOUT_BASE,
    EPOCAS_BASE,
    LEARNING_RATE_BASE,
    RUTA_RESULTADOS,
    SPLITS_PERMITIDOS,
    cargar_mejor_configuracion_por_arquitectura,
    cargar_resultados_sensibilidad,
    construir_configuraciones,
    crear_pipelines_sensibilidad,
    normalizar_config,
    registrar_resultados_existentes,
    seleccionar_mejor_por_arquitectura,
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Artefacto oficial de la etapa de sensibilidad."""
    return cargar_resultados_sensibilidad()


def _resultado(model_name: str, config_id: str, config: dict[str, Any], metricas: dict[str, float]) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "config_id": config_id,
        "grupo": config_id,
        "config": config,
        "es_referencia": config_id.startswith("ref"),
        "validation": metricas,
    }


def test_configuraciones_cubren_tres_grupos_y_solo_cambian_un_factor() -> None:
    configuraciones = construir_configuraciones()
    assert len(configuraciones) == 7
    assert {config["grupo"] for config in configuraciones} == {"referencia", "learning_rate", "dropout", "epochs"}

    base = {"learning_rate": LEARNING_RATE_BASE, "dropout": DROPOUT_BASE, "epochs": EPOCAS_BASE}
    assert (LEARNING_RATE_BASE, DROPOUT_BASE, EPOCAS_BASE) == (1e-4, 0.3, 3)

    for config in configuraciones:
        distintos = [clave for clave in base if config[clave] != base[clave]]
        if config["grupo"] == "referencia":
            assert distintos == []
        else:
            assert len(distintos) == 1, f"{config['id']} cambia más de un hiperparámetro"
            assert distintos == config["variante_de"]


def test_la_etapa_de_sensibilidad_no_admite_el_split_test() -> None:
    assert SPLITS_PERMITIDOS == ("train", "val")
    with pytest.raises(ValueError):
        crear_pipelines_sensibilidad(("test",))
    with pytest.raises(ValueError):
        crear_pipelines_sensibilidad(("train", "val", "test"))


def test_artefacto_de_sensibilidad_no_contiene_test(payload: dict[str, Any]) -> None:
    assert payload["test_utilizado"] is False
    assert payload["splits_utilizados"] == ["train", "val"]
    assert "test" not in payload["distribucion"]
    for item in payload["resultados"]:
        assert "test" not in item
        assert "test" not in item["validation"]
        assert set(item["validation"]) >= set(CRITERIO_METRICAS)


def test_artefacto_cubre_tres_arquitecturas_y_veintiuna_configuraciones(payload: dict[str, Any]) -> None:
    assert len(payload["resultados"]) == 21
    assert set(payload["mejor_por_arquitectura"]) == set(NOMBRES_MODELOS)
    for model_name in NOMBRES_MODELOS:
        configuraciones = [item for item in payload["resultados"] if item["model_name"] == model_name]
        assert len(configuraciones) == 7


def test_mejor_configuracion_por_arquitectura_es_la_de_mayor_balanced_accuracy(payload: dict[str, Any]) -> None:
    for model_name, info in payload["mejor_por_arquitectura"].items():
        candidatos = [item for item in payload["resultados"] if item["model_name"] == model_name]
        mejor = max(candidatos, key=lambda item: item["validation"]["balanced_accuracy"])
        assert info["config_id"] == mejor["config_id"] == "lr_1e-3"
        assert info["config"]["learning_rate"] == 1e-3
        assert info["validation"]["balanced_accuracy"] == pytest.approx(mejor["validation"]["balanced_accuracy"])
        assert info["delta_balanced_accuracy_vs_referencia"] > 0


def test_ganador_global_es_mobilenetv2_con_la_sensibilidad(payload: dict[str, Any]) -> None:
    ganador = payload["ganador_global"]
    assert ganador["model_name"] == "MobileNetV2"
    assert ganador["config_id"] == "lr_1e-3"
    assert ganador["config"] == {"learning_rate": 1e-3, "dropout": 0.3, "epochs": 3}
    assert ganador["validation"]["balanced_accuracy"] == pytest.approx(0.9590, abs=5e-4)
    assert payload["comparacion_modelos"]["winner"]["model_name"] == "MobileNetV2"
    assert [item["model_name"] for item in payload["comparacion_modelos"]["results"]] == [
        "MobileNetV2",
        "VGG16",
        "ResNet50",
    ]


def test_la_seleccion_ignora_el_test_evenente_que_exista() -> None:
    base_config = {"learning_rate": 1e-4, "dropout": 0.3, "epochs": 3}
    buen_val = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "specificity": 0.9,
                "balanced_accuracy": 0.90, "f1": 0.9, "roc_auc": 0.95}
    peor_val = {"accuracy": 0.7, "precision": 0.7, "recall": 0.7, "specificity": 0.7,
                "balanced_accuracy": 0.70, "f1": 0.7, "roc_auc": 0.75}

    resultados = [
        _resultado("VGG16", "ref_lr1e-4_do0.3_ep3", base_config, buen_val),
        _resultado("VGG16", "lr_1e-3", {"learning_rate": 1e-3, "dropout": 0.3, "epochs": 3}, peor_val),
    ]
    # Un resultado con métricas de test no puede entrar en la selección de la etapa.
    contaminado = dict(resultados[0])
    contaminado["test"] = {"balanced_accuracy": 0.99}
    with pytest.raises(ValueError):
        seleccionar_mejor_por_arquitectura([contaminado])

    seleccion = seleccionar_mejor_por_arquitectura(resultados)
    assert seleccion["VGG16"]["config_id"] == "ref_lr1e-4_do0.3_ep3"


def test_el_flujo_final_parte_de_las_tres_configuraciones(payload: dict[str, Any]) -> None:
    """Cada arquitectura conserva su mejor configuración; ninguna se privilegia por ser la global."""
    configuraciones = cargar_mejor_configuracion_por_arquitectura()

    assert set(configuraciones) == set(NOMBRES_MODELOS)
    for model_name, info in configuraciones.items():
        assert info["config_id"] == payload["mejor_por_arquitectura"][model_name]["config_id"]
        assert info["config"] == normalizar_config(payload["mejor_por_arquitectura"][model_name]["config"])
        assert info["validation"]["balanced_accuracy"] == pytest.approx(
            payload["mejor_por_arquitectura"][model_name]["validation"]["balanced_accuracy"]
        )
    assert configuraciones["VGG16"]["validation"]["balanced_accuracy"] < (
        payload["ganador_global"]["validation"]["balanced_accuracy"]
    )


def test_normalizar_config_acepta_la_clave_epochs_max() -> None:
    assert normalizar_config({"learning_rate": 3e-4, "dropout": 0.3, "epochs_max": 5}) == {
        "learning_rate": 3e-4,
        "dropout": 0.3,
        "epochs": 5,
    }
    with pytest.raises(ValueError):
        normalizar_config({"learning_rate": 3e-4, "dropout": 0.3})


def test_registrar_resultados_existentes_exige_el_barrido_completo(tmp_path: Path) -> None:
    origen = tmp_path / "resultados.json"
    origen.write_text(json.dumps({"resultados": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        registrar_resultados_existentes(origen)

    origen.write_text(
        json.dumps(
            {
                "configuraciones": construir_configuraciones(),
                "resultados": {
                    "VGG16/lr_1e-3": {
                        "modelo": "VGG16",
                        "config_id": "lr_1e-3",
                        "grupo": "learning_rate",
                        "config": {"learning_rate": 1e-3, "dropout": 0.3, "epochs_max": 3},
                        "estado": "completado",
                        "validation": {
                            "accuracy": 0.9,
                            "precision": 0.9,
                            "recall": 0.9,
                            "specificity": 0.9,
                            "balanced_accuracy": 0.9,
                            "f1": 0.9,
                            "roc_auc": 0.95,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        registrar_resultados_existentes(origen)


def test_la_sensibilidad_publica_un_unico_artefacto() -> None:
    """La etapa de sensibilidad escribe un único artefacto, con métricas de validación."""
    import json

    assert RUTA_RESULTADOS.name == "sensitivity_results.json"
    assert RUTA_RESULTADOS.parent.name == "models"
    assert [ruta.name for ruta in RUTA_RESULTADOS.parent.glob("*sensitivity*")] == [
        "sensitivity_results.json"
    ]
    datos = json.loads(RUTA_RESULTADOS.read_text(encoding="utf-8"))
    assert set(datos) >= {"configuracion_base", "configuraciones", "resultados", "mejor_por_arquitectura"}
    assert datos["splits_utilizados"] == ["train", "val"]
    assert datos["test_utilizado"] is False


def test_la_cli_expone_el_comando_de_sensibilidad() -> None:
    from click.testing import CliRunner

    from src.cli import cli

    resultado = CliRunner().invoke(cli, ["--help"])
    assert resultado.exit_code == 0
    assert "sensibilidad" in resultado.output

    ayuda = CliRunner().invoke(cli, ["sensibilidad", "--help"])
    assert ayuda.exit_code == 0
    assert "--recalcular" in ayuda.output
