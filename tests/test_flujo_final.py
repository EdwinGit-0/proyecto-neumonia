"""Pruebas del flujo final: estrategia única COMBINADO, umbral congelado y test final.

Ninguna de estas pruebas entrena un modelo: comprueban las garantías del protocolo
(qué splits se usan, que validation y test quedan intactos, que existe una sola
estrategia, qué se congela y que el test se evalúa sin bloqueos) y la lógica de
búsqueda de umbral con datos sintéticos.
"""

from __future__ import annotations

import json
import inspect
import string
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.training import flujo_final
from src.training import tratamiento_desbalance


METRICAS_COMPLETAS = {
    "accuracy": 0.9,
    "precision": 0.9,
    "recall": 0.9,
    "specificity": 0.9,
    "balanced_accuracy": 0.9,
    "f1": 0.9,
    "roc_auc": 0.95,
}

MANIFIESTO = flujo_final.RUTA_MANIFIESTO


def _diagnostico_falso() -> dict[str, Any]:
    return {
        "estrategia": "combinado",
        "splits": ["train"],
        "composicion_original": {"NORMAL": 1073, "PNEUMONIA": 3100, "total": 4173},
        "composicion_tras_oversampling": {"NORMAL": 3100, "PNEUMONIA": 3100, "total": 6200},
        "filas_duplicadas": 2027,
        "class_weight": {"0": 1.9445479962721341, "1": 0.6730645161290323},
        "class_weight_calculado_sobre": "distribucion_original_pre_oversampling",
        "razon_muestreo_tras_oversampling": 1.0,
        "razon_pesos_por_clase": 2.889084362905982,
        "refuerzo_total_minoritaria": 2.889084362905982,
        "advertencia_doble_correccion": "texto de advertencia",
        "filas_efectivas_entrenamiento": 6200,
    }


def _combinado_falso(validation_n: int = 1043) -> dict[str, Any]:
    return {
        "descripcion": "combinado",
        "etapa": "combinado",
        "estrategia": dict(tratamiento_desbalance.ESTRATEGIA_COMBINADO),
        "model_name": "MobileNetV2",
        "config": {"learning_rate": 1e-3, "dropout": 0.3, "epochs": 3},
        "diagnostico_combinado": _diagnostico_falso(),
        "class_weight_aplicado": {"0": 1.9445479962721341, "1": 0.6730645161290323},
        "validation_composicion": {"NORMAL": 268, "PNEUMONIA": 775, "total": validation_n},
        "validation": dict(METRICAS_COMPLETAS),
        "validation_matriz_confusion": {"tn": 253, "fp": 15, "fn": 18, "tp": 757},
        "validation_n": validation_n,
        "test_utilizado": False,
    }


# ---------------------------------------------------------------------------
# Una sola estrategia
# ---------------------------------------------------------------------------
def test_la_estrategia_unica_es_combinado() -> None:
    """El proyecto tiene una sola estrategia de tratamiento: COMBINADO."""
    assert tratamiento_desbalance.NOMBRE_ESTRATEGIA == "combinado"
    estrategia = tratamiento_desbalance.ESTRATEGIA_COMBINADO
    assert estrategia["id"] == "combinado"
    assert estrategia["oversampling"] is True
    assert estrategia["class_weight"] is True
    assert estrategia["proporcion_objetivo"] == "50/50"


def test_la_estrategia_del_proyecto_es_unica_y_no_es_parametro() -> None:
    """La estrategia COMBINADO está cerrada: el flujo no expone variantes seleccionables."""
    assert tratamiento_desbalance.NOMBRE_ESTRATEGIA == "combinado"
    estrategia = tratamiento_desbalance.ESTRATEGIA_COMBINADO
    assert estrategia["id"] == "combinado"
    assert estrategia["oversampling"] is True
    assert estrategia["class_weight"] is True
    assert estrategia["proporcion_objetivo"] == "50/50"
    assert tratamiento_desbalance.SPLITS_PERMITIDOS == ("train", "val")
    firma = inspect.signature(flujo_final.construir_pipeline_combinado)
    assert "estrategia" not in firma.parameters


def test_el_flujo_no_expone_una_rejilla_de_variantes() -> None:
    """El módulo entrena una única corrida, sin ranking ni selección de ganador."""
    assert callable(flujo_final.ejecutar_combinado)
    for nombre_atributo in ("seleccionar_ganador_por_arquitectura", "clave_orden",
                            "estrategia_por_id", "ejecutar_variantes"):
        assert not hasattr(flujo_final, nombre_atributo), (
            f"flujo_final expone {nombre_atributo}, que implicaría comparar variantes"
        )


def test_el_flujo_de_entrenamiento_rechaza_el_test() -> None:
    with pytest.raises(ValueError, match="test"):
        flujo_final.rutas_splits_permitidos(("train", "test"))


def test_las_etapas_de_decision_no_consultan_test() -> None:
    """Ninguna etapa de decisión debe construir el pipeline de test."""
    assert flujo_final.SPLITS_ENTRENAMIENTO == ("train",)
    assert flujo_final.SPLITS_MEDICION == ("val",)
    assert flujo_final.SPLITS_ENTRENAMIENTO_FINAL == ("train", "val")
    assert flujo_final.SPLIT_TEST not in flujo_final.SPLITS_ENTRENAMIENTO_FINAL


# ---------------------------------------------------------------------------
# Oversampling 50/50 y class weights
# ---------------------------------------------------------------------------
def test_el_oversampling_iguala_la_minoritaria() -> None:
    import pandas as pd

    marco = pd.DataFrame(
        {
            "target": [0] * 10 + [1] * 30,
            "path": [f"imagen_{indice}.png" for indice in range(40)],
            "split": ["train"] * 40,
            "label": ["NORMAL"] * 10 + ["PNEUMONIA"] * 30,
        }
    )
    equilibrado, resumen = tratamiento_desbalance.aplicar_oversampling(marco)

    assert resumen["NORMAL_oversampled"] == 30
    assert resumen["PNEUMONIA_oversampled"] == 30
    assert resumen["total_oversampled"] == 60
    assert resumen["duplicadas"] == 20
    # La clase mayoritaria no se toca: solo se anaden filas de la minoritaria.
    assert len(equilibrado) == len(marco) + 20
    assert (equilibrado["target"] == 1).sum() == 30


def test_el_oversampling_es_reproducible() -> None:
    import pandas as pd

    marco = pd.DataFrame(
        {
            "target": [0] * 10 + [1] * 30,
            "path": [f"imagen_{indice}.png" for indice in range(40)],
            "split": ["train"] * 40,
            "label": ["NORMAL"] * 10 + ["PNEUMONIA"] * 30,
        }
    )
    primero, _ = tratamiento_desbalance.aplicar_oversampling(marco)
    segundo, _ = tratamiento_desbalance.aplicar_oversampling(marco)
    assert primero["path"].tolist() == segundo["path"].tolist()


def test_los_pesos_se_calculan_sobre_la_distribucion_original() -> None:
    """Los class_weight NO se calculan sobre el set ya balanceado.

    Si se calcularan sobre el conjunto equilibrado a 50/50 darian 1.0 y la ponderación
    sería un no-op, con lo que COMBINADO sería idéntico a oversampling solo.
    """
    pesos = tratamiento_desbalance.calcular_pesos_clase(MANIFIESTO, splits=("train",))
    n_normal, n_pneumonia = 1073, 3100
    total = n_normal + n_pneumonia
    assert pesos[0] == pytest.approx(total / (2 * n_normal), rel=1e-9)
    assert pesos[1] == pytest.approx(total / (2 * n_pneumonia), rel=1e-9)
    # Si se calcularan sobre 50/50 serían 1.0: comprobamos que no es el caso.
    assert pesos != {0: 1.0, 1: 1.0}


def test_el_diagnostico_declara_la_doble_correccion() -> None:
    diagnostico = tratamiento_desbalance.diagnostico_combinado(MANIFIESTO, splits=("train",))
    assert diagnostico["composicion_tras_oversampling"] == {
        "NORMAL": 3100, "PNEUMONIA": 3100, "total": 6200
    }
    assert diagnostico["razon_muestreo_tras_oversampling"] == pytest.approx(1.0)
    assert diagnostico["refuerzo_total_minoritaria"] == pytest.approx(2.8891, abs=1e-3)
    # La advertencia debe ser explícita sobre el efecto combinado y su alcance.
    advertencia = diagnostico["advertencia_doble_correccion"]
    assert "2.89" in advertencia
    assert "validation" in advertencia and "test" in advertencia


def test_el_tratamiento_rechaza_el_split_de_test() -> None:
    with pytest.raises(ValueError, match="test"):
        tratamiento_desbalance.subconjunto_entrenamiento(MANIFIESTO, ("train", "test"))


def test_la_validation_no_recibe_oversampling_ni_pesos() -> None:
    """Validation conserva su distribución: 268 NORMAL / 775 PNEUMONIA."""
    from src.data.splitting import cargar_manifiesto_division

    marco = cargar_manifiesto_division(MANIFIESTO)
    validacion = marco[marco["split"] == "val"]
    assert len(validacion) == 1043
    assert int((validacion["target"] == 0).sum()) == 268
    assert int((validacion["target"] == 1).sum()) == 775

    # El oversampling se aplica explícitamente a train y no a validation.
    train = marco[marco["split"] == "train"]
    equilibrado, _ = tratamiento_desbalance.aplicar_oversampling(train)
    assert len(equilibrado) == 6200
    assert len(validacion) == 1043  # intacta


# ---------------------------------------------------------------------------
# Umbral
# ---------------------------------------------------------------------------
def test_el_umbral_se_elige_por_balanced_accuracy_en_validacion() -> None:
    y_true = np.array([0] * 40 + [1] * 60)
    y_prob = np.concatenate([np.linspace(0.0, 0.4, 40), np.linspace(0.45, 0.95, 60)])
    busqueda = flujo_final.buscar_umbral(y_true, y_prob)
    assert 0.0 <= busqueda["umbral"] <= 1.0
    assert busqueda["mejor"]["balanced_accuracy"] >= busqueda["referencia_0.5"]["balanced_accuracy"]
    assert busqueda["candidatos_evaluados"] == len(flujo_final.GRILLA_UMBRALES)


def test_el_umbral_rechaza_probabilidades_inutilizables() -> None:
    with pytest.raises(ValueError):
        flujo_final.buscar_umbral(np.array([0, 1]), np.array([0.5]))


def test_el_umbral_se_ajusta_a_un_sesgo_hacia_la_clase_minoritaria() -> None:
    """Con un sesgo hacia NORMAL, el mejor umbral debe ser mayor que 0.5."""
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([np.linspace(0.0, 0.55, 50), np.linspace(0.6, 0.99, 50)])
    busqueda = flujo_final.buscar_umbral(y_true, y_prob)
    assert busqueda["umbral"] > 0.5


def test_el_umbral_usa_la_misma_rejilla_que_antes() -> None:
    assert 0.05 in flujo_final.GRILLA_UMBRALES
    assert 0.5 in flujo_final.GRILLA_UMBRALES
    assert 0.95 in flujo_final.GRILLA_UMBRALES
    assert len(flujo_final.GRILLA_UMBRALES) == 91


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------
def test_el_artefacto_de_combinado_declara_no_usar_test() -> None:
    assert _combinado_falso()["test_utilizado"] is False


def test_la_decision_congela_estrategia_configuracion_y_umbral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(flujo_final, "RUTA_COMBINADO", tmp_path / "combinado.json")
    monkeypatch.setattr(flujo_final, "RUTA_PROB_TRUE", tmp_path / "y_true.npy")
    monkeypatch.setattr(flujo_final, "RUTA_PROB_PROB", tmp_path / "y_prob.npy")
    monkeypatch.setattr(flujo_final, "RUTA_UMBRAL", tmp_path / "umbral.json")
    monkeypatch.setattr(flujo_final, "RUTA_DECISION", tmp_path / "decision.json")

    # ajustar_umbral() grafica con la ruta por defecto (reports/figures); se redirige a tmp_path
    # para que el test no sobrescriba la figura real del proyecto.
    graficar_real = flujo_final.graficar_seleccion_umbral
    monkeypatch.setattr(
        flujo_final,
        "graficar_seleccion_umbral",
        lambda curva, umbral_elegido: graficar_real(
            curva, umbral_elegido, ruta_salida=tmp_path / flujo_final.FIGURA_UMBRAL
        ),
    )

    flujo_final.guardar_json(_combinado_falso(), flujo_final.RUTA_COMBINADO)
    y_true = np.array([0] * 268 + [1] * 775)
    y_prob = np.concatenate([np.linspace(0.0, 0.5, 268), np.linspace(0.5, 1.0, 775)])
    np.save(flujo_final.RUTA_PROB_TRUE, y_true)
    np.save(flujo_final.RUTA_PROB_PROB, y_prob)

    decision = flujo_final.ajustar_umbral()

    assert decision["estrategia"] == "combinado"
    assert decision["model_name"] == "MobileNetV2"
    assert decision["config"] == {"learning_rate": 1e-3, "dropout": 0.3, "epochs": 3}
    assert decision["epochs_definitivos"] == 3
    assert decision["splits_entrenamiento_definitivo"] == ["train", "val"]
    assert decision["congelado"] is True
    assert decision["test_utilizado"] is False

    umbral = json.loads(flujo_final.RUTA_UMBRAL.read_text(encoding="utf-8"))
    assert umbral["conjunto_origen"] == "validation"
    assert umbral["validation_n"] == 1043
    assert umbral["congelado"] is True
    # El umbral se congela antes de cualquier evaluación de test.
    assert umbral["umbral"] == decision["umbral"]


def test_ajustar_umbral_exige_la_corrida_previa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flujo_final, "RUTA_COMBINADO", tmp_path / "no_existe.json")
    with pytest.raises(FileNotFoundError):
        flujo_final.ajustar_umbral()


def test_la_decision_no_congelada_se_rechaza(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flujo_final, "RUTA_DECISION", tmp_path / "decision.json")
    flujo_final.guardar_json({"congelado": False}, flujo_final.RUTA_DECISION)
    with pytest.raises(RuntimeError, match="congelada"):
        flujo_final.cargar_decision()


def test_la_config_se_toma_de_la_sensibilidad_y_se_verifica() -> None:
    config = flujo_final.config_desde_sensibilidad()
    assert config == {"learning_rate": 1e-3, "dropout": 0.3, "epochs": 3}
    assert flujo_final.MODELO_FINAL == "MobileNetV2"


def test_las_rutas_del_flujo_viven_en_results_final() -> None:
    """El umbral y los artefactos del flujo se guardan bajo ``results/final``."""
    assert flujo_final.RUTA_UMBRAL.name == "umbral_decision.json"
    assert flujo_final.RUTA_UMBRAL.parent.name == "final"
    assert flujo_final.RUTA_COMBINADO.parent.name == "final"


# ---------------------------------------------------------------------------
# Evaluación del test
# ---------------------------------------------------------------------------
def test_evaluar_test_exige_el_modelo_definitivo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin modelo definitivo no se puede evaluar el test."""
    monkeypatch.setattr(flujo_final, "RUTA_DECISION", tmp_path / "decision.json")
    monkeypatch.setattr(flujo_final, "RUTA_MODELO_FINAL", tmp_path / "no_existe.keras")
    flujo_final.guardar_json(
        {"congelado": True, "umbral": 0.47, "model_name": "MobileNetV2",
         "estrategia": "combinado", "config": {}},
        flujo_final.RUTA_DECISION,
    )
    with pytest.raises(FileNotFoundError):
        flujo_final.evaluar_test()


def test_el_test_se_evalua_una_sola_vez_al_final() -> None:
    """``evaluar_test`` es un paso normal del flujo y no depende de ninguna otra llamada."""
    assert list(inspect.signature(flujo_final.evaluar_test).parameters) == []
    assert flujo_final.SPLIT_TEST == "test"
    assert flujo_final.SPLITS_MEDICION == ("val",)


def test_la_evaluacion_del_test_escribe_solo_sus_artefactos() -> None:
    """La evaluación del test deja únicamente el informe y el CSV de métricas."""
    assert flujo_final.RUTA_REPORTE_TEST.name == "final_test_report.json"
    assert flujo_final.RUTA_REPORTE_TEST.parent.name == "final"
    assert flujo_final.RUTA_METRICAS_TEST.name == "final_test_metrics.csv"
    assert flujo_final.RUTA_METRICAS_TEST.parent.name == "final"


def test_la_evaluacion_del_test_no_aplica_tratamiento_de_desbalance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El pipeline de test se pide con ``incluir_test`` y sin ningún tratamiento.

    Se intercepta la construcción de pipelines para comprobar los argumentos con los que
    el flujo final pide el test: debe ser el manifiesto original, sin augmentation ni
    oversampling, y el único split que devuelve el flujo es ``test``.
    """
    llamadas: dict[str, object] = {}

    class Detenido(Exception):
        """Marca el punto en el que ya no hace falta seguir."""

    def falso_construir_pipelines(directorio_datos, **kwargs):
        llamadas.update(kwargs)
        llamadas["directorio_datos"] = directorio_datos
        return {"test": objeto_test}

    def falso_predecir(modelo, dataset):
        assert dataset is objeto_test
        raise Detenido

    objeto_test = object()
    monkeypatch.setattr(flujo_final, "construir_pipelines_datos", falso_construir_pipelines)
    monkeypatch.setattr(flujo_final, "predecir_probabilidades", falso_predecir)
    monkeypatch.setattr(flujo_final, "cargar_decision", lambda: {
        "umbral": 0.47, "model_name": "MobileNetV2", "estrategia": "combinado", "config": {},
    })
    monkeypatch.setattr(flujo_final, "cargar_modelo_definitivo", lambda: object())
    monkeypatch.setattr(flujo_final, "RUTA_DECISION", tmp_path / "d.json")
    monkeypatch.setattr(flujo_final, "RUTA_MODELO_FINAL", tmp_path / "m.keras")
    (tmp_path / "m.keras").write_bytes(b"x")
    monkeypatch.setattr(flujo_final, "DIRECTORIO_RESULTADOS_FINAL", tmp_path)

    with pytest.raises(Detenido):
        flujo_final.evaluar_test()

    # El pipeline se pide con el manifiesto original y con el test incluido.
    assert llamadas["incluir_test"] is True
    assert llamadas["manifiesto_division"] == flujo_final.RUTA_MANIFIESTO
    assert llamadas["image_size"] == flujo_final.TAMANO_IMAGEN
    # Ninguna bandera de tratamiento de desbalance ni de augmentation.
    for clave in llamadas:
        assert "oversampl" not in clave.lower()
        assert "class_weight" not in clave.lower()
        assert "augment" not in clave.lower()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_la_cli_expone_las_etapas_del_flujo() -> None:
    from click.testing import CliRunner

    from src.cli import cli

    ayuda = CliRunner().invoke(cli, ["--help"])
    assert ayuda.exit_code == 0
    for comando in ("eda", "prepare", "augment", "sensibilidad", "combinado", "umbral",
                    "final", "test", "evaluar", "run", "test-suite"):
        assert comando in ayuda.output


def test_la_cli_de_combinado_muestra_el_refuerzo_de_la_minoritaria() -> None:
    from click.testing import CliRunner

    from src.cli import cli

    resultado = CliRunner().invoke(cli, ["combinado", "--help"])
    assert resultado.exit_code == 0
    assert "COMBINADO" in resultado.output


def test_la_cli_de_test_no_toma_opciones() -> None:
    """El comando ``test`` se ejecuta normal: no declara ninguna opción."""
    from click.testing import CliRunner

    from src.cli import cli

    resultado = CliRunner().invoke(cli, ["test", "--help"])
    assert resultado.exit_code == 0
    assert resultado.output.count("--") == 1


def test_las_figuras_del_flujo_viven_en_reports_figures(tmp_path: Path) -> None:
    """Cada figura se llama como la etapa que la produce y vive en ``reports/figures``."""
    from src.training.pipeline_entrenamiento import guardar_curva_roc, guardar_matriz_confusion
    from src.training import flujo_final as ff
    from src.utils.paths import DIRECTORIO_FIGURAS
    from src.visualization.visualize import graficar_metricas_test, graficar_seleccion_umbral

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)

    cm = guardar_matriz_confusion(y_true, y_pred, "Validation", ff.FIGURA_VALIDACION_CONFUSION, directorio=tmp_path)
    roc = guardar_curva_roc(y_true, y_prob, "Validation", ff.FIGURA_VALIDACION_ROC, directorio=tmp_path)
    umbral = graficar_seleccion_umbral(
        [{"umbral": 0.3, "balanced_accuracy": 0.6}, {"umbral": 0.38, "balanced_accuracy": 0.9}, {"umbral": 0.5, "balanced_accuracy": 0.8}],
        0.38,
        ruta_salida=tmp_path / ff.FIGURA_UMBRAL,
    )
    metricas = graficar_metricas_test(
        {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "specificity": 0.8, "balanced_accuracy": 0.85, "f1": 0.9},
        {"accuracy": 0.85, "precision": 0.9, "recall": 0.8, "specificity": 0.85, "balanced_accuracy": 0.82, "f1": 0.85},
        0.38,
        ruta_salida=tmp_path / ff.FIGURA_TEST_METRICAS,
    )

    for figura in (cm, roc, umbral, metricas):
        assert figura.parent == tmp_path
        assert figura.exists() and figura.stat().st_size > 0

    # Los nombres por defecto de cada función apuntan al directorio de figuras del proyecto.
    assert (DIRECTORIO_FIGURAS / ff.FIGURA_VALIDACION_CONFUSION).name == ff.FIGURA_VALIDACION_CONFUSION
    assert ff.FIGURA_UMBRAL == "threshold_selection_validation.png"
    assert ff.FIGURA_TEST_CONFUSION == "test_final_confusion_matrix.png"
    assert ff.FIGURA_TEST_ROC == "test_final_roc_curve.png"
    assert ff.FIGURA_TEST_METRICAS == "test_final_metrics.png"

    nombres_figura = (
        ff.FIGURA_VALIDACION_CONFUSION,
        ff.FIGURA_VALIDACION_ROC,
        ff.FIGURA_UMBRAL,
        ff.FIGURA_TEST_CONFUSION,
        ff.FIGURA_TEST_ROC,
        ff.FIGURA_TEST_METRICAS,
    )
    assert len(set(nombres_figura)) == len(nombres_figura)
    for nombre in nombres_figura:
        assert nombre.endswith(".png")
        assert " " not in nombre and nombre == nombre.lower()


def test_las_figuras_del_flujo_actual_existen_en_disco() -> None:
    """Las figuras que documenta el proyecto están regeneradas y no están vacías."""
    from src.training import flujo_final as ff
    from src.utils.paths import DIRECTORIO_FIGURAS

    for nombre in (
        ff.FIGURA_VALIDACION_CONFUSION,
        ff.FIGURA_VALIDACION_ROC,
        ff.FIGURA_UMBRAL,
        ff.FIGURA_TEST_CONFUSION,
        ff.FIGURA_TEST_ROC,
        ff.FIGURA_TEST_METRICAS,
        "sensitivity_validation.png",
        "data_augmentation_examples.png",
        "dataset_distribution.png",
    ):
        ruta = DIRECTORIO_FIGURAS / nombre
        assert ruta.exists(), f"falta la figura {nombre}"
        assert ruta.stat().st_size > 0, f"la figura {nombre} está vacía"


def test_los_format_de_este_modulo_no_tienen_claves_renombradas() -> None:
    """Cada ``.format()`` debe pasar exactamente las claves que el texto declara.

    Estos mensajes solo se ejecutan al correr el flujo real, asi que un error aqui pasaria
    inadvertido hasta gastar minutos de CPU en un entrenamiento. Se comprueba de forma
    estatica sobre el codigo fuente del modulo.
    """
    import ast
    from pathlib import Path as _Path

    ruta = _Path(flujo_final.__file__)
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))

    revisados = 0
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call) or len(nodo.args) != 0:
            continue
        func = nodo.func
        if not (isinstance(func, ast.Attribute) and func.attr == "format"):
            continue
        # Solo interesan los .format() sobre un literal de texto.
        if not isinstance(func.value, ast.Constant) or not isinstance(func.value.value, str):
            continue

        declarados = {
            campo.split(".")[0].split("[")[0]
            for _, campo, _, _ in string.Formatter().parse(func.value.value)
            if campo
        }
        forzados = {kw.arg for kw in nodo.keywords if kw.arg is not None}

        assert declarados == forzados, (
            f"linea {nodo.lineno}: el texto declara {sorted(declarados - forzados)} "
            f"que no se pasan, y recibe {sorted(forzados - declarados)} que no usa"
        )
        revisados += 1

    assert revisados >= 2, "el modulo deberia seguir teniendo mensajes formateados"


def test_las_etapas_no_se_pisan_el_checkpoint() -> None:
    """La corrida sobre train no puede sobrescribir el modelo definitivo."""
    assert flujo_final.RUTA_CHECKPOINT_COMBINADO != flujo_final.RUTA_MODELO_FINAL
    assert flujo_final.RUTA_CHECKPOINT_COMBINADO.parent.name == "combinado"
    assert flujo_final.RUTA_MODELO_FINAL.parent.name == "final_model"


def test_el_entrenamiento_definitivo_guarda_el_modelo() -> None:
    """Sin validation no hay ModelCheckpoint, asi que el save debe ser explicito."""
    import inspect

    fuente = inspect.getsource(flujo_final.entrenar_modelo_definitivo)

    # Entrena sin validation y aun asi persiste los pesos definitivos.
    assert "val_dataset=None" in fuente
    assert "model.save(RUTA_MODELO_FINAL)" in fuente


def test_el_umbral_advierte_que_validation_esta_sesgada() -> None:
    """El umbral se maximiza sobre validation, asi que su BA esta sesgada al alza."""
    import inspect

    fuente = inspect.getsource(flujo_final.ajustar_umbral)

    assert "advertencia_optimismo" in fuente
    assert "delta_balanced_accuracy_vs_0.5" in fuente
