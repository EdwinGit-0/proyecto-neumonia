"""Pruebas del flujo final: estrategia única COMBINADO, umbral congelado y test final.

Ninguna de estas pruebas entrena un modelo: comprueban las garantías del protocolo
(qué splits se usan, que validation y test quedan intactos, que existe una sola
estrategia, qué se congela y que el test se evalúa sin bloqueos) y la lógica de
búsqueda de umbral con datos sintéticos.
"""

from __future__ import annotations

import json
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


def test_no_existe_la_comparacion_de_estrategias() -> None:
    """La comparación/ranking de estrategias debe haber desaparecido del flujo."""
    for nombre_atributo in ("ESTRATEGIAS", "NOMBRES_ESTRATEGIAS", "estrategia_por_id",
                            "ejecutar_estrategias", "seleccionar_ganador_por_arquitectura",
                            "escribir_tabla_csv", "clave_orden", "RUTA_TABLA", "RUTA_RESULTADOS"):
        assert not hasattr(flujo_final, nombre_atributo), (
            f"flujo_final todavia expone {nombre_atributo}, que pertenece a la comparacion de estrategias"
        )


def test_las_estrategias_retiradas_no_existen_como_opciones() -> None:
    """Los identificadores de las alternativas retiradas no deben quedar como código vivo.

    Se ignoran docstrings y comentarios: el módulo menciona ``sin_tratamiento`` a
    propósito, para documentar que el test ya se evaluó con esa estrategia. Lo que no
    debe existir es la alternativa como opción seleccionable.
    """
    import io
    import tokenize
    from pathlib import Path as _Path

    import src.training.flujo_final as modulo_flujo
    import src.training.tratamiento_desbalance as modulo_tratamiento

    for modulo in (modulo_flujo, modulo_tratamiento):
        ruta = _Path(modulo.__file__)
        with open(ruta, "rb") as archivo:
            fuente = archivo.read()
        codigo = "".join(
            token.string
            for token in tokenize.tokenize(io.BytesIO(fuente).readline)
            if token.type not in (tokenize.COMMENT, tokenize.STRING)
        )
        for retirado in ("sin_tratamiento", "ganador_global", "mejor_por_arquitectura"):
            assert retirado not in codigo, (
                f"{retirado} sigue presente como codigo en {ruta.name}"
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


def test_las_rutas_del_flujo_no_terminan_en_test() -> None:
    """El umbral se guarda junto al resultado final, no junto a los artefactos de estrategias."""
    assert flujo_final.RUTA_UMBRAL.name == "umbral_decision.json"
    assert flujo_final.RUTA_UMBRAL.parent.name == "final"
    # El umbral ya no vive junto a los artefactos de estrategias.
    assert "estrategias" not in str(flujo_final.RUTA_UMBRAL)
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


def test_el_flujo_no_conserva_mecanismo_de_bloqueo_del_test() -> None:
    """El test es un conjunto de medición normal: nada en el código lo bloquea.

    Este proyecto eliminó el registro de usos y el guard de contaminación, así que
    ``evaluar_test`` no debe recibir ningún parámetro de reconocimiento ni consultar
    ningún historial previo.
    """
    import inspect

    parametros = inspect.signature(flujo_final.evaluar_test).parameters
    assert list(parametros) == [], f"evaluar_test no debe pedir parámetros: {list(parametros)}"

    fuente = inspect.getsource(flujo_final)
    for prohibido in (
        "registro_test",
        "anotar_uso_test",
        "consultar_registro_test",
        "reconocer_contaminacion",
        "test_ya_utilizado",
        "evaluar_test_una_vez",
    ):
        assert prohibido not in fuente, f"queda una referencia a {prohibido!r}"


def test_la_evaluacion_del_test_no_depende_de_ningun_historial() -> None:
    """``results/registro_test.json`` ya no existe ni debe volver a crearse."""
    from src.utils.paths import DIRECTORIO_RESULTADOS

    assert not (DIRECTORIO_RESULTADOS / "registro_test.json").exists()
    assert not (DIRECTORIO_RESULTADOS / "historico").exists()


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
    for comando in ("combinado", "umbral", "final", "test", "evaluar", "run"):
        assert comando in ayuda.output
    # La comparación de estrategias ya no es un comando.
    assert "estrategias" not in ayuda.output


def test_la_cli_de_combinado_muestra_el_refuerzo_de_la_minoritaria() -> None:
    from click.testing import CliRunner

    from src.cli import cli

    resultado = CliRunner().invoke(cli, ["combinado", "--help"])
    assert resultado.exit_code == 0
    assert "COMBINADO" in resultado.output


def test_la_cli_de_test_no_pide_reconocer_contaminacion() -> None:
    """El comando ``test`` se ejecuta normal, sin banderas de historial."""
    from click.testing import CliRunner

    from src.cli import cli

    resultado = CliRunner().invoke(cli, ["test", "--help"])
    assert resultado.exit_code == 0
    assert "--reconocer-contaminacion" not in resultado.output
    assert "contaminaci" not in resultado.output.lower()


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
