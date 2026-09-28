"""Flujo final del proyecto: estrategia única COMBINADO, umbral congelado y test final.

La estrategia de desbalance del proyecto es ``combinado`` (oversampling 50/50 + class weights)
sobre la arquitectura elegida por la sensibilidad de hiperparámetros. COMBINADO es una premisa
metodológica, no el resultado de comparar variantes. Este módulo ejecuta el flujo de principio
a fin.

Etapas:

1. ``ejecutar_combinado``: entrena MobileNetV2 con COMBINADO sobre ``train`` y mide
   exclusivamente sobre ``validation``, que queda intacta.
2. ``ajustar_umbral``: elige el umbral que maximiza el balanced accuracy con las
   probabilidades de ``validation`` y lo congela en ``results/final/umbral_decision.json``.
3. ``entrenar_modelo_definitivo``: reentrena COMBINADO sobre ``train + val`` con las
   épocas ya decididas.
4. ``evaluar_test``: evalúa el modelo definitivo sobre el test original usando el
   umbral congelado y guarda el informe final.

El test se trata como un conjunto de measurement normal: se lee una vez, al final del
flujo, después de que el umbral ya está congelado. No aplica oversampling, class weights
ni augmentation: usa sus imágenes y su distribución originales.

Uso:

    python -m src.training.flujo_final combinado
    python -m src.training.flujo_final umbral
    python -m src.training.flujo_final final
    python -m src.training.flujo_final test
    python -m src.training.flujo_final todo
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.datasets import construir_pipelines_datos
from src.data.splitting import cargar_manifiesto_division
from src.models.architectures import construir_modelo_proyecto
from src.models.evaluation import (
    calcular_reporte_metricas,
    evaluar_baseline_mayoritaria,
    evaluar_criterio_exito,
)
from src.training.tratamiento_desbalance import (
    ESTRATEGIA_COMBINADO,
    NOMBRE_ESTRATEGIA,
    construir_train_combinado,
)
from src.training.pipeline_entrenamiento import (
    TAMANO_IMAGEN,
    TAMANO_LOTE,
    UMBRAL_BASE,
    construir_pipeline_augmentation,
    entrenar_corrida,
    guardar_curva_roc,
    guardar_matriz_confusion,
    predecir_probabilidades,
)
from src.training.sensitivity import cargar_mejor_configuracion_por_arquitectura
from src.visualization.visualize import (
    graficar_metricas_test,
    graficar_seleccion_umbral,
)
from src.utils.paths import (
    DIRECTORIO_DATOS,
    DIRECTORIO_RESULTADOS_FINAL,
    RAIZ_PROYECTO,
    asegurar_directorio,
)
from src.utils.reproducibility import SEMILLA, reiniciar_semilla


RUTA_MANIFIESTO = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"

# Arquitectura e hiperparámetros fijados por la etapa de sensibilidad. No se vuelve a
# buscar: este flujo no hace selección de modelos.
MODELO_FINAL = "MobileNetV2"
CONFIG_FINAL_ESPERADA = {"learning_rate": 1e-3, "dropout": 0.3, "epochs": 3}

RUTA_COMBINADO = DIRECTORIO_RESULTADOS_FINAL / "combinado_validacion.json"
RUTA_TABLA_COMBINADO = DIRECTORIO_RESULTADOS_FINAL / "combinado_validacion.csv"
RUTA_PROB_TRUE = DIRECTORIO_RESULTADOS_FINAL / "validacion_y_true.npy"
RUTA_PROB_PROB = DIRECTORIO_RESULTADOS_FINAL / "validacion_y_prob.npy"
RUTA_UMBRAL = DIRECTORIO_RESULTADOS_FINAL / "umbral_decision.json"
RUTA_DECISION = DIRECTORIO_RESULTADOS_FINAL / "decision_modelo.json"
RUTA_MODELO_FINAL = DIRECTORIO_RESULTADOS_FINAL / "final_model" / "best_model.keras"
RUTA_ENTRENAMIENTO = DIRECTORIO_RESULTADOS_FINAL / "final_model_training.json"
RUTA_CHECKPOINT_COMBINADO = DIRECTORIO_RESULTADOS_FINAL / "combinado" / "best_model.keras"
RUTA_REPORTE_TEST = DIRECTORIO_RESULTADOS_FINAL / "final_test_report.json"
RUTA_METRICAS_TEST = DIRECTORIO_RESULTADOS_FINAL / "final_test_metrics.csv"

# Figuras del flujo actual, todas en reports/figures con el nombre de la etapa que
# las produce.
FIGURA_VALIDACION_CONFUSION = "validation_combinado_confusion_matrix.png"
FIGURA_VALIDACION_ROC = "validation_combinado_roc_curve.png"
FIGURA_UMBRAL = "threshold_selection_validation.png"
FIGURA_TEST_CONFUSION = "test_final_confusion_matrix.png"
FIGURA_TEST_ROC = "test_final_roc_curve.png"
FIGURA_TEST_METRICAS = "test_final_metrics.png"

SPLIT_TEST = "test"
SPLITS_ENTRENAMIENTO = ("train",)
SPLITS_ENTRENAMIENTO_FINAL = ("train", "val")
SPLITS_MEDICION = ("val",)

METRICAS_REPORTE = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f1",
    "roc_auc",
)

# Rejilla de umbrales evaluada sobre validation: 0.05 a 0.95 en pasos de 0.01, más el
# 0.5 por defecto. El desempate favorece el umbral más cercano a 0.5.
GRILLA_UMBRALES = tuple(
    sorted({round(0.05 + 0.01 * indice, 2) for indice in range(91)} | {UMBRAL_BASE})
)


def _marcar(mensaje: str) -> None:
    print(f"[flujo_final] {mensaje}", flush=True)


def guardar_json(datos: dict[str, Any], ruta: Path) -> Path:
    """Guardar un JSON de forma atómica."""
    asegurar_directorio(Path(ruta).parent)
    temporal = Path(ruta).with_suffix(Path(ruta).suffix + ".tmp")
    temporal.write_text(json.dumps(datos, indent=2, default=str), encoding="utf-8")
    temporal.replace(Path(ruta))
    return Path(ruta)


def cargar_json(ruta: Path) -> dict[str, Any]:
    """Cargar un JSON del flujo."""
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecute antes las etapas previas del flujo final.")
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def rutas_splits_permitidos(splits: Sequence[str]) -> None:
    """Impedir que se construya un pipeline con el split de prueba."""
    if SPLIT_TEST in splits:
        raise ValueError(
            "El flujo de entrenamiento y decisión no puede construir el pipeline de test. "
            "La evaluación de test solo existe en evaluar_test()."
        )


def config_desde_sensibilidad() -> dict[str, Any]:
    """Recuperar la configuración que la sensibilidad ya eligió para MobileNetV2.

    El flujo no busca hiperparámetros: los lee del artefacto de sensibilidad, que es la
    evidencia que justifica la arquitectura y evita presentar una decisión como si se
    hubiera tomado aquí.
    """
    configuraciones = cargar_mejor_configuracion_por_arquitectura()
    if MODELO_FINAL not in configuraciones:
        raise RuntimeError(
            f"La sensibilidad no registró {MODELO_FINAL}. Ejecute 'neumonia sensibilidad' antes."
        )
    config = dict(configuraciones[MODELO_FINAL]["config"])
    esperada = CONFIG_FINAL_ESPERADA
    discrepancia = {
        clave: (config.get(clave), valor)
        for clave, valor in esperada.items()
        if config.get(clave) != valor
    }
    if discrepancia:
        raise RuntimeError(
            f"La configuración de sensibilidad para {MODELO_FINAL} no coincide con la fijada para "
            f"este flujo: {discrepancia}. Se esperaba {esperada}. No se continúa para no entrenar "
            "con hiperparámetros distintos de los documentados."
        )
    return config


def construir_pipeline_validacion() -> tf.data.Dataset:
    """Construir el pipeline de validación **sin** oversampling ni reponderación.

    ``validation`` conserva su distribución original (268 NORMAL / 775 PNEUMONIA):
    no se duplica ninguna fila y no se aplica ningún ``class_weight``.
    """
    rutas_splits_permitidos(SPLITS_MEDICION)
    construidos = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        manifiesto_division=RUTA_MANIFIESTO,
    )
    faltantes = [split for split in SPLITS_MEDICION if split not in construidos]
    if faltantes:
        raise ValueError(f"El manifiesto no contiene los splits solicitados: {faltantes}")
    return construidos["val"]


def construir_pipeline_combinado(
    splits: Sequence[str] = SPLITS_ENTRENAMIENTO, batch_size: int = TAMANO_LOTE
) -> tuple[tf.data.Dataset, dict[int, float], dict[str, Any]]:
    """Pipeline de entrenamiento con COMBINADO más los class weights de la loss."""
    rutas_splits_permitidos(splits)
    dataset, pesos, diagnostico = construir_train_combinado(
        RUTA_MANIFIESTO,
        image_size=TAMANO_IMAGEN,
        batch_size=batch_size,
        splits=splits,
    )
    return construir_pipeline_augmentation(dataset), pesos, diagnostico


def _composicion_manifiesto(splits: Sequence[str]) -> dict[str, int]:
    manifiesto = cargar_manifiesto_division(RUTA_MANIFIESTO)
    filas = manifiesto[manifiesto["split"].isin(list(splits))]
    return {
        "NORMAL": int((filas["target"] == 0).sum()),
        "PNEUMONIA": int((filas["target"] == 1).sum()),
        "total": int(len(filas)),
    }


# ---------------------------------------------------------------------------
# Etapa 1: COMBINADO sobre train, medido en validation
# ---------------------------------------------------------------------------
def ejecutar_combinado(recalcular: bool = False) -> dict[str, Any]:
    """Entrenar MobileNetV2 con COMBINADO sobre train y medirlo en validation.

    No se compara con ninguna otra estrategia: la elección de COMBINADO es una premisa
    del proyecto. El resultado se guarda aunque exista, salvo que se pida ``recalcular``.
    """
    asegurar_directorio(DIRECTORIO_RESULTADOS_FINAL)
    config = config_desde_sensibilidad()

    if RUTA_COMBINADO.exists() and not recalcular:
        _marcar("combinado ya completado, se reutiliza (use --recalcular para rehacerlo)")
        return cargar_json(RUTA_COMBINADO)

    _marcar(
        f"entrenando {MODELO_FINAL}/{NOMBRE_ESTRATEGIA} sobre train "
        f"lr={config['learning_rate']:g} dropout={config['dropout']:g} epochs={config['epochs']}"
    )

    tf.keras.backend.clear_session()
    reiniciar_semilla(SEMILLA)
    train_dataset, class_weight, diagnostico = construir_pipeline_combinado(SPLITS_ENTRENAMIENTO)
    _marcar(
        "oversampling: {NORMAL} + {PNEUMONIA} = {total} filas efectivas | class_weight={cw}".format(
            NORMAL=diagnostico["composicion_tras_oversampling"]["NORMAL"],
            PNEUMONIA=diagnostico["composicion_tras_oversampling"]["PNEUMONIA"],
            total=diagnostico["composicion_tras_oversampling"]["total"],
            cw=diagnostico["class_weight"],
        )
    )
    _marcar(diagnostico["advertencia_doble_correccion"])

    val_dataset = construir_pipeline_validacion()

    model = construir_modelo_proyecto(
        model_name=MODELO_FINAL,
        dropout=float(config["dropout"]),
        learning_rate=float(config["learning_rate"]),
    )
    entrenamiento = entrenar_corrida(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=int(config["epochs"]),
        ruta_checkpoint=RUTA_CHECKPOINT_COMBINADO,
        class_weight=class_weight,
    )
    del train_dataset
    tf.keras.backend.clear_session()

    y_true, y_prob = predecir_probabilidades(model, val_dataset)
    del val_dataset

    metricas = calcular_reporte_metricas(y_true, y_prob, threshold=UMBRAL_BASE)
    y_pred = (y_prob >= UMBRAL_BASE).astype(int)
    matriz = {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }

    np.save(RUTA_PROB_TRUE, y_true)
    np.save(RUTA_PROB_PROB, y_prob)

    figura_validacion_confusion = guardar_matriz_confusion(
        y_true,
        y_pred,
        f"Validation - {MODELO_FINAL} COMBINADO (umbral {UMBRAL_BASE:g})",
        FIGURA_VALIDACION_CONFUSION,
    )
    figura_validacion_roc = guardar_curva_roc(
        y_true,
        y_prob,
        f"Validation - {MODELO_FINAL} COMBINADO",
        FIGURA_VALIDACION_ROC,
        etiqueta_adicional="MobileNetV2 COMBINADO",
    )

    payload = {
        "descripcion": (
            "Entrenamiento de MobileNetV2 con la estrategia unica COMBINADO "
            "(oversampling 50/50 + class weights) sobre train, medido exclusivamente "
            "sobre validation. Validation no recibe oversampling ni class weights."
        ),
        "etapa": "combinado",
        "estrategia": dict(ESTRATEGIA_COMBINADO),
        "model_name": MODELO_FINAL,
        "config": config,
        "config_origen": "etapa de sensibilidad (sin nueva busqueda de hiperparametros)",
        "split_manifest": str(RUTA_MANIFIESTO),
        "seed": SEMILLA,
        "batch_size": TAMANO_LOTE,
        "tamano_imagen": list(TAMANO_IMAGEN),
        "umbral_medicion": UMBRAL_BASE,
        "splits_entrenamiento": list(SPLITS_ENTRENAMIENTO),
        "splits_medicion": list(SPLITS_MEDICION),
        "diagnostico_combinado": diagnostico,
        "class_weight_aplicado": {str(clase): peso for clase, peso in class_weight.items()},
        "class_weight_aplicado_solo_a": "funcion de perdida del entrenamiento",
        "validation_composicion": _composicion_manifiesto(("val",)),
        "validation": {metrica: float(metricas[metrica]) for metrica in METRICAS_REPORTE},
        "validation_matriz_confusion": matriz,
        "validation_n": int(y_true.size),
        "segundos_entrenamiento": entrenamiento["segundos_entrenamiento"],
        "epochs_ejecutadas": entrenamiento["epochs_ejecutadas"],
        "early_stopping": entrenamiento["early_stopping"],
        "historial": entrenamiento["historial"],
        "figuras": {
            "matriz_confusion": str(figura_validacion_confusion),
            "curva_roc": str(figura_validacion_roc),
        },
        "test_utilizado": False,
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }
    guardar_json(payload, RUTA_COMBINADO)
    escribir_tabla_validacion(payload)
    _marcar(
        f"validation: balanced accuracy {metricas['balanced_accuracy']:.4f} "
        f"(recall {metricas['recall']:.4f}, specificity {metricas['specificity']:.4f})"
    )
    _marcar(f"figuras: {figura_validacion_confusion}, {figura_validacion_roc}")
    return payload


def escribir_tabla_validacion(payload: dict[str, Any], ruta: Path = RUTA_TABLA_COMBINADO) -> Path:
    """Escribir las métricas de validación del modelo COMBINADO en CSV."""
    fila = {"split": "validation", "umbral": UMBRAL_BASE}
    fila.update({metrica: payload["validation"][metrica] for metrica in METRICAS_REPORTE})
    fila.update({f"matriz_{clave}": valor for clave, valor in payload["validation_matriz_confusion"].items()})
    asegurar_directorio(Path(ruta).parent)
    pd.DataFrame([fila]).to_csv(ruta, index=False, encoding="utf-8")
    return Path(ruta)


# ---------------------------------------------------------------------------
# Etapa 2: umbral de decisión sobre validación
# ---------------------------------------------------------------------------
def buscar_umbral(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Buscar el umbral que maximiza el balanced accuracy en el conjunto dado.

    El empate se resuelve eligiendo el umbral más cercano a 0.5, de modo que el
    criterio sea determinista y no premie artefactos de la rejilla.
    """
    if y_true.size != y_prob.size or y_true.size == 0:
        raise ValueError("Las probabilidades de validación no son utilizables para buscar umbral.")

    evaluados: list[dict[str, float]] = []
    for umbral in GRILLA_UMBRALES:
        metricas = calcular_reporte_metricas(y_true, y_prob, threshold=umbral)
        evaluados.append(
            {
                "umbral": float(umbral),
                "balanced_accuracy": float(metricas["balanced_accuracy"]),
                "f1": float(metricas["f1"]),
                "accuracy": float(metricas["accuracy"]),
                "precision": float(metricas["precision"]),
                "recall": float(metricas["recall"]),
                "specificity": float(metricas["specificity"]),
            }
        )

    mejor = max(
        evaluados,
        key=lambda item: (item["balanced_accuracy"], -abs(item["umbral"] - UMBRAL_BASE)),
    )
    referencia = next(item for item in evaluados if abs(item["umbral"] - UMBRAL_BASE) < 1e-9)
    return {
        "umbral": mejor["umbral"],
        "criterio": "maximo balanced_accuracy en validation; empate -> mas cercano a 0.5",
        "candidatos_evaluados": len(evaluados),
        "mejor": mejor,
        "referencia_0.5": referencia,
        "delta_balanced_accuracy_vs_0.5": mejor["balanced_accuracy"] - referencia["balanced_accuracy"],
        "curva": evaluados,
    }


def ajustar_umbral() -> dict[str, Any]:
    """Elegir y congelar el umbral de COMBINADO usando solo las probabilidades de validation.

    El resultado va a ``results/final/umbral_decision.json``, y queda congelado antes
    de que el test se lea por primera vez.
    """
    combinado = cargar_json(RUTA_COMBINADO)
    if not RUTA_PROB_TRUE.exists() or not RUTA_PROB_PROB.exists():
        raise FileNotFoundError(
            "Faltan las probabilidades de validación. Ejecute antes 'neumonia combinado'."
        )

    y_true = np.load(RUTA_PROB_TRUE)
    y_prob = np.load(RUTA_PROB_PROB)
    if int(y_true.size) != int(combinado["validation_n"]):
        raise RuntimeError(
            f"Las probabilidades guardadas ({y_true.size}) no corresponden a la validation medida "
            f"({combinado['validation_n']})."
        )
    busqueda = buscar_umbral(y_true, y_prob)

    umbral = {
        "descripcion": (
            "Umbral de decision de la estrategia COMBINADO, congelado antes de cualquier "
            "evaluacion de test. Se calcula exclusivamente con las probabilidades de "
            "validation del modelo COMBINADO."
        ),
        "estrategia": NOMBRE_ESTRATEGIA,
        "model_name": combinado["model_name"],
        "config": dict(combinado["config"]),
        "umbral": busqueda["umbral"],
        "criterio": busqueda["criterio"],
        "candidatos_evaluados": busqueda["candidatos_evaluados"],
        "origen_probabilidades": str(RUTA_PROB_PROB),
        "conjunto_origen": "validation",
        "validation_n": int(y_true.size),
        "validation_composicion": combinado["validation_composicion"],
        "balanced_accuracy_validacion": busqueda["mejor"]["balanced_accuracy"],
        "balanced_accuracy_validacion_umbral_0.5": busqueda["referencia_0.5"]["balanced_accuracy"],
        "delta_balanced_accuracy_vs_0.5": busqueda["delta_balanced_accuracy_vs_0.5"],
        "metricas_umbral_elegido": busqueda["mejor"],
        "curva_umbrales": busqueda["curva"],
        "figura_seleccion_umbral": str(
            graficar_seleccion_umbral(busqueda["curva"], busqueda["umbral"])
        ),
        "advertencia_optimismo": (
            "El umbral se eligio maximizando balanced_accuracy sobre validation, y ese mismo "
            "conjunto es el que se reporta. El balanced_accuracy de validation esta por tanto "
            "optimistamente sesgado al alza. Con "
            f"{busqueda['delta_balanced_accuracy_vs_0.5']:+.4f} frente a 0.5, el umbral 0.5 "
            "rinde practicamente igual y la diferencia no es interpretable."
        ),
        "test_utilizado": False,
        "congelado": True,
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }
    guardar_json(umbral, RUTA_UMBRAL)

    decision = {
        "descripcion": (
            "Decision cerrada sobre train y validation antes de cualquier evaluacion de test. "
            "Estrategia, modelo, hiperparametros y umbral quedan congelados aqui."
        ),
        "estrategia": NOMBRE_ESTRATEGIA,
        "definicion_estrategia": dict(ESTRATEGIA_COMBINADO),
        "model_name": combinado["model_name"],
        "config": dict(combinado["config"]),
        "validation_ganadora": dict(combinado["validation"]),
        "validation_matriz_confusion": dict(combinado["validation_matriz_confusion"]),
        "diagnostico_combinado": combinado["diagnostico_combinado"],
        "umbral": busqueda["umbral"],
        "criterio_umbral": busqueda["criterio"],
        "epochs_definitivos": int(combinado["config"]["epochs"]),
        "splits_entrenamiento_definitivo": list(SPLITS_ENTRENAMIENTO_FINAL),
        "test_utilizado": False,
        "congelado": True,
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }
    guardar_json(decision, RUTA_DECISION)
    _marcar(
        f"umbral congelado: {busqueda['umbral']} "
        f"(balanced accuracy validacion {busqueda['mejor']['balanced_accuracy']:.4f} "
        f"frente a {busqueda['referencia_0.5']['balanced_accuracy']:.4f} en 0.5)"
    )
    _marcar(f"figura: {umbral['figura_seleccion_umbral']}")
    return decision


def cargar_decision() -> dict[str, Any]:
    """Cargar la decisión congelada del modelo definitivo."""
    decision = cargar_json(RUTA_DECISION)
    if not decision.get("congelado"):
        raise RuntimeError("La decisión del modelo definitivo no está congelada.")
    return decision


# ---------------------------------------------------------------------------
# Etapa 3: modelo definitivo sobre train + val
# ---------------------------------------------------------------------------
def entrenar_modelo_definitivo() -> dict[str, Any]:
    """Reentrenar COMBINADO sobre ``train + val`` con las épocas ya decididas.

    COMBINADO se aplica solo a los datos de entrenamiento: el oversampling equilibra a
    50/50 las 5.216 filas de ``train + val`` y los class weights se derivan de esa misma
    distribución original. ``test`` no participa.
    """
    decision = cargar_decision()
    asegurar_directorio(RUTA_MODELO_FINAL.parent)

    _marcar(
        f"entrenando modelo definitivo {decision['model_name']}/{decision['estrategia']} "
        f"sobre {'+'.join(SPLITS_ENTRENAMIENTO_FINAL)} con {decision['epochs_definitivos']} epochs"
    )

    tf.keras.backend.clear_session()
    reiniciar_semilla(SEMILLA)
    train_dataset, class_weight, diagnostico = construir_pipeline_combinado(
        SPLITS_ENTRENAMIENTO_FINAL
    )
    _marcar(
        "oversampling train+val: {NORMAL} + {PNEUMONIA} = {total} filas efectivas".format(
            NORMAL=diagnostico["composicion_tras_oversampling"]["NORMAL"],
            PNEUMONIA=diagnostico["composicion_tras_oversampling"]["PNEUMONIA"],
            total=diagnostico["composicion_tras_oversampling"]["total"],
        )
    )

    model = construir_modelo_proyecto(
        model_name=decision["model_name"],
        dropout=float(decision["config"]["dropout"]),
        learning_rate=float(decision["config"]["learning_rate"]),
    )
    entrenamiento = entrenar_corrida(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        epochs=int(decision["epochs_definitivos"]),
        ruta_checkpoint=RUTA_MODELO_FINAL,
        class_weight=class_weight,
    )
    del train_dataset
    # Sin validation no hay ModelCheckpoint, asi que el modelo se guarda aqui. Se guardan los
    # pesos finales, que son los unicos definitivos: al no haber validacion no existe "mejor epoch".
    model.save(RUTA_MODELO_FINAL)
    del model
    tf.keras.backend.clear_session()

    if not RUTA_MODELO_FINAL.exists():
        raise RuntimeError(f"El entrenamiento definitivo no produjo {RUTA_MODELO_FINAL}.")

    composicion = _composicion_manifiesto(SPLITS_ENTRENAMIENTO_FINAL)
    registro = {
        "descripcion": (
            "Modelo definitivo: estrategia unica COMBINADO reentrenada sobre train + val. "
            "El numero de epochs y el umbral ya estaban decididos sobre validation."
        ),
        "model_name": decision["model_name"],
        "estrategia": NOMBRE_ESTRATEGIA,
        "config": dict(decision["config"]),
        "splits_entrenamiento": list(SPLITS_ENTRENAMIENTO_FINAL),
        "composicion_entrenamiento": composicion,
        "diagnostico_combinado": diagnostico,
        "class_weight": {str(clase): peso for clase, peso in class_weight.items()},
        "class_weight_aplicado_solo_a": "funcion de perdida del entrenamiento",
        "epochs": int(decision["epochs_definitivos"]),
        "epochs_ejecutadas": entrenamiento["epochs_ejecutadas"],
        "segundos_entrenamiento": entrenamiento["segundos_entrenamiento"],
        "umbral": decision["umbral"],
        "modelo": str(RUTA_MODELO_FINAL),
        "test_utilizado": False,
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }
    guardar_json(registro, RUTA_ENTRENAMIENTO)
    _marcar(f"modelo definitivo guardado en {RUTA_MODELO_FINAL}")
    return registro


def cargar_modelo_definitivo() -> tf.keras.Model:
    """Cargar el modelo definitivo congelado."""
    if not RUTA_MODELO_FINAL.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_MODELO_FINAL}. Ejecute antes 'neumonia final'."
        )
    return tf.keras.models.load_model(RUTA_MODELO_FINAL)


# ---------------------------------------------------------------------------
# Etapa 4: evaluación del test con el umbral ya congelado
# ---------------------------------------------------------------------------
def _composicion_test() -> dict[str, int]:
    manifiesto = cargar_manifiesto_division(RUTA_MANIFIESTO)
    filas = manifiesto[manifiesto["split"] == SPLIT_TEST]
    return {
        "total": int(len(filas)),
        "NORMAL": int((filas["label"] == "NORMAL").sum()),
        "PNEUMONIA": int((filas["label"] == "PNEUMONIA").sum()),
    }


def evaluar_test() -> dict[str, Any]:
    """Evaluar el modelo definitivo COMBINADO sobre el test original.

    El test se lee una sola vez, al final del flujo, con el umbral ya congelado a partir
    de ``validation``. No se aplica ningún tratamiento de desbalance: las imágenes y la
    distribución son las originales del manifiesto.
    """
    decision = cargar_decision()
    if not RUTA_MODELO_FINAL.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_MODELO_FINAL}. Ejecute antes el entrenamiento del modelo definitivo."
        )

    composicion = _composicion_test()
    if composicion["total"] != 624:
        raise RuntimeError(
            f"El manifiesto declara {composicion['total']} filas de test; se esperaban 624. "
            "No se evalúa un test que no sea el original."
        )

    _marcar("cargando el test original")
    test_dataset = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        manifiesto_division=RUTA_MANIFIESTO,
        incluir_test=True,
    )[SPLIT_TEST]

    model = cargar_modelo_definitivo()
    y_true, y_prob = predecir_probabilidades(model, test_dataset)
    del model, test_dataset
    tf.keras.backend.clear_session()

    umbral = float(decision["umbral"])
    metricas = calcular_reporte_metricas(y_true, y_prob, threshold=umbral)
    metricas_referencia = calcular_reporte_metricas(y_true, y_prob, threshold=UMBRAL_BASE)
    y_pred = (y_prob >= umbral).astype(int)
    y_pred_referencia = (y_prob >= UMBRAL_BASE).astype(int)

    baseline = evaluar_baseline_mayoritaria(y_true)
    criterios = {
        "umbral_congelado": evaluar_criterio_exito(
            {"balanced_accuracy": metricas["balanced_accuracy"], "recall": metricas["recall"],
             "specificity": metricas["specificity"]},
            baseline,
        ),
        "referencia_0.5": evaluar_criterio_exito(
            {"balanced_accuracy": metricas_referencia["balanced_accuracy"],
             "recall": metricas_referencia["recall"],
             "specificity": metricas_referencia["specificity"]},
            baseline,
        ),
    }

    asegurar_directorio(DIRECTORIO_RESULTADOS_FINAL)
    etiqueta = f"{decision['model_name']} COMBINADO, umbral {umbral:g}"
    figura_confusion = guardar_matriz_confusion(
        y_true,
        y_pred,
        f"Test final - {etiqueta}",
        FIGURA_TEST_CONFUSION,
    )
    figura_roc = guardar_curva_roc(
        y_true,
        y_prob,
        f"Test final - {decision['model_name']} COMBINADO",
        FIGURA_TEST_ROC,
        etiqueta_adicional=etiqueta,
    )
    figura_metricas = graficar_metricas_test(
        {m: float(metricas[m]) for m in METRICAS_REPORTE},
        {m: float(metricas_referencia[m]) for m in METRICAS_REPORTE},
        umbral,
    )

    reporte = {
        "descripcion": (
            "Evaluacion del test original con el modelo definitivo COMBINADO, "
            "usando el umbral congelado a partir de validation."
        ),
        "model_name": decision["model_name"],
        "estrategia": decision["estrategia"],
        "config": dict(decision["config"]),
        "umbral": umbral,
        "umbral_referencia": UMBRAL_BASE,
        "modelo": str(RUTA_MODELO_FINAL),
        "split_manifest": str(RUTA_MANIFIESTO),
        "test": {
            "total": composicion["total"],
            "NORMAL": composicion["NORMAL"],
            "PNEUMONIA": composicion["PNEUMONIA"],
            "evaluadas": int(y_true.size),
        },
        "metricas_umbral_congelado": {m: float(metricas[m]) for m in METRICAS_REPORTE},
        "metricas_referencia_0.5": {m: float(metricas_referencia[m]) for m in METRICAS_REPORTE},
        "matriz_confusion": {
            "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
            "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
            "matriz_referencia_0.5": {
                "tn": int(((y_true == 0) & (y_pred_referencia == 0)).sum()),
                "fp": int(((y_true == 0) & (y_pred_referencia == 1)).sum()),
                "fn": int(((y_true == 1) & (y_pred_referencia == 0)).sum()),
                "tp": int(((y_true == 1) & (y_pred_referencia == 1)).sum()),
            },
        },
        "baseline_clase_mayoritaria": baseline,
        "criterio_exito": criterios,
        "figuras": {
            "matriz_confusion": str(figura_confusion),
            "curva_roc": str(figura_roc),
            "metricas": str(figura_metricas),
        },
        "decision_previa": {
            "validation_ganadora": dict(decision.get("validation_ganadora", {})),
            "criterio_umbral": decision.get("criterio_umbral"),
        },
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }
    guardar_json(reporte, RUTA_REPORTE_TEST)
    escribir_tabla_test(reporte)
    np.save(DIRECTORIO_RESULTADOS_FINAL / "test_y_true.npy", y_true)
    np.save(DIRECTORIO_RESULTADOS_FINAL / "test_y_prob.npy", y_prob)
    _marcar(
        f"test evaluado: balanced accuracy {metricas['balanced_accuracy']:.4f} "
        f"con umbral {umbral} (accuracy {metricas['accuracy']:.4f})"
    )
    return reporte


def escribir_tabla_test(reporte: dict[str, Any], ruta: Path = RUTA_METRICAS_TEST) -> Path:
    """Escribir las métricas de test en CSV."""
    filas = []
    for clave, etiqueta in (
        ("metricas_umbral_congelado", f"umbral_congelado_{reporte['umbral']}"),
        ("metricas_referencia_0.5", "umbral_0.5"),
    ):
        for metrica, valor in reporte[clave].items():
            filas.append({"umbral": etiqueta, "metrica": metrica, "valor": valor})
    asegurar_directorio(Path(ruta).parent)
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8")
    return Path(ruta)


def ejecutar_flujo_completo(recalcular_combinado: bool = False) -> dict[str, Any]:
    """Ejecutar el flujo entero: train/val, umbral, modelo definitivo y test final.

    El orden importa: el umbral se congela a partir de ``validation`` antes de que el
    modelo definitivo se reentrene y antes de leer el test.
    """
    combinado = ejecutar_combinado(recalcular=recalcular_combinado)
    decision = ajustar_umbral()
    final = entrenar_modelo_definitivo()
    test = evaluar_test()
    return {
        "combinado": combinado,
        "decision": decision,
        "modelo_definitivo": final,
        "test": test,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flujo final del proyecto de neumonía")
    parser.add_argument(
        "etapa",
        choices=["combinado", "umbral", "final", "test", "todo"],
        help="Etapa a ejecutar",
    )
    parser.add_argument("--recalcular", action="store_true", help="Rehacer la etapa aunque exista su artefacto")
    argumentos = parser.parse_args()

    if argumentos.etapa == "combinado":
        print(json.dumps(ejecutar_combinado(recalcular=argumentos.recalcular), indent=2, default=str))
    elif argumentos.etapa == "umbral":
        print(json.dumps(ajustar_umbral(), indent=2, default=str))
    elif argumentos.etapa == "final":
        print(json.dumps(entrenar_modelo_definitivo(), indent=2, default=str))
    elif argumentos.etapa == "test":
        print(json.dumps(evaluar_test(), indent=2, default=str))
    else:
        print(json.dumps(ejecutar_flujo_completo(argumentos.recalcular), indent=2, default=str))
