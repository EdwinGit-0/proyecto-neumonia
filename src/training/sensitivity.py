"""Etapa de sensibilidad de hiperparámetros previa a la selección de arquitecturas.

La etapa responde a una pregunta simple: ¿la configuración base del proyecto
(``lr=1e-4``, ``dropout=0.3``, ``3`` épocas) es la mejor de su neighbourhood para
cada arquitectura? Se evalúan tres factores de uno en uno:

- ``learning_rate``: 1e-4 (base), 3e-4, 1e-3
- ``dropout``: 0.3 (base), 0.2, 0.5
- ``epochs``: 3 (base), 5, 10

La configuración base es la referencia común de los tres grupos, de modo que se
ejecuta una sola vez por arquitectura: 7 configuraciones × 3 arquitecturas = 21
entrenamientos.

Reglas de la etapa:

- Solo intervienen ``train`` y ``val``; el split ``test`` no se carga ni se
  consulta en ningún punto (validado por :data:`SPLITS_PERMITIDOS`).
- La configuración base del proyecto no se modifica: la sensibilidad identifica
  la mejor configuración **por arquitectura**, no sustituye la configuración
  documentada del entrenamiento base.
- La selección usa el criterio jerárquico del proyecto: Balanced Accuracy >
  ROC-AUC > F1 > Accuracy, calculado exclusivamente sobre ``validation``.
- Los resultados se guardan en ``models/sensitivity_results.json``. La evaluación
  sobre ``test`` es una etapa posterior y separada, registrada en
  ``models/sensitivity_test_audit.json``; nunca forma parte de este artefacto ni
  participa en la selección.

Uso:

    python -m src.training.sensitivity
    python -m src.training.sensitivity --recalcular
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.models import Model

from src.data.datasets import construir_pipelines_datos
from src.data.splitting import cargar_manifiesto_division
from src.models.architectures import NOMBRES_MODELOS
from src.models.comparison import construir_tabla_resultados
from src.models.evaluation import calcular_matriz_confusion, calcular_reporte_metricas, seleccionar_mejor_modelo
from src.training.run_real_training import construir_pipeline_augmentation
from src.utils.paths import DIRECTORIO_DATOS, DIRECTORIO_MODELOS, RAIZ_PROYECTO, asegurar_directorio

SEMILLA = 42
TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 16

LEARNING_RATE_BASE = 1e-4
DROPOUT_BASE = 0.3
EPOCAS_BASE = 3

CRITERIO_METRICAS = ("balanced_accuracy", "roc_auc", "f1", "accuracy")
METRICAS_REPORTE = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f1",
    "roc_auc",
)

GRUPOS_HIPERPARAMETROS: dict[str, list[Any]] = {
    "learning_rate": [LEARNING_RATE_BASE, 3e-4, 1e-3],
    "dropout": [DROPOUT_BASE, 0.2, 0.5],
    "epochs": [EPOCAS_BASE, 5, 10],
}

# La etapa de sensibilidad no puede observar el test original.
SPLITS_PERMITIDOS = ("train", "val")

RUTA_MANIFIESTO = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"
RUTA_RESULTADOS = DIRECTORIO_MODELOS / "sensitivity_results.json"
RUTA_AUDITORIA_TEST = DIRECTORIO_MODELOS / "sensitivity_test_audit.json"
RUTA_RESULTADOS_ENTRENAMIENTO = DIRECTORIO_MODELOS / "model_results.json"
DIRECTORIO_CHECKPOINTS = DIRECTORIO_MODELOS / "sensitivity_checkpoints"

CLAVE_COMPARACION_BASE = "validation_comparison_configuracion_base"
CLAVE_COMPARACION = "validation_comparison"


def construir_configuraciones() -> list[dict[str, Any]]:
    """Devolver las 7 configuraciones únicas, modificando un solo factor a la vez.

    La base ``(1e-4, 0.3, 3)`` es a la vez la configuración original del proyecto
    y la referencia común de los tres grupos, por lo que se evalúa una sola vez.
    """
    return [
        {
            "id": "ref_lr1e-4_do0.3_ep3",
            "grupo": "referencia",
            "learning_rate": LEARNING_RATE_BASE,
            "dropout": DROPOUT_BASE,
            "epochs": EPOCAS_BASE,
            "es_referencia": True,
            "variante_de": ["learning_rate", "dropout", "epochs"],
        },
        {"id": "lr_3e-4", "grupo": "learning_rate", "learning_rate": 3e-4, "dropout": DROPOUT_BASE, "epochs": EPOCAS_BASE, "es_referencia": False, "variante_de": ["learning_rate"]},
        {"id": "lr_1e-3", "grupo": "learning_rate", "learning_rate": 1e-3, "dropout": DROPOUT_BASE, "epochs": EPOCAS_BASE, "es_referencia": False, "variante_de": ["learning_rate"]},
        {"id": "do_0.2", "grupo": "dropout", "learning_rate": LEARNING_RATE_BASE, "dropout": 0.2, "epochs": EPOCAS_BASE, "es_referencia": False, "variante_de": ["dropout"]},
        {"id": "do_0.5", "grupo": "dropout", "learning_rate": LEARNING_RATE_BASE, "dropout": 0.5, "epochs": EPOCAS_BASE, "es_referencia": False, "variante_de": ["dropout"]},
        {"id": "ep_5", "grupo": "epochs", "learning_rate": LEARNING_RATE_BASE, "dropout": DROPOUT_BASE, "epochs": 5, "es_referencia": False, "variante_de": ["epochs"]},
        {"id": "ep_10", "grupo": "epochs", "learning_rate": LEARNING_RATE_BASE, "dropout": DROPOUT_BASE, "epochs": 10, "es_referencia": False, "variante_de": ["epochs"]},
    ]


def construir_modelo_sensibilidad(model_name: str, dropout: float, learning_rate: float) -> Model:
    """Construir la arquitectura base del proyecto con ``dropout`` y ``lr`` variables.

    Réplica exacta de ``src/models/architectures.py`` (base ImageNet congelada y
    cabeza GAP -> Dense(128, relu) -> Dropout -> Dense(1, sigmoid) compilada con
    Adam y ``binary_crossentropy``). Solo cambia el valor de los dos
    hiperparámetros que la etapa varía.
    """
    if model_name not in NOMBRES_MODELOS:
        raise ValueError(f"Modelo no compatible: {model_name}")

    constructores = {"VGG16": VGG16, "ResNet50": ResNet50, "MobileNetV2": MobileNetV2}
    base_model = constructores[model_name](include_top=False, weights="imagenet", input_shape=(*TAMANO_IMAGEN, 3))
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def crear_pipelines_sensibilidad(splits: tuple[str, ...] = SPLITS_PERMITIDOS) -> dict[str, tf.data.Dataset]:
    """Crear los pipelines de la etapa leyendo el manifiesto original sin regenerarlo.

    Se rechaza cualquier split fuera de :data:`SPLITS_PERMITIDOS`, de modo que la
    etapa no pueda leer el test original ni por error.
    """
    no_permitidos = [split for split in splits if split not in SPLITS_PERMITIDOS]
    if no_permitidos:
        raise ValueError(
            f" splits no permitidos en la etapa de sensibilidad: {no_permitidos}. "
            f"Solo se admiten {list(SPLITS_PERMITIDOS)}."
        )

    manifiesto = cargar_manifiesto_division(RUTA_MANIFIESTO)
    disponibles = [split for split in splits if not manifiesto[manifiesto["split"] == split].empty]
    if not disponibles:
        raise ValueError(f"El manifiesto no contiene los splits solicitados: {list(splits)}")

    construidos = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        manifiesto_division=RUTA_MANIFIESTO,
    )
    pipelines = {split: construidos[split] for split in disponibles}
    if "train" in pipelines:
        pipelines["train"] = construir_pipeline_augmentation(pipelines["train"])
    return pipelines


def evaluar_dataset(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, Any]:
    """Evaluar un modelo sobre un dataset y devolver las siete métricas del proyecto."""
    etiquetas: list[np.ndarray] = []
    probabilidades: list[np.ndarray] = []
    for features, labels in dataset:
        predicciones = model.predict(features, verbose=0)
        etiquetas.append(labels.numpy().astype(int))
        probabilidades.append(predicciones.ravel())

    y_true = np.concatenate(etiquetas)
    y_prob = np.concatenate(probabilidades)
    metricas = calcular_reporte_metricas(y_true, y_prob)
    matriz = calcular_matriz_confusion(y_true, y_prob)
    metricas_texto = {metrica: float(metricas[metrica]) for metrica in METRICAS_REPORTE}
    return {
        **metricas_texto,
        "n_evaluadas": int(y_true.size),
        "n_clase_0": int((y_true == 0).sum()),
        "n_clase_1": int((y_true == 1).sum()),
        "matriz_confusion": {
            "tn": int(matriz[0, 0]),
            "fp": int(matriz[0, 1]),
            "fn": int(matriz[1, 0]),
            "tp": int(matriz[1, 1]),
        },
    }


def clave_orden(item: dict[str, Any]) -> tuple[float, ...]:
    """Devolver la clave del criterio jerárquico del proyecto sobre ``validation``."""
    validacion = item["validation"]
    return tuple(float(validacion[metrica]) for metrica in CRITERIO_METRICAS)


def ruta_checkpoint(model_name: str, config_id: str) -> Path:
    """Devolver la ruta del checkpoint de una configuración de la etapa."""
    return DIRECTORIO_CHECKPOINTS / model_name.lower() / config_id / "best_model.keras"


def sha256_archivo(ruta: Path) -> str:
    """Devolver el hash SHA-256 de un archivo."""
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def normalizar_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalizar las claves de una configuración a ``learning_rate``, ``dropout`` y ``epochs``."""
    epochs = config.get("epochs", config.get("epochs_max"))
    if epochs is None:
        raise ValueError(f"La configuración no indica el número de épocas: {config}")
    return {
        "learning_rate": float(config["learning_rate"]),
        "dropout": float(config["dropout"]),
        "epochs": int(epochs),
    }


def _validar_resultados_sin_test(resultados: list[dict[str, Any]]) -> None:
    """Comprobar que ningún resultado de la etapa contiene métricas de test."""
    for item in resultados:
        if "test" in item or "test" in item.get("validation", {}):
            raise ValueError(
                f"La etapa de sensibilidad no admite resultados de test: {item.get('model_name')}/{item.get('config_id')}"
            )


def seleccionar_mejor_por_arquitectura(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Seleccionar la mejor configuración de cada arquitectura usando solo ``validation``."""
    if not resultados:
        raise ValueError("No hay resultados de validación para seleccionar.")
    _validar_resultados_sin_test(resultados)

    seleccion: dict[str, Any] = {}
    for model_name in NOMBRES_MODELOS:
        candidatos = [item for item in resultados if item["model_name"] == model_name]
        if not candidatos:
            continue
        ordenados = sorted(candidatos, key=clave_orden, reverse=True)
        ganador = ordenados[0]
        referencia = next((item for item in candidatos if item.get("es_referencia")), None)
        seleccion[model_name] = {
            "config_id": ganador["config_id"],
            "grupo": ganador["grupo"],
            "config": dict(ganador["config"]),
            "validation": dict(ganador["validation"]),
            "delta_balanced_accuracy_vs_referencia": (
                float(ganador["validation"]["balanced_accuracy"])
                - float(referencia["validation"]["balanced_accuracy"])
                if referencia
                else None
            ),
            "referencia_original": (
                {
                    "config_id": referencia["config_id"],
                    "config": dict(referencia["config"]),
                    "validation": dict(referencia["validation"]),
                }
                if referencia
                else None
            ),
            "tabla_ordenada": [
                {
                    "config_id": item["config_id"],
                    "grupo": item["grupo"],
                    "config": dict(item["config"]),
                    **{metrica: float(item["validation"][metrica]) for metrica in METRICAS_REPORTE},
                }
                for item in ordenados
            ],
        }
    return seleccion


def seleccionar_ganador_global(seleccion_por_arquitectura: dict[str, Any]) -> dict[str, Any]:
    """Elegir la mejor configuración global aplicando el criterio de validación."""
    candidatos: list[dict[str, Any]] = []
    for model_name, info in seleccion_por_arquitectura.items():
        if model_name not in NOMBRES_MODELOS:
            continue
        candidato = {
            "model_name": model_name,
            "config_id": info["config_id"],
            "config": dict(info["config"]),
            "validation": dict(info["validation"]),
        }
        candidato.update({metrica: float(info["validation"][metrica]) for metrica in CRITERIO_METRICAS})
        candidatos.append(candidato)
    if not candidatos:
        raise ValueError("No hay configuraciones candidatas para el ganador global.")
    elegido = seleccionar_mejor_modelo(candidatos)
    metricas_texto = {metrica: float(elegido["validation"][metrica]) for metrica in CRITERIO_METRICAS}
    return {
        "model_name": elegido["model_name"],
        "config_id": elegido["config_id"],
        "config": elegido["config"],
        "criterio_seleccion": list(CRITERIO_METRICAS),
        "validation": metricas_texto,
    }


def construir_comparacion_modelos(seleccion_por_arquitectura: dict[str, Any]) -> dict[str, Any]:
    """Construir la comparación de modelos con la mejor configuración de cada arquitectura.

    La forma (``winner`` y ``results``) es la esperada por
    :func:`src.models.comparison.resumir_comparacion_modelos` y por ``neumonia evaluate``.
    """
    resultados: list[dict[str, Any]] = []
    for model_name, info in seleccion_por_arquitectura.items():
        if model_name not in NOMBRES_MODELOS:
            continue
        metricas_texto = {metrica: float(info["validation"][metrica]) for metrica in METRICAS_REPORTE}
        matriz = info["validation"].get("matriz_confusion")
        entrada: dict[str, Any] = {
            "model_name": model_name,
            "config_id": info["config_id"],
            "learning_rate": float(info["config"]["learning_rate"]),
            "dropout": float(info["config"]["dropout"]),
            "epochs": int(info["config"]["epochs"]),
            "origen": "sensibilidad",
            **metricas_texto,
        }
        if matriz is not None:
            entrada["confusion_matrix"] = [
                [int(matriz["tn"]), int(matriz["fp"])],
                [int(matriz["fn"]), int(matriz["tp"])],
            ]
        resultados.append(entrada)
    if not resultados:
        raise ValueError("No hay resultados por arquitectura para comparar.")
    ordenados = construir_tabla_resultados(resultados)
    return {"winner": seleccionar_mejor_modelo(ordenados), "results": ordenados}


def construir_payload_sensibilidad(
    resultados: list[dict[str, Any]],
    configuraciones: list[dict[str, Any]] | None = None,
    procedencia: dict[str, Any] | None = None,
    segundos_totales: float | None = None,
) -> dict[str, Any]:
    """Construir el artefacto de la etapa a partir de las métricas de ``validation``."""
    if not resultados:
        raise ValueError("No hay resultados de sensibilidad para construir el artefacto.")
    _validar_resultados_sin_test(resultados)

    configuraciones = configuraciones or construir_configuraciones()
    seleccion = seleccionar_mejor_por_arquitectura(resultados)
    ganador_global = seleccionar_ganador_global(seleccion)
    comparacion = construir_comparacion_modelos(seleccion)

    manifiesto = cargar_manifiesto_division(RUTA_MANIFIESTO)
    distribucion: dict[str, dict[str, int]] = {}
    for nombre_split in SPLITS_PERMITIDOS:
        registros = manifiesto[manifiesto["split"] == nombre_split]
        distribucion[nombre_split] = {
            "NORMAL": int((registros["label"] == "NORMAL").sum()),
            "PNEUMONIA": int((registros["label"] == "PNEUMONIA").sum()),
        }

    return {
        "descripcion": (
            "Etapa de sensibilidad de hiperparametros previa a la seleccion de arquitecturas. "
            "Un factor por vez: learning rate, dropout y numero maximo de epocas."
        ),
        "etapa": "sensibilidad",
        "split_manifest": str(RUTA_MANIFIESTO),
        "manifiesto_sha256": sha256_archivo(RUTA_MANIFIESTO),
        "seed": SEMILLA,
        "batch_size": TAMANO_LOTE,
        "tamano_imagen": list(TAMANO_IMAGEN),
        "configuracion_base": {
            "learning_rate": LEARNING_RATE_BASE,
            "dropout": DROPOUT_BASE,
            "epochs": EPOCAS_BASE,
        },
        "grupos": {clave: list(valores) for clave, valores in GRUPOS_HIPERPARAMETROS.items()},
        "criterio_seleccion": list(CRITERIO_METRICAS),
        "splits_utilizados": list(SPLITS_PERMITIDOS),
        "test_utilizado": False,
        "distribucion": distribucion,
        "entorno": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpus": [str(dispositivo) for dispositivo in tf.config.list_physical_devices("GPU")],
        },
        "procedencia": procedencia or {},
        "configuraciones": configuraciones,
        "resultados": resultados,
        "mejor_por_arquitectura": seleccion,
        "comparacion_modelos": comparacion,
        "ganador_global": ganador_global,
        "segundos_totales": segundos_totales,
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }


def guardar_resultados_sensibilidad(payload: dict[str, Any], ruta: Path = RUTA_RESULTADOS) -> Path:
    """Guardar el artefacto de la etapa de forma atómica."""
    asegurar_directorio(Path(ruta).parent)
    temporal = Path(ruta).with_suffix(".json.tmp")
    temporal.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporal.replace(Path(ruta))
    return Path(ruta)


def cargar_resultados_sensibilidad(ruta: Path = RUTA_RESULTADOS) -> dict[str, Any]:
    """Cargar el artefacto de la etapa de sensibilidad."""
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute 'neumonia sensibilidad --recalcular' para generar la etapa."
        )
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def registrar_resultados_existentes(ruta_origen: str | Path) -> dict[str, Any]:
    """Registrar en el artefacto oficial resultados de validación ya medidos.

    Permite consolidar un barrido ya ejecutado sin volver a entrenar: los
    resultados se leen de un JSON con la misma estructura y se validan antes de
    guardarse (21 Runs completados, métricas de ``validation`` sin ``test``).
    """
    origen = Path(ruta_origen)
    if not origen.exists():
        raise FileNotFoundError(f"No existe el archivo de resultados de origen: {origen}")

    datos = json.loads(origen.read_text(encoding="utf-8"))
    registros = datos.get("resultados", {})
    if not isinstance(registros, dict) or not registros:
        raise ValueError(f"El archivo de origen no contiene resultados utilizables: {origen}")

    resultados: list[dict[str, Any]] = []
    for clave, item in registros.items():
        if item.get("estado") != "completado":
            raise ValueError(f"Resultado incompleto en el origen: {clave}")
        resultados.append(
            {
                "model_name": item["modelo"],
                "config_id": item["config_id"],
                "grupo": item["grupo"],
                "config": normalizar_config(item["config"]),
                "es_referencia": bool(item.get("es_referencia", False)),
                "variante_de": list(item.get("variante_de", [])),
                "epochs_ejecutadas": int(item.get("epochs_ejecutadas", 0)),
                "early_stopping": bool(item.get("early_stopping", False)),
                "learning_rate_final": float(item.get("learning_rate_final", 0.0)),
                "segundos_entrenamiento": float(item.get("segundos_entrenamiento", 0.0)),
                "checkpoint": item.get("checkpoint"),
                "validation": dict(item["validation"]),
            }
        )

    esperados = len(construir_configuraciones()) * len(NOMBRES_MODELOS)
    if len(resultados) != esperados:
        raise ValueError(f"Se esperaban {esperados} resultados y se encontraron {len(resultados)} en {origen}")

    return construir_payload_sensibilidad(
        resultados,
        configuraciones=datos.get("configuraciones"),
        procedencia={
            "tipo": "resultados_ya_medidos",
            "archivo_origen": str(origen),
            "sha256_origen": sha256_archivo(origen),
            "nota": (
                "Barrido ejecutado con el mismo dataset, division, preprocesamiento, augmentation y semilla "
                "que el entrenamiento base. No se vuelven a entrenar los checkpoints."
            ),
        },
        segundos_totales=datos.get("segundos_totales"),
    )


def _ejecutar_configuracion(
    model_name: str,
    config: dict[str, Any],
    pipelines: dict[str, tf.data.Dataset],
) -> dict[str, Any]:
    """Entrenar una configuración de la etapa y devolver sus métricas de validación."""
    checkpoint = ruta_checkpoint(model_name, config["id"])
    asegurar_directorio(checkpoint.parent)
    if checkpoint.exists():
        checkpoint.unlink()

    print(f"[sensibilidad] entrenando {model_name}/{config['id']} "
          f"lr={config['learning_rate']} dropout={config['dropout']} epochs={config['epochs']}", flush=True)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    inicio = time.time()
    model = construir_modelo_sensibilidad(
        model_name=model_name,
        dropout=float(config["dropout"]),
        learning_rate=float(config["learning_rate"]),
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint), monitor="val_loss", mode="min", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    historial = model.fit(
        pipelines["train"],
        validation_data=pipelines["val"],
        epochs=int(config["epochs"]),
        batch_size=TAMANO_LOTE,
        callbacks=callbacks,
        verbose=2,
    )
    segundos = time.time() - inicio

    modelo_cargado = tf.keras.models.load_model(checkpoint) if checkpoint.exists() else model
    metricas = evaluar_dataset(modelo_cargado, pipelines["val"])
    learning_rate_final = float(tf.keras.backend.get_value(modelo_cargado.optimizer.learning_rate.numpy()))
    del modelo_cargado, model
    tf.keras.backend.clear_session()

    return {
        "model_name": model_name,
        "config_id": config["id"],
        "grupo": config["grupo"],
        "config": {
            "learning_rate": float(config["learning_rate"]),
            "dropout": float(config["dropout"]),
            "epochs": int(config["epochs"]),
        },
        "es_referencia": bool(config.get("es_referencia", False)),
        "variante_de": list(config.get("variante_de", [])),
        "epochs_ejecutadas": int(len(historial.history["loss"])),
        "early_stopping": bool(len(historial.history["loss"]) < int(config["epochs"])),
        "learning_rate_final": learning_rate_final,
        "segundos_entrenamiento": float(segundos),
        "checkpoint": str(checkpoint),
        "validation": metricas,
    }


def ejecutar_barrido_sensibilidad() -> dict[str, Any]:
    """Ejecutar los 21 entrenamientos de la etapa y construir el artefacto."""
    configuraciones = construir_configuraciones()
    inicio_total = time.time()
    resultados: list[dict[str, Any]] = []

    for model_name in NOMBRES_MODELOS:
        pipelines = crear_pipelines_sensibilidad()
        for config in configuraciones:
            resultados.append(_ejecutar_configuracion(model_name, config, pipelines))
        del pipelines

    return construir_payload_sensibilidad(
        resultados,
        configuraciones=configuraciones,
        procedencia={
            "tipo": "barrido_ejecutado",
            "nota": "21 entrenamientos (3 arquitecturas x 7 configuraciones) generados por la propia etapa.",
        },
        segundos_totales=float(time.time() - inicio_total),
    )


def actualizar_comparacion_modelos(payload: dict[str, Any]) -> dict[str, Any]:
    """Publicar en ``model_results.json`` la comparación basada en sensibilidad.

    La comparación histórica de la configuración base se conserva íntegra bajo
    ``validation_comparison_configuracion_base``; no se altera ningún otro
    resultado (incluido ``final_test``).
    """
    if not RUTA_RESULTADOS_ENTRENAMIENTO.exists():
        raise FileNotFoundError(f"No existe {RUTA_RESULTADOS_ENTRENAMIENTO}.")

    resultados = json.loads(RUTA_RESULTADOS_ENTRENAMIENTO.read_text(encoding="utf-8"))
    if CLAVE_COMPARACION_BASE not in resultados:
        resultados[CLAVE_COMPARACION_BASE] = resultados[CLAVE_COMPARACION]
    resultados[CLAVE_COMPARACION] = payload["comparacion_modelos"]
    resultados["sensibilidad"] = {
        "descripcion": (
            "Etapa de sensibilidad previa a la seleccion de arquitecturas. "
            "La comparacion usa la mejor configuracion por arquitectura, medida solo sobre validation."
        ),
        "results_file": str(RUTA_RESULTADOS),
        "test_file": str(RUTA_AUDITORIA_TEST),
        "criterio_seleccion": list(CRITERIO_METRICAS),
        "configuracion_base": payload["configuracion_base"],
        "grupos": payload["grupos"],
        "configuraciones_por_arquitectura": {
            model_name: info["config"] for model_name, info in payload["mejor_por_arquitectura"].items()
        },
        "ganador_global": payload["ganador_global"],
        "test_utilizado_en_sensibilidad": False,
    }

    temporal = RUTA_RESULTADOS_ENTRENAMIENTO.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(resultados, indent=2, default=str), encoding="utf-8")
    temporal.replace(RUTA_RESULTADOS_ENTRENAMIENTO)
    return resultados


def cargar_punto_de_partida_sensibilidad(ruta: Path = RUTA_RESULTADOS) -> dict[str, Any]:
    """Devolver la configuración ganadora de la sensibilidad que se debe optimizar.

    Es el punto de partida de la etapa de optimización de MobileNetV2: la
    arquitectura elegida por la sensibilidad y su mejor configuración, medidas
    solo sobre ``validation``.
    """
    payload = cargar_resultados_sensibilidad(ruta)
    ganador = payload["ganador_global"]
    punto_de_partida = {
        "model_name": ganador["model_name"],
        "config_id": ganador["config_id"],
        "config": dict(ganador["config"]),
        "validation": dict(ganador["validation"]),
        "validation_completa": dict(payload["mejor_por_arquitectura"][ganador["model_name"]]["validation"]),
        "criterio_seleccion": list(CRITERIO_METRICAS),
        "origen": str(ruta),
    }
    if punto_de_partida["model_name"] != "MobileNetV2":
        raise ValueError(
            "La optimización de MobileNetV2 solo puede aplicarse si la sensibilidad elige MobileNetV2; "
            f"la configuración seleccionada fue {punto_de_partida['model_name']}."
        )
    return punto_de_partida


def resumir_sensibilidad(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Devolver la tabla de resultados ordenada por el criterio del proyecto."""
    filas: list[dict[str, Any]] = []
    for item in payload["resultados"]:
        metricas_texto = {metrica: float(item["validation"][metrica]) for metrica in METRICAS_REPORTE}
        filas.append(
            {
                "model_name": item["model_name"],
                "config_id": item["config_id"],
                "learning_rate": float(item["config"]["learning_rate"]),
                "dropout": float(item["config"]["dropout"]),
                "epochs": int(item["config"]["epochs"]),
                **metricas_texto,
            }
        )
    return construir_tabla_resultados(filas)


def registrar_auditoria_test(ruta_origen: str | Path) -> dict[str, Any]:
    """Registrar en un artefacto separado la evaluación sobre TEST ya realizada.

    La auditoría de TEST **no** forma parte de la etapa de sensibilidad: se
    ejecuta únicamente después de cerrar la selección sobre ``validation`` y no
    interviene en ninguna decisión. Se guarda en ``models/sensitivity_test_audit.json``
    para dejar constancia del paso de TEST del modelo ganador en el flujo del
    proyecto.
    """
    origen = Path(ruta_origen)
    if not origen.exists():
        raise FileNotFoundError(f"No existe el archivo de auditoría de origen: {origen}")

    datos = json.loads(origen.read_text(encoding="utf-8"))
    payload = cargar_resultados_sensibilidad()
    ganador_global = payload["ganador_global"]

    modelos = {
        model_name: {
            "config_id": registro["config_id"],
            "config": dict(registro["config"]),
            "estado": registro.get("estado"),
            "test": registro.get("test"),
        }
        for model_name, registro in datos.get("modelos", {}).items()
    }

    auditoria = {
        "descripcion": (
            "Auditoria de TEST del modelo ganador de la sensibilidad. "
            "Etapa posterior e independiente: no forma parte de models/sensitivity_results.json "
            "y no interviene en la seleccion."
        ),
        "split_manifest": str(RUTA_MANIFIESTO),
        "n_test": int(datos.get("n_test", 0)),
        "clases_test": dict(datos.get("clases_test", {})),
        "seleccion_por_arquitectura": {
            model_name: info["config_id"] for model_name, info in payload["mejor_por_arquitectura"].items()
        },
        "ganador_global": ganador_global,
        "modelos": modelos,
        "test_del_ganador": modelos.get(ganador_global["model_name"], {}).get("test"),
        "procedencia": {
            "tipo": "auditoria_ya_medida",
            "archivo_origen": str(origen),
            "sha256_origen": sha256_archivo(origen),
        },
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }

    asegurar_directorio(RUTA_AUDITORIA_TEST.parent)
    RUTA_AUDITORIA_TEST.write_text(json.dumps(auditoria, indent=2, default=str), encoding="utf-8")
    return auditoria


def refrescar_derivados(payload: dict[str, Any]) -> dict[str, Any]:
    """Recalcular las secciones derivadas del artefacto a partir de ``resultados``.

    Así la comparación publicada y el ganador global siempre coinciden con el
    código, incluso si el artefacto se volvió a cargar sin reentrenar.
    """
    _validar_resultados_sin_test(payload["resultados"])
    payload["mejor_por_arquitectura"] = seleccionar_mejor_por_arquitectura(payload["resultados"])
    payload["ganador_global"] = seleccionar_ganador_global(payload["mejor_por_arquitectura"])
    payload["comparacion_modelos"] = construir_comparacion_modelos(payload["mejor_por_arquitectura"])
    return payload


def ejecutar_analisis_sensibilidad(recalcular: bool = False, generar_figura: bool = True) -> dict[str, Any]:
    """Ejecutar la etapa de sensibilidad: barrido completo o reutilización del artefacto.

    Guarda ``models/sensitivity_results.json``, publica la comparación de
    ``model_results.json`` y genera la figura de la etapa. No evalúa el test.
    """
    if recalcular:
        payload = ejecutar_barrido_sensibilidad()
    else:
        payload = refrescar_derivados(cargar_resultados_sensibilidad())

    guardar_resultados_sensibilidad(payload)
    actualizar_comparacion_modelos(payload)

    if generar_figura:
        from src.visualization.visualize import graficar_sensibilidad

        graficar_sensibilidad(payload)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Etapa de sensibilidad de hiperparametros (sin TEST)")
    parser.add_argument("--recalcular", action="store_true", help="Reentrenar los 21 experimentos")
    argumentos = parser.parse_args()
    artefacto = ejecutar_analisis_sensibilidad(recalcular=argumentos.recalcular)
    print(json.dumps(artefacto["comparacion_modelos"], indent=2, default=str))
    print(json.dumps(artefacto["ganador_global"], indent=2, default=str))
