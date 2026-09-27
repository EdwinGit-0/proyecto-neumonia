"""Optimización de MobileNetV2 sobre el ganador del análisis de sensibilidad.

Este módulo implementa el flujo definitivo del proyecto, que sustituye al tuning
antiguo (que partía de ``lr=1e-4`` / ``lr=3e-4`` y volvía a explorar learning rate,
dropout y épocas)::

    análisis de sensibilidad (ya realizado)
        -> selección de MobileNetV2 ganador
        -> evaluación inicial en test
        -> optimización de MobileNetV2
        -> evaluación final en test

Reglas del flujo
----------------
- Las 21 pruebas de sensibilidad **no se vuelven a ejecutar**. Su artefacto
  :data:`RUTA_SENSIBILIDAD` es la entrada de esta etapa y se lee en modo lectura.
- El punto de partida lo aporta :func:`cargar_punto_de_partida_sensibilidad`: la
  arquitectura ganadora y su mejor configuración, medidas solo sobre
  ``validation``. No hay una nueva selección de arquitectura en esta etapa.
- ``learning_rate``, ``dropout`` y ``epochs`` quedan **fijos** en los valores del
  ganador: esos tres factores ya fueron evaluados en las 21 pruebas.
- Lo único que se varía, un factor por experimento, es lo que quedó pendiente:
  tratamiento del desbalance y descongelado del backbone.
- La comparación y la decisión usan **exclusivamente validation**, con la regla
  jerárquica acordada: Balanced Accuracy > ROC-AUC > F1 > Accuracy.
- ``test`` se consulta en dos momentos del flujo y solo para medir, nunca para
  decidir: la evaluación inicial (antes de optimizar) y la evaluación final (una
  sola vez, sobre el modelo ya elegido).

Uso
---
    python -m src.training.optimizacion_mobilenetv2
    python -m src.training.optimizacion_mobilenetv2 --reusar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.data.datasets import construir_pipelines_datos
from src.data.splitting import cargar_manifiesto_division
from src.training.run_real_training import construir_pipeline_augmentation, evaluar_modelo
from src.training.sensitivity import (
    CRITERIO_METRICAS,
    RUTA_AUDITORIA_TEST,
    RUTA_MANIFIESTO,
    RUTA_RESULTADOS as RUTA_SENSIBILIDAD,
    clave_orden,
    crear_pipelines_sensibilidad,
    cargar_punto_de_partida_sensibilidad,
    evaluar_dataset,
)
from src.training.tuning_desbalance_clases import calcular_pesos_clase, construir_train_oversampling
from src.training.tuning_mobilenetv2 import construir_modelo_mobilenetv2_tunable
from src.utils.paths import DIRECTORIO_DATOS, DIRECTORIO_MODELOS, RAIZ_PROYECTO, asegurar_directorio
from src.utils.reproducibility import (
    SEMILLA,
    configurar_reproducibilidad,
    descripcion_reproducibilidad,
    reiniciar_semilla,
)


TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 16

RUTA_RESULTADOS = DIRECTORIO_MODELOS / "optimization_results.json"
DIRECTORIO_CHECKPOINTS = DIRECTORIO_MODELOS / "mobilenetv2_optimizacion"
DIRECTORIO_PARCIALES = DIRECTORIO_CHECKPOINTS / "_resultados_parciales"

# Índices de capa del extractor de MobileNetV2 (154 capas) en los que empieza cada
# bloque. `construir_modelo_mobilenetv2_tunable(fine_tune_from=...)` descongela todo
# lo que va desde el índice indicado, así que cada valor define una configuración de
# fine-tuning distinta y con sentido sobre el código existente.
INDICE_FINETUNE_BLOQUE16 = 143
INDICE_FINETUNE_BLOQUE13 = 116
INDICE_FINETUNE_BLOQUE10 = 90


def construir_variantes(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Describir los experimentos de optimización pendientes.

    ``config`` es la configuración ganadora de la sensibilidad. Se devuelve
    **una variante por factor pendiente**, todas con la misma arquitectura y los
    mismos ``learning_rate`` / ``dropout`` / ``epochs`` que el punto de partida.

    No se incluyen variantes de learning rate, dropout ni epochs: esos factores
    ya fueron evaluados en las 21 pruebas de sensibilidad. Tampoco se incluyen
    variaciones de la proporción de oversampling porque
    :func:`construir_train_oversampling` solo implementa el equilibrado completo
    (NORMAL igualado a PNEUMONIA) y no admite un factor configurable.
    """
    return [
        {
            "id": "oversampling_normal",
            "grupo": "desbalance",
            "descripcion": "Oversampling de NORMAL hasta equipararlo con PNEUMONIA en train",
            "tratamiento": "oversampling",
            "fine_tune_from": None,
            "capa_descongelada": None,
        },
        {
            "id": "class_weight",
            "grupo": "desbalance",
            "descripcion": "Pesos de clase balanceados calculados sobre train",
            "tratamiento": "class_weight",
            "fine_tune_from": None,
            "capa_descongelada": None,
        },
        {
            "id": "finetune_block16",
            "grupo": "fine_tuning",
            "descripcion": "Descongelar block_16_expand (indice 143, ultimo bloque)",
            "tratamiento": "ninguno",
            "fine_tune_from": INDICE_FINETUNE_BLOQUE16,
            "capa_descongelada": "block_16_expand",
        },
        {
            "id": "finetune_block13",
            "grupo": "fine_tuning",
            "descripcion": "Descongelar block_13_expand (indice 116, bloques 13-16)",
            "tratamiento": "ninguno",
            "fine_tune_from": INDICE_FINETUNE_BLOQUE13,
            "capa_descongelada": "block_13_expand",
        },
        {
            "id": "finetune_block10",
            "grupo": "fine_tuning",
            "descripcion": "Descongelar block_10_expand (indice 90, bloques 10-16)",
            "tratamiento": "ninguno",
            "fine_tune_from": INDICE_FINETUNE_BLOQUE10,
            "capa_descongelada": "block_10_expand",
        },
    ]


def sha256_archivo(ruta: Path) -> str:
    """Devolver el SHA-256 de un archivo."""
    hash_ = hashlib.sha256()
    with ruta.open("rb") as file_handle:
        for bloque in iter(lambda: file_handle.read(1 << 20), b""):
            hash_.update(bloque)
    return hash_.hexdigest()


def descripcion_entorno() -> dict[str, Any]:
    """Describir el entorno de ejecución."""
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "seed": SEMILLA,
        "batch_size": TAMANO_LOTE,
        "image_size": list(TAMANO_IMAGEN),
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "reproducibilidad": descripcion_reproducibilidad(SEMILLA),
    }


def guardar_json(payload: dict[str, Any], ruta: Path) -> Path:
    """Escribir un artefacto JSON con indentación."""
    asegurar_directorio(ruta.parent)
    ruta.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return ruta


def ruta_relativa(ruta: Path) -> str:
    """Expresar una ruta como relativa al proyecto cuando pertenece a su árbol.

    Si la ruta queda fuera del proyecto (por ejemplo cuando los checkpoints se
    dirigen a un directorio externo en un ejercicio de diagnóstico) se devuelve
    la ruta absoluta en lugar de fallar.
    """
    try:
        return str(ruta.relative_to(RAIZ_PROYECTO))
    except ValueError:
        return str(ruta)


def ruta_checkpoint_sensibilidad(config_id: str) -> Path:
    """Devolver la ruta del checkpoint del ganador de la sensibilidad."""
    return (
        RAIZ_PROYECTO
        / "experiments"
        / "sensibilidad_hiperparametros"
        / "checkpoints"
        / "mobilenetv2"
        / config_id
        / "best_model.keras"
    )


def test_auditado_sensibilidad(model_name: str, config_id: str) -> dict[str, Any] | None:
    """Devolver el TEST ya auditado para una configuración de la sensibilidad, si existe."""
    if not RUTA_AUDITORIA_TEST.exists():
        return None
    payload = json.loads(RUTA_AUDITORIA_TEST.read_text(encoding="utf-8"))
    registro = payload.get("modelos", {}).get(model_name, {})
    if registro.get("config_id") != config_id:
        return None
    return registro.get("test")


def construir_pipeline_test() -> tf.data.Dataset:
    """Construir el pipeline de test.

    Se aísla en su propia función para dejar constancia de que el test solo se
    carga en los dos puntos del flujo que lo necesitan: la evaluación inicial y
    la evaluación final. Durante el entrenamiento y la selección no se invoca.
    """
    pipelines = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        manifiesto_division=RUTA_MANIFIESTO,
    )
    if "test" not in pipelines:
        raise ValueError("El manifiesto no contiene el split test.")
    return pipelines["test"]


def registrar_test_inicial(punto_de_partida: dict[str, Any]) -> dict[str, Any]:
    """Evaluar en test el MobileNetV2 ganador de la sensibilidad.

    Es la primera de las dos evaluaciones de test del flujo. Se ejecuta sobre el
    checkpoint que la sensibilidad ya entrenó, por lo que no vuelve a entrenar
    ninguna de las 21 pruebas. El resultado no interviene en ninguna decisión de
    la etapa de optimización.
    """
    model_name = punto_de_partida["model_name"]
    config_id = punto_de_partida["config_id"]
    checkpoint = ruta_checkpoint_sensibilidad(config_id)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint del ganador de la sensibilidad: {checkpoint}. "
            "Sin ese archivo no se puede realizar la evaluación inicial en test."
        )

    print(f"[optimizacion] test inicial sobre el checkpoint de la sensibilidad: {checkpoint}", flush=True)
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(checkpoint)
    test = evaluar_modelo(model, construir_pipeline_test(), f"{model_name}-sensibilidad", "test")
    del model
    tf.keras.backend.clear_session()

    return {
        "id": "test_inicial",
        "modelo": f"{model_name}-{config_id}",
        "descripcion": "Evaluacion inicial en test del MobileNetV2 ganador de la sensibilidad",
        "checkpoint": ruta_relativa(checkpoint),
        "checkpoint_sha256": sha256_archivo(checkpoint),
        "config": dict(punto_de_partida["config"]),
        "validation_origen": dict(punto_de_partida["validation_completa"]),
        "test": test,
        "test_auditado_sensibilidad": test_auditado_sensibilidad(model_name, config_id),
        "nota": "Medicion previa a la optimizacion; no participa en ninguna seleccion.",
    }


def construir_train_de_variante(
    variante: dict[str, Any], train_dataset: tf.data.Dataset
) -> tuple[tf.data.Dataset, dict[str, Any], dict[Any, float]]:
    """Devolver el train efectivo de una variante, su detalle y sus class_weight."""
    tratamiento = variante["tratamiento"]
    if tratamiento == "oversampling":
        oversampleado, info = construir_train_oversampling(RUTA_MANIFIESTO, TAMANO_IMAGEN, TAMANO_LOTE)
        # Se mantiene la augmentation del flujo base para que el unico factor que
        # cambia sea el oversampling: los duplicados reciben una augmentation
        # distinta en cada epoca, que es el beneficio de combinar ambas tecnicas.
        return construir_pipeline_augmentation(oversampleado), info, {}
    if tratamiento == "class_weight":
        pesos = calcular_pesos_clase(RUTA_MANIFIESTO)
        return train_dataset, {"class_weight": {str(clase): peso for clase, peso in pesos.items()}}, pesos
    return train_dataset, {}, {}


def construir_pipelines_de_la_etapa() -> dict[str, tf.data.Dataset]:
    """Construir ``train`` (con augmentation) y ``validation`` leyendo el manifiesto.

    No incluye ``test``: el test solo se carga en :func:`registrar_test_inicial` y
    :func:`registrar_test_final`.

    **Debe invocarse después de** :func:`src.utils.reproducibility.reiniciar_semilla`.
    El ``shuffle`` y las capas de augmentation capturan su semilla en el momento de
    crearse, así que construirlos antes de fijar la semilla es lo que hacía que dos
    ejecuciones del mismo código entrenaran sobre imágenes distintas.
    """
    return crear_pipelines_sensibilidad(("train", "val"))


def callbacks_de_la_etapa(ruta_checkpoint: Path) -> list[Any]:
    """Devolver los callbacks de la etapa.

    Son los mismos que usaron las 21 pruebas de sensibilidad, para que el punto de
    partida y las variantes de optimización sean comparables.
    """
    asegurar_directorio(ruta_checkpoint.parent)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ruta_checkpoint), monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]


def entrenar_variante(
    variante: dict[str, Any],
    config: dict[str, Any],
    val_dataset: tf.data.Dataset,
) -> dict[str, Any]:
    """Entrenar una variante de optimización y medirla sobre validation.

    El orden de las operaciones es lo que hace comparables las variantes:

    1. ``clear_session()`` para liberar el grafo de la variante anterior.
    2. ``reiniciar_semilla(SEMILLA)`` para fijar el estado aleatorio.
    3. Construir el pipeline de ``train`` **después** de la semilla, porque el
       ``shuffle`` y las capas de augmentation capturan su semilla al crearse.
    4. Construir el modelo, cuya inicialización también depende de esa semilla.

    Con este orden las cinco variantes ven el mismo orden de ejemplos y la misma
    secuencia de augmentation: lo único que cambia entre ellas es el factor que se
    quiere medir.
    """
    identificador = variante["id"]
    ruta_checkpoint = DIRECTORIO_CHECKPOINTS / identificador / "best_model.keras"
    asegurar_directorio(ruta_checkpoint.parent)
    if ruta_checkpoint.exists():
        ruta_checkpoint.unlink()

    print(f"[optimizacion] entrenando {identificador} ({variante['descripcion']})", flush=True)

    tf.keras.backend.clear_session()
    reiniciar_semilla(SEMILLA)
    inicio = time.time()

    pipelines = construir_pipelines_de_la_etapa()
    train_efectivo, info_train, class_weight = construir_train_de_variante(variante, pipelines["train"])
    model = construir_modelo_mobilenetv2_tunable(
        dropout=float(config["dropout"]),
        learning_rate=float(config["learning_rate"]),
        fine_tune_from=variante["fine_tune_from"],
    )
    capas_entrenables = int(sum(1 for capa in model.layers if capa.trainable))
    capas_totales = int(len(model.layers))

    historial = model.fit(
        train_efectivo,
        validation_data=val_dataset,
        epochs=int(config["epochs"]),
        batch_size=TAMANO_LOTE,
        class_weight=class_weight or None,
        callbacks=callbacks_de_la_etapa(ruta_checkpoint),
        verbose=2,
    )
    segundos = time.time() - inicio

    modelo_cargado = tf.keras.models.load_model(ruta_checkpoint) if ruta_checkpoint.exists() else model
    metricas = evaluar_dataset(modelo_cargado, val_dataset)
    learning_rate_final = float(tf.keras.backend.get_value(modelo_cargado.optimizer.learning_rate.numpy()))
    del modelo_cargado, model, train_efectivo, pipelines
    tf.keras.backend.clear_session()

    return {
        "id": identificador,
        "grupo": variante["grupo"],
        "descripcion": variante["descripcion"],
        "config": dict(config),
        "tratamiento": variante["tratamiento"],
        "fine_tune_from": variante["fine_tune_from"],
        "capa_descongelada": variante["capa_descongelada"],
        "capas_entrenables": capas_entrenables,
        "capas_totales": capas_totales,
        "train": info_train,
        "class_weight": {str(clase): peso for clase, peso in class_weight.items()} or None,
        "epochs_ejecutadas": int(len(historial.history["loss"])),
        "early_stopping": bool(len(historial.history["loss"]) < int(config["epochs"])),
        "learning_rate_final": learning_rate_final,
        "segundos_entrenamiento": float(segundos),
        "checkpoint": ruta_relativa(ruta_checkpoint),
        "history": {clave: [float(valor) for valor in valores] for clave, valores in historial.history.items()},
        "validation": metricas,
    }


def seleccionar_ganador_optimizacion(
    candidatos: list[dict[str, Any]], referencia: dict[str, Any]
) -> dict[str, Any]:
    """Elegir el modelo final de la etapa usando solo validation.

    Aplica el criterio jerarquico acordado: Balanced Accuracy, ROC-AUC, F1 y
    Accuracy, en ese orden. El ``test`` no participa. El punto de partida se
    conserva unicamente como referencia informative para saber si la etapa mejora
    o no al modelo de partida, nunca como filtro.
    """
    if not candidatos:
        raise ValueError("No hay resultados de optimizacion.")

    ordenados = sorted(candidatos, key=clave_orden, reverse=True)
    ganador = ordenados[0]

    return {
        "criterio": " > ".join(CRITERIO_METRICAS),
        "split": "validation",
        "ganador_id": ganador["id"],
        "ganador_grupo": ganador["grupo"],
        "ganador_checkpoint": ganador.get("checkpoint"),
        "ganador_validation": {metrica: float(ganador["validation"][metrica]) for metrica in CRITERIO_METRICAS},
        "supera_punto_de_partida": bool(clave_orden(ganador) > clave_orden(referencia)),
        "punto_de_partida_validation": {
            metrica: float(referencia["validation"][metrica]) for metrica in CRITERIO_METRICAS
        },
        "tabla_ordenada": [
            {
                "id": item["id"],
                "grupo": item["grupo"],
                **{metrica: float(item["validation"][metrica]) for metrica in CRITERIO_METRICAS},
            }
            for item in ordenados
        ],
    }


def registrar_test_final(
    seleccion: dict[str, Any],
    candidatos: list[dict[str, Any]],
    test_inicial: dict[str, Any],
) -> dict[str, Any]:
    """Evaluar en test el modelo elegido por la optimización, una sola vez.

    Si el modelo elegido es el propio punto de partida, no se vuelve a medir: se
    reutiliza la evaluación inicial, porque es el mismo checkpoint.
    """
    ganador_id = seleccion["ganador_id"]
    registro = next((item for item in candidatos if item["id"] == ganador_id), None)

    if registro is None:
        return {
            "id": "test_final",
            "modelo": f"MobileNetV2-{seleccion['ganador_id']}",
            "descripcion": "El modelo final es el punto de partida de la sensibilidad; no se reevalua.",
            "procedencia": {"tipo": "test_inicial_reutilizado", "motivo": "Es el mismo checkpoint."},
            "test": test_inicial["test"],
        }

    checkpoint = RAIZ_PROYECTO / registro["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"No existe el checkpoint del modelo final: {checkpoint}")

    print(f"[optimizacion] test final sobre el modelo elegido: {ganador_id}", flush=True)
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(checkpoint)
    test = evaluar_modelo(model, construir_pipeline_test(), f"MobileNetV2-{ganador_id}", "test")
    del model
    tf.keras.backend.clear_session()

    return {
        "id": "test_final",
        "modelo": f"MobileNetV2-{ganador_id}",
        "descripcion": "Evaluacion final en test del modelo elegido por la optimizacion",
        "checkpoint": ruta_relativa(checkpoint),
        "checkpoint_sha256": sha256_archivo(checkpoint),
        "validation": dict(registro["validation"]),
        "test": test,
        "nota": "Unica evaluacion de test posterior a la seleccion; el test no intervino en la decision.",
    }


def ejecutar_optimizacion(
    punto_de_partida: dict[str, Any], reusar: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Entrenar las variantes de optimización pendientes y seleccionar con validation."""
    asegurar_directorio(DIRECTORIO_PARCIALES)
    config = dict(punto_de_partida["config"])

    # El pipeline de validation se construye una sola vez y se comparte: con la
    # semilla fijada es determinista, y asi todas las variantes se miden sobre
    # exactamente los mismos batches de validation.
    reiniciar_semilla(SEMILLA)
    pipelines = construir_pipelines_de_la_etapa()
    val_dataset = pipelines["val"]
    del pipelines

    manifiesto = cargar_manifiesto_division(RUTA_MANIFIESTO)
    conteos = {
        "NORMAL_train": int(((manifiesto["split"] == "train") & (manifiesto["target"] == 0)).sum()),
        "PNEUMONIA_train": int(((manifiesto["split"] == "train") & (manifiesto["target"] == 1)).sum()),
        "n_val": int((manifiesto["split"] == "val").sum()),
    }
    print(f"[optimizacion] train/validation: {conteos}", flush=True)

    referencia = {
        "id": "punto_de_partida_sensibilidad",
        "grupo": "punto_de_partida",
        "descripcion": (
            f"MobileNetV2 ganador de la sensibilidad ({punto_de_partida['config_id']}), "
            "sin tratamiento de desbalance y con backbone congelado"
        ),
        "config": config,
        "tratamiento": "ninguno",
        "fine_tune_from": None,
        "entrenado_en_esta_etapa": False,
        "checkpoint": ruta_relativa(ruta_checkpoint_sensibilidad(punto_de_partida["config_id"])),
        "validation": dict(punto_de_partida["validation_completa"]),
    }

    candidatos: list[dict[str, Any]] = [referencia]
    for variante in construir_variantes(config):
        ruta_parcial = DIRECTORIO_PARCIALES / f"{variante['id']}.json"
        if reusar and ruta_parcial.exists():
            print(f"[optimizacion] reutilizando resultado parcial de {variante['id']}", flush=True)
            candidatos.append(json.loads(ruta_parcial.read_text(encoding="utf-8")))
            continue
        registro = entrenar_variante(variante, config, val_dataset)
        guardar_json(registro, ruta_parcial)
        candidatos.append(registro)

    seleccion = seleccionar_ganador_optimizacion(candidatos, referencia)
    return candidatos, seleccion


def ejecutar_flujo_optimizacion(reusar: bool = False) -> dict[str, Any]:
    """Ejecutar el flujo completo desde el ganador de la sensibilidad.

    Orden: selección de la sensibilidad (solo lectura) -> test inicial ->
    optimización sobre validation -> test final.
    """
    asegurar_directorio(DIRECTORIO_MODELOS)
    # Debe ser lo primero: enable_op_determinism() solo afecta a las operaciones
    # creadas despues, y el pipeline de datos captura su semilla al construirse.
    configurar_reproducibilidad(SEMILLA)

    if not RUTA_SENSIBILIDAD.exists():
        raise FileNotFoundError(
            f"No existe el artefacto de sensibilidad {RUTA_SENSIBILIDAD}. "
            "Las 21 pruebas ya fueron realizadas en el proyecto; no se vuelven a ejecutar."
        )

    print("[optimizacion] leyendo el ganador del analisis de sensibilidad (sin reentrenar)...", flush=True)
    punto_de_partida = cargar_punto_de_partida_sensibilidad()
    print(
        f"[optimizacion] punto de partida: {punto_de_partida['model_name']} {punto_de_partida['config_id']} "
        f"lr={punto_de_partida['config']['learning_rate']} dropout={punto_de_partida['config']['dropout']} "
        f"epochs={punto_de_partida['config']['epochs']} "
        f"BA={punto_de_partida['validation']['balanced_accuracy']:.6f}",
        flush=True,
    )

    test_inicial = registrar_test_inicial(punto_de_partida)
    candidatos, seleccion = ejecutar_optimizacion(punto_de_partida, reusar=reusar)
    test_final = registrar_test_final(seleccion, candidatos, test_inicial)

    payload = {
        "descripcion": (
            "Flujo definitivo del proyecto: analisis de sensibilidad (ya realizado) -> seleccion de "
            "MobileNetV2 ganador -> test inicial -> optimizacion de MobileNetV2 -> test final. "
            "No se repite ninguna de las 21 pruebas ni se vuelven a evaluar learning rate, dropout "
            "o epocas."
        ),
        "etapa": "optimizacion_posterior_a_sensibilidad",
        "ejecucion": "final_reproducible",
        "nota_ejecucion": (
            "Ejecucion final reproducible. El pipeline de datos se construye despues de fijar "
            "SEMILLA=42, el shuffle y las capas de augmentation reciben esa semilla y las "
            "operaciones de TensorFlow son deterministas, de modo que repetir el flujo produce "
            "los mismos numeros. Las ejecuciones anteriores no eran comparables entre si: el "
            "shuffle y la augmentation se creaban sin semilla, asi que cada corrida entrenaba "
            "sobre imagenes distintas. Sus artefactos se conservan en models/historico/ y en "
            "experiments/optimizacion_pendiente_mobilenetv2/."
        ),
        "resultados_anteriores_no_reproducibles": {
            "motivo": "Pipeline de tf.data sin semilla en shuffle ni en las capas de augmentation.",
            "artefacto": "models/historico/optimization_results_ALEATORIO_NO_CONTROLADO.json",
            "checkpoints": "models/historico/mobilenetv2_optimizacion_ALEATORIO_NO_CONTROLADO/",
            "experimento_historico": "experiments/optimizacion_pendiente_mobilenetv2/",
        },
        "reproducibilidad": descripcion_reproducibilidad(SEMILLA),
        "criterio_seleccion": " > ".join(CRITERIO_METRICAS),
        "test_utilizado_para_seleccion": False,
        "split_manifest": ruta_relativa(RUTA_MANIFIESTO),
        "manifiesto_sha256": sha256_archivo(RUTA_MANIFIESTO),
        "sensibilidad": {
            "artefacto": ruta_relativa(RUTA_SENSIBILIDAD),
            "sha256": sha256_archivo(RUTA_SENSIBILIDAD),
            "numero_de_pruebas": 21,
            "reutilizado_sin_reejecutar": True,
        },
        "punto_de_partida": punto_de_partida,
        "factores_fijos": {
            "learning_rate": punto_de_partida["config"]["learning_rate"],
            "dropout": punto_de_partida["config"]["dropout"],
            "epochs": punto_de_partida["config"]["epochs"],
            "motivo": "Ya evaluados en las 21 pruebas de sensibilidad.",
        },
        "aspectos_pendientes": {
            "desbalance": ["oversampling_normal", "class_weight"],
            "fine_tuning": ["finetune_block16", "finetune_block13", "finetune_block10"],
            "variaciones_de_proporcion_oversampling": (
                "No se prueban: construir_train_oversampling solo implementa el equilibrado completo."
            ),
        },
        "entorno": descripcion_entorno(),
        "resultados": candidatos,
        "seleccion": seleccion,
        "test_inicial": test_inicial,
        "test_final": test_final,
        "modelo_final": test_final["modelo"],
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }
    guardar_json(payload, RUTA_RESULTADOS)
    return payload


def resumir_optimizacion(payload: dict[str, Any]) -> str:
    """Devolver un resumen legible del flujo para la consola."""
    lineas = []
    punto = payload["punto_de_partida"]
    lineas.append(
        f"Punto de partida (sensibilidad): {punto['model_name']} {punto['config_id']} "
        f"lr={punto['config']['learning_rate']} dropout={punto['config']['dropout']} "
        f"epochs={punto['config']['epochs']} | BA validation "
        f"{punto['validation']['balanced_accuracy']:.4f}"
    )
    inicial = payload["test_inicial"]["test"]
    lineas.append(
        f"Test inicial: BA {inicial['balanced_accuracy']:.4f} | ROC-AUC {inicial['roc_auc']:.4f} | "
        f"F1 {inicial['f1']:.4f} | Accuracy {inicial['accuracy']:.4f}"
    )
    lineas.append("Optimizacion (decision sobre validation):")
    for item in payload["seleccion"]["tabla_ordenada"]:
        lineas.append(
            f"  {item['id']:<24} BA {item['balanced_accuracy']:.4f} | ROC-AUC {item['roc_auc']:.4f} | "
            f"F1 {item['f1']:.4f} | Accuracy {item['accuracy']:.4f}"
        )
    if payload["seleccion"].get("motivo"):
        lineas.append(f"  Motivo: {payload['seleccion']['motivo']}")
    final = payload["test_final"]["test"]
    lineas.append(
        f"Test final ({payload['modelo_final']}): BA {final['balanced_accuracy']:.4f} | "
        f"ROC-AUC {final['roc_auc']:.4f} | F1 {final['f1']:.4f} | Accuracy {final['accuracy']:.4f}"
    )
    return "\n".join(lineas)


def main() -> None:
    """Punto de entrada del módulo."""
    parser = argparse.ArgumentParser(
        description="Optimizacion de MobileNetV2 sobre el ganador de la sensibilidad (sin repetir las 21 pruebas)."
    )
    parser.add_argument(
        "--reusar",
        action="store_true",
        help="Reutilizar resultados parciales de variantes ya entrenadas.",
    )
    args = parser.parse_args()
    payload = ejecutar_flujo_optimizacion(reusar=args.reusar)
    print("\n" + resumir_optimizacion(payload))
    print(f"\nArtefacto: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
