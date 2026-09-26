"""Etapa de optimización pendiente, partiendo del ganador de las 21 pruebas de sensibilidad.

Contexto
--------
Las 21 pruebas de sensibilidad (7 configuraciones x 3 arquitecturas) ya evaluaron
``learning_rate`` x ``dropout`` x ``epochs``. Su ganador global es::

    MobileNetV2 + lr=0.001 + dropout=0.3 + 3 epochs  (validation BA 0.959042)

Ese ganador es el **punto de partida** de esta etapa y se reutiliza tal cual: no se
vuelve a entrenar ni a medir, porque repetirlo sería repetir una de las 21 pruebas.
El control "sin tratamiento de desbalance, backbone congelado" ya está cubierto por
esa misma configuración, de modo que tampoco se reentrena aquí.

Qué queda realmente pendiente y existe en el código
---------------------------------------------------
1. Tratamiento del desbalance (implementado en ``src/training/tuning_desbalance_clases.py``):
   - ``oversampling_normal``: equipara NORMAL hasta PNEUMONIA dentro de ``train``.
   - ``class_weight``: pesos balanceados calculados sobre ``train``.
   El constructor ``construir_train_oversampling`` solo implementa el equilibrado
   completo (factor 1.0x sobre NORMAL). No admite un factor configurable, así que
   **no hay variaciones de proporción que probar sin inventar código nuevo**.
2. Fine-tuning (implementado en ``src/training/tuning_mobilenetv2.py`` mediante
   ``construir_modelo_mobilenetv2_tunable(fine_tune_from=...)``): el parámetro es un
   índice de capa del extractor de MobileNetV2 (154 capas) y descongela todo lo que
   va desde ese índice. Los índices-significativo son los inicios de bloque
   verificados empíricamente: 143 -> ``block_16_expand`` (11 capas), 116 ->
   ``block_13_expand`` (38 capas, el que ya usa el tuning antiguo) y 90 ->
   ``block_10_expand`` (64 capas).

Reglas de la etapa
------------------
- Solo se varía **un factor** por experimento respecto al punto de partida.
- ``learning_rate``, ``dropout`` y ``epochs`` quedan fijos en 0.001 / 0.3 / 3.
- La comparación y la selección usan **exclusivamente validation**, con la regla
  jerárquica del proyecto: Balanced Accuracy > ROC-AUC > F1 > Accuracy.
- ``test`` no se carga durante la selección. Solo se consulta una vez, al final,
  mediante ``--evaluar-test``, que exige que la selección ya esté written en disco.
- El manifiesto, los 21 resultados de sensibilidad y los artefactos del proyecto no
  se modifican.

Uso
---
    python -m experiments.optimizacion_pendiente_mobilenetv2.run_optimizacion
    python -m experiments.optimizacion_pendiente_mobilenetv2.run_optimizacion --reusar
    python -m experiments.optimizacion_pendiente_mobilenetv2.run_optimizacion --evaluar-test
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

from src.data.splitting import cargar_manifiesto_division
from src.training.run_real_training import construir_pipeline_augmentation
from src.training.sensitivity import (
    CRITERIO_METRICAS,
    clave_orden,
    crear_pipelines_sensibilidad,
    evaluar_dataset,
)
from src.training.tuning_desbalance_clases import calcular_pesos_clase, construir_train_oversampling
from src.training.tuning_mobilenetv2 import construir_modelo_mobilenetv2_tunable
from src.utils.paths import DIRECTORIO_DATOS, RAIZ_PROYECTO, asegurar_directorio


SEMILLA = 42
TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 16

# Punto de partida: ganador de las 21 pruebas de sensibilidad. Inmutable en esta etapa.
LR_BASE = 0.001
DROPOUT_BASE = 0.3
EPOCAS_BASE = 3
CONFIG_ID_REFERENCIA = "MobileNetV2/lr_1e-3"

RUTA_MANIFIESTO = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"
RUTA_SENSIBILIDAD = (
    RAIZ_PROYECTO / "experiments" / "sensibilidad_hiperparametros" / "resultados" / "sensibilidad_resultados.json"
)
DIRECTORIO_RAIZ = RAIZ_PROYECTO / "experiments" / "optimizacion_pendiente_mobilenetv2"
DIRECTORIO_CHECKPOINTS = DIRECTORIO_RAIZ / "checkpoints"
DIRECTORIO_RESULTADOS = DIRECTORIO_RAIZ / "resultados"
DIRECTORIO_PARCIALES = DIRECTORIO_RESULTADOS / "parciales"
RUTA_RESULTADOS = DIRECTORIO_RESULTADOS / "optimizacion_pendiente.json"


VARIANTES: list[dict[str, Any]] = [
    {
        "id": "oversampling_normal",
        "grupo": "desbalance",
        "descripcion": "Oversampling de NORMAL hasta equiparar con PNEUMONIA en train",
        "tratamiento": "oversampling",
        "fine_tune_from": None,
        "indice_capa": None,
    },
    {
        "id": "class_weight",
        "grupo": "desbalance",
        "descripcion": "Pesos de clase balanceados calculados sobre train",
        "tratamiento": "class_weight",
        "fine_tune_from": None,
        "indice_capa": None,
    },
    {
        "id": "finetune_block16",
        "grupo": "fine_tuning",
        "descripcion": "Descongelar block_16_expand (indice 143, 11 capas del extractor)",
        "tratamiento": "ninguno",
        "fine_tune_from": 143,
        "indice_capa": "block_16_expand",
    },
    {
        "id": "finetune_block13",
        "grupo": "fine_tuning",
        "descripcion": "Descongelar block_13_expand (indice 116, 38 capas, el del tuning antiguo)",
        "tratamiento": "ninguno",
        "fine_tune_from": 116,
        "indice_capa": "block_13_expand",
    },
    {
        "id": "finetune_block10",
        "grupo": "fine_tuning",
        "descripcion": "Descongelar block_10_expand (indice 90, 64 capas del extractor)",
        "tratamiento": "ninguno",
        "fine_tune_from": 90,
        "indice_capa": "block_10_expand",
    },
]


def sha256_archivo(ruta: Path) -> str:
    """Devolver el SHA-256 de un archivo."""
    hash_ = hashlib.sha256()
    with ruta.open("rb") as file_handle:
        for bloque in iter(lambda: file_handle.read(1 << 20), b""):
            hash_.update(bloque)
    return hash_.hexdigest()


def descripcion_arquitectura() -> dict[str, Any]:
    """Describir el entorno de ejecución."""
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "semilla": SEMILLA,
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }


def cargar_referencia() -> dict[str, Any]:
    """Recuperar el ganador de las 21 pruebas sin volver a entrenarlo ni reevaluarlo."""
    if not RUTA_SENSIBILIDAD.exists():
        raise FileNotFoundError(f"No existe el artefacto de sensibilidad: {RUTA_SENSIBILIDAD}")

    payload = json.loads(RUTA_SENSIBILIDAD.read_text(encoding="utf-8"))
    registro = payload["resultados"][CONFIG_ID_REFERENCIA]
    if registro["config"]["learning_rate"] != LR_BASE or registro["config"]["dropout"] != DROPOUT_BASE:
        raise ValueError("La referencia registrada no coincide con el punto de partida declarado.")

    test_auditado = payload.get("auditoria_test", {}).get("modelos", {}).get("MobileNetV2", {}).get("test")

    return {
        "id": "referencia_sensibilidad_lr1e-3",
        "grupo": "punto_de_partida",
        "descripcion": "Ganador de las 21 pruebas: sin tratamiento de desbalance y backbone congelado",
        "config": {
            "learning_rate": LR_BASE,
            "dropout": DROPOUT_BASE,
            "epochs": EPOCAS_BASE,
        },
        "tratamiento": "ninguno",
        "fine_tune_from": None,
        "indice_capa": None,
        "entrenado_en_esta_etapa": False,
        "procedencia": {
            "artefacto": str(RUTA_SENSIBILIDAD.relative_to(RAIZ_PROYECTO)),
            "config_id": CONFIG_ID_REFERENCIA,
            "sha256": sha256_archivo(RUTA_SENSIBILIDAD),
            "nota": "Se reutiliza tal cual; repetirla seria repetir una de las 21 pruebas.",
        },
        "validation": dict(registro["validation"]),
        "history": dict(registro.get("history", {})),
        "epochs_ejecutadas": int(registro.get("epochs_ejecutadas", EPOCAS_BASE)),
        "early_stopping": bool(registro.get("early_stopping", False)),
        "learning_rate_final": float(registro.get("learning_rate_final", LR_BASE)),
        "segundos_entrenamiento": float(registro.get("segundos_entrenamiento", 0.0)),
        "checkpoint": registro.get("checkpoint"),
        "capas_entrenables": None,
        "capas_totales": None,
        "test_registrado_auditoria": test_auditado,
    }


def construir_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, dict[str, Any]]:
    """Crear train (con augmentation) y validation. Test no se construye aquí."""
    pipelines = crear_pipelines_sensibilidad(("train", "val"))
    train_dataset = pipelines["train"]
    val_dataset = pipelines["val"]

    manifiesto = cargar_manifiesto_division(RUTA_MANIFIESTO)
    conteos = {
        "NORMAL_train": int(((manifiesto["split"] == "train") & (manifiesto["target"] == 0)).sum()),
        "PNEUMONIA_train": int(((manifiesto["split"] == "train") & (manifiesto["target"] == 1)).sum()),
        "n_val": int((manifiesto["split"] == "val").sum()),
    }
    return train_dataset, val_dataset, conteos


def construir_train_para_variante(
    variante: dict[str, Any], train_dataset: tf.data.Dataset
) -> tuple[tf.data.Dataset, dict[str, Any], dict[Any, float]]:
    """Devolver el train efectivo, su conteo y los class_weight de la variante."""
    tratamiento = variante["tratamiento"]
    if tratamiento == "oversampling":
        oversampleado, info = construir_train_oversampling(RUTA_MANIFIESTO, TAMANO_IMAGEN, TAMANO_LOTE)
        # Se conserva la augmentation del flujo base para que el único factor que
        # cambia sea el oversampling (los duplicados reciben augmentation distinta
        # en cada epoca, que es justamente el beneficio de combinar ambas).
        return construir_pipeline_augmentation(oversampleado), info, {}
    if tratamiento == "class_weight":
        pesos = calcular_pesos_clase(RUTA_MANIFIESTO)
        return train_dataset, {"class_weight": {str(k): v for k, v in pesos.items()}}, pesos
    return train_dataset, {}, {}


def callbacks_de_la_etapa(ruta_checkpoint: Path) -> list[Any]:
    """Replica exacta de los callbacks usados en las 21 pruebas de sensibilidad."""
    asegurar_directorio(ruta_checkpoint.parent)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ruta_checkpoint), monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]


def _evaluar(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Calcular métricas y matriz de confusión a partir de etiquetas y probabilidades."""
    from src.models.evaluation import calcular_matriz_confusion, calcular_reporte_metricas

    reporte = calcular_reporte_metricas(y_true, y_prob)
    matriz = calcular_matriz_confusion(y_true, y_prob)
    metricas = {metrica: float(reporte[metrica]) for metrica in (
        "accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1", "roc_auc"
    )}
    return {
        **metricas,
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


def entrenar_variante(
    variante: dict[str, Any],
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
) -> dict[str, Any]:
    """Entrenar una variante y devolver su métricas de validación."""
    identificador = variante["id"]
    ruta_checkpoint = DIRECTORIO_CHECKPOINTS / identificador / "best_model.keras"
    if ruta_checkpoint.exists():
        ruta_checkpoint.unlink()

    print(
        f"[optimizacion] entrenando {identificador} "
        f"tratamiento={variante['tratamiento']} fine_tune_from={variante['fine_tune_from']} "
        f"(lr={LR_BASE} dropout={DROPOUT_BASE} epochs={EPOCAS_BASE})",
        flush=True,
    )

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    inicio = time.time()

    train_efectivo, info_train, class_weight = construir_train_para_variante(variante, train_dataset)
    model = construir_modelo_mobilenetv2_tunable(
        dropout=DROPOUT_BASE,
        learning_rate=LR_BASE,
        fine_tune_from=variante["fine_tune_from"],
    )
    capas_entrenables = int(sum(1 for capa in model.layers if capa.trainable))
    capas_totales = int(len(model.layers))

    historial = model.fit(
        train_efectivo,
        validation_data=val_dataset,
        epochs=EPOCAS_BASE,
        batch_size=TAMANO_LOTE,
        class_weight=class_weight or None,
        callbacks=callbacks_de_la_etapa(ruta_checkpoint),
        verbose=2,
    )
    segundos = time.time() - inicio

    modelo_cargado = tf.keras.models.load_model(ruta_checkpoint) if ruta_checkpoint.exists() else model
    metricas_val = evaluar_dataset(modelo_cargado, val_dataset)
    learning_rate_final = float(tf.keras.backend.get_value(modelo_cargado.optimizer.learning_rate.numpy()))
    del modelo_cargado, model, train_efectivo
    tf.keras.backend.clear_session()

    return {
        "id": identificador,
        "grupo": variante["grupo"],
        "descripcion": variante["descripcion"],
        "config": {"learning_rate": LR_BASE, "dropout": DROPOUT_BASE, "epochs": EPOCAS_BASE},
        "tratamiento": variante["tratamiento"],
        "fine_tune_from": variante["fine_tune_from"],
        "indice_capa": variante["indice_capa"],
        "entrenado_en_esta_etapa": True,
        "procedencia": {"tipo": "entrenado_en_esta_etapa", "nota": "Sin acceso a test."},
        "train": info_train,
        "class_weight": {str(k): v for k, v in class_weight.items()} or None,
        "capas_entrenables": capas_entrenables,
        "capas_totales": capas_totales,
        "epochs_ejecutadas": int(len(historial.history["loss"])),
        "early_stopping": bool(len(historial.history["loss"]) < EPOCAS_BASE),
        "learning_rate_final": learning_rate_final,
        "segundos_entrenamiento": float(segundos),
        "checkpoint": str(ruta_checkpoint.relative_to(RAIZ_PROYECTO)),
        "history": {clave: [float(v) for v in valores] for clave, valores in historial.history.items()},
        "validation": metricas_val,
    }


def seleccionar_ganador(candidatos: list[dict[str, Any]]) -> dict[str, Any]:
    """Seleccionar por validation con la regla jerarquica del proyecto."""
    ordenados = sorted(candidatos, key=lambda item: clave_orden(item), reverse=True)
    ganador = ordenados[0]
    referencia = next((item for item in candidatos if item["grupo"] == "punto_de_partida"), None)
    return {
        "criterio": " > ".join(CRITERIO_METRICAS),
        "split": "validation",
        "ganador_id": ganador["id"],
        "ganador_grupo": ganador["grupo"],
        "ganador_validation": {metrica: ganador["validation"][metrica] for metrica in CRITERIO_METRICAS},
        "supera_referencia": (
            clave_orden(ganador) > clave_orden(referencia) if referencia is not None else None
        ),
        "tabla_ordenada": [
            {
                "id": item["id"],
                "grupo": item["grupo"],
                **{metrica: item["validation"][metrica] for metrica in CRITERIO_METRICAS},
            }
            for item in ordenados
        ],
    }


def guardar_json(payload: dict[str, Any], ruta: Path) -> None:
    """Escribir un JSON con indentación y registrar su SHA-256."""
    asegurar_directorio(ruta.parent)
    ruta.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def ejecutar(reusar: bool) -> dict[str, Any]:
    """Ejecutar la seleccion sobre validation, sin tocar test."""
    asegurar_directorio(DIRECTORIO_PARCIALES)

    referencia = cargar_referencia()
    print(f"[optimizacion] punto de partida (reutilizado): {referencia['id']} "
          f"BA={referencia['validation']['balanced_accuracy']:.6f}", flush=True)

    train_dataset, val_dataset, conteos = construir_datasets()
    print(f"[optimizacion] train: {conteos}", flush=True)

    resultados: list[dict[str, Any]] = [referencia]
    for variante in VARIANTES:
        ruta_parcial = DIRECTORIO_PARCIALES / f"{variante['id']}.json"
        if reusar and ruta_parcial.exists():
            print(f"[optimizacion] reutilizando parcial {variante['id']}", flush=True)
            resultados.append(json.loads(ruta_parcial.read_text(encoding="utf-8")))
            continue
        registro = entrenar_variante(variante, train_dataset, val_dataset)
        guardar_json(registro, ruta_parcial)
        resultados.append(registro)

    seleccion = seleccionar_ganador(resultados)
    payload = {
        "descripcion": (
            "Etapa de optimizacion pendiente ejecutada sobre el ganador de las 21 pruebas de "
            "sensibilidad. No se repiten learning_rate / dropout / epochs: solo se varia el "
            "tratamiento del desbalance o el descongelado del backbone, un factor por experimento."
        ),
        "advertencia": (
            "La seleccion se realizo exclusivamente sobre validation. El campo 'test' se anade "
            "solo tras cerrar la seleccion, mediante --evaluar-test."
        ),
        "punto_de_partida": {
            "origen": "21 pruebas de sensibilidad (7 configuraciones x 3 arquitecturas)",
            "modelo": "MobileNetV2",
            "config": {"learning_rate": LR_BASE, "dropout": DROPOUT_BASE, "epochs": EPOCAS_BASE},
            "validation_balanced_accuracy": referencia["validation"]["balanced_accuracy"],
            "nota": "Se reutiliza el resultado registrado; no se reentrena.",
        },
        "aspectos_pendientes_considerados": {
            "desbalance": ["oversampling_normal", "class_weight"],
            "fine_tuning": ["finetune_block16", "finetune_block13", "finetune_block10"],
            "variaciones_proporcion_oversampling": (
                "No se prueban: construir_train_oversampling solo implementa el equilibrado completo "
                "(factor 1.0x) y no admite un factor configurable."
            ),
        },
        "entorno": descripcion_arquitectura(),
        "split_manifest": str(RUTA_MANIFIESTO.relative_to(RAIZ_PROYECTO)),
        "manifiesto_sha256": sha256_archivo(RUTA_MANIFIESTO),
        "criterio_seleccion": " > ".join(CRITERIO_METRICAS),
        "conteos_train_val": conteos,
        "resultados": resultados,
        "seleccion": seleccion,
        "test": None,
        "test_registrado_auditoria_referencia": referencia.get("test_registrado_auditoria"),
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }
    guardar_json(payload, RUTA_RESULTADOS)
    return payload


def evaluar_test_ganador() -> dict[str, Any]:
    """Evaluar en test el ganador ya seleccionado. Unico punto donde se lee test."""
    if not RUTA_RESULTADOS.exists():
        raise FileNotFoundError(
            "No existe la seleccion. Ejecuta primero la etapa sin --evaluar-test."
        )
    payload = json.loads(RUTA_RESULTADOS.read_text(encoding="utf-8"))
    if payload.get("test") is not None:
        print("[optimizacion] el test ya fue evaluado; no se repite.")
        return payload

    id_ganador = payload["seleccion"]["ganador_id"]
    registro = next(item for item in payload["resultados"] if item["id"] == id_ganador)

    if registro["grupo"] == "punto_de_partida":
        test = registro.get("test_registrado_auditoria")
        if test is None:
            raise ValueError("La referencia no tiene test auditado; no se puede cerrar la etapa.")
        procedencia = {
            "tipo": "auditoria_de_sensibilidad",
            "nota": (
                "El ganador es el propio punto de partida y su test ya fue medido en la "
                "auditoria de sensibilidad sobre el mismo checkpoint. Se reutiliza; no se reevalua."
            ),
            "checkpoint": registro.get("checkpoint"),
        }
        print(f"[optimizacion] test del ganador (referencia) reutilizado de la auditoria: "
              f"BA={test['balanced_accuracy']:.6f}", flush=True)
    else:
        from src.data.datasets import construir_pipelines_datos

        print(f"[optimizacion] evaluando test del ganador: {id_ganador}", flush=True)
        pipelines = construir_pipelines_datos(
            DIRECTORIO_DATOS,
            image_size=TAMANO_IMAGEN,
            batch_size=TAMANO_LOTE,
            manifiesto_division=RUTA_MANIFIESTO,
        )
        model = tf.keras.models.load_model(RAIZ_PROYECTO / registro["checkpoint"])
        test = _evaluar(*_predicciones(model, pipelines["test"]))
        procedencia = {"tipo": "evaluado_en_esta_etapa", "checkpoint": registro["checkpoint"]}
        del model
        tf.keras.backend.clear_session()

    payload["test"] = {
        "id_ganador": id_ganador,
        "grupo_ganador": registro["grupo"],
        "test": test,
        "procedencia": procedencia,
        "advertencia": "Test se consulto una sola vez, tras cerrar la seleccion sobre validation.",
    }
    payload["finalizado_test"] = datetime.now(timezone.utc).isoformat()
    guardar_json(payload, RUTA_RESULTADOS)
    return payload


def _predicciones(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Devolver etiquetas y probabilidades de un dataset."""
    etiquetas: list[np.ndarray] = []
    probabilidades: list[np.ndarray] = []
    for features, labels in dataset:
        etiquetas.append(labels.numpy().astype(int))
        probabilidades.append(model.predict(features, verbose=0).ravel())
    return np.concatenate(etiquetas), np.concatenate(probabilidades)


def imprimir_tabla(payload: dict[str, Any]) -> None:
    """Imprimir la tabla de resultados solicitada."""
    print("\n=== RESULTADOS (selection sobre validation) ===")
    header = f"{'Experimento':<28}{'BA':>10}{'ROC-AUC':>10}{'F1':>10}{'Acc':>10}"
    print(header)
    print("-" * len(header))
    for item in payload["seleccion"]["tabla_ordenada"]:
        print(
            f"{item['id']:<28}{item['balanced_accuracy']:>10.6f}"
            f"{item['roc_auc']:>10.6f}{item['f1']:>10.6f}{item['accuracy']:>10.6f}"
        )
    print(f"\nGanador: {payload['seleccion']['ganador_id']}")
    if payload.get("test"):
        test = payload["test"]["test"]
        print(
            f"TEST final ({payload['test']['id_ganador']}): BA={test['balanced_accuracy']:.6f} "
            f"ROC-AUC={test['roc_auc']:.6f} F1={test['f1']:.6f} Acc={test['accuracy']:.6f}"
        )


def main() -> None:
    """Punto de entrada."""
    parser = argparse.ArgumentParser(description="Optimizacion pendiente desde el ganador de sensibilidad.")
    parser.add_argument("--reusar", action="store_true", help="Reutilizar resultados parciales ya guardados.")
    parser.add_argument("--evaluar-test", action="store_true", help="Evaluar en test al ganador ya seleccionado.")
    args = parser.parse_args()

    if args.evaluar_test:
        payload = evaluar_test_ganador()
    else:
        payload = ejecutar(reusar=args.reusar)
    imprimir_tabla(payload)


if __name__ == "__main__":
    main()
