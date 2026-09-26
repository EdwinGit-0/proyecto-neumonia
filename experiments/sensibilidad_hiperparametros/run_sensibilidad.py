"""Revisión experimental retrospectiva: análisis de sensibilidad de hiperparámetros.

Experimento AISLADO. No modifica el proyecto definitivo:
- no escribe en src/, models/, reports/ ni data/
- no modifica el manifiesto de división (solo lo lee)
- no toca la monografía ni los resultados existentes

Objetivo: comprobar si un análisis básico de sensibilidad (learning rate, dropout y
número máximo de épocas, variando uno por vez) habría cambiado el comportamiento o la
selección de VGG16, ResNet50 y MobileNetV2.

Reglas:
- Mismo dataset, división, preprocesamiento, augmentation, semilla (42) y arquitectura
  base congelada que el proyecto original.
- Solo TRAIN y VALIDATION participan en la comparación y en la selección.
- TEST (624 imágenes) se carga y evalúa únicamente al final, como auditoría
  retrospectiva de la configuración ya seleccionada sobre VALIDATION.

Uso:
    python -m experiments.sensibilidad_hiperparametros.run_sensibilidad
    python experiments/sensibilidad_hiperparametros/run_sensibilidad.py --modelo MobileNetV2
    python experiments/sensibilidad_hiperparametros/run_sensibilidad.py --rehacer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.models import Model

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from src.data.datasets import construir_pipelines_datos
from src.data.splitting import cargar_manifiesto_division
from src.models.evaluation import calcular_matriz_confusion, calcular_reporte_metricas
from src.training.run_real_training import construir_pipeline_augmentation
from src.utils.paths import DIRECTORIO_DATOS

# ---------------------------------------------------------------------------
# Constantes: replican exactamente las del entrenamiento original
# ---------------------------------------------------------------------------
SEMILLA = 42
TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 16
APRENDIZAJE_BASE = 1e-4
DROPOUT_BASE = 0.3
EPOCAS_BASE = 3

CRITERIO_SELECCION = ("balanced_accuracy", "roc_auc", "f1", "accuracy")
NOMBRES_MODELOS = ("VGG16", "ResNet50", "MobileNetV2")

DIRECTORIO_EXPERIMENTO = RAIZ_PROYECTO / "experiments" / "sensibilidad_hiperparametros"
DIRECTORIO_RESULTADOS = DIRECTORIO_EXPERIMENTO / "resultados"
DIRECTORIO_CHECKPOINTS = DIRECTORIO_EXPERIMENTO / "checkpoints"
RUTA_MANIFIESTO = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"
RUTA_RESULTADOS = DIRECTORIO_RESULTADOS / "sensibilidad_resultados.json"
RUTA_SELECCION = DIRECTORIO_RESULTADOS / "seleccion.json"
RUTA_AUDITORIA = DIRECTORIO_RESULTADOS / "auditoria_test.json"
RUTA_REFERENCIA_MONOGRAFIA = RAIZ_PROYECTO / "models" / "model_results.json"


def rutas_solo_lectura() -> list[Path]:
    """Rutas del proyecto que este experimento solo puede leer."""
    return [RUTA_MANIFIESTO, RUTA_REFERENCIA_MONOGRAFIA]


def verificar_aislamiento() -> None:
    """Comprobar que ninguna ruta de escritura sale del directorio del experimento."""
    permitidos = {DIRECTORIO_EXPERIMENTO.resolve()}
    for ruta in (DIRECTORIO_RESULTADOS, DIRECTORIO_CHECKPOINTS, DIRECTORIO_EXPERIMENTO / "logs"):
        if ruta.resolve() not in permitidos and DIRECTORIO_EXPERIMENTO.resolve() not in ruta.resolve().parents:
            raise RuntimeError(f"Ruta de escritura fuera del experimento: {ruta}")


def sha256_archivo(ruta: Path) -> str:
    """Devolver el hash SHA-256 de un archivo."""
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _marcar(mensaje: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Configuraciones: se modifica un solo hiperparámetro por configuración
# ---------------------------------------------------------------------------
def construir_configuraciones() -> list[dict[str, Any]]:
    """Devolver las 7 configuraciones únicas por modelo (3 grupos, sin combinarlas).

   learning rate: 1e-4, 3e-4, 1e-3   (dropout 0.3, epochs 3)
    dropout:      0.2, 0.3, 0.5       (lr 1e-4, epochs 3)
    epochs:       3, 5, 10            (lr 1e-4, dropout 0.3)

    La configuración (1e-4, 0.3, 3) es a la vez la original del proyecto y la
    referencia común de los tres grupos, por lo que se ejecuta una sola vez.
    """
    return [
        {
            "id": "ref_lr1e-4_do0.3_ep3",
            "grupo": "referencia",
            "learning_rate": 1e-4,
            "dropout": 0.3,
            "epochs": 3,
            "es_referencia": True,
            "variante_de": ["learning_rate", "dropout", "epochs"],
        },
        {"id": "lr_3e-4", "grupo": "learning_rate", "learning_rate": 3e-4, "dropout": 0.3, "epochs": 3, "es_referencia": False, "variante_de": ["learning_rate"]},
        {"id": "lr_1e-3", "grupo": "learning_rate", "learning_rate": 1e-3, "dropout": 0.3, "epochs": 3, "es_referencia": False, "variante_de": ["learning_rate"]},
        {"id": "do_0.2", "grupo": "dropout", "learning_rate": 1e-4, "dropout": 0.2, "epochs": 3, "es_referencia": False, "variante_de": ["dropout"]},
        {"id": "do_0.5", "grupo": "dropout", "learning_rate": 1e-4, "dropout": 0.5, "epochs": 3, "es_referencia": False, "variante_de": ["dropout"]},
        {"id": "ep_5", "grupo": "epochs", "learning_rate": 1e-4, "dropout": 0.3, "epochs": 5, "es_referencia": False, "variante_de": ["epochs"]},
        {"id": "ep_10", "grupo": "epochs", "learning_rate": 1e-4, "dropout": 0.3, "epochs": 10, "es_referencia": False, "variante_de": ["epochs"]},
    ]


# ---------------------------------------------------------------------------
# Modelo: réplica exacta de src/models/architectures.py con dropout y lr parametrizados
# ---------------------------------------------------------------------------
def construir_modelo_sensibilidad(model_name: str, dropout: float, learning_rate: float) -> Model:
    """Construir el modelo base del proyecto con dropout y learning rate variables.

    Réplica de src/models/architectures.py: base ImageNet congelada y cabeza
    GAP -> Dense(128, relu) -> Dropout -> Dense(1, sigmoid), compilada con Adam,
    binary_crossentropy y accuracy. No se modifica el archivo original.
    """
    if model_name not in NOMBRES_MODELOS:
        raise ValueError(f"Modelo no compatible: {model_name}")

    constructores = {"VGG16": VGG16, "ResNet50": ResNet50, "MobileNetV2": MobileNetV2}
    base_model = constructores[model_name](include_top=False, weights="imagenet", input_shape=(224, 224, 3))
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


# ---------------------------------------------------------------------------
# Pipelines y evaluación
# ---------------------------------------------------------------------------
def crear_pipelines(splits: tuple[str, ...] = ("train", "val")) -> dict[str, tf.data.Dataset]:
    """Crear los pipelines de datos leyendo el manifiesto original sin regenerarlo.

    Se construye un pipeline nuevo por configuración para que todas las
    configuraciones partan de exactamente el mismo orden de datos y del mismo
    estado de augmentation, de modo que las diferencias se deban solo al
    hiperparámetro modificado.
    """
    manifiesto = cargar_manifiesto_division(RUTA_MANIFIESTO)
    disponibles = {
        split
        for split in ("train", "val", "test")
        if not manifiesto[manifiesto["split"] == split].empty
    }
    pedidos = [split for split in splits if split in disponibles]
    if not pedidos:
        raise ValueError(f"El manifiesto no contiene los splits solicitados: {splits}")

    construidos = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        manifiesto_division=RUTA_MANIFIESTO,
    )
    pipelines = {split: construidos[split] for split in pedidos}
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
    return {
        "accuracy": float(metricas["accuracy"]),
        "precision": float(metricas["precision"]),
        "recall": float(metricas["recall"]),
        "specificity": float(metricas["specificity"]),
        "balanced_accuracy": float(metricas["balanced_accuracy"]),
        "f1": float(metricas["f1"]),
        "roc_auc": float(metricas["roc_auc"]),
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


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def cargar_resultados() -> dict[str, Any]:
    """Cargar el JSON de resultados si existe."""
    if RUTA_RESULTADOS.exists():
        return json.loads(RUTA_RESULTADOS.read_text(encoding="utf-8"))
    return {
        "descripcion": "Revisión experimental retrospectiva: sensibilidad de learning rate, dropout y épocas.",
        "advertencia": "Experimento aislado. No modifica src/, models/, reports/, data/ ni la monografía.",
        "semilla": SEMILLA,
        "tamano_imagen": list(TAMANO_IMAGEN),
        "batch_size": TAMANO_LOTE,
        "criterio_seleccion": list(CRITERIO_SELECCION),
        "entorno": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpus": [str(d) for d in tf.config.list_physical_devices("GPU")],
            "cpu": platform.processor(),
        },
        "manifiesto": {
            "ruta": str(RUTA_MANIFIESTO),
            "sha256_inicio": sha256_archivo(RUTA_MANIFIESTO),
        },
        "configuraciones": construir_configuraciones(),
        "resultados": {},
    }


def guardar_resultados(datos: dict[str, Any]) -> None:
    """Guardar el JSON de resultados de forma atómica."""
    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    temporal = RUTA_RESULTADOS.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(datos, indent=2), encoding="utf-8")
    temporal.replace(RUTA_RESULTADOS)


def ruta_checkpoint(model_name: str, config_id: str) -> Path:
    """Devolver la ruta del checkpoint de una configuración."""
    return DIRECTORIO_CHECKPOINTS / model_name.lower() / config_id / "best_model.keras"


# ---------------------------------------------------------------------------
# Ejecución de una configuración
# ---------------------------------------------------------------------------
def ejecutar_configuracion(
    model_name: str,
    config: dict[str, Any],
    registro_existente: dict[str, Any] | None,
    rehacer: bool,
) -> dict[str, Any]:
    """Entrenar una configuración y registrar sus métricas de validación.

    Si ya existe un resultado completo y no se pide rehacer, se omite el
    entrenamiento y se devuelven las métricas guardadas.
    """
    clave = f"{model_name}/{config['id']}"
    if registro_existente and not rehacer and registro_existente.get("estado") == "completado":
        _marcar(f"SALTADO {clave} (ya completado)")
        return registro_existente

    checkpoint = ruta_checkpoint(model_name, config["id"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        checkpoint.unlink()

    _marcar(f"ENTRENO {clave} lr={config['learning_rate']} dropout={config['dropout']} epochs_max={config['epochs']}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)

    inicio = time.time()
    pipelines = crear_pipelines(("train", "val"))
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

    resultado = {
        "modelo": model_name,
        "config_id": config["id"],
        "grupo": config["grupo"],
        "config": {
            "learning_rate": float(config["learning_rate"]),
            "dropout": float(config["dropout"]),
            "epochs_max": int(config["epochs"]),
        },
        "es_referencia": bool(config.get("es_referencia", False)),
        "variante_de": list(config.get("variante_de", [])),
        "epochs_ejecutadas": int(len(historial.history["loss"])),
        "early_stopping": bool(len(historial.history["loss"]) < int(config["epochs"])),
        "validation": metricas,
        "history": {
            clave: [float(v) for v in valores]
            for clave, valores in historial.history.items()
        },
        "learning_rate_final": float(float(tf.keras.backend.get_value(modelo_cargado.optimizer.learning_rate.numpy()))),
        "segundos_entrenamiento": float(segundos),
        "checkpoint": str(checkpoint),
        "estado": "completado",
        "finalizado": datetime.now(timezone.utc).isoformat(),
    }

    del modelo_cargado, model, pipelines
    tf.keras.backend.clear_session()
    _marcar(
        f"FIN {clave} bal_acc={metricas['balanced_accuracy']:.4f} roc_auc={metricas['roc_auc']:.4f} "
        f"f1={metricas['f1']:.4f} acc={metricas['accuracy']:.4f} ({segundos/60:.1f} min)"
    )
    return resultado


# ---------------------------------------------------------------------------
# Selección sobre validación
# ---------------------------------------------------------------------------
def clave_orden(item: dict[str, Any]) -> tuple[float, ...]:
    """Devolver la clave de ordenación del criterio jerárquico del proyecto."""
    validacion = item["validation"]
    return tuple(float(validacion[metrica]) for metrica in CRITERIO_SELECCION)


def seleccionar_por_validacion(resultados: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seleccionar la mejor configuración de un modelo usando solo validación.

    Criterio jerárquico del proyecto: Balanced Accuracy > ROC-AUC > F1 > Accuracy.
    """
    if not resultados:
        raise ValueError("No hay resultados de validación para seleccionar.")
    ordenados = sorted(resultados, key=clave_orden, reverse=True)
    return ordenados[0], ordenados


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def ejecutar_seleccion(datos: dict[str, Any]) -> dict[str, Any]:
    """Seleccionar la mejor configuración de cada modelo y la global, solo con validación."""
    seleccion: dict[str, Any] = {}
    for model_name in NOMBRES_MODELOS:
        resultados = [item for clave, item in datos["resultados"].items() if clave.startswith(f"{model_name}/")]
        if not resultados:
            continue
        ganador, ordenados = seleccionar_por_validacion(resultados)
        referencia = next((item for item in resultados if item["es_referencia"]), None)
        seleccion[model_name] = {
            "ganador": ganador["config_id"],
            "config": ganador["config"],
            "validation": ganador["validation"],
            "referencia_original": {
                "config_id": referencia["config_id"],
                "validation": referencia["validation"],
            } if referencia else None,
            "tabla_ordenada": [
                {
                    "config_id": item["config_id"],
                    "grupo": item["grupo"],
                    "balanced_accuracy": item["validation"]["balanced_accuracy"],
                    "roc_auc": item["validation"]["roc_auc"],
                    "f1": item["validation"]["f1"],
                    "accuracy": item["validation"]["accuracy"],
                }
                for item in ordenados
            ],
        }

    globales = [
        {
            "modelo": model_name,
            "config_id": seleccion[model_name]["ganador"],
            "validation": seleccion[model_name]["validation"],
        }
        for model_name in NOMBRES_MODELOS
        if model_name in seleccion
    ]
    if globales:
        elegido = max(globales, key=clave_orden)
        seleccion["ganador_global"] = {
            "modelo": elegido["modelo"],
            "config_id": elegido["config_id"],
            "validation_por_modelo": {
                clave: valor for clave, valor in elegido["validation"].items() if clave in CRITERIO_SELECCION
            },
        }
    return seleccion


def ejecutar_auditoria_test(datos: dict[str, Any], seleccion: dict[str, Any]) -> dict[str, Any]:
    """Evaluar sobre TEST las configuraciones ya seleccionadas en VALIDATION.

    Auditoría retrospectiva: el test no participa en ninguna decisión. Este paso solo
    se ejecuta si la selección previa está completa y guardada.
    """
    if not (RUTA_SELECCION.exists() and RUTA_RESULTADOS.exists()):
        raise RuntimeError("La selección sobre validación debe completarse antes de auditar TEST.")

    pipelines = crear_pipelines(("test",))
    manifiesto = cargar_manifiesto_division(RUTA_MANIFIESTO)
    filas_test = int((manifiesto["split"] == "test").sum())
    auditoria: dict[str, Any] = {
        "advertencia": "Auditoría retrospectiva. TEST no se usó para seleccionar ni para ajustar hiperparámetros.",
        "n_test": filas_test,
        "clases_test": {
            "NORMAL": int(((manifiesto["split"] == "test") & (manifiesto["label"] == "NORMAL")).sum()),
            "PNEUMONIA": int(((manifiesto["split"] == "test") & (manifiesto["label"] == "PNEUMONIA")).sum()),
        },
        "modelos": {},
    }

    for model_name, info in seleccion.items():
        if model_name == "ganador_global":
            continue
        config_id = info["ganador"]
        checkpoint = ruta_checkpoint(model_name, config_id)
        if not checkpoint.exists():
            auditoria["modelos"][model_name] = {
                "config_id": config_id,
                "estado": "no_ejecutado",
                "motivo": f"Checkpoint no encontrado: {checkpoint}",
            }
            _marcar(f"AUDITORIA {model_name}: checkpoint no encontrado")
            continue

        _marcar(f"AUDITORIA TEST {model_name} config={config_id}")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEMILLA)
        modelo = tf.keras.models.load_model(checkpoint)
        metricas = evaluar_dataset(modelo, pipelines["test"])
        auditoria["modelos"][model_name] = {
            "config_id": config_id,
            "config": info["config"],
            "estado": "completado",
            "test": metricas,
        }
        del modelo
        tf.keras.backend.clear_session()

    RUTA_AUDITORIA.write_text(json.dumps(auditoria, indent=2), encoding="utf-8")
    return auditoria


def main() -> int:
    """Ejecutar el barrido de sensibilidad y, al final, la auditoría sobre TEST."""
    parser = argparse.ArgumentParser(description="Sensibilidad de hiperparámetros (experimento aislado)")
    parser.add_argument("--modelo", choices=list(NOMBRES_MODELOS), help="Ejecutar solo un modelo")
    parser.add_argument("--rehacer", action="store_true", help="Reentrenar aunque exista resultado")
    parser.add_argument("--sin-test", action="store_true", help="No ejecutar la auditoría sobre TEST")
    args = parser.parse_args()

    verificar_aislamiento()
    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    DIRECTORIO_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    datos = cargar_resultados()
    hashes_inicio = {str(ruta): sha256_archivo(ruta) for ruta in rutas_solo_lectura()}

    modelos = [args.modelo] if args.modelo else list(NOMBRES_MODELOS)
    configuraciones = construir_configuraciones()
    inicio_total = time.time()

    for model_name in modelos:
        for config in configuraciones:
            clave = f"{model_name}/{config['id']}"
            datos["resultados"][clave] = ejecutar_configuracion(
                model_name=model_name,
                config=config,
                registro_existente=datos["resultados"].get(clave),
                rehacer=args.rehacer,
            )
            guardar_resultados(datos)

    if args.modelo or args.sin_test:
        _marcar("Sweep terminado (sin auditoría de TEST en esta ejecución).")
        return 0

    faltan = [
        f"{m}/{c['id']}"
        for m in NOMBRES_MODELOS
        for c in configuraciones
        if datos["resultados"].get(f"{m}/{c['id']}", {}).get("estado") != "completado"
    ]
    if faltan:
        _marcar(f"INCOMPLETO, no se selecciona ni se audita TEST. Faltan: {faltan}")
        return 1

    seleccion = ejecutar_seleccion(datos)
    RUTA_SELECCION.write_text(json.dumps(seleccion, indent=2), encoding="utf-8")
    _marcar(f"SELECCION guardada: {json.dumps(seleccion.get('ganador_global'), default=str)}")

    auditoria = ejecutar_auditoria_test(datos, seleccion)
    _marcar("AUDITORIA TEST completada")

    hashes_fin = {str(ruta): sha256_archivo(ruta) for ruta in rutas_solo_lectura()}
    datos["manifiesto"]["sha256_fin"] = hashes_fin[str(RUTA_MANIFIESTO)]
    datos["manifiesto"]["sin_modificar"] = bool(
        hashes_inicio[str(RUTA_MANIFIESTO)] == hashes_fin[str(RUTA_MANIFIESTO)]
    )
    datos["referencia_monografia_sha256"] = {
        str(RUTA_REFERENCIA_MONOGRAFIA): {
            "inicio": hashes_inicio[str(RUTA_REFERENCIA_MONOGRAFIA)],
            "fin": hashes_fin[str(RUTA_REFERENCIA_MONOGRAFIA)],
            "sin_modificar": bool(
                hashes_inicio[str(RUTA_REFERENCIA_MONOGRAFIA)] == hashes_fin[str(RUTA_REFERENCIA_MONOGRAFIA)]
            ),
        }
    }
    datos["seleccion"] = seleccion
    datos["auditoria_test"] = auditoria
    datos["segundos_totales"] = float(time.time() - inicio_total)
    datos["finalizado"] = datetime.now(timezone.utc).isoformat()
    guardar_resultados(datos)

    modificados = [ruta for ruta, valor in datos["referencia_monografia_sha256"].items() if not valor["sin_modificar"]]
    if modificados:
        _marcar(f"ATENCION: archivos del proyecto modificados: {modificados}")
    else:
        _marcar("Integridad verificada: manifiesto y resultados del proyecto sin cambios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
