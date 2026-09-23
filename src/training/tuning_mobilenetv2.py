"""Tuning de hiperparámetros y fine-tuning para MobileNetV2.

Reglas del experimento:
- Solo se utilizan TRAIN y VALIDATION para seleccionar/ajustar configuraciones.
- El TEST permanece completamente independiente y solo se evalúa al final
  la configuración ganadora.
- La configuración original (baseline) se conserva intacta en
  models/mobilenetv2 y en models/model_results.json.
- El pipeline de datos, división, etiquetas y augmentation no cambian.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from src.data.datasets import construir_pipelines_datos
from src.data.splitting import crear_manifiesto_division_estratificada
from src.models.evaluation import calcular_reporte_metricas
from src.training.run_real_training import construir_pipeline_augmentation, configurar_ejecucion, evaluar_modelo
from src.utils.paths import DIRECTORIO_DATOS, DIRECTORIO_FIGURAS, DIRECTORIO_MODELOS, RAIZ_PROYECTO, asegurar_directorio


SEMILLA = 42
TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 16
EPOCAS = 3
INDICE_FINETUNE = 116  # unfreeze desde block_13_expand (últimas ~38 capas del extractor)

CRITERIO_METRICAS = ("balanced_accuracy", "roc_auc", "f1", "accuracy")
TOLERANCIA_RECALL = 0.05


EXPERIMENTOS: list[dict[str, Any]] = [
    {"nombre": "lr_5e-5", "learning_rate": 5e-5, "dropout": 0.3, "fine_tune_from": None},
    {"nombre": "lr_3e-4", "learning_rate": 3e-4, "dropout": 0.3, "fine_tune_from": None},
    {"nombre": "dropout_0.5", "learning_rate": 1e-4, "dropout": 0.5, "fine_tune_from": None},
    {"nombre": "finetune_block13", "learning_rate": 1e-4, "dropout": 0.3, "fine_tune_from": INDICE_FINETUNE},
]


def construir_modelo_mobilenetv2_tunable(
    dropout: float = 0.3,
    learning_rate: float = 1e-4,
    fine_tune_from: int | None = None,
) -> Model:
    """Crear MobileNetV2 con la misma cabeza que el baseline pero tunable.

    Réplica fiel de src/models/architectures.py (GAP, Dense 128 relu,
    Dropout, Dense 1 sigmoid). Si fine_tune_from no es None se descongelan
    las capas del extractor desde ese índice (sin tocar las cabezas).
    """
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    if fine_tune_from is not None:
        if fine_tune_from <= 0 or fine_tune_from >= len(base_model.layers):
            raise ValueError(f"fine_tune_from inválido: {fine_tune_from}")
        for layer in base_model.layers[fine_tune_from:]:
            layer.trainable = True

    inputs = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def cargar_baseline_metricas(ruta_resultados: Path) -> dict[str, Any]:
    """Leer las métricas de validation del MobileNetV2 original desde model_results.json."""
    with ruta_resultados.open(encoding="utf-8") as file_handle:
        resultados = json.load(file_handle)
    baseline = dict(resultados["models"]["MobileNetV2"]["validation"])
    for key in CRITERIO_METRICAS:
        baseline[key] = float(baseline[key])
    return baseline


def _metricas_esenciales(etiquetas: Any, probabilidades: Any) -> dict[str, float]:
    """Calcular las métricas numéricas de un experimento."""
    reporte = calcular_reporte_metricas(
        np.asarray(etiquetas).astype(int),
        np.asarray(probabilidades).astype(float),
    )
    return {
        "accuracy": float(reporte["accuracy"]),
        "precision": float(reporte["precision"]),
        "recall": float(reporte["recall"]),
        "specificity": float(reporte["specificity"]),
        "balanced_accuracy": float(reporte["balanced_accuracy"]),
        "f1": float(reporte["f1"]),
        "roc_auc": float(reporte["roc_auc"]),
    }


def evaluar_metricas_sin_figuras(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, float]:
    """Evaluar métricas sobre un dataset sin guardar figuras ni listas largas."""
    etiquetas: list[Any] = []
    probabilidades: list[Any] = []
    for features, labels in dataset:
        predicciones = model.predict(features, verbose=0)
        etiquetas.extend(labels.numpy().astype(int).tolist())
        probabilidades.extend(predicciones.ravel().tolist())
    return _metricas_esenciales(etiquetas, probabilidades)


def entramar_experimento(experimento: dict[str, Any], train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> dict[str, Any]:
    """Entrenar una configuración y devolver sus métricas de validación.

    Si el checkpoint de la configuración ya existe (por ejemplo por una
    ejecución previa fallida), se reutiliza y se calculan las métricas sin
    volver a entrenar.
    """
    nombre = experimento["nombre"]
    directorio_experimento = DIRECTORIO_MODELOS / "mobilenetv2_tuning" / nombre
    asegurar_directorio(directorio_experimento)
    best_model_path = directorio_experimento / "best_model.keras"

    if best_model_path.exists():
        print(f"[resume] Checkpoint existente para {nombre}; se evalúa sin reentrenar.")
        model = tf.keras.models.load_model(best_model_path)
        metricas = evaluar_metricas_sin_figuras(model, val_dataset)
        metricas["training_time_seconds"] = 0.0
        metricas["epochs"] = 0
        metricas["history"] = {}
        return metricas

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    model = construir_modelo_mobilenetv2_tunable(
        dropout=experimento["dropout"],
        learning_rate=experimento["learning_rate"],
        fine_tune_from=experimento["fine_tune_from"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", mode="min", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCAS,
        batch_size=TAMANO_LOTE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start_time

    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)

    metricas = evaluar_metricas_sin_figuras(model, val_dataset)
    metricas["training_time_seconds"] = float(elapsed)
    metricas["epochs"] = int(len(history.history["loss"]))
    metricas["history"] = {
        "loss": [float(value) for value in history.history["loss"]],
        "val_loss": [float(value) for value in history.history["val_loss"]],
        "accuracy": [float(value) for value in history.history["accuracy"]],
        "val_accuracy": [float(value) for value in history.history["val_accuracy"]],
    }
    return metricas


def construir_tabla_experimentos(resultados: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordenar experimentos por el criterio del proyecto: Balanced Acc, ROC-AUC, F1, Acc."""
    return sorted(
        resultados,
        key=lambda item: (
            -float(item.get("balanced_accuracy", 0.0)),
            -float(item.get("roc_auc", 0.0)),
            -float(item.get("f1", 0.0)),
            -float(item.get("accuracy", 0.0)),
        ),
    )


def seleccionar_ganador(resultados: list[dict[str, Any]], baseline: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Seleccionar la mejor configuración con el criterio del proyecto.

    Prioridad: Balanced Accuracy, ROC-AUC, F1, Accuracy. Se aplica además la
    precaución de no sacrificar Recall: si el mejor por criterio cae más de
    TOLERANCIA_RECALL respecto al baseline, se prefiere el mejor candidato
    que no produzca esa caída de sensibilidad.
    """
    if not resultados:
        raise ValueError("No hay resultados de experimentos.")

    ordenados = construir_tabla_experimentos(resultados)
    mejor_por_criterio = ordenados[0]
    recall_baseline = float(baseline["recall"])
    recall_mejor = float(mejor_por_criterio["recall"])

    if recall_mejor >= recall_baseline - TOLERANCIA_RECALL:
        return mejor_por_criterio, None

    candidatos_sin_caida = [
        item
        for item in ordenados
        if float(item.get("recall", 0.0)) >= recall_baseline - TOLERANCIA_RECALL
    ]
    if not candidatos_sin_caida:
        return mejor_por_criterio, "Sin candidatos dentro de la tolerancia de recall; se conserva el mejor por criterio."
    alternativo = candidatos_sin_caida[0]
    motivo = (
        f"El mejor por criterio ({mejor_por_criterio['experimento']}) baja recall "
        f"a {recall_mejor:.4f} vs baseline {recall_baseline:.4f}; se elige "
        f"{alternativo['experimento']} que respeta la sensibilidad."
    )
    return alternativo, motivo


def cargar_test_baseline(ruta_resultados: Path) -> dict[str, Any]:
    """Leer las métricas de TEST del MobileNetV2 original desde model_results.json."""
    with ruta_resultados.open(encoding="utf-8") as file_handle:
        resultados = json.load(file_handle)
    test = dict(resultados["final_test"])
    return test


def entrenar_y_evaluar_tuning() -> dict[str, Any]:
    """Ejecutar todos los experimentos de tuning, seleccionar y evaluar en TEST."""
    asegurar_directorio(DIRECTORIO_MODELOS)
    asegurar_directorio(DIRECTORIO_FIGURAS)
    configurar_ejecucion()

    ruta_manifiesto = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"
    _ = crear_manifiesto_division_estratificada(DIRECTORIO_DATOS, ruta_manifiesto, random_state=SEMILLA)
    datasets = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        manifiesto_division=ruta_manifiesto,
    )
    train_dataset = construir_pipeline_augmentation(datasets["train"])
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    ruta_resultados = DIRECTORIO_MODELOS / "model_results.json"
    baseline = cargar_baseline_metricas(ruta_resultados)
    baseline["experimento"] = "baseline_original"

    experimentos: list[dict[str, Any]] = []
    for experimento in EXPERIMENTOS:
        print(f"\n===== EXPERIMENTO: {experimento['nombre']} =====")
        metricas = entramar_experimento(experimento, train_dataset, val_dataset)
        metricas["experimento"] = experimento["nombre"]
        metricas["config"] = {
            "learning_rate": float(experimento["learning_rate"]),
            "dropout": float(experimento["dropout"]),
            "fine_tune_from": experimento["fine_tune_from"],
        }
        experimentos.append(metricas)
        resumen_exp = {clave: metricas[clave] for clave in CRITERIO_METRICAS + ("recall", "specificity")}
        print(json.dumps(resumen_exp, indent=2, default=str))

    tabla = construir_tabla_experimentos(experimentos)
    ganador, motivo = seleccionar_ganador(experimentos, baseline)

    gana_tuning = (
        float(ganador["balanced_accuracy"]) > float(baseline["balanced_accuracy"])
        or (
            float(ganador["balanced_accuracy"]) == float(baseline["balanced_accuracy"])
            and float(ganador["roc_auc"]) > float(baseline["roc_auc"])
        )
    )

    if not gana_tuning:
        print("\n===== EL TUNING NO SUPERA EL BASELINE; SE CONSERVA LA CONFIGURACIÓN ORIGINAL =====")
        final_val = dict(baseline)
        final_test = cargar_test_baseline(ruta_resultados)
        mejor_modelo = "baseline_original"
        ruta_modelo_ganador = str(DIRECTORIO_MODELOS / "mobilenetv2" / "best_model.keras")
        config_ganadora = {"learning_rate": 1e-4, "dropout": 0.3, "fine_tune_from": None}
        tiempo_ganador = float(baseline.get("training_time_seconds", 0.0))
    else:
        # Reentrenar la configuración ganadora desde cero y evaluar sobre TEST (único uso del test).
        print(f"\n===== REENTRENO DE LA CONFIGURACIÓN GANADORA: {ganador['experimento']} =====")
        config_ganadora = next(item["config"] for item in experimentos if item["experimento"] == ganador["experimento"])
        directorio_ganador = DIRECTORIO_MODELOS / "mobilenetv2_ajustado"
        asegurar_directorio(directorio_ganador)
        best_model_path = directorio_ganador / "best_model.keras"
        if best_model_path.exists():
            best_model_path.unlink()

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEMILLA)
        model = construir_modelo_mobilenetv2_tunable(
            dropout=config_ganadora["dropout"],
            learning_rate=config_ganadora["learning_rate"],
            fine_tune_from=config_ganadora["fine_tune_from"],
        )
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", mode="min", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ]
        start_time = time.time()
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCAS,
            batch_size=TAMANO_LOTE,
            callbacks=callbacks,
            verbose=1,
        )
        tiempo_ganador = float(time.time() - start_time)
        model = tf.keras.models.load_model(best_model_path)

        modelo_ajustado = "MobileNetV2-ajustado"
        final_val = evaluar_modelo(model, val_dataset, modelo_ajustado, "validation")
        final_test = evaluar_modelo(model, test_dataset, modelo_ajustado, "test")
        mejor_modelo = f"ganador_{ganador['experimento']}"
        ruta_modelo_ganador = str(best_model_path)
        motivo = f"Supera la balanced_accuracy o empata con mejor ROC-AUC del baseline. {motivo or ''}".strip()

    payload = {
        "descripcion": "Tuning experimental de MobileNetV2. Solo TRAIN/VALIDATION para selección; TEST solo al final.",
        "split_manifest": str(ruta_manifiesto),
        "seed": SEMILLA,
        "epochs": EPOCAS,
        "batch_size": TAMANO_LOTE,
        "criterio_seleccion": ["balanced_accuracy", "roc_auc", "f1", "accuracy"],
        "baseline": baseline,
        "experimentos": experimentos,
        "tabla_ordenada": tabla,
        "seleccion": {
            "ganador": ganador["experimento"],
            "config": config_ganadora,
            "motivo_tolerancia_recall": motivo,
            "gana_tuning": bool(gana_tuning),
        },
        "mejor_modelo": mejor_modelo,
        "final_validation": final_val,
        "final_test": final_test,
        "training_time_seconds_ganador": float(tiempo_ganador),
        "ruta_modelo_ganador": ruta_modelo_ganador,
    }

    ruta_salida = DIRECTORIO_MODELOS / "mobilenetv2_tuning_results.json"
    ruta_salida.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    resultado = entrenar_y_evaluar_tuning()
    print(json.dumps(resultado["seleccion"], indent=2, default=str))
    print(json.dumps(resultado["final_test"], indent=2, default=str))