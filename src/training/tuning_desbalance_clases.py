"""Tratamiento del desbalance de clases sobre el MobileNetV2 ajustado (lr=3e-4).

Dos experimentos independientes (no combinados):
- class_weight: pesos por clase aplicados únicamente durante el entrenamiento.
- oversampling: duplicación de la clase minoritaria NORMAL solo dentro de TRAIN.

Reglas:
- Se mantiene el MobileNetV2 ajustado (lr=3e-4) como baseline de comparación.
- VALIDATION y TEST no se modifican ni se usan para las decisiones de selección.
- La selección usa el criterio del proyecto: Balanced Acc -> ROC-AUC -> F1 -> Acc.
- Los experimentos previos (tuning base) no se modifican.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data.datasets import construir_pipelines_datos
from src.data.preprocessing import cargar_imagen, imagen_a_arreglo
from src.data.splitting import cargar_manifiesto_division, crear_manifiesto_division_estratificada
from src.training.run_real_training import construir_pipeline_augmentation, configurar_ejecucion, evaluar_modelo
from src.training.tuning_mobilenetv2 import (
    CRITERIO_METRICAS,
    EPOCAS,
    SEMILLA,
    TAMANO_IMAGEN,
    TAMANO_LOTE,
    construir_modelo_mobilenetv2_tunable,
    construir_tabla_experimentos,
    evaluar_metricas_sin_figuras,
)
from src.utils.paths import DIRECTORIO_DATOS, DIRECTORIO_FIGURAS, DIRECTORIO_MODELOS, RAIZ_PROYECTO, asegurar_directorio


CONFIG_AJUSTADO = {
    "learning_rate": 3e-4,
    "dropout": 0.3,
    "fine_tune_from": None,
    "epochs": EPOCAS,
    "batch_size": TAMANO_LOTE,
    "seed": SEMILLA,
}
NOMBRE_AJUSTADO = "mobileNetV2_ajustado_lr3e-4"
EXPERIMENTOS_DESBALANCE = ("class_weight", "oversampling")


def construir_dataset_tensor(df: pd.DataFrame, image_size: tuple[int, int], batch_size: int = 32) -> tf.data.Dataset:
    """Replica exacta del constructor interna de datasets (misma shuffle/batch/prefetch)."""
    def generator():
        for _, row in df.iterrows():
            image = cargar_imagen(row["path"])
            feature = imagen_a_arreglo(image, image_size)
            label = float(row["target"])
            yield feature, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    return dataset.shuffle(buffer_size=max(1000, len(df))).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def construir_train_oversampling(ruta_manifiesto: Path, image_size: tuple[int, int], batch_size: int) -> tf.data.Dataset:
    """Aumentar la clase minoritaria NORMAL dentro de TRAIN hasta equiparar recuentos."""
    dataframe = cargar_manifiesto_division(ruta_manifiesto)
    df_train = dataframe[dataframe["split"] == "train"].copy()
    df_normal = df_train[df_train["target"] == 0]
    df_pneumonia = df_train[df_train["target"] == 1]

    faltantes = len(df_pneumonia) - len(df_normal)
    df_normal_extra = df_normal.sample(n=faltantes, replace=True, random_state=SEMILLA)
    df_train_oversampled = pd.concat([df_train, df_normal_extra], ignore_index=True)
    df_train_oversampled = df_train_oversampled.sample(frac=1.0, random_state=SEMILLA).reset_index(drop=True)

    return construir_dataset_tensor(df_train_oversampled, image_size=image_size, batch_size=batch_size), {
        "NORMAL_original": int(len(df_normal)),
        "PNEUMONIA_original": int(len(df_pneumonia)),
        "NORMAL_oversampled": int(df_train_oversampled["target"].value_counts().get(0)),
        "total_train_oversampled": int(len(df_train_oversampled)),
        "duplicadas_NORMAL": int(faltantes),
    }


def calcular_pesos_clase(ruta_manifiesto: Path) -> dict[str, float]:
    """Pesos balanceados por clase calculados sobre TRAIN."""
    dataframe = cargar_manifiesto_division(ruta_manifiesto)
    y_train = (dataframe[dataframe["split"] == "train"]["target"]).to_numpy().astype(int)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    return {int(clase): float(peso) for clase, peso in zip([0, 1], weights)}


def entrenar_variante_desbalance(
    nombre: str,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    use_class_weight: bool = False,
    pesos_clase: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Entrenar una variante del MobileNetV2 ajustado y devolver sus métricas de validación."""
    directorio_variante = DIRECTORIO_MODELOS / "mobilenetv2_tuning" / nombre
    asegurar_directorio(directorio_variante)
    best_model_path = directorio_variante / "best_model.keras"

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
        dropout=CONFIG_AJUSTADO["dropout"],
        learning_rate=CONFIG_AJUSTADO["learning_rate"],
        fine_tune_from=CONFIG_AJUSTADO["fine_tune_from"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", mode="min", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    fit_kwargs: dict[str, Any] = {
        "validation_data": val_dataset,
        "epochs": CONFIG_AJUSTADO["epochs"],
        "batch_size": CONFIG_AJUSTADO["batch_size"],
        "callbacks": callbacks,
        "verbose": 1,
    }
    if use_class_weight:
        fit_kwargs["class_weight"] = pesos_clase

    start_time = time.time()
    history = model.fit(train_dataset, **fit_kwargs)
    elapsed = time.time() - start_time

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


def evaluar_modelo_ajustado_actual(val_dataset: tf.data.Dataset) -> dict[str, Any]:
    """Reevaluar el MobileNetV2 ajustado guardado (lr=3e-4) sobre VALIDATION."""
    ruta_modelo = DIRECTORIO_MODELOS / "mobilenetv2_ajustado" / "best_model.keras"
    if not ruta_modelo.exists():
        raise FileNotFoundError("No existe el modelo ajustado actual en models/mobilenetv2_ajustado.")
    model = tf.keras.models.load_model(ruta_modelo)
    metricas = evaluar_metricas_sin_figuras(model, val_dataset)
    metricas["training_time_seconds"] = 0.0
    metricas["epochs"] = 0
    metricas["history"] = {}
    return metricas


def seleccionar_ganador_desbalance(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Seleccionar la mejor configuración con el criterio del proyecto."""
    if not resultados:
        raise ValueError("No hay resultados de experimentos.")
    return construir_tabla_experimentos(resultados)[0]


def entrenar_y_evaluar_desbalance() -> dict[str, Any]:
    """Ejecutar class_weight y oversampling, seleccionar y evaluar en TEST si procede."""
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
    train_dataset_normal = construir_pipeline_augmentation(datasets["train"])
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    pesos_clase = calcular_pesos_clase(ruta_manifiesto)

    print(f"\n===== REFERENCIA: {NOMBRE_AJUSTADO} (reevaluado en VALIDATION) =====")
    referencia = evaluar_modelo_ajustado_actual(val_dataset)
    referencia["experimento"] = NOMBRE_AJUSTADO
    referencia["config"] = dict(CONFIG_AJUSTADO)
    referencia["config"]["class_weight"] = None
    referencia["config"]["oversampling"] = False

    train_oversampled_info: dict[str, Any] = {}
    experimentos: list[dict[str, Any]] = [referencia]

    for nombre in EXPERIMENTOS_DESBALANCE:
        print(f"\n===== EXPERIMENTO: {nombre} =====")
        if nombre == "class_weight":
            metricas = entrenar_variante_desbalance(
                "class_weight",
                train_dataset_normal,
                val_dataset,
                use_class_weight=True,
                pesos_clase=pesos_clase,
            )
            metricas["config"] = dict(CONFIG_AJUSTADO)
            metricas["config"]["class_weight"] = pesos_clase
            metricas["config"]["oversampling"] = False
        else:
            train_oversampled, info = construir_train_oversampling(ruta_manifiesto, TAMANO_IMAGEN, TAMANO_LOTE)
            train_oversampled_info = info
            train_oversampled = construir_pipeline_augmentation(train_oversampled)
            metricas = entrenar_variante_desbalance("oversampling", train_oversampled, val_dataset, use_class_weight=False)
            metricas["config"] = dict(CONFIG_AJUSTADO)
            metricas["config"]["class_weight"] = None
            metricas["config"]["oversampling"] = info

        metricas["experimento"] = nombre
        experimentos.append(metricas)
        resumen_exp = {clave: metricas[clave] for clave in CRITERIO_METRICAS + ("recall", "specificity")}
        print(json.dumps(resumen_exp, indent=2, default=str))

    tabla = construir_tabla_experimentos(experimentos)
    ganador = seleccionar_ganador_desbalance(experimentos)

    ganador_cambia = ganador["experimento"] != NOMBRE_AJUSTADO

    if not ganador_cambia:
        print("\n===== NINGÚN EXPERIMENTO SUPERA DE FORMA CLARA AL AJUSTADO; SE MANTIENE MobileNetV2-ajustado (lr=3e-4) =====")
        ruta_resultados_tuning = DIRECTORIO_MODELOS / "mobilenetv2_tuning_results.json"
        with ruta_resultados_tuning.open(encoding="utf-8") as file_handle:
            tuning = json.load(file_handle)
        final_val = dict(tuning["final_validation"])
        final_test = dict(tuning["final_test"])
        modelo_final = "MobileNetV2-ajustado (lr=3e-4) — sin RE-train de TEST (ya evaluado en la fase previa)"
        ruta_modelo_final = str(DIRECTORIO_MODELOS / "mobilenetv2_ajustado" / "best_model.keras")
        tiempo_final = float(tuning["training_time_seconds_ganador"])
        motivo = "class_weight y oversampling no mejoran el criterio del proyecto frente al ajustado actual."
    else:
        print(f"\n===== ELEGIDO: {ganador['experimento']}; REENTRENO Y EVALUACIÓN EN TEST =====")
        config_ganadora = ganador["config"]
        nombre_directorio = f"mobilenetv2_desbalance_{ganador['experimento']}"
        directorio_ganador = DIRECTORIO_MODELOS / nombre_directorio
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
        fit_kwargs: dict[str, Any] = {
            "validation_data": val_dataset,
            "epochs": CONFIG_AJUSTADO["epochs"],
            "batch_size": CONFIG_AJUSTADO["batch_size"],
            "callbacks": callbacks,
            "verbose": 1,
        }

        if ganador["experimento"] == "class_weight":
            train_final = train_dataset_normal
            fit_kwargs["class_weight"] = pesos_clase
        else:
            train_final, _ = construir_train_oversampling(ruta_manifiesto, TAMANO_IMAGEN, TAMANO_LOTE)
            train_final = construir_pipeline_augmentation(train_final)

        start_time = time.time()
        model.fit(train_final, **fit_kwargs)
        tiempo_final = float(time.time() - start_time)
        model = tf.keras.models.load_model(best_model_path)

        modelo_final = f"MobileNetV2-{ganador['experimento']}"
        final_val = evaluar_modelo(model, val_dataset, modelo_final, "validation")
        final_test = evaluar_modelo(model, test_dataset, modelo_final, "test")
        ruta_modelo_final = str(best_model_path)
        motivo = f"{ganador['experimento']} es el mejor según Balanced Acc -> ROC-AUC -> F1 -> Acc y supera al ajustado actual."

    payload = {
        "descripcion": "Experimentos de desbalance de clases (class_weight y oversampling) sobre el MobileNetV2 ajustado lr=3e-4.",
        "reglas": {
            "class_weight_solo_entrenamiento": True,
            "oversampling_solo_train": True,
            "validation_test_intactos": True,
            "experimentos_por_separado": True,
            "test_solo_evaluacion_final": True,
        },
        "config_ajustado": CONFIG_AJUSTADO,
        "pesos_clase_balanceados": pesos_clase,
        "oversampling_train": train_oversampled_info,
        "tabla_seleccion": tabla,
        "seleccion": ganador,
        "ganador_cambia": bool(ganador_cambia),
        "motivo_seleccion": motivo,
        "modelo_final": modelo_final,
        "final_validation": final_val,
        "final_test": final_test,
        "training_time_seconds_final": float(tiempo_final),
        "ruta_modelo_final": ruta_modelo_final,
    }

    ruta_salida = DIRECTORIO_MODELOS / "mobilenetv2_desbalance_results.json"
    ruta_salida.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    resultado = entrenar_y_evaluar_desbalance()
    print(json.dumps(resultado["seleccion"], indent=2, default=str))
    print(json.dumps(resultado["tabla_seleccion"], indent=2, default=str))