"""Pipeline integral de entrenamiento y evaluación para el proyecto de radiografías de tórax."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve

from src.data.datasets import construir_pipelines_datos
from src.data.splitting import crear_manifiesto_division_estratificada
from src.models.architectures import construir_modelo_transferencia
from src.models.comparison import resumir_comparacion_modelos
from src.models.evaluation import (
    calcular_matriz_confusion,
    calcular_reporte_metricas,
    evaluar_baseline_mayoritaria,
    evaluar_criterio_exito,
)
from src.utils.paths import DIRECTORIO_DATOS, DIRECTORIO_FIGURAS, DIRECTORIO_MODELOS, RAIZ_PROYECTO, asegurar_directorio


SEMILLA = 42
TAMANO_IMAGEN = (224, 224)
TAMANO_LOTE = 16
EPOCAS = 3


def configurar_ejecucion() -> None:
    """Configurar TensorFlow para un entrenamiento reproducible en este entorno."""
    tf.keras.utils.set_random_seed(SEMILLA)
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True) if tf.config.list_physical_devices("GPU") else None


def crear_capas_augmentation() -> list:
    """Devolver las capas de data augmentation usadas en el entrenamiento.

    Se descarta el volteo horizontal porque las radiografías de tórax pueden
    contener información de lateralidad anatómica y marcadores L/R; un volteo
    horizontal podría invertir artificialmente esa información.
    """
    return [
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
    ]


def construir_pipeline_augmentation(train_dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Aplicar augmentation ligera únicamente al conjunto de entrenamiento."""
    augmentation = tf.keras.Sequential(crear_capas_augmentation())

    def augment_examples(features: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return augmentation(features, training=True), labels

    return train_dataset.map(augment_examples, num_parallel_calls=tf.data.AUTOTUNE)


def guardar_matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split_name: str) -> Path:
    """Guardar la gráfica de la matriz de confusión normalizada para un modelo."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm = cm.astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(cm_norm, cmap="Blues")
    ax.set_title(f"Matriz de confusión - {model_name}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
    ax.set_yticklabels(["NORMAL", "PNEUMONIA"])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    ruta_salida = DIRECTORIO_FIGURAS / f"confusion_matrix_{split_name.lower()}_{model_name.lower()}.png"
    plt.savefig(ruta_salida, dpi=200)
    plt.close(fig)
    return ruta_salida


def guardar_curva_roc(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, split_name: str) -> Path:
    """Guardar la curva ROC para un modelo."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{model_name}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"Curva ROC - {model_name}")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.legend()
    plt.tight_layout()
    ruta_salida = DIRECTORIO_FIGURAS / f"roc_curve_{split_name.lower()}_{model_name.lower()}.png"
    plt.savefig(ruta_salida, dpi=200)
    plt.close()
    return ruta_salida


def evaluar_modelo(model: tf.keras.Model, dataset: tf.data.Dataset, model_name: str, split_name: str) -> dict[str, Any]:
    """Evaluar un modelo entrenado sobre un conjunto del dataset y devolver las métricas."""
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []
    for features, labels in dataset:
        predictions = model.predict(features, verbose=0)
        y_true_list.append(labels.numpy().astype(int))
        y_prob_list.append(predictions.ravel())

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = calcular_reporte_metricas(y_true, y_prob)
    cm = calcular_matriz_confusion(y_true, y_prob)
    confusion_path = guardar_matriz_confusion(y_true, y_pred, model_name, split_name)
    roc_path = guardar_curva_roc(y_true, y_prob, model_name, split_name)

    return {
        "model_name": model_name,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "confusion_matrix": cm.tolist(),
        "confusion_plot": str(confusion_path),
        "roc_plot": str(roc_path),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


def entrenar_y_evaluar_modelos() -> dict[str, Any]:
    """Entrenar todos los modelos, seleccionar por validación y evaluar solo al ganador en prueba."""
    asegurar_directorio(DIRECTORIO_MODELOS)
    asegurar_directorio(DIRECTORIO_FIGURAS)
    configurar_ejecucion()

    ruta_manifiesto = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"
    manifiesto_division = crear_manifiesto_division_estratificada(DIRECTORIO_DATOS, ruta_manifiesto, random_state=SEMILLA)
    datasets = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        manifiesto_division=ruta_manifiesto,
    )

    if "train" not in datasets or "val" not in datasets or "test" not in datasets:
        raise ValueError("El pipeline del dataset está incompleto. Verifique la estructura real del dataset.")

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    train_dataset = construir_pipeline_augmentation(train_dataset)

    validation_results: list[dict[str, Any]] = []
    model_artifacts: dict[str, Any] = {}
    for model_name in ["VGG16", "ResNet50", "MobileNetV2"]:
        tf.keras.backend.clear_session()
        print(f"\nEntrenando {model_name}...")
        start_time = time.time()
        model = construir_modelo_transferencia(model_name=model_name, input_shape=(224, 224, 3), classes=2, weights="imagenet")
        directorio_modelo = DIRECTORIO_MODELOS / model_name.lower()
        asegurar_directorio(directorio_modelo)
        best_model_path = directorio_modelo / "best_model.keras"
        if best_model_path.exists():
            best_model_path.unlink()
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor="val_loss", mode="min", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ]

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

        validation_evaluation = evaluar_modelo(model, val_dataset, model_name, "validation")
        validation_evaluation["training_time_seconds"] = float(elapsed)
        validation_evaluation["epochs"] = int(EPOCAS)
        validation_evaluation["history"] = {
            "loss": [float(value) for value in history.history["loss"]],
            "val_loss": [float(value) for value in history.history["val_loss"]],
            "accuracy": [float(value) for value in history.history["accuracy"]],
            "val_accuracy": [float(value) for value in history.history["val_accuracy"]],
        }
        model_artifacts[model_name] = {
            "model_path": str(best_model_path),
            "validation": validation_evaluation,
        }
        validation_results.append(validation_evaluation)

    summary = resumir_comparacion_modelos(validation_results)
    winner_name = summary["winner"]["model_name"]
    winner_model_path = Path(model_artifacts[winner_name]["model_path"])
    winner_model = tf.keras.models.load_model(winner_model_path)
    final_test = evaluar_modelo(winner_model, test_dataset, winner_name, "test")

    baseline_test = evaluar_baseline_mayoritaria(np.asarray(final_test["y_true"]))
    criterio_exito = evaluar_criterio_exito(final_test, baseline_test)

    results_payload = {
        "split_manifest": str(ruta_manifiesto),
        "split_random_state": SEMILLA,
        "split_ratios": {"train": 0.80, "val": 0.20},
        "test_original": {"total": 624, "intacto": True, "descripcion": "El test original del dataset crudo no participa en entrenamiento ni en seleccion de modelos."},
        "split_distribution": {
            split_name: {
                "total": int((manifiesto_division["split"] == split_name).sum()),
                "NORMAL": int(((manifiesto_division["split"] == split_name) & (manifiesto_division["label"] == "NORMAL")).sum()),
                "PNEUMONIA": int(((manifiesto_division["split"] == split_name) & (manifiesto_division["label"] == "PNEUMONIA")).sum()),
            }
            for split_name in ["train", "val", "test"]
        },
        "models": model_artifacts,
        "validation_comparison": summary,
        "baseline_test": baseline_test,
        "criterio_exito": criterio_exito,
        "final_test": final_test,
    }

    result_path = DIRECTORIO_MODELOS / "model_results.json"
    result_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    return results_payload


if __name__ == "__main__":
    result = entrenar_y_evaluar_modelos()
    print(json.dumps(result["validation_comparison"], indent=2, default=str))
    print(json.dumps(result["final_test"], indent=2, default=str))
