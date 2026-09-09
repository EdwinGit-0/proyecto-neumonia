"""End-to-end training and evaluation pipeline for the chest X-ray project."""

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

from src.data.datasets import build_data_pipelines
from src.data.splitting import create_stratified_split_manifest
from src.models.architectures import build_transfer_model
from src.models.comparison import summarize_model_comparison
from src.models.evaluation import compute_confusion_matrix, compute_metrics_report
from src.utils.paths import DATA_DIR, FIGURES_DIR, MODELS_DIR, PROJECT_ROOT, ensure_directory


SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3


def configure_runtime() -> None:
    """Configure TensorFlow for reproducible training in this environment."""
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True) if tf.config.list_physical_devices("GPU") else None


def build_augmentation_pipeline(train_dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Apply light augmentation only to the training split."""
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.05),
        ]
    )

    def augment_examples(features: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return augmentation(features, training=True), labels

    return train_dataset.map(augment_examples, num_parallel_calls=tf.data.AUTOTUNE)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split_name: str) -> Path:
    """Save the normalized confusion matrix plot for a model."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm = cm.astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(cm_norm, cmap="Blues")
    ax.set_title(f"Confusion matrix - {model_name}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
    ax.set_yticklabels(["NORMAL", "PNEUMONIA"])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    output_path = FIGURES_DIR / f"confusion_matrix_{split_name.lower()}_{model_name.lower()}.png"
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, split_name: str) -> Path:
    """Save the ROC curve for a model."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{model_name}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    output_path = FIGURES_DIR / f"roc_curve_{split_name.lower()}_{model_name.lower()}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, model_name: str, split_name: str) -> dict[str, Any]:
    """Evaluate a trained model on one dataset split and return metrics."""
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []
    for features, labels in dataset:
        predictions = model.predict(features, verbose=0)
        y_true_list.append(labels.numpy().astype(int))
        y_prob_list.append(predictions.ravel())

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics_report(y_true, y_prob)
    cm = compute_confusion_matrix(y_true, y_prob)
    confusion_path = save_confusion_matrix(y_true, y_pred, model_name, split_name)
    roc_path = save_roc_curve(y_true, y_prob, model_name, split_name)

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


def train_and_evaluate_models() -> dict[str, Any]:
    """Train all models, select on validation, and evaluate only the winner on test."""
    ensure_directory(MODELS_DIR)
    ensure_directory(FIGURES_DIR)
    configure_runtime()

    manifest_path = PROJECT_ROOT / "data" / "interim" / "stratified_split_70_15_15.csv"
    split_manifest = create_stratified_split_manifest(DATA_DIR, manifest_path, random_state=SEED)
    datasets = build_data_pipelines(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        split_manifest=manifest_path,
    )

    if "train" not in datasets or "val" not in datasets or "test" not in datasets:
        raise ValueError("The dataset pipeline is incomplete. Check the real dataset structure.")

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    train_dataset = build_augmentation_pipeline(train_dataset)

    validation_results: list[dict[str, Any]] = []
    model_artifacts: dict[str, Any] = {}
    for model_name in ["VGG16", "ResNet50", "MobileNetV2"]:
        tf.keras.backend.clear_session()
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        model = build_transfer_model(model_name=model_name, input_shape=(224, 224, 3), classes=2, weights="imagenet")
        model_dir = MODELS_DIR / model_name.lower()
        ensure_directory(model_dir)
        best_model_path = model_dir / "best_model.keras"
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
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )
        elapsed = time.time() - start_time
        if best_model_path.exists():
            model = tf.keras.models.load_model(best_model_path)

        validation_evaluation = evaluate_model(model, val_dataset, model_name, "validation")
        validation_evaluation["training_time_seconds"] = float(elapsed)
        validation_evaluation["epochs"] = int(EPOCHS)
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

    summary = summarize_model_comparison(validation_results)
    winner_name = summary["winner"]["model_name"]
    winner_model_path = Path(model_artifacts[winner_name]["model_path"])
    winner_model = tf.keras.models.load_model(winner_model_path)
    final_test = evaluate_model(winner_model, test_dataset, winner_name, "test")
    results_payload = {
        "split_manifest": str(manifest_path),
        "split_random_state": SEED,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_distribution": {
            split_name: {
                "total": int((split_manifest["split"] == split_name).sum()),
                "NORMAL": int(((split_manifest["split"] == split_name) & (split_manifest["label"] == "NORMAL")).sum()),
                "PNEUMONIA": int(((split_manifest["split"] == split_name) & (split_manifest["label"] == "PNEUMONIA")).sum()),
            }
            for split_name in ["train", "val", "test"]
        },
        "models": model_artifacts,
        "validation_comparison": summary,
        "final_test": final_test,
    }

    result_path = MODELS_DIR / "model_results.json"
    result_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    return results_payload


if __name__ == "__main__":
    result = train_and_evaluate_models()
    print(json.dumps(result["validation_comparison"], indent=2, default=str))
    print(json.dumps(result["final_test"], indent=2, default=str))
