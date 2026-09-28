from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.preprocessing import (
    build_class_distribution,
    compute_image_statistics,
    load_image,
    normalize_image,
    resize_image,
)
from src.models.architectures import MODEL_NAMES, build_transfer_model
from src.models.evaluation import compute_confusion_matrix, compute_metrics_report, select_best_model
from src.data.splitting import create_stratified_split_manifest


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))
    image.save(image_path)
    return image_path


def test_load_image_returns_pil_image(sample_image: Path) -> None:
    image = load_image(sample_image)
    assert image.size == (64, 64)
    assert image.mode == "RGB"


def test_resize_image_changes_shape(sample_image: Path) -> None:
    image = load_image(sample_image)
    resized = resize_image(image, target_size=(32, 32))
    assert resized.size == (32, 32)


def test_normalize_image_scales_to_unit_interval(sample_image: Path) -> None:
    image = load_image(sample_image)
    normalized = normalize_image(image)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_compute_image_statistics_matches_image_values(sample_image: Path) -> None:
    image = load_image(sample_image)
    stats = compute_image_statistics(image)
    assert stats["width"] == 64
    assert stats["height"] == 64
    assert stats["channels"] == 3
    assert stats["mean"] >= 0.0
    assert stats["std"] >= 0.0


def test_build_class_distribution_counts_labels() -> None:
    records = pd.DataFrame(
        [
            {"split": "train", "label": "NORMAL"},
            {"split": "train", "label": "PNEUMONIA"},
            {"split": "train", "label": "NORMAL"},
            {"split": "val", "label": "PNEUMONIA"},
        ]
    )
    distribution = build_class_distribution(records)
    assert distribution["train"]["NORMAL"] == 2
    assert distribution["train"]["PNEUMONIA"] == 1
    assert distribution["val"]["PNEUMONIA"] == 1


def test_model_names_include_required_models() -> None:
    assert set(MODEL_NAMES) == {"VGG16", "ResNet50", "MobileNetV2"}


def test_build_transfer_model_uses_expected_backbone_name() -> None:
    model = build_transfer_model("VGG16", input_shape=(32, 32, 3), classes=2, weights=None)
    assert model is not None
    assert model.layers[0].input_shape[0][1:4] == (32, 32, 3)


def test_build_transfer_model_raises_for_unknown_backbone() -> None:
    with pytest.raises(ValueError):
        build_transfer_model("UnknownModel")


def test_compute_confusion_matrix_values() -> None:
    labels = np.array([0, 1, 1, 0])
    predictions = np.array([0, 1, 0, 0])
    matrix = compute_confusion_matrix(labels, predictions)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 2
    assert matrix[1, 1] == 1


def test_compute_metrics_report_contains_expected_fields() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics_report(y_true, y_pred, average="binary")
    assert {"accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc"}.issubset(metrics.keys())


def test_select_best_model_prefers_balanced_performance() -> None:
    results = [
        {"model_name": "ResNet50", "balanced_accuracy": 0.50, "roc_auc": 0.99, "f1": 0.90, "accuracy": 0.90},
        {"model_name": "MobileNetV2", "balanced_accuracy": 0.78, "roc_auc": 0.95, "f1": 0.88, "accuracy": 0.84},
    ]
    winner = select_best_model(results)
    assert winner["model_name"] == "MobileNetV2"


def test_select_best_model_uses_roc_auc_as_tie_breaker() -> None:
    results = [
        {"model_name": "MobileNetV2", "balanced_accuracy": 0.78, "roc_auc": 0.95, "f1": 0.85, "accuracy": 0.84},
        {"model_name": "VGG16", "balanced_accuracy": 0.78, "roc_auc": 0.92, "f1": 0.88, "accuracy": 0.85},
    ]
    winner = select_best_model(results)
    assert winner["model_name"] == "MobileNetV2"


def test_build_transfer_model_can_use_binary_head() -> None:
    model = build_transfer_model("MobileNetV2", input_shape=(64, 64, 3), classes=2, weights=None)
    assert model.output_shape[-1] == 1


def test_build_transfer_model_uses_imagenet_weights_by_default() -> None:
    model = build_transfer_model("ResNet50", input_shape=(224, 224, 3), classes=2, weights=None)
    assert model is not None


def test_build_transfer_model_handles_unsupported_shape() -> None:
    with pytest.raises(ValueError):
        build_transfer_model("VGG16", input_shape=(8, 8, 3), classes=2)


def test_resize_image_rejects_invalid_target_size() -> None:
    image = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError):
        resize_image(image, target_size=(0, 0))


def test_load_image_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        load_image(missing)


def test_compute_metrics_report_handles_empty_predictions() -> None:
    with pytest.raises(ValueError):
        compute_metrics_report(np.array([]), np.array([]))


def test_select_best_model_rejects_empty_results() -> None:
    with pytest.raises(ValueError):
        select_best_model([])


def test_create_stratified_split_manifest_is_reproducible_and_disjoint(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = data_dir / "train" / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(10):
            image = Image.new("RGB", (16, 16), color=(index, 20 if label == "NORMAL" else 40, 30))
            image.save(folder / f"{label}_{index}.png")

    first = create_stratified_split_manifest(data_dir, tmp_path / "first.csv", random_state=42)
    second = create_stratified_split_manifest(data_dir, tmp_path / "second.csv", random_state=42)

    assert first[["path", "split"]].equals(second[["path", "split"]])
    assert first.groupby("split")["label"].count().sum() == 20
    assert set(first["split"]) == {"train", "val", "test"}
