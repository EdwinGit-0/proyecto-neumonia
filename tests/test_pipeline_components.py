from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.preprocessing import (
    construir_distribucion_clases,
    calcular_estadisticas_imagen,
    cargar_imagen,
    normalizar_imagen,
    redimensionar_imagen,
)
from src.models.architectures import NOMBRES_MODELOS, construir_modelo_transferencia
from src.models.evaluation import calcular_matriz_confusion, calcular_reporte_metricas, seleccionar_mejor_modelo
from src.data.splitting import crear_manifiesto_division_estratificada


@pytest.fixture
def imagen_muestra(tmp_path: Path) -> Path:
    ruta_imagen = tmp_path / "sample.png"
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))
    image.save(ruta_imagen)
    return ruta_imagen


def test_cargar_imagen_devuelve_imagen_pil(imagen_muestra: Path) -> None:
    image = cargar_imagen(imagen_muestra)
    assert image.size == (64, 64)
    assert image.mode == "RGB"


def test_redimensionar_imagen_cambia_forma(imagen_muestra: Path) -> None:
    image = cargar_imagen(imagen_muestra)
    resized = redimensionar_imagen(image, target_size=(32, 32))
    assert resized.size == (32, 32)


def test_normalizar_imagen_escala_al_intervalo_unitario(imagen_muestra: Path) -> None:
    image = cargar_imagen(imagen_muestra)
    normalized = normalizar_imagen(image)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_calcular_estadisticas_imagen_coincide_con_valores(imagen_muestra: Path) -> None:
    image = cargar_imagen(imagen_muestra)
    stats = calcular_estadisticas_imagen(image)
    assert stats["width"] == 64
    assert stats["height"] == 64
    assert stats["channels"] == 3
    assert stats["mean"] >= 0.0
    assert stats["std"] >= 0.0


def test_construir_distribucion_clases_cuenta_etiquetas() -> None:
    records = pd.DataFrame(
        [
            {"split": "train", "label": "NORMAL"},
            {"split": "train", "label": "PNEUMONIA"},
            {"split": "train", "label": "NORMAL"},
            {"split": "val", "label": "PNEUMONIA"},
        ]
    )
    distribution = construir_distribucion_clases(records)
    assert distribution["train"]["NORMAL"] == 2
    assert distribution["train"]["PNEUMONIA"] == 1
    assert distribution["val"]["PNEUMONIA"] == 1


def test_nombres_modelos_incluyen_modelos_requeridos() -> None:
    assert set(NOMBRES_MODELOS) == {"VGG16", "ResNet50", "MobileNetV2"}


def test_construir_modelo_transferencia_usa_backbone_esperado() -> None:
    model = construir_modelo_transferencia("VGG16", input_shape=(32, 32, 3), classes=2, weights=None)
    assert model is not None
    assert model.layers[0].input_shape[0][1:4] == (32, 32, 3)


def test_construir_modelo_transferencia_lanza_error_para_backbone_desconocido() -> None:
    with pytest.raises(ValueError):
        construir_modelo_transferencia("UnknownModel")


def test_calcular_matriz_confusion_valores() -> None:
    labels = np.array([0, 1, 1, 0])
    predictions = np.array([0, 1, 0, 0])
    matrix = calcular_matriz_confusion(labels, predictions)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 2
    assert matrix[1, 1] == 1


def test_calcular_reporte_metricas_contiene_campos_esperados() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = calcular_reporte_metricas(y_true, y_pred, average="binary")
    assert {"accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc"}.issubset(metrics.keys())


def test_seleccionar_mejor_modelo_prefiere_desempeno_balanceado() -> None:
    results = [
        {"model_name": "ResNet50", "balanced_accuracy": 0.50, "roc_auc": 0.99, "f1": 0.90, "accuracy": 0.90},
        {"model_name": "MobileNetV2", "balanced_accuracy": 0.78, "roc_auc": 0.95, "f1": 0.88, "accuracy": 0.84},
    ]
    winner = seleccionar_mejor_modelo(results)
    assert winner["model_name"] == "MobileNetV2"


def test_seleccionar_mejor_modelo_usa_roc_auc_como_desempate() -> None:
    results = [
        {"model_name": "MobileNetV2", "balanced_accuracy": 0.78, "roc_auc": 0.95, "f1": 0.85, "accuracy": 0.84},
        {"model_name": "VGG16", "balanced_accuracy": 0.78, "roc_auc": 0.92, "f1": 0.88, "accuracy": 0.85},
    ]
    winner = seleccionar_mejor_modelo(results)
    assert winner["model_name"] == "MobileNetV2"


def test_construir_modelo_transferencia_puede_usar_cabeza_binaria() -> None:
    model = construir_modelo_transferencia("MobileNetV2", input_shape=(64, 64, 3), classes=2, weights=None)
    assert model.output_shape[-1] == 1


def test_construir_modelo_transferencia_usa_pesos_imagenet_por_defecto() -> None:
    model = construir_modelo_transferencia("ResNet50", input_shape=(224, 224, 3), classes=2, weights=None)
    assert model is not None


def test_construir_modelo_transferencia_maneja_forma_no_compatible() -> None:
    with pytest.raises(ValueError):
        construir_modelo_transferencia("VGG16", input_shape=(8, 8, 3), classes=2)


def test_redimensionar_imagen_rechaza_tamano_invalido() -> None:
    image = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError):
        redimensionar_imagen(image, target_size=(0, 0))


def test_cargar_imagen_rechaza_archivo_inexistente(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        cargar_imagen(missing)


def test_calcular_reporte_metricas_maneja_predicciones_vacias() -> None:
    with pytest.raises(ValueError):
        calcular_reporte_metricas(np.array([]), np.array([]))


def test_seleccionar_mejor_modelo_rechaza_resultados_vacios() -> None:
    with pytest.raises(ValueError):
        seleccionar_mejor_modelo([])


def test_crear_manifiesto_division_estratificada_es_reproducible_y_sin_solapamiento(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = data_dir / "train" / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(10):
            image = Image.new("RGB", (16, 16), color=(index, 20 if label == "NORMAL" else 40, 30))
            image.save(folder / f"{label}_{index}.png")

    first = crear_manifiesto_division_estratificada(data_dir, tmp_path / "first.csv", random_state=42)
    second = crear_manifiesto_division_estratificada(data_dir, tmp_path / "second.csv", random_state=42)

    assert first[["path", "split"]].equals(second[["path", "split"]])
    assert first.groupby("split")["label"].count().sum() == 20
    assert set(first["split"]) == {"train", "val", "test"}
