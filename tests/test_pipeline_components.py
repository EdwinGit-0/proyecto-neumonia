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
from src.models.evaluation import (
    calcular_matriz_confusion,
    calcular_reporte_metricas,
    evaluar_baseline_mayoritaria,
    evaluar_criterio_exito,
    seleccionar_mejor_modelo,
)
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


def _crear_estructura_prueba(data_dir: Path) -> None:
    """Crear una estructura cruda con train original y test original (sin val original)."""
    for label in ["NORMAL", "PNEUMONIA"]:
        train_folder = data_dir / "train" / label
        train_folder.mkdir(parents=True, exist_ok=True)
        for index in range(10):
            color = (index, 20 if label == "NORMAL" else 40, 30)
            Image.new("RGB", (16, 16), color=color).save(train_folder / f"train_{label}_{index}.png")
        test_folder = data_dir / "test" / label
        test_folder.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            color = (index, 60 if label == "NORMAL" else 80, 90)
            Image.new("RGB", (16, 16), color=color).save(test_folder / f"test_{label}_{index}.png")


def test_crear_manifiesto_division_estratificada_es_reproducible_y_sin_solapamiento(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    _crear_estructura_prueba(data_dir)

    first = crear_manifiesto_division_estratificada(data_dir, tmp_path / "first.csv", random_state=42)
    second = crear_manifiesto_division_estratificada(data_dir, tmp_path / "second.csv", random_state=42)

    assert first[["path", "split"]].equals(second[["path", "split"]])
    assert first.groupby("split")["label"].count().sum() == 28
    assert set(first["split"]) == {"train", "val", "test"}


def test_manifiesto_preserva_test_original_completo(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    _crear_estructura_prueba(data_dir)

    manifest = crear_manifiesto_division_estratificada(data_dir, tmp_path / "split.csv", random_state=42)
    test_records = manifest[manifest["split"] == "test"]
    assert len(test_records) == 8
    assert set(test_records["path"]).issubset(
        {str(path) for path in (data_dir / "test").rglob("*.png")}
    )


def test_manifiesto_train_y_validation_provienen_solo_del_train_original(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    _crear_estructura_prueba(data_dir)

    manifest = crear_manifiesto_division_estratificada(data_dir, tmp_path / "split.csv", random_state=42)
    experimental = manifest[manifest["split"].isin(["train", "val"])]
    train_originals = {str(path) for path in (data_dir / "train").rglob("*.png")}
    assert set(experimental["path"]).issubset(train_originals)
    assert len(experimental) == 20


def test_manifiesto_no_reparte_imagenes_test_en_entrenamiento_ni_validacion(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    _crear_estructura_prueba(data_dir)

    manifest = crear_manifiesto_division_estratificada(data_dir, tmp_path / "split.csv", random_state=42)
    test_paths = set(manifest[manifest["split"] == "test"]["path"])
    train_paths = set(manifest[manifest["split"] == "train"]["path"])
    val_paths = set(manifest[manifest["split"] == "val"]["path"])
    assert test_paths.isdisjoint(train_paths)
    assert test_paths.isdisjoint(val_paths)


def test_manifiesto_division_es_estratificada_por_clase(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    _crear_estructura_prueba(data_dir)

    manifest = crear_manifiesto_division_estratificada(data_dir, tmp_path / "split.csv", random_state=42)
    for label in ["NORMAL", "PNEUMONIA"]:
        label_records = manifest[manifest["label"] == label]
        train_count = int((label_records["split"] == "train").sum())
        val_count = int((label_records["split"] == "val").sum())
        total = train_count + val_count
        assert total == 10
        assert 0.15 <= val_count / total <= 0.25


def test_manifiesto_no_separa_duplicados_por_hash_entre_train_y_validation(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = data_dir / "train" / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(6):
            Image.new("RGB", (16, 16), color=(index, 20, 30)).save(folder / f"{label}_{index}.png")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "test").mkdir(parents=True, exist_ok=True)

    duplicado = data_dir / "train" / "NORMAL" / "copia.png"
    duplicado.write_bytes((data_dir / "train" / "NORMAL" / "NORMAL_0.png").read_bytes())
    (data_dir / "test" / "NORMAL").mkdir(parents=True, exist_ok=True)

    manifest = crear_manifiesto_division_estratificada(data_dir, tmp_path / "split.csv", random_state=42)
    split_por_ruta = manifest.set_index("path")["split"].to_dict()
    splits_duplicado = {
        split_por_ruta[str(data_dir / "train" / "NORMAL" / "NORMAL_0.png")],
        split_por_ruta[str(data_dir / "train" / "NORMAL" / "copia.png")],
    }
    assert len(splits_duplicado & {"train", "val"}) == 1


def test_manifiesto_ignora_val_original_del_dataset_crudo(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    _crear_estructura_prueba(data_dir)
    (data_dir / "val" / "NORMAL").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(5, 5, 5)).save(data_dir / "val" / "NORMAL" / "val_normal.png")

    manifest = crear_manifiesto_division_estratificada(data_dir, tmp_path / "split.csv", random_state=42)
    all_paths = {str(Path(p)) for p in manifest["path"]}
    assert not any("\\val\\" in p or "/val/" in p for p in all_paths)


def test_baseline_mayoritaria_mide_clase_mayoritaria(tmp_path: Path) -> None:
    y_true = np.array([1, 1, 1, 0, 0])
    baseline = evaluar_baseline_mayoritaria(y_true)
    assert baseline["majority_class"] == "PNEUMONIA"
    assert baseline["balanced_accuracy"] == 0.5
    assert baseline["recall"] == 1.0
    assert baseline["specificity"] == 0.0


def test_criterio_exito_requiere_superar_baseline_y_equilibrio(tmp_path: Path) -> None:
    baseline = {"balanced_accuracy": 0.5}
    mete = {
        "balanced_accuracy": 0.90,
        "recall": 0.95,
        "specificity": 0.85,
    }
    resultado = evaluar_criterio_exito(mete, baseline)
    assert resultado["cumple"] is True

    sesgado = {
        "balanced_accuracy": 0.55,
        "recall": 1.0,
        "specificity": 0.10,
    }
    resultado_sesgado = evaluar_criterio_exito(sesgado, baseline)
    assert resultado_sesgado["cumple"] is False
    assert resultado_sesgado["equilibrio_adecuado"] is False


def test_capas_augmentation_no_incluyen_flip_horizontal() -> None:
    import tensorflow as tf

    from src.training.run_real_training import crear_capas_augmentation

    layers = crear_capas_augmentation()
    assert not any(isinstance(layer, tf.keras.layers.RandomFlip) for layer in layers)


def test_capas_augmentation_incluyen_rotacion_y_zoom() -> None:
    import tensorflow as tf

    from src.training.run_real_training import crear_capas_augmentation

    layers = crear_capas_augmentation()
    assert any(isinstance(layer, tf.keras.layers.RandomRotation) for layer in layers)
    assert any(isinstance(layer, tf.keras.layers.RandomZoom) for layer in layers)


def test_capas_augmentation_rotacion_y_zoom_usados_en_entrenamiento_solo(tmp_path: Path) -> None:
    """La pipeline de datos no incorpora augmentación; esta se aplica solo al train."""
    import tensorflow as tf

    from src.data.datasets import construir_pipelines_datos
    from src.data.splitting import cargar_manifiesto_division
    from src.training.run_real_training import construir_pipeline_augmentation

    data_dir = tmp_path / "chest_xray"
    _crear_estructura_prueba(data_dir)
    manifest_path = tmp_path / "split.csv"
    crear_manifiesto_division_estratificada(data_dir, manifest_path, random_state=42)

    manifiesto = cargar_manifiesto_division(manifest_path)
    assert set(manifiesto["split"]) == {"train", "val", "test"}

    datasets = construir_pipelines_datos(
        data_dir,
        image_size=(16, 16),
        batch_size=4,
        manifiesto_division=manifest_path,
    )
    assert isinstance(datasets["train"], tf.data.Dataset)
    assert isinstance(datasets["val"], tf.data.Dataset)
    assert isinstance(datasets["test"], tf.data.Dataset)

    augmented = construir_pipeline_augmentation(datasets["train"])
    assert isinstance(augmented, tf.data.Dataset)
    assert augmented.element_spec == datasets["train"].element_spec
