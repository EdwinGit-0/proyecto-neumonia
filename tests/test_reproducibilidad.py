"""Pruebas de reproducibilidad del pipeline de datos.

Estas pruebas fijan la causa raiz detectada en la auditoria: el ``shuffle`` del
train y las capas de augmentation se construian sin semilla, de modo que dos
ejecuciones del mismo codigo con ``SEMILLA = 42`` entrenaban sobre imagenes
distintas. No entrenan ningun modelo: solo comprueban que la misma semilla
produce el mismo pipeline.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.datasets import construir_pipelines_datos
from src.data.splitting import crear_manifiesto_division_estratificada
from src.training import run_real_training, tuning_desbalance_clases, tuning_mobilenetv2
from src.training import optimizacion_mobilenetv2
from src.utils import reproducibility
from src.utils.reproducibility import SEMILLA, configurar_reproducibilidad, reiniciar_semilla


def _crear_dataset_minimo(data_dir: Path, Manifest: Path) -> None:
    """Crear un dataset sintetico con train original y test original."""
    for etiqueta in ["NORMAL", "PNEUMONIA"]:
        for carpeta, cantidad in [("train", 10), ("test", 4)]:
            destino = data_dir / carpeta / etiqueta
            destino.mkdir(parents=True, exist_ok=True)
            for indice in range(cantidad):
                color = (indice * 7, 20 if etiqueta == "NORMAL" else 40, 30)
                Image.new("RGB", (16, 16), color=color).save(destino / f"{carpeta}_{etiqueta}_{indice}.png")
    crear_manifiesto_division_estratificada(data_dir, Manifest, random_state=SEMILLA)


def _primeros_lotes(dataset, cuantos: int = 2) -> list[tuple[bytes, tuple[int, ...]]]:
    """Devolver las huellas de los primeros lotes de un dataset."""
    lotes = []
    for features, labels in dataset.take(cuantos):
        huellas = np.ascontiguousarray(features.numpy()).tobytes()
        lotes.append((huellas, tuple(int(valor) for valor in labels.numpy())))
    return lotes


def test_configurar_reproducibilidad_activa_el_determinismo() -> None:
    """La configuracion commun deja constancia de como se fijaron las semillas."""
    configurar_reproducibilidad(SEMILLA)
    descripcion = reproducibility.descripcion_reproducibilidad(SEMILLA)

    assert descripcion["semilla"] == SEMILLA
    assert descripcion["operaciones_deterministas"] is True
    assert descripcion["tensorflow_deterministic_ops"] == "1"
    assert descripcion["shuffle_semilla"] is True
    assert descripcion["augmentation_semilla"] is True
    assert descripcion["augmentation_determinista_en_tfdata"] is True


def test_todo_el_proyecto_usa_la_misma_semilla() -> None:
    """No puede haber una semilla distinta escondida en cada modulo."""
    assert SEMILLA == 42
    assert run_real_training.SEMILLA == SEMILLA
    assert optimizacion_mobilenetv2.SEMILLA == SEMILLA
    assert reproducibility.SEMILLA == SEMILLA


def test_las_capas_de_augmentation_reciben_la_semilla() -> None:
    """RandomRotation y RandomZoom deben llevar ``seed`` explicito."""
    import tensorflow as tf

    capas = run_real_training.crear_capas_augmentation()
    aleatorias = [
        capa
        for capa in capas
        if isinstance(capa, (tf.keras.layers.RandomRotation, tf.keras.layers.RandomZoom))
    ]

    assert len(aleatorias) == 2
    for capa in aleatorias:
        assert capa.seed == SEMILLA


def test_reiniciar_semilla_restablece_el_estado_aleatorio() -> None:
    """Dos reinicios seguidos producen la misma secuencia de numeros."""
    import tensorflow as tf

    reiniciar_semilla(SEMILLA)
    primera = tf.random.normal((4,)).numpy().tolist()
    reiniciar_semilla(SEMILLA)
    segunda = tf.random.normal((4,)).numpy().tolist()

    assert primera == segunda


def test_el_shuffle_del_train_es_reproducible(tmp_path: Path) -> None:
    """Dos pipelines construidos con la misma semilla entregan los mismos lotes."""
    configurar_reproducibilidad(SEMILLA)
    data_dir = tmp_path / "chest_xray"
    manifiesto = tmp_path / "split.csv"
    _crear_dataset_minimo(data_dir, manifiesto)

    primero = construir_pipelines_datos(data_dir, image_size=(16, 16), batch_size=4, manifiesto_division=manifiesto)
    segundo = construir_pipelines_datos(data_dir, image_size=(16, 16), batch_size=4, manifiesto_division=manifiesto)

    assert _primeros_lotes(primero["train"]) == _primeros_lotes(segundo["train"])


def test_la_augmentation_del_train_es_reproducible(tmp_path: Path) -> None:
    """Dos pipelines de augmentation con la misma semilla dan los mismos pixeles.

    Sin ``seed`` en las capas y sin ``deterministic=True`` en el ``map``, estas dos
    ejecuciones divergen: es exactamente el defecto que se corrigio. Cada pipeline
    se construye sobre su propio dataset base porque ``shuffle`` usa
    ``reshuffle_each_iteration=True`` y volver a iterar el mismo objeto re-baraja.
    """
    configurar_reproducibilidad(SEMILLA)
    data_dir = tmp_path / "chest_xray"
    manifiesto = tmp_path / "split.csv"
    _crear_dataset_minimo(data_dir, manifiesto)

    primero = run_real_training.construir_pipeline_augmentation(
        construir_pipelines_datos(data_dir, image_size=(16, 16), batch_size=4, manifiesto_division=manifiesto)["train"]
    )
    segundo = run_real_training.construir_pipeline_augmentation(
        construir_pipelines_datos(data_dir, image_size=(16, 16), batch_size=4, manifiesto_division=manifiesto)["train"]
    )

    assert _primeros_lotes(primero) == _primeros_lotes(segundo)


def test_el_oversampling_es_reproducible(tmp_path: Path) -> None:
    """La variante con oversampling tambien entrega siempre el mismo orden."""
    configurar_reproducibilidad(SEMILLA)
    data_dir = tmp_path / "chest_xray"
    manifiesto = tmp_path / "split.csv"
    _crear_dataset_minimo(data_dir, manifiesto)

    primero, resumen_primero = tuning_desbalance_clases.construir_train_oversampling(
        manifiesto, image_size=(16, 16), batch_size=4
    )
    segundo, resumen_segundo = tuning_desbalance_clases.construir_train_oversampling(
        manifiesto, image_size=(16, 16), batch_size=4
    )

    assert _primeros_lotes(primero) == _primeros_lotes(segundo)
    assert resumen_primero == resumen_segundo


def test_el_criterio_de_seleccion_no_incluye_la_tolerancia_de_recall() -> None:
    """La seleccion se define solo por las cuatro metricas acordadas."""
    from src.training.tuning_mobilenetv2 import CRITERIO_METRICAS

    assert tuple(CRITERIO_METRICAS) == ("balanced_accuracy", "roc_auc", "f1", "accuracy")
    assert not hasattr(tuning_mobilenetv2, "TOLERANCIA_RECALL")


def test_el_modelo_also_es_reproducible() -> None:
    """Dos construcciones del modelo tras reiniciar la semilla dan los mismos pesos."""
    import numpy as np

    vectores = []
    for _ in range(2):
        reiniciar_semilla(SEMILLA)
        model = tuning_mobilenetv2.construir_modelo_mobilenetv2_tunable(dropout=0.3, learning_rate=1e-3)
        vectores.append(np.concatenate([np.asarray(pesos).ravel() for pesos in model.get_weights()]))

    assert vectores[0].shape == vectores[1].shape
    assert np.array_equal(vectores[0], vectores[1])


def test_la_construccion_del_pipeline_ocurre_despues_de_fijar_la_semilla() -> None:
    """El orden documentado en ``entrenar_variante`` debe cumplirse de verdad."""
    import inspect

    fuente = inspect.getsource(optimizacion_mobilenetv2.entrenar_variante)
    cuerpo = fuente.split('"""', 2)[-1]

    assert cuerpo.index("reiniciar_semilla") < cuerpo.index("construir_pipeline")


@pytest.mark.parametrize(
    "variante_id, indice, capa",
    [
        ("finetune_block16", 143, "block_16_expand"),
        ("finetune_block13", 116, "block_13_expand"),
        ("finetune_block10", 90, "block_10_expand"),
    ],
)
def test_las_variantes_de_fine_tuning_conservan_su_bloque(
    variante_id: str, indice: int, capa: str
) -> None:
    """Cada variante de fine-tuning sigue descongelando el bloque que le toca."""
    variantes = {
        variante["id"]: variante
        for variante in optimizacion_mobilenetv2.construir_variantes(
            {"learning_rate": 1e-3, "dropout": 0.3, "epochs": 3}
        )
    }

    assert variantes[variante_id]["fine_tune_from"] == indice
    assert variantes[variante_id]["capa_descongelada"] == capa
