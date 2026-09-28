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
from src.training import flujo_final, pipeline_entrenamiento, sensitivity, tratamiento_desbalance
from src.utils import reproducibility
from src.utils.reproducibility import SEMILLA, configurar_reproducibilidad, reiniciar_semilla


def _crear_dataset_minimo(data_dir: Path, manifiesto: Path) -> None:
    """Crear un dataset sintetico con train original y test original."""
    for etiqueta in ["NORMAL", "PNEUMONIA"]:
        for carpeta, cantidad in [("train", 10), ("test", 4)]:
            destino = data_dir / carpeta / etiqueta
            destino.mkdir(parents=True, exist_ok=True)
            for indice in range(cantidad):
                color = (indice * 7, 20 if etiqueta == "NORMAL" else 40, 30)
                Image.new("RGB", (16, 16), color=color).save(destino / f"{carpeta}_{etiqueta}_{indice}.png")
    crear_manifiesto_division_estratificada(data_dir, manifiesto, random_state=SEMILLA)


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
    assert pipeline_entrenamiento.SEMILLA == SEMILLA
    assert flujo_final.SEMILLA == SEMILLA
    assert sensitivity.SEMILLA == SEMILLA
    assert tratamiento_desbalance.SEMILLA == SEMILLA
    assert reproducibility.SEMILLA == SEMILLA


def test_las_capas_de_augmentation_reciben_la_semilla() -> None:
    """RandomRotation y RandomZoom deben llevar ``seed`` explicito."""
    import tensorflow as tf

    capas = pipeline_entrenamiento.crear_capas_augmentation()
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

    primero = pipeline_entrenamiento.construir_pipeline_augmentation(
        construir_pipelines_datos(data_dir, image_size=(16, 16), batch_size=4, manifiesto_division=manifiesto)["train"]
    )
    segundo = pipeline_entrenamiento.construir_pipeline_augmentation(
        construir_pipelines_datos(data_dir, image_size=(16, 16), batch_size=4, manifiesto_division=manifiesto)["train"]
    )

    assert _primeros_lotes(primero) == _primeros_lotes(segundo)


def test_el_oversampling_es_reproducible(tmp_path: Path) -> None:
    """La variante con oversampling tambien entrega siempre el mismo orden."""
    configurar_reproducibilidad(SEMILLA)
    data_dir = tmp_path / "chest_xray"
    manifiesto = tmp_path / "split.csv"
    _crear_dataset_minimo(data_dir, manifiesto)

    primero, pesos_primero, diagnostico_primero = tratamiento_desbalance.construir_train_combinado(
        manifiesto, image_size=(16, 16), batch_size=4
    )
    segundo, pesos_segundo, diagnostico_segundo = tratamiento_desbalance.construir_train_combinado(
        manifiesto, image_size=(16, 16), batch_size=4
    )

    assert _primeros_lotes(primero) == _primeros_lotes(segundo)
    assert pesos_primero == pesos_segundo
    assert diagnostico_primero == diagnostico_segundo


def test_el_oversampling_solo_afecta_a_los_splits_pedidos(tmp_path: Path) -> None:
    """La composicion oversampleada sale solo de los splits indicados, nunca de test."""
    configurar_reproducibilidad(SEMILLA)
    data_dir = tmp_path / "chest_xray"
    manifiesto = tmp_path / "split.csv"
    _crear_dataset_minimo(data_dir, manifiesto)

    from src.data.splitting import cargar_manifiesto_division

    marco = cargar_manifiesto_division(manifiesto)
    filas_train = int((marco["split"] == "train").sum())
    filas_combinado = int(marco["split"].isin(["train", "val"]).sum())

    _, _, diagnostico = tratamiento_desbalance.construir_train_combinado(
        manifiesto, image_size=(16, 16), batch_size=4, splits=("train",)
    )
    _, _, diagnostico_combinado = tratamiento_desbalance.construir_train_combinado(
        manifiesto, image_size=(16, 16), batch_size=4, splits=("train", "val")
    )
    resumen = diagnostico["detalle_oversampling"]
    resumen_combinado = diagnostico_combinado["detalle_oversampling"]

    assert diagnostico["splits"] == ["train"]
    assert resumen["total_original"] == filas_train
    assert diagnostico_combinado["splits"] == ["train", "val"]
    assert resumen_combinado["total_original"] == filas_combinado
    assert resumen["total_oversampled"] == 2 * max(
        resumen["NORMAL_original"], resumen["PNEUMONIA_original"]
    )
    assert resumen_combinado["total_oversampled"] == 2 * max(
        resumen_combinado["NORMAL_original"], resumen_combinado["PNEUMONIA_original"]
    )


def test_el_tratamiento_del_desbalance_rechaza_el_test(tmp_path: Path) -> None:
    """Ni oversampling ni class weights pueden aplicarse al test original."""
    configurar_reproducibilidad(SEMILLA)
    data_dir = tmp_path / "chest_xray"
    manifiesto = tmp_path / "split.csv"
    _crear_dataset_minimo(data_dir, manifiesto)

    with pytest.raises(ValueError, match="test"):
        tratamiento_desbalance.aplicar_oversampling(
            tratamiento_desbalance.subconjunto_entrenamiento(manifiesto, ("test",))
        )
    with pytest.raises(ValueError, match="test"):
        tratamiento_desbalance.calcular_pesos_clase(manifiesto, splits=("test",))


def test_el_criterio_de_seleccion_es_el_acordado() -> None:
    """La seleccion se define solo por las cuatro metricas acordadas."""
    assert tuple(sensitivity.CRITERIO_METRICAS) == ("balanced_accuracy", "roc_auc", "f1", "accuracy")
    assert not hasattr(flujo_final, "TOLERANCIA_RECALL")
    assert not hasattr(sensitivity, "TOLERANCIA_RECALL")


def test_el_modelo_also_es_reproducible() -> None:
    """Dos construcciones del modelo tras reiniciar la semilla dan los mismos pesos."""
    from src.models.architectures import construir_modelo_proyecto

    vectores = []
    for _ in range(2):
        reiniciar_semilla(SEMILLA)
        model = construir_modelo_proyecto("MobileNetV2", dropout=0.3, learning_rate=1e-3)
        vectores.append(np.concatenate([np.asarray(pesos).ravel() for pesos in model.get_weights()]))

    assert vectores[0].shape == vectores[1].shape
    assert np.array_equal(vectores[0], vectores[1])


def test_la_construccion_del_pipeline_ocurre_despues_de_fijar_la_semilla() -> None:
    """El orden documentado en la corrida debe cumplirse de verdad."""
    import inspect

    fuente = inspect.getsource(flujo_final.ejecutar_combinado)

    assert fuente.index("reiniciar_semilla") < fuente.index("construir_pipeline_combinado")


def test_el_flujo_final_entrena_una_sola_configuracion() -> None:
    """El flujo final usa COMBINADO sobre una sola arquitectura: no hay rejilla que recorrer."""
    import src.training.tratamiento_desbalance as tratamiento_desbalance

    assert tratamiento_desbalance.NOMBRE_ESTRATEGIA == "combinado"
    assert flujo_final.MODELO_FINAL == "MobileNetV2"
    assert flujo_final.CONFIG_FINAL_ESPERADA == {"learning_rate": 1e-3, "dropout": 0.3, "epochs": 3}
