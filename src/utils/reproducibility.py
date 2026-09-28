"""Reproducibilidad del proyecto: una sola semilla y una sola forma de fijarla.

Antes de este modulo, cada etapa fijaba la semilla a su manera y siempre despues
de construir los pipelines de ``tf.data``. Como ``Dataset.shuffle`` y las capas de
augmentation se creaban sin semilla, dos ejecuciones del mismo codigo con
``SEMILLA = 42`` entrenaban sobre imagenes distintas y producian resultados
distintos. Este modulo centraliza el control:

- ``SEMILLA`` es la unica semilla del proyecto.
- ``configurar_reproducibilidad()`` debe llamarse **antes** de construir cualquier
  pipeline, modelo o callback.
- Las capas de augmentation y los ``shuffle`` reciben ``seed=SEMILLA``, de modo que
  usan generacion sin estado y el resultado no depende del orden de ejecucion.

No se modifica la logica conceptual del preprocesamiento ni de la augmentation
(siguen siendo rotacion y zoom ligeros, sin volteo horizontal); solo se hace que
sean reproducibles.
"""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


SEMILLA = 42

_configurado = False


def configurar_reproducibilidad(semilla: int = SEMILLA) -> None:
    """Fijar la semilla y activar las operaciones deterministas de TensorFlow.

    Es idempotente: se puede llamar varias veces sin efectos adverse. Debe
    invocarse antes de construir cualquier ``tf.data.Dataset``, modelo o callback,
    porque ``enable_op_determinism()`` solo afecta a las operaciones creadas
    despues de la llamada.
    """
    global _configurado

    # Se asignan de forma directa y no con setdefault: si el entorno ya traia un
    # valor distinto, heredado de otra ejecucion, debe quedar corregido.
    # PYTHONHASHSEED solo tiene efecto si se fija antes de que arranque el
    # interprete; dentro del proceso solo se puede registrar el valor pretendido.
    os.environ["PYTHONHASHSEED"] = str(semilla)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if not _configurado:
        tf.config.experimental.enable_op_determinism()
        _configurado = True

    random.seed(semilla)
    np.random.seed(semilla)
    tf.keras.utils.set_random_seed(semilla)


def reiniciar_semilla(semilla: int = SEMILLA) -> None:
    """Reiniciar el estado aleatorio antes de entrenar una corrida.

    Cada corrida la llama justo despues de
    ``tf.keras.backend.clear_session()`` y antes de construir su pipeline, de modo
    que todas ven exactamente el mismo orden de ejemplos y la misma secuencia de
    augmentation.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    tf.keras.utils.set_random_seed(semilla)


def descripcion_reproducibilidad(semilla: int = SEMILLA) -> dict[str, object]:
    """Describir el estado de reproducibilidad para dejarlo en los artefactos.

    TensorFlow 2.15 no expone ni ``are_op_determinism_enabled()`` ni un lector de
    la semilla global, asi que se informa de la bandera que activa este propio
    modulo y de las variables de entorno que realmente gobiernan el determinismo.
    """
    return {
        "semilla": semilla,
        "operaciones_deterministas": bool(_configurado),
        "tensorflow_deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "shuffle_semilla": True,
        "augmentation_semilla": True,
        "augmentation_determinista_en_tfdata": True,
    }
