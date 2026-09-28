"""Tratamiento del desbalance de clases: estrategia única COMBINADO.

Este módulo implementa la **única** estrategia de tratamiento del desbalance que usa
el proyecto: ``COMBINADO`` = oversampling 50/50 + class weights.

Ya no existe comparación de estrategias. La decisión de tratar el desbalance con ambas
mecanismos a la vez es una decisión del proyecto, cerrada antes de cualquier evaluación
de test; este módulo solo la ejecuta y deja constancia de cómo se calculó cada cosa.

Qué significa exactamente COMBINADO
-----------------------------------

``train`` original = 1.073 NORMAL + 3.100 PNEUMONIA (4.173 imágenes, desbalance 2,89:1).

1. **Oversampling 50/50** sobre ``train``. Se añade NORMAL con muestreo **con
   reemplazo** hasta igualar el recuento de la clase mayoritaria, dejando
   3.100 NORMAL + 3.100 PNEUMONIA = **6.200 imágenes efectivas**. La clase
   mayoritaria nunca se toca. El aumento ocurre **solo** en los splits de
   entrenamiento: ``validation`` y ``test`` conservan su distribución original.

2. **Class weights** balances, calculados sobre la distribución de
   ``trainOriginal``, es decir **antes** del oversampling, y aplicados **solo** a la
   función de pérdida del entrenamiento. Nunca se aplican a ``validation`` ni a
   ``test``: esos conjuntos se miden con la distribución real, sin reponderar.

Advertencia sobre la doble corrección
-------------------------------------

Los ``class_weight`` balanceados deben calcularse sobre la distribución **anterior**
al oversampling. Si se calcularan sobre el conjunto ya equilibrado a 50/50, ambos
pesos serían exactamente ``1.0`` y la ponderación sería un no-op, con lo que
COMBINADO sería idéntico bit a bit a "oversampling solo".

La consecuencia de calcularlos sobre la distribución original es que las dos
mecanismos se **combinan**: el oversampling iguala la frecuencia de muestreo
(razón 1,00) y los pesos añaden además una razón de 2,89, de modo que la
minoritaria recibe un refuerzo total de 2,89x respecto de la mayoritaria. Es decir,
COMBINADO **no se limita a corregir** el desbalance: lo corrige y además desplaza el
equilibrio hacia NORMAL. El efecto neto medido en validación fue una especificidad
mayor y un recall menor que con las demás alternativas.

Este comportamiento es intencionado y queda registrado en cada artefacto que genera
el flujo (``refuerzo_total_minoritaria``), de modo que sea auditable y no una
sorpresa. La razón se expone en :func:`diagnostico_combinado`.

Nada de este módulo lee ``validation`` ni ``test`` para el tratamiento: la única
excepción es leer el manifiesto completo, que contiene las etiquetas de los tres
splits, y del que aquí solo se seleccionan filas de entrenamiento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data.preprocessing import cargar_imagen, imagen_a_arreglo
from src.data.splitting import cargar_manifiesto_division
from src.utils.reproducibility import SEMILLA


# Únicos splits sobre los que se permite aplicar el tratamiento. El split de test
# queda excluido por construcción: el tratamiento nunca lo alcanza.
SPLITS_PERMITIDOS = ("train", "val")

# Definición de la estrategia. No hay lista de estrategias porque hay una sola.
ESTRATEGIA_COMBINADO: dict[str, Any] = {
    "id": "combinado",
    "descripcion": (
        "Oversampling 50/50 de la minoritaria NORMAL en train (muestreo con reemplazo) "
        "mas class weights balanceados calculados sobre la distribucion original de train. "
        "El tratamiento se aplica solo a los splits de entrenamiento; validation y test "
        "conservan su distribucion y nunca se reponderan."
    ),
    "oversampling": True,
    "class_weight": True,
    "class_weights_sobre": "distribucion_original_pre_oversampling",
    "proporcion_objetivo": "50/50",
}

NOMBRE_ESTRATEGIA = ESTRATEGIA_COMBINADO["id"]


def construir_dataset_tensor(
    df: pd.DataFrame, image_size: tuple[int, int], batch_size: int = 32
) -> tf.data.Dataset:
    """Réplica exacta del constructor interno de ``src.data.datasets`` (mismo shuffle/batch/prefetch)."""

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
    return (
        dataset.shuffle(buffer_size=max(1000, len(df)), seed=SEMILLA, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def subconjunto_entrenamiento(
    ruta_manifiesto: Path, splits: Sequence[str] = SPLITS_PERMITIDOS
) -> pd.DataFrame:
    """Devolver las filas del manifiesto que forman un conjunto de entrenamiento.

    ``splits=("train",)`` reproduce el ``train`` original (donde se elige la
    configuración) y ``splits=("train", "val")`` el conjunto combinado que entrena el
    modelo definitivo. El split ``test`` se rechaza.
    """
    prohibidos = [split for split in splits if split == "test"]
    if prohibidos:
        raise ValueError(
            f"El tratamiento del desbalance no puede aplicarse a test: {prohibidos}. "
            "COMBINADO se aplica solo a train (y a train+val en el modelo definitivo)."
        )

    dataframe = cargar_manifiesto_division(ruta_manifiesto)
    seleccion = dataframe[dataframe["split"].isin(list(splits))].copy()
    if seleccion.empty:
        raise ValueError(f"El manifiesto no contiene filas para los splits {list(splits)}")
    return seleccion


def aplicar_oversampling(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Igualar la minoritaria con la mayoritaria duplicando filas con reemplazo.

    Se opera solo sobre el dataframe de entrenamiento recibido: la clase mayoritaria
    nunca se toca y no hay ninguna forma de que ``validation`` o ``test`` queden
    alterados, porque no forman parte del dataframe de entrada.
    """
    df_normal = df[df["target"] == 0]
    df_pneumonia = df[df["target"] == 1]
    if df_normal.empty or df_pneumonia.empty:
        raise ValueError("El oversampling necesita ambas clases presentes en el entrenamiento.")

    minoritaria, mayoritaria = (
        (df_normal, df_pneumonia) if len(df_normal) <= len(df_pneumonia) else (df_pneumonia, df_normal)
    )
    etiqueta_minoritaria = "NORMAL" if len(df_normal) <= len(df_pneumonia) else "PNEUMONIA"
    faltantes = len(mayoritaria) - len(minoritaria)
    extra = minoritaria.sample(n=faltantes, replace=True, random_state=SEMILLA)
    equilibrado = pd.concat([df, extra], ignore_index=True)
    equilibrado = equilibrado.sample(frac=1.0, random_state=SEMILLA).reset_index(drop=True)

    conteo = equilibrado["target"].value_counts()
    return equilibrado, {
        "clase_minoritaria": etiqueta_minoritaria,
        "NORMAL_original": int(len(df_normal)),
        "PNEUMONIA_original": int(len(df_pneumonia)),
        "NORMAL_oversampled": int(conteo.get(0, 0)),
        "PNEUMONIA_oversampled": int(conteo.get(1, 0)),
        "total_original": int(len(df)),
        "total_oversampled": int(len(equilibrado)),
        "duplicadas": int(faltantes),
    }


def calcular_pesos_clase(
    ruta_manifiesto: Path, splits: Sequence[str] = ("train",)
) -> dict[int, float]:
    """Pesos balanceados por clase, calculados sobre la distribución **sin** oversampling.

    Se derivan del recuento real de los splits indicados **antes** de duplicar filas.
    Es deliberado: calculados sobre el conjunto ya equilibrado a 50/50 darían
    ``1.0`` para ambas clases y la ponderación no tendría efecto. Véase el docstring
    del módulo para la consecuencia de esta decisión.
    """
    df_entrenamiento = subconjunto_entrenamiento(ruta_manifiesto, splits)
    y_train = df_entrenamiento["target"].to_numpy().astype(int)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    return {int(clase): float(peso) for clase, peso in zip([0, 1], weights)}


def diagnostico_combinado(
    ruta_manifiesto: Path, splits: Sequence[str] = ("train",)
) -> dict[str, Any]:
    """Describir numéricamente qué hace COMBINADO, para que quede en los artefactos.

    Calcula, sobre los splits indicados y sin entrenar nada:

    - la composición antes y después del oversampling;
    - los ``class_weight`` que se aplicarán a la loss;
    - la razón de refuerzo total de la minoritaria frente a la mayoritaria, que es el
      número que revela que COMBINADO sobrecorrige el desbalance.
    """
    df = subconjunto_entrenamiento(ruta_manifiesto, splits)
    equilibrado, resumen = aplicar_oversampling(df)
    pesos = calcular_pesos_clase(ruta_manifiesto, splits)

    razon_muestreo = resumen["NORMAL_oversampled"] / max(1, resumen["PNEUMONIA_oversampled"])
    razon_pesos = pesos[0] / pesos[1]
    refuerzo = razon_muestreo * razon_pesos

    return {
        "estrategia": NOMBRE_ESTRATEGIA,
        "splits": list(splits),
        "composicion_original": {
            "NORMAL": resumen["NORMAL_original"],
            "PNEUMONIA": resumen["PNEUMONIA_original"],
            "total": resumen["total_original"],
        },
        "composicion_tras_oversampling": {
            "NORMAL": resumen["NORMAL_oversampled"],
            "PNEUMONIA": resumen["PNEUMONIA_oversampled"],
            "total": resumen["total_oversampled"],
        },
        "filas_duplicadas": resumen["duplicadas"],
        "class_weight": {str(clase): peso for clase, peso in pesos.items()},
        "class_weight_calculado_sobre": "distribucion_original_pre_oversampling",
        "razon_muestreo_tras_oversampling": razon_muestreo,
        "razon_pesos_por_clase": razon_pesos,
        "refuerzo_total_minoritaria": refuerzo,
        "advertencia_doble_correccion": (
            "El oversampling iguala la frecuencia (razon %.2f) y los class_weights anaden "
            "una razon de %.2f, de modo que NORMAL recibe un refuerzo total de %.2fx frente a "
            "PNEUMONIA. COMBINADO no solo corrige el desbalance: lo invierte parcialmente. "
            "Los class_weight NO se aplican a validation ni a test." % (razon_muestreo, razon_pesos, refuerzo)
        ),
        "filas_efectivas_entrenamiento": int(len(equilibrado)),
    }


def construir_train_combinado(
    ruta_manifiesto: Path,
    image_size: tuple[int, int],
    batch_size: int,
    splits: Sequence[str] = ("train",),
) -> tuple[tf.data.Dataset, dict[int, float], dict[str, Any]]:
    """Construir el pipeline de entrenamiento de COMBINADO y devolver sus class weights.

    Devuelve el dataset ya equilibrado a 50/50, los ``class_weight`` que la loss debe
    aplicar y un diagnóstico auditable. ``validation`` y ``test`` no se tocan.
    """
    df_entrenamiento = subconjunto_entrenamiento(ruta_manifiesto, splits)
    equilibrado, resumen = aplicar_oversampling(df_entrenamiento)
    pesos = calcular_pesos_clase(ruta_manifiesto, splits)

    diagnostico = diagnostico_combinado(ruta_manifiesto, splits)
    diagnostico["splits"] = list(splits)
    diagnostico["detalle_oversampling"] = resumen

    dataset = construir_dataset_tensor(equilibrado, image_size=image_size, batch_size=batch_size)
    return dataset, pesos, diagnostico
