Comandos
========

La interfaz recomendada del proyecto es la CLI ``neumonia``, registrada con la
instalación del paquete (``pip install -e .``). Comandos disponibles:

.. code-block:: powershell

   neumonia eda           # Análisis exploratorio de datos
   neumonia prepare       # Manifiesto estratificado 80/20 (test original intacto) y pipelines
   neumonia augment       # Figura de ejemplos de augmentación
   neumonia sensibilidad  # Sensibilidad de hiperparámetros: 21 corridas sobre train/validation
   neumonia combinado      # MobileNetV2 con la estrategia única COMBINADO, medido en validation
   neumonia umbral        # Congela el umbral de decisión con las probabilidades de validation
   neumonia final         # Reentrena el modelo definitivo sobre train + validation
   neumonia test          # Evaluación del test original con el modelo definitivo
   neumonia evaluar       # Muestra los artefactos guardados sin volver a entrenar
   neumonia test-suite    # Ejecuta la suite con pytest
   neumonia run           # Flujo completo del proyecto, incluido el test final

Orden obligatorio
-----------------

Las decisiones se cierran enteramente sobre ``train`` y ``validation``; el test
original se carga **al final**, con el umbral ya congelado:

.. code-block:: powershell

   neumonia combinado      # fija arquitectura e hiperparámetros ya, mide en validation
   neumonia umbral        # decide el umbral con validation
   neumonia final         # entrena con train + validation la decisión ya cerrada
   neumonia test          # evaluación del test original
   neumonia evaluar       # muestra el informe final guardado

Tratamiento de desbalance
------------------------

MobileNetV2 con **COMBINADO** (oversampling 50/50 sobre ``train`` más
``class_weight`` en la función de pérdida) es la única estrategia de desbalance
del proyecto: es una premisa metodológica cerrada, no el resultado de comparar
variantes.

El tratamiento se aplica **solo** a los splits de entrenamiento.
``validation`` y ``test`` conservan su distribución original y nunca se
reponderan.

Test final
----------

``neumonia test`` se ejecuta como cualquier otra etapa: lee el test original
(624 imágenes: 234 ``NORMAL`` y 390 ``PNEUMONIA``), aplica el umbral congelado y
guarda el informe final y el CSV de métricas en ``results/final/``, más la
matriz de confusión, la curva ROC y el gráfico de métricas en
``reports/figures/``.

No lleva opciones: el umbral, la arquitectura y la estrategia ya están cerrados
antes de que el test se lea, y el test no participa en ninguna decisión.

El Makefile conserva reglas heredadas del template original (incluidas las de
sincronización con S3) y **no** es el entry point recomendado para este
proyecto; el dataset se administra con DVC y las etapas se ejecutan con la
CLI, tal como se describe en ``references/guia_ejecucion.md``.
