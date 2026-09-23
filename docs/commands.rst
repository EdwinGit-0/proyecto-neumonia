Comandos
========

La interfaz recomendada del proyecto es la CLI ``neumonia``, registrada con la
instalación del paquete (``pip install -e .``). Comandos disponibles:

.. code-block:: powershell

   neumonia eda          # Análisis exploratorio de datos
   neumonia prepare      # Manifiesto estratificado 80/20 (train original) y pipelines
   neumonia augment      # Figura de ejemplos de augmentación
   neumonia train        # Entrenar, evaluar, comparar y seleccionar VGG16, ResNet50 y MobileNetV2
   neumonia tune         # Tuning experimental de MobileNetV2 (lr, dropout y fine-tuning)
   neumonia desbalance   # Experimentos de desbalance: class weights y oversampling
   neumonia evaluate     # Consultar resultados guardados sin volver a entrenar
   neumonia test         # Ejecutar la suite con pytest
   neumonia run          # Flujo base completo (EDA -> prepare -> augment -> train -> evaluate -> test)

El Makefile conserva reglas heredadas del template original (incluidas las de
sincronización con S3) y **no** es el entry point recomendado para este
proyecto; el dataset se administra con DVC y las etapas se ejecutan con la
CLI, tal como se describe en ``references/guia_ejecucion.md``.