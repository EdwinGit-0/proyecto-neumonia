Primeros pasos
==============

Este documento describe cómo preparar el proyecto en una instalación limpia.

El dataset crudo se administra con **DVC** (remoto Google Drive) y no con S3.
Para recuperarlo:

.. code-block:: powershell

   dvc pull data/raw/chest_xray.dvc

Después se puede ejecutar el flujo base con la CLI del proyecto:

.. code-block:: powershell

   neumonia eda
   neumonia prepare
   neumonia augment
   neumonia train
   neumonia evaluate
   neumonia test

Las etapas de mejora del modelo final (tuning y desbalance) se ejecutan de
forma independiente:

.. code-block:: powershell

   neumonia tune
   neumonia desbalance

La guía detallada de ejecución está en ``references/guia_ejecucion.md`` y la
documentación técnica en ``references/documentacion_proyecto.md``.