Primeros pasos
==============

Este documento describe cómo preparar el proyecto en una instalación limpia.

El dataset crudo se administra con **DVC** (remoto Google Drive) y no con S3.
Para recuperarlo:

.. code-block:: powershell

   dvc pull data/raw/chest_xray.dvc

Después se puede ejecutar el flujo completo con la CLI del proyecto:

.. code-block:: powershell

   neumonia eda
   neumonia prepare
   neumonia augment
   neumonia sensibilidad
   neumonia combinado
   neumonia umbral
   neumonia final
   neumonia test
   neumonia test-suite

``neumonia combinado``, ``neumonia umbral`` y ``neumonia final`` son las tres
etapas del flujo de decisión y se ejecutan en ese orden. Las tres solo usan
``train`` y ``validation``.

``neumonia test`` cierra el flujo: evalúa el test original (624 imágenes) con el
modelo definitivo y el umbral ya congelado. No interviene en ninguna decisión
previa y no lleva banderas de confirmación.

En este repositorio el entorno ya está preparado en ``.venv``. Si el Python del
sistema no tiene TensorFlow, usar el intérprete del entorno:

.. code-block:: powershell

   .\.venv\Scripts\python.exe -m src.cli combinado

La guía detallada de ejecución está en ``references/guia_ejecucion.md`` y la
documentación técnica en ``references/documentacion_proyecto.md``.
