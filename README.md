proyecto-neumonia
==============================

Clasificacion de imagenes de rayos X de torax para la identificacion de neumonia mediante redes neuronales convolucionales

# Proyecto de clasificación de neumonía

Clasificación de imágenes de rayos X de tórax en las clases `NORMAL` y
`PNEUMONIA` mediante redes convolucionales con transferencia de aprendizaje.

## Estado actual

El EDA, la preparación de imágenes, el entrenamiento, la evaluación y la
comparación de tres modelos fueron ejecutados sobre el dataset real. El modelo
seleccionado actualmente es **MobileNetV2**, elegido únicamente con las métricas
de `validation` del reparto experimental 70/15/15 (train 4.099, validation 879
y test 878, sin duplicados entre conjuntos). El conjunto `test` quedó reservado
y solo se utilizó para la evaluación final de MobileNetV2.

Resultados guardados: `models/model_results.json`.

## Dataset

El dataset está gestionado mediante DVC y se recupera en
`data/raw/chest_xray/`. Contiene 5.856 imágenes JPEG con esta estructura:

```text
data/raw/chest_xray/
├── train/NORMAL/
├── train/PNEUMONIA/
├── val/NORMAL/
├── val/PNEUMONIA/
├── test/NORMAL/
└── test/PNEUMONIA/
```

La estructura anterior es la del dataset crudo. Para el modelado se utiliza la
partición experimental estratificada 70/15/15 (semilla 42) registrada en
`data/interim/stratified_split_70_15_15.csv`, sin duplicados entre los tres
conjuntos.

La configuración DVC usa un remoto Google Drive. El detalle técnico completo
está en [references/documentacion_proyecto.md](references/documentacion_proyecto.md).

## Instalación

En PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para recuperar los datos:

```powershell
dvc pull data/raw/chest_xray.dvc
```

## Ejecución

EDA:

```powershell
.\.venv\Scripts\python.exe -m src.data.make_dataset
```

Entrenamiento y evaluación real de los tres modelos:

```powershell
.\.venv\Scripts\python.exe -m src.training.run_real_training
```

Este comando vuelve a entrenar los modelos. Para consultar los resultados ya
existentes, leer `models/model_results.json` sin ejecutarlo.

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Organización

- `src/data/`: EDA, preprocesamiento y datasets TensorFlow.
- `src/models/`: arquitecturas, métricas y comparación.
- `src/training/`: utilidades y entry point del entrenamiento real.
- `notebooks/`: notebook de comprensión de datos y EDA.
- `models/`: checkpoints y resultados.
- `reports/figures/`: figuras EDA y evaluación.
- `references/`: documentación técnica y guía de ejecución.
- `tests/`: 23 pruebas automatizadas.

El despliegue productivo y la validación clínica están fuera del alcance de
este proyecto académico.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
