# Documentación técnica del proyecto

## 1. Información general

**Título:** clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA` mediante modelos convolucionales con transferencia de aprendizaje.

**Objetivo:** entrenar, evaluar y comparar VGG16, ResNet50 y MobileNetV2 con una configuración comparable, y seleccionar el mejor modelo sobre el conjunto de validación con una regla reproducible.

**Alcance:** análisis exploratorio, preparación de imágenes, entrenamiento experimental, evaluación y comparación de tres arquitecturas. El resultado es académico y experimental.

**Limitaciones:** no es un sistema clínico ni una herramienta de diagnóstico; no existe validación clínica externa; el dataset está desbalanceado; no se realiza calibración ni despliegue productivo.

## 2. Dataset

El proyecto utiliza un dataset de radiografías de tórax administrado con DVC, restaurado en `data/raw/chest_xray/` con la estructura `train/`, `val/` y `test/`, cada uno con las clases `NORMAL` y `PNEUMONIA`. La clase `NORMAL` se codifica como `0` y `PNEUMONIA` como `1`.

Estructura cruda del dataset (5.856 imágenes `.jpeg`):

| División | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| train | 1.341 | 3.875 | 5.216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |
| **Total** | **1.583** | **4.273** | **5.856** |

El `val` original de 16 imágenes pertenece a la estructura cruda del dataset y **no se utilizó como validación experimental**.

Para el modelado se genera un reparto experimental estratificado 70/15/15 (semilla 42, agrupación por hash SHA-256 para evitar duplicados entre conjuntos) durante la preparación de datos. El manifiesto es `data/interim/stratified_split_70_15_15.csv`:

| Conjunto | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| train | 1.108 | 2.991 | 4.099 |
| validation | 238 | 641 | 879 |
| test | 237 | 641 | 878 |
| **Total** | **1.583** | **4.273** | **5.856** |

El EDA verificó las imágenes y no detectó archivos corruptos, pero detectó 32 rutas duplicadas por SHA-256 dentro de los splits crudos. El manifiesto experimental agrupa por hash, por lo que no hay duplicados entre `train`, `validation` y `test`. El archivo `data/raw/chest_xray.dvc` registra 5.856 archivos y 1.236.482.806 bytes.

## 3. Gestión de datos

El dataset se gestiona con DVC: Git conserva el puntero `data/raw/chest_xray.dvc` y `.gitignore` excluye `data/raw/chest_xray/`. El remoto configurado es `google_drive` (`gdrive://1_-_KTx4WphjoXLyfWFwFqxV_DxgWFj3v`); no se incluyen credenciales ni secretos.

Recuperación del dataset:

```powershell
dvc pull data/raw/chest_xray.dvc
```

## 4. Estructura del proyecto

```text
proyecto-neumonia/
├── data/
│   ├── interim/
│   │   └── stratified_split_70_15_15.csv
│   └── raw/
│       ├── chest_xray.dvc
│       └── chest_xray/
├── models/
│   ├── vgg16/best_model.keras
│   ├── resnet50/best_model.keras
│   ├── mobilenetv2/best_model.keras
│   └── model_results.json
├── notebooks/
│   └── 01_comprension_datos_eda.ipynb
├── references/
│   ├── documentacion_proyecto.md
│   ├── guia_ejecucion.md
│   └── instrucciones_copilot.md
├── reports/figures/
├── src/
│   ├── cli.py
│   ├── data/
│   │   ├── make_dataset.py
│   │   ├── preprocessing.py
│   │   ├── splitting.py
│   │   └── datasets.py
│   ├── features/build_features.py
│   ├── models/
│   │   ├── architectures.py
│   │   ├── evaluation.py
│   │   ├── comparison.py
│   │   ├── train_model.py
│   │   └── predict_model.py
│   ├── training/
│   │   ├── training.py
│   │   └── run_real_training.py
│   ├── utils/paths.py
│   └── visualization/visualize.py
├── tests/
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini
```

`build_features.py`, `train_model.py` y `predict_model.py` existen pero están vacíos y no son entry points del flujo ejecutado. El entrenamiento real se invoca a través de la CLI (`neumonia train`), que llama a `entrenar_y_evaluar_modelos` en `src/training/run_real_training.py`.

## 5. CLI

La forma recomendada de ejecutar el flujo es la interfaz de línea de comandos `neumonia`, que invoca las funciones existentes sin duplicar lógica. Se registra con la instalación del proyecto (`pip install -r requirements.txt`, que incluye `-e .`; si falta, `pip install -e .`).

| Comando | Función |
|---|---|
| `neumonia eda` | Análisis exploratorio de datos existente. |
| `neumonia prepare` | Genera el manifiesto estratificado 70/15/15 y verifica los pipelines. |
| `neumonia augment` | Genera la figura de ejemplos de augmentación. |
| `neumonia train` | Entrena VGG16, ResNet50 y MobileNetV2; evalúa, compara y selecciona. |
| `neumonia evaluate` | Muestra los resultados guardados sin volver a entrenar. |
| `neumonia test` | Ejecuta la suite de pruebas con pytest. |
| `neumonia run` | Ejecuta el flujo completo: EDA, preparación, augmentación, entrenamiento, evaluación y tests. |

## 6. Preparación de datos y augmentación

El reparto experimental lo genera `crear_manifiesto_division_estratificada` (`src/data/splitting.py`): estratificado por clase, semilla 42, proporción 70/15/15 y sin duplicados por contenido entre conjuntos.

El preprocesamiento (`src/data/preprocessing.py`) carga cada imagen con Pillow y la convierte a RGB, la redimensiona a `224 x 224` con interpolación bilinear y la normaliza a `[0, 1]` (división por 255). El código no utiliza el `preprocess_input` específico de cada backbone.

`src/data/datasets.py` construye los `tf.data.Dataset` con tensores `224 x 224 x 3`, mezcla los ejemplos, agrupa en batches y usa `prefetch` automático. La data augmentation se aplica únicamente sobre `train` con `RandomFlip("horizontal")`, `RandomRotation(0.05)` y `RandomZoom(0.05)`; `validation` y `test` no reciben augmentation. No se implementó oversampling, undersampling ni class weights. La figura de ejemplos se genera con `graficar_ejemplos_augmentation` (`src/visualization/visualize.py`) en `reports/figures/data_augmentation_examples.png`.

## 7. Configuración experimental de los modelos

Los tres modelos usan Keras Applications con pesos ImageNet, `include_top=False` y base convolucional congelada. Sobre la base se añade:

```text
GlobalAveragePooling2D
Dense(128, activation="relu")
Dropout(0.3)
Dense(1, activation="sigmoid")
```

- Entrada: `224 x 224 x 3`.
- Función de pérdida: `binary_crossentropy`.
- Optimizador: Adam con learning rate `1e-4`.
- Batch size: 16.
- Máximo de epochs: 3 (semilla 42).
- `ModelCheckpoint` monitorizando `val_loss` (guarda solo el mejor modelo).
- `EarlyStopping` con paciencia 3 y restauración de mejores pesos.
- `ReduceLROnPlateau` con factor 0.5, paciencia 2 y learning rate mínimo `1e-6`.

## 8. Entrenamiento, evaluación y selección

El flujo real está implementado en `src/training/run_real_training.py` y se ejecuta con `neumonia train` (o `neumonia run`). Establece la semilla 42, regenera el manifiesto, construye los datasets, aplica augmentation a `train`, entrena cada modelo y evalúa su checkpoint seleccionado por `val_loss`.

La comparación y la selección se realizan **exclusivamente** con las métricas sobre `validation` (879 imágenes); `test` (878 imágenes) queda reservado y solo se usa para la evaluación final del modelo ganador. Las predicciones binarias usan umbral 0.5 y las curvas ROC se generan con probabilidades. La regla de selección es, en orden:

1. Balanced Accuracy;
2. ROC-AUC;
3. F1;
4. Accuracy.

Esta regla reemplazó a la anterior `Recall -> Accuracy -> F1`, que podía seleccionar un modelo que detectara todos los positivos mientras clasificaba mal todos los normales.

## 9. Resultados reales

Valores verificados contra `models/model_results.json`.

Resultados sobre `validation` (879 imágenes), usados para la selección:

| Modelo | Balanced Accuracy | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.9293 | 0.9431 | 0.9624 | 0.9594 | 0.8992 | 0.9609 | 0.9844 |
| VGG16 | 0.8124 | 0.8862 | 0.8826 | 0.9735 | 0.6513 | 0.9258 | 0.9608 |
| ResNet50 | 0.5000 | 0.7292 | 0.7292 | 1.0000 | 0.0000 | 0.8434 | 0.8956 |

ResNet50 predijo los 879 casos de `validation` como `PNEUMONIA` (TN = 0, FP = 238, FN = 0, TP = 641): Recall 1.0, pero Specificity 0.0 y Balanced Accuracy 0.5. El Recall aislado no representa un desempeño global adecuado para este problema.

Resultados finales sobre `test` (878 imágenes), correspondientes solo al modelo ganador:

| Métrica | Valor |
|---:|---:|
| Accuracy | 0.9294 |
| Balanced Accuracy | 0.9064 |
| Precision | 0.9474 |
| Recall/Sensitivity | 0.9563 |
| Specificity | 0.8565 |
| F1 | 0.9519 |
| ROC-AUC | 0.9734 |

**Modelo seleccionado: MobileNetV2.**

## 10. Testing

La suite se ejecuta con `pytest` (también disponible con `neumonia test`). Contiene **23 pruebas** y la ejecución final terminó con **23 passed**.

Las pruebas cubren carga y transformación de imágenes, estadísticas y distribución del dataset, construcción de arquitecturas, validación de entradas, matrices de confusión, métricas, selección de modelos, generación del manifiesto estratificado y condiciones de error.

## 11. Reproducibilidad

La ejecución validada utilizó Python 3.10.11 en el entorno `.venv` con las dependencias de `requirements.txt` (TensorFlow CPU 2.15.0, NumPy, Pandas, Pillow, Matplotlib, scikit-learn, pytest, click).

Flujo de ejecución:

```powershell
dvc pull data/raw/chest_xray.dvc
neumonia eda
neumonia prepare
neumonia augment
neumonia train
neumonia evaluate
neumonia test
```

`neumonia train` consume bastante tiempo y recursos. `neumonia evaluate` consulta los resultados guardados sin volver a entrenar. Como alternativa, cada comando puede invocarse con `python -m src.cli <comando>`.

## 12. Limitaciones

- El `val` original del dataset crudo contiene solo 16 imágenes y no se usó como validación experimental; la validación experimental tiene 879 imágenes.
- Existe desbalance entre `NORMAL` y `PNEUMONIA`.
- Hay duplicados exactos dentro de algunos splits crudos, aunque no entre ellos.
- El trabajo es académico y experimental; no hay validación clínica ni despliegue.
- La normalización es general a `[0, 1]` y no específica de cada backbone.
- Los modelos no deben usarse como herramientas de diagnóstico médico.

## 13. Estado final

| Etapa | Estado |
|---|---|
| EDA | Completado |
| Preparación | Completada |
| Augmentación | Completada |
| Modelado | Completado |
| Entrenamiento | Completado |
| Evaluación | Completada |
| Comparación | Completada |
| Selección | MobileNetV2 |
| Testing | 23/23 pruebas aprobadas |
| Documentación | Actualizada |
| Despliegue productivo | Fuera del alcance |