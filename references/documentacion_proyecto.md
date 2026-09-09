# Documentación técnica del proyecto

## 1. Información general

**Título:** Clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA` mediante modelos convolucionales con transferencia de aprendizaje.

**Objetivo general:** construir y comparar un pipeline reproducible de ciencia de datos y aprendizaje profundo para clasificar radiografías del dataset real gestionado con DVC.

**Objetivos específicos:**

- inspeccionar y documentar la estructura y calidad del dataset;
- preparar imágenes para entrada de modelos convolucionales;
- entrenar VGG16, ResNet50 y MobileNetV2 con una configuración comparable;
- evaluar y comparar los modelos sobre el conjunto de validación y realizar la evaluación final del modelo seleccionado sobre el conjunto de prueba;
- comparar sus resultados con métricas apropiadas para clasificación médica binaria;
- seleccionar el modelo con mejor desempeño global según una regla reproducible.

**Alcance:** análisis exploratorio, preparación de imágenes, entrenamiento experimental, evaluación y comparación de tres arquitecturas. El resultado es académico y experimental.

**Limitaciones:** no es un sistema clínico ni una herramienta de diagnóstico; no existe validación clínica externa; el `val` original del dataset crudo es muy pequeño (16 imágenes) y no se utiliza como validación experimental; el dataset está desbalanceado; no se realiza calibración clínica ni despliegue productivo.

## 2. Dataset

El proyecto utiliza el dataset de radiografías de tórax administrado mediante DVC. El repositorio no contiene una ficha de procedencia externa más específica que la estructura `chest_xray` y la referencia almacenada en DVC, por lo que no se atribuye aquí una fuente adicional no documentada.

La ruta recuperada es `data/raw/chest_xray/` y su estructura es:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

La clase `NORMAL` se codifica como objetivo `0` y `PNEUMONIA` como objetivo `1`.

Esta es la estructura original del dataset crudo (`data/raw/chest_xray/`):

| División | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| train | 1.341 | 3.875 | 5.216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |
| **Total** | **1.583** | **4.273** | **5.856** |

La partición experimental utilizada para modelado no es esa estructura cruda: se genera en la preparación de datos con `create_stratified_split_manifest` (`src/data/splitting.py`), estratificada por clase, con semilla 42 y proporción 70/15/15. El manifiesto es `data/interim/stratified_split_70_15_15.csv`:

| Conjunto | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| train | 1.108 | 2.991 | 4.099 |
| validation | 238 | 641 | 879 |
| test | 237 | 641 | 878 |
| **Total** | **1.583** | **4.273** | **5.856** |

Todas las imágenes actuales son `.jpeg`. El análisis de archivos observó tamaños entre 5.441 y 2.414.342 bytes. La función EDA verificó las imágenes y no detectó archivos corruptos.

En la estructura cruda se detectaron 30 grupos de archivos con contenido idéntico, que agrupan 62 rutas; tomando una ruta representante por grupo, implica 32 rutas duplicadas adicionales. La auditoría por SHA-256 no detectó duplicados exactos entre `train`, `val` y `test` crudos; los duplicados encontrados están dentro del mismo split. El reparto experimental 70/15/15 se construye agrupando por hash SHA-256, por lo que garantiza que no existan duplicados entre los tres conjuntos.

El manifiesto [data/raw/chest_xray.dvc](../data/raw/chest_xray.dvc) registra 5.856 archivos y un tamaño de 1.236.482.806 bytes.

## 3. Gestión de datos

El dataset está gestionado con DVC. El archivo de seguimiento es `data/raw/chest_xray.dvc`; no se versionan las imágenes directamente en Git. La configuración de DVC se encuentra en `.dvc/config` y define un remoto Google Drive con URL `gdrive://1_-_KTx4WphjoXLyfWFwFqxV_DxgWFj3v`. No se incluyen credenciales ni secretos.

Para recuperar el dataset en una copia del proyecto se requiere tener DVC y el acceso autorizado al remoto configurado:

```powershell
python -m pip install "dvc[gdrive]"
dvc pull data/raw/chest_xray.dvc
```

`.gitignore` excluye `data/raw/chest_xray/`, mientras que Git conserva el archivo `.dvc`, que funciona como puntero y metadatos del contenido. El archivo `.dvcignore` no añade reglas específicas.

## 4. Estructura actual del proyecto

```text
proyecto-neumonia/
├── data/
│   ├── external/
│   ├── interim/
│   │   └── stratified_split_70_15_15.csv
│   ├── processed/
│   └── raw/
│       ├── chest_xray.dvc
│       └── chest_xray/
├── docs/
├── models/
│   ├── vgg16/best_model.keras
│   ├── resnet50/best_model.keras
│   ├── mobilenetv2/best_model.keras
│   └── model_results.json
├── notebooks/
│   └── 01_comprension_datos_eda.ipynb
├── references/
│   ├── instrucciones_copilot.md
│   ├── documentacion_proyecto.md
│   └── guia_ejecucion.md
├── reports/figures/
├── src/
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

`train_model.py`, `predict_model.py` y `build_features.py` existen, pero están vacíos. No son entry points del flujo ejecutado. El entrenamiento y la evaluación reales se ejecutan desde `src/training/run_real_training.py`.

## 5. Metodología CRISP-DM

1. **Comprensión del negocio:** se definió el problema académico de distinguir radiografías normales de radiografías con neumonía, dando importancia a la detección de positivos.
2. **Comprensión de los datos:** realizada mediante EDA sobre el dataset real, incluyendo estructura, distribución, dimensiones, tamaños, muestras, corrupción y duplicados.
3. **Preparación de los datos:** realizada con el reparto estratificado 70/15/15 (`data/interim/stratified_split_70_15_15.csv`, train 4.099, validation 879 y test 878, sin duplicados entre conjuntos), y mediante carga, conversión RGB, redimensionamiento, normalización y creación de datasets TensorFlow.
4. **Modelado:** realizado con VGG16, ResNet50 y MobileNetV2 con transferencia de aprendizaje.
5. **Evaluación:** selección del modelo únicamente con las métricas de `validation` y evaluación final del ganador sobre `test`, con matriz de confusión, métricas binarias y curvas ROC.
6. **Despliegue:** fuera del alcance actual. No se implementó una API ni una aplicación de producción.

## 6. EDA

El notebook utilizado es [notebooks/01_comprension_datos_eda.ipynb](../notebooks/01_comprension_datos_eda.ipynb). Las funciones reutilizables están en `src/data/make_dataset.py`.

El EDA realiza:

- validación de las carpetas `train`, `val`, `test` y sus dos clases;
- conteo por división y clase;
- lectura de dimensiones, modo, extensión y tamaño de archivo;
- verificación de imágenes corruptas;
- detección de duplicados mediante SHA-256;
- generación de distribución de clases;
- generación de histograma de tamaños;
- generación de dispersión de dimensiones;
- selección de muestras visuales de `NORMAL` y `PNEUMONIA`.

Las figuras generadas en `reports/figures/` son:

- `dataset_distribution.png`;
- `file_size_distribution.png`;
- `image_dimensions.png`;
- `sample_normal.png`;
- `sample_pneumonia.png`.

La conclusión real es que el dataset está fuertemente desbalanceado hacia `PNEUMONIA`, especialmente en el `train` original, que el `val` original de la estructura cruda contiene solo 16 imágenes y que no se detectaron imágenes corruptas. También existen duplicados dentro de los splits crudos, sin duplicados exactos entre ellos. El `val` de 16 imágenes es descriptivo del dataset crudo y no se utiliza como validación experimental: para el modelado se genera el reparto 70/15/15 en la preparación de datos, con validation de 879 imágenes y test de 878.

## 7. Preparación de imágenes

La partición experimental la genera `src/data/splitting.py` con `create_stratified_split_manifest`: estratificada por clase, con semilla 42, proporción 70/15/15 y agrupación por hash SHA-256 para evitar duplicados entre conjuntos. El manifiesto resultante es `data/interim/stratified_split_70_15_15.csv` (train 4.099, validation 879 y test 878).

`src/data/preprocessing.py` implementa:

- carga con Pillow y conversión a RGB;
- redimensionamiento a `224 x 224` mediante interpolación bilinear;
- conversión a `float32` y división por 255, dejando valores en `[0, 1]`;
- construcción de un DataFrame con ruta, split, etiqueta y objetivo numérico.

`src/data/datasets.py` convierte el DataFrame en `tf.data.Dataset`, genera tensores con forma `224 x 224 x 3`, asigna `0` a `NORMAL` y `1` a `PNEUMONIA`, mezcla los ejemplos, agrupa en batches y usa `prefetch` automático.

Durante la ejecución real se utilizó batch size 16. La data augmentation se aplicó únicamente a `train` mediante `RandomFlip("horizontal")`, `RandomRotation(0.05)` y `RandomZoom(0.05)`. `val` y `test` no reciben augmentation. No se implementó oversampling, undersampling ni ponderación de clases.

El código no utiliza `preprocess_input` específico de cada backbone; la transformación implementada es la normalización general a `[0, 1]`.

## 8. Modelos

Se evaluaron tres arquitecturas de Keras Applications:

- VGG16;
- ResNet50;
- MobileNetV2.

Las tres usan pesos ImageNet, `include_top=False` y base convolucional congelada. Sobre cada base se añadió:

```text
GlobalAveragePooling2D
Dense(128, activation="relu")
Dropout(0.3)
Dense(1, activation="sigmoid")
```

La función de pérdida es `binary_crossentropy` y el optimizador es Adam con learning rate `1e-4`. La entrada es `224 x 224 x 3`.

La ejecución real utiliza tres epochs y batch size 16. Sus callbacks son:

- `ModelCheckpoint`, monitorizando `val_loss` y guardando solo el mejor modelo;
- `EarlyStopping` con paciencia 3 y restauración de mejores pesos;
- `ReduceLROnPlateau`, factor 0.5, paciencia 2 y learning rate mínimo `1e-6`.

La utilidad separada `src/training/training.py` contiene defaults reutilizables distintos, pero no fue el entry point de la ejecución experimental final.

## 9. Entrenamiento

Los tres modelos fueron entrenados sobre el subconjunto `train` del reparto experimental 70/15/15 (4.099 imágenes). Los checkpoints finales están en:

- `models/vgg16/best_model.keras`;
- `models/resnet50/best_model.keras`;
- `models/mobilenetv2/best_model.keras`.

El proceso real está implementado en `src/training/run_real_training.py`. Se estableció la semilla 42, se generó el manifiesto estratificado 70/15/15, se construyeron los datasets, se aplicó augmentation a `train`, se entrenó cada backbone y se evaluó su checkpoint seleccionado por `val_loss` sobre `validation` (879 imágenes).

## 10. Evaluación

La evaluación se divide en dos fases. Para la comparación y la selección, cada modelo se evalúa sobre el subconjunto `validation` (879 imágenes: 238 `NORMAL` y 641 `PNEUMONIA`). El conjunto `test` (878 imágenes: 237 `NORMAL` y 641 `PNEUMONIA`) queda reservado y se utiliza exclusivamente para la evaluación final del modelo ganador (MobileNetV2).

La matriz se interpreta como `[[TN, FP], [FN, TP]]`, con `PNEUMONIA` como clase positiva:

- **TN:** normales correctamente clasificados;
- **FP:** normales clasificados incorrectamente como neumonía;
- **FN:** neumonías no detectadas;
- **TP:** neumonías correctamente detectadas.

Las métricas son:

- **Accuracy:** proporción total de clasificaciones correctas;
- **Precision:** proporción de predicciones positivas que realmente son neumonía;
- **Recall/Sensitivity:** proporción de neumonías detectadas;
- **Specificity:** proporción de casos normales correctamente rechazados como neumonía;
- **F1:** media armónica entre Precision y Recall;
- **ROC-AUC:** capacidad de ordenar positivos por encima de negativos a través de umbrales;
- **Balanced Accuracy:** media entre Sensitivity y Specificity.

Las predicciones binarias se obtienen con umbral 0.5. Las curvas ROC se generan con las probabilidades, no con las etiquetas binarias.

## 11. Resultados reales

Los valores siguientes fueron verificados contra `models/model_results.json` y las predicciones almacenadas.

Resultados sobre `validation` (879 imágenes), utilizados para la comparación y la selección del modelo:

| Modelo | Balanced Accuracy | Accuracy | Precision | Recall/Sensitivity | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.9293 | 0.9431 | 0.9624 | 0.9594 | 0.8992 | 0.9609 | 0.9844 |
| VGG16 | 0.8124 | 0.8862 | 0.8826 | 0.9735 | 0.6513 | 0.9258 | 0.9608 |
| ResNet50 | 0.5000 | 0.7292 | 0.7292 | 1.0000 | 0.0000 | 0.8434 | 0.8949 |

Resultados finales sobre `test` (878 imágenes), correspondientes únicamente al modelo ganador, MobileNetV2:

| Métrica | Valor |
|---:|---:|
| Accuracy | 0.9294 |
| Balanced Accuracy | 0.9064 |
| Precision | 0.9474 |
| Recall/Sensitivity | 0.9563 |
| Specificity | 0.8565 |
| F1 | 0.9519 |
| ROC-AUC | 0.9733 |

## 12. Comparación y selección

La regla final de selección es, en orden:

1. Balanced Accuracy;
2. ROC-AUC;
3. F1;
4. Accuracy.

La comparación se realiza únicamente con las métricas sobre `validation`; el conjunto `test` no participa en la selección.

La regla anterior `Recall -> Accuracy -> F1` fue abandonada porque podía seleccionar un modelo que detectara todos los positivos mientras clasificaba incorrectamente todos los normales.

ResNet50 obtuvo sobre `validation`:

```text
TN = 0
FP = 238
FN = 0
TP = 641
```

Por tanto, con umbral 0.5 predijo los 879 casos de `validation` como `PNEUMONIA`. Su Recall es 1.0, pero su Specificity es 0.0 y su Balanced Accuracy es 0.5. Recall aislado no representa un desempeño global adecuado para este problema.

El modelo seleccionado es **MobileNetV2**, por obtener la mayor Balanced Accuracy sobre `validation`, además de la mayor Accuracy, F1 y ROC-AUC, con Recall de 0.9594 y Specificity de 0.8992. Esta decisión se basa únicamente en las métricas de `validation`; `test` se utilizó después, solo para la evaluación final de MobileNetV2 (Accuracy 0.9294, Balanced Accuracy 0.9064, Precision 0.9474, Recall 0.9563, Specificity 0.8565, F1 0.9519 y ROC-AUC 0.9733).

## 13. Testing

Se utiliza `pytest`. La suite actual contiene 23 pruebas y finalizó con `23 passed`.

Las pruebas cubren carga y transformación de imágenes, estadísticas y distribución del dataset, construcción de arquitecturas, validación de entradas, matrices de confusión, métricas, selección de modelos, generación del manifiesto estratificado y condiciones de error.

## 14. Reproducibilidad

La ejecución validada utilizó Python 3.10.11 en el entorno `.venv`. Las dependencias se declaran en `requirements.txt`, incluyendo TensorFlow CPU 2.15.0, NumPy, Pandas, Pillow, Matplotlib, scikit-learn y pytest.

DVC utiliza el remoto Google Drive configurado en `.dvc/config`. La recuperación se realiza con:

```powershell
dvc pull data/raw/chest_xray.dvc
```

Comandos principales del flujo actual:

```powershell
.\.venv\Scripts\python.exe -m src.data.make_dataset
.\.venv\Scripts\python.exe -m src.training.run_real_training
.\.venv\Scripts\python.exe -m pytest
```

El segundo comando vuelve a entrenar y evaluar los tres modelos, por lo que no debe ejecutarse si solo se desea consultar los artefactos ya generados.

## 15. Limitaciones

- El `val` original del dataset crudo contiene solamente 16 imágenes, pero no se utiliza para la selección de modelos; la validación experimental tiene 879 imágenes.
- Existe desbalance entre `NORMAL` y `PNEUMONIA`.
- Hay duplicados exactos dentro de algunos splits, aunque no entre splits.
- El trabajo es académico y experimental.
- No se realizó validación clínica ni comparación con especialistas.
- No se evaluó generalización sobre otra institución o población.
- Los modelos no deben utilizarse como herramientas de diagnóstico médico.
- La normalización es general a `[0, 1]` y no específica de cada backbone.
- No se implementó despliegue ni calibración de probabilidades.

## 16. Estado final

| Etapa | Estado |
|---|---|
| Comprensión del negocio | Completada en alcance académico |
| Comprensión de los datos / EDA | Completada |
| Preparación | Completada para el pipeline ejecutado |
| Modelado | Completado para VGG16, ResNet50 y MobileNetV2 |
| Entrenamiento | Completado con artefactos guardados |
| Evaluación | Completada sobre `validation` para la selección y sobre `test` solo para el modelo final (MobileNetV2) |
| Comparación | Completada con criterio reproducible |
| Selección | Completada: MobileNetV2 |
| Testing | Completado: 23 pruebas aprobadas |
| Documentación | Completada con este documento y la guía de ejecución |
| Despliegue productivo | Fuera del alcance |
