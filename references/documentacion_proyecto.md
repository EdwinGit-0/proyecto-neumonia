# Documentación técnica del proyecto

## 1. Información general

**Título:** Clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA` mediante modelos convolucionales con transferencia de aprendizaje.

**Objetivo general:** construir y comparar un pipeline reproducible de ciencia de datos y aprendizaje profundo para clasificar radiografías del dataset real gestionado con DVC.

**Objetivos específicos:**

- inspeccionar y documentar la estructura y calidad del dataset;
- preparar imágenes para entrada de modelos convolucionales;
- entrenar VGG16, ResNet50 y MobileNetV2 con una configuración comparable;
- evaluar los modelos sobre el conjunto `test`;
- comparar sus resultados con métricas apropiadas para clasificación médica binaria;
- seleccionar el modelo con mejor desempeño global según una regla reproducible.

**Alcance:** análisis exploratorio, preparación de imágenes, entrenamiento experimental, evaluación y comparación de tres arquitecturas. El resultado es académico y experimental.

**Limitaciones:** no es un sistema clínico ni una herramienta de diagnóstico; no existe validación clínica externa; el conjunto `val` es muy pequeño; el dataset está desbalanceado; no se realiza calibración clínica ni despliegue productivo.

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

| División | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| train | 1.341 | 3.875 | 5.216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |
| **Total** | **1.583** | **4.273** | **5.856** |

Todas las imágenes actuales son `.jpeg`. El análisis de archivos observó tamaños entre 5.441 y 2.414.342 bytes. La función EDA verificó las imágenes y no detectó archivos corruptos.

Se detectaron 30 grupos de archivos con contenido idéntico, correspondientes a 30 rutas duplicadas adicionales dentro de sus respectivos grupos. La auditoría por SHA-256 no detectó duplicados exactos entre `train`, `val` y `test`; los duplicados encontrados están dentro del mismo split. Esta condición debe considerarse al interpretar la independencia efectiva de algunos ejemplos.

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
│   ├── doc_proyecto.md
│   ├── guia_monografia.pdf
│   ├── instrucciones_copilot.md
│   ├── documentacion_proyecto.md
│   └── guia_ejecucion.md
├── reports/figures/
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   ├── preprocessing.py
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
3. **Preparación de los datos:** realizada mediante carga, conversión RGB, redimensionamiento, normalización y creación de datasets TensorFlow.
4. **Modelado:** realizado con VGG16, ResNet50 y MobileNetV2 con transferencia de aprendizaje.
5. **Evaluación:** realizada sobre `test` con matriz de confusión, métricas binarias y curvas ROC.
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

La conclusión real es que el dataset está fuertemente desbalanceado hacia `PNEUMONIA`, especialmente en `train`, que `val` contiene solo 16 imágenes y que no se detectaron imágenes corruptas. También existen duplicados dentro de splits, sin duplicados exactos entre splits.

## 7. Preparación de imágenes

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

Los tres modelos fueron entrenados sobre el dataset real. Los checkpoints finales están en:

- `models/vgg16/best_model.keras`;
- `models/resnet50/best_model.keras`;
- `models/mobilenetv2/best_model.keras`.

El proceso real está implementado en `src/training/run_real_training.py`. Se estableció la semilla 42, se construyeron los datasets, se aplicó augmentation a `train`, se entrenó cada backbone y se evaluó su checkpoint seleccionado por `val_loss`.

## 10. Evaluación

La evaluación utiliza exclusivamente el split `test`, con 624 predicciones por modelo: 234 casos `NORMAL` y 390 casos `PNEUMONIA`.

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

Los valores siguientes fueron verificados contra `models/model_results.json` y las predicciones almacenadas:

| Modelo | Balanced Accuracy | Accuracy | Precision | Recall/Sensitivity | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.7876 | 0.8381 | 0.7992 | 0.9897 | 0.5855 | 0.8843 | 0.9539 |
| VGG16 | 0.7179 | 0.7821 | 0.7510 | 0.9744 | 0.4615 | 0.8482 | 0.9155 |
| ResNet50 | 0.5000 | 0.6250 | 0.6250 | 1.0000 | 0.0000 | 0.7692 | 0.8354 |

## 12. Comparación y selección

La regla final de selección es, en orden:

1. Balanced Accuracy;
2. ROC-AUC;
3. F1;
4. Accuracy.

La regla anterior `Recall -> Accuracy -> F1` fue abandonada porque podía seleccionar un modelo que detectara todos los positivos mientras clasificaba incorrectamente todos los normales.

ResNet50 obtuvo:

```text
TN = 0
FP = 234
FN = 0
TP = 390
```

Por tanto, con umbral 0.5 predijo los 624 casos como `PNEUMONIA`. Su Recall es 1.0, pero su Specificity es 0.0 y su Balanced Accuracy es 0.5. Recall aislado no representa un desempeño global adecuado para este problema.

El modelo seleccionado es **MobileNetV2**, porque obtiene la mayor Balanced Accuracy, Accuracy, F1 y ROC-AUC, manteniendo además un Recall de 0.9897 y una Specificity de 0.5855. Esta decisión se basa únicamente en los resultados experimentales existentes.

## 13. Testing

Se utiliza `pytest`. La suite actual contiene 22 pruebas y finalizó con `22 passed`.

Las pruebas cubren carga y transformación de imágenes, estadísticas y distribución del dataset, construcción de arquitecturas, validación de entradas, matrices de confusión, métricas, selección de modelos y condiciones de error.

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

- `val` contiene solamente 16 imágenes, por lo que sus métricas de seguimiento son inestables.
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
| Evaluación | Completada sobre `test` |
| Comparación | Completada con criterio reproducible |
| Selección | Completada: MobileNetV2 |
| Testing | Completado: 22 pruebas aprobadas |
| Documentación | Completada con este documento y la guía de ejecución |
| Despliegue productivo | Fuera del alcance |
