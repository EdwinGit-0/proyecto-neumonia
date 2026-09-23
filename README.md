# Proyecto de clasificación de neumonía

Clasificación de imágenes de rayos X de tórax en las clases `NORMAL` y `PNEUMONIA` mediante redes neuronales convolucionales con transferencia de aprendizaje.

## 1. Información general

**Título:** Clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA`.

**Objetivo:** entrenar, evaluar y comparar tres arquitecturas de redes convolucionales con una configuración comparable y seleccionar el modelo con mejor desempeño sobre el conjunto de validación.

**Alcance:** análisis exploratorio, preparación de imágenes, entrenamiento experimental, evaluación y comparación de modelos.

> **Nota:** este proyecto tiene finalidad académica y experimental. No constituye un sistema clínico ni una herramienta de diagnóstico médico.

---

## 2. Dataset

El proyecto utiliza el dataset **Chest X-Ray Images (Pneumonia)**, administrado mediante DVC.

El dataset crudo contiene **5.856 imágenes JPEG** distribuidas originalmente de la siguiente manera:

| División  |    NORMAL | PNEUMONIA |     Total |
| --------- | --------: | --------: | --------: |
| train     |     1.341 |     3.875 |     5.216 |
| val       |         8 |         8 |        16 |
| test      |       234 |       390 |       624 |
| **Total** | **1.583** | **4.273** | **5.856** |

La clase `NORMAL` se codifica como `0` y `PNEUMONIA` como `1`.

El `val` original contiene solamente 16 imágenes, por lo que **no se utiliza en los experimentos**.

### División experimental

Para el modelado se utilizan únicamente las 5.216 imágenes del `train` original. Estas se dividen aproximadamente en:

* **80 % para entrenamiento:** 4.173 imágenes.
* **20 % para validación:** 1.043 imágenes.
* **Test:** las 624 imágenes del `test` original permanecen intactas.

La división es estratificada por clase, utiliza semilla `42` y agrupa imágenes con contenido idéntico mediante SHA-256 para evitar que el mismo contenido quede distribuido entre diferentes conjuntos.

| Conjunto            |    NORMAL | PNEUMONIA |     Total |
| ------------------- | --------: | --------: | --------: |
| Train               |     1.073 |     3.100 |     4.173 |
| Validation          |       268 |       775 |     1.043 |
| Test                |       234 |       390 |       624 |
| **Total utilizado** | **1.575** | **4.265** | **5.840** |

El manifiesto de esta división se encuentra en:

```text
data/interim/stratified_split_train80_val20_test_original.csv
```

---

## 3. Preparación de las imágenes

Las imágenes se procesan de la siguiente manera:

* Conversión a RGB.
* Redimensionamiento a `224 × 224` píxeles.
* Interpolación bilinear.
* Conversión a `float32`.
* Normalización de valores al rango `[0, 1]`.
* Entrada final de los modelos: `224 × 224 × 3`.

La augmentación se aplica **únicamente al conjunto de entrenamiento**:

* `RandomRotation(0.05)`
* `RandomZoom(0.05)`

No se aplica augmentación a `validation` ni a `test`.

El `RandomFlip("horizontal")` fue descartado porque una inversión horizontal puede alterar información de lateralidad anatómica y marcadores `L/R` presentes en algunas radiografías.

El entrenamiento original de los tres modelos no utiliza oversampling, undersampling ni class weights. En fases posteriores (tuning y tratamiento del desbalance) sí se aplicó **oversampling de la clase minoritaria `NORMAL` únicamente sobre el `train`** del modelo final, y de forma *experimental* se probaron **class weights** (ver sección 5). En ningún momento se modificaron `validation` ni `test`.

---

## 4. Modelos

Se compararon tres arquitecturas mediante transferencia de aprendizaje con pesos de ImageNet:

* **VGG16**
* **ResNet50**
* **MobileNetV2**

Las bases convolucionales permanecen congeladas y los tres modelos utilizan la misma cabeza de clasificación:

```text
GlobalAveragePooling2D
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(1, Sigmoid)
```

### Configuración de entrenamiento

| Parámetro            | Configuración           |
| -------------------- | ----------------------- |
| Pérdida              | Binary Crossentropy     |
| Optimizador          | Adam                    |
| Learning rate        | `3e-4` (ajustado) | `1e-4` (original)  |
| Batch size           | `16`                    |
| Máximo de epochs     | `3`                     |
| Semilla              | `42`                    |
| Checkpoint           | `val_loss`              |
| Early Stopping       | paciencia 3             |
| ReduceLROnPlateau    | factor 0.5, paciencia 2 |
| Learning rate mínimo | `1e-6`                  |

---

## 5. Selección del modelo

Todas las decisiones se realizan **exclusivamente sobre validation (1.043 imágenes)**.

El conjunto `test` de 624 imágenes permanece reservado para la evaluación final y no participa en:

* entrenamiento;
* selección del modelo;
* ajuste de hiperparámetros.

La regla de selección es jerárquica y se aplicó en cada etapa:

1. Balanced Accuracy
2. ROC-AUC
3. F1
4. Accuracy

El proceso completo tuvo estas etapas:

1. **Modelo inicial:** MobileNetV2 fue seleccionado sobre validation entre VGG16, ResNet50 y MobileNetV2.
2. **Tuning de MobileNetV2:** se probaron learning rate (`5e-5`, `3e-4`), dropout (`0.5`) y fine-tuning de las últimas capas (`block_13+`). La configuración con `lr=3e-4` (dropout `0.3`, base congelada) obtuvo la mejor Balanced Accuracy en validation y fue seleccionada.
3. **Tratamiento del desbalance:** sobre el MobileNetV2 ajustado (`lr=3e-4`) se probaron por separado **class weights** y **oversampling** de la clase minoritaria `NORMAL` únicamente en `train`.
4. **Configuración final:** el **oversampling** obtuvo la mejor Balanced Accuracy en validation y fue seleccionado como modelo final.

El modelo final actual es:

```text
models/mobilenetv2_desbalance_oversampling/best_model.keras
```

### Configuración final del modelo

| Parámetro           | Valor                                                            |
| ------------------- | ---------------------------------------------------------------- |
| Arquitectura        | MobileNetV2 (ImageNet), base convolucional congelada             |
| Cabeza              | `GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)`      |
| Learning rate       | `3e-4`                                                           |
| Batch size          | `16`                                                             |
| Máximo de epochs    | `3`                                                              |
| Semilla             | `42`                                                             |
| Oversampling        | `NORMAL` duplicada en `train` (1 073 → 3 100; total 6 200)       |
| Augmentación        | `RandomRotation(0.05)` y `RandomZoom(0.05)`, solo en `train`     |
| Early Stopping      | paciencia 3, `restore_best_weights=True`                         |
| ModelCheckpoint     | `val_loss`, `mode="min"`, guarda solo el mejor                   |
| ReduceLROnPlateau   | factor 0.5, paciencia 2, mínimo `1e-6`                           |

---

## 6. Criterio de éxito

El criterio de éxito se evalúa únicamente sobre el **test original e independiente**.

Se utiliza como baseline una estrategia sencilla que predice siempre la clase mayoritaria (`PNEUMONIA`).

El baseline obtiene:

| Métrica           | Baseline |
| ----------------- | -------: |
| Balanced Accuracy |   0.5000 |
| Sensitivity       |   1.0000 |
| Specificity       |   0.0000 |

El modelo seleccionado debe:

1. Superar el baseline en Balanced Accuracy.
2. Obtener una sensibilidad superior a `0.5`.
3. Obtener una especificidad superior a `0.5`.

Este criterio se utiliza con fines **académicos y experimentales**, no como criterio clínico.

---

## 7. Resultados

### Validation

Los resultados utilizados para seleccionar el modelo fueron:

| Modelo          | Balanced Accuracy |   Accuracy |  Precision |     Recall | Specificity |         F1 |    ROC-AUC |
| --------------- | ----------------: | ---------: | ---------: | ---------: | ----------: | ---------: | ---------: |
| **MobileNetV2** |        **0.9412** | **0.9434** | **0.9773** |     0.9458 |  **0.9366** | **0.9613** | **0.9887** |
| VGG16           |            0.8765 |     0.9108 |     0.9338 | **0.9471** |      0.8060 |     0.9404 |     0.9643 |
| ResNet50        |            0.5000 |     0.7430 |     0.7430 | **1.0000** |      0.0000 |     0.8526 |     0.8878 |

MobileNetV2 fue seleccionado como modelo inicial por presentar el mejor desempeño según la regla jerárquica definida.

ResNet50 clasificó todos los casos de validation como `PNEUMONIA`, por lo que obtuvo sensibilidad de 1.0 pero especificidad de 0.0.

### Tuning de MobileNetV2 (validation)

Sobre MobileNetV2 se probaron learning rate, dropout y fine-tuning. Resultados sobre validation:

| Configuración                     | Balanced Acc | ROC-AUC |   F1 | Accuracy | Recall | Specificity |
| --------------------------------- | -----------: | ------: | ---: | -------: | -----: | ----------: |
| Original (lr `1e-4`)              |       0.9412 |  0.9887 | 0.9613 |   0.9434 | 0.9458 |      0.9366 |
| lr `5e-5`                         |       0.9415 |  0.9862 | 0.9683 |   0.9530 | 0.9652 |      0.9179 |
| **lr `3e-4` (seleccionada)**       |     **0.9547** | **0.9928** | **0.9741** | **0.9616** | 0.9690 | 0.9403 |
| dropout `0.5`                      |       0.9423 |  0.9884 | 0.9703 |   0.9559 | 0.9703 |      0.9142 |
| fine-tuning `block_13+`            |       0.6194 |  0.9978 | 0.8837 |   0.8044 | 1.0000 |      0.2388 |

El fine-tuning de las últimas capas degradó gravemente la especificidad (0.2388) y se descartó. La configuración `lr=3e-4` (dropout `0.3`, base congelada) fue seleccionada como **MobileNetV2 ajustado**.

### Tratamiento del desbalance (validation)

Sobre el ajustado (`lr=3e-4`) se probaron por separado **class weights** y **oversampling** de `NORMAL` en `train`:

| Configuración                         | Balanced Acc | ROC-AUC |   F1 | Accuracy | Recall | Specificity |
| ------------------------------------- | -----------: | ------: | ---: | -------: | -----: | ----------: |
| Ajustado `lr=3e-4` (referencia)       |       0.9555 |  0.9933 | 0.9774 |   0.9664 | 0.9781 |      0.9328 |
| Class weights (solo entrenamiento)    |       0.9530 |  0.9925 | 0.9644 |   0.9482 | 0.9432 |      0.9627 |
| **Oversampling `NORMAL` en train**     |     **0.9598** | **0.9936** | **0.9780** | **0.9674** | **0.9755** | **0.9440** |

El oversampling lideró según el criterio (Balanced Accuracy) y conservó una sensibilidad alta, por lo que fue seleccionado como **modelo final**. Los class weights mejoraron la especificidad pero redujeron el recall (0.9432) y la Balanced Accuracy, por lo que se descartaron.

> El resultado del oversampling es un hallazgo experimental de este dataset y no demuestra que el desbalance fuera la única causa de las diferencias observadas.

### Test final

Evaluación sobre las **624 imágenes del test original** (conjunto nunca usado para decidir):

| Métrica              | Original MobileNetV2 | Ajustado `lr=3e-4` | **Final (oversampling)** |
| -------------------- | -------------------: | -----------------: | -----------------------: |
| Accuracy             |               0.8381 |             0.8478 |               **0.8878** |
| Balanced Accuracy    |               0.7885 |             0.7987 |               **0.8590** |
| Precision            |               0.8004 |             0.8067 |                   0.8636 |
| Recall / Sensitivity |               0.9872 |             0.9949 |                   0.9744 |
| Specificity          |               0.5897 |             0.6026 |               **0.7436** |
| F1                   |               0.8840 |             0.8909 |               **0.9157** |
| ROC-AUC              |               0.9545 |             0.9633 |                   0.9612 |

Matriz de confusión del modelo final en test:

```text
[[174, 60],
 [ 10, 380]]
```

Donde:

* TN = 174
* FP = 60
* FN = 10
* TP = 380

El modelo final mantiene una sensibilidad alta (0.9744) con una especificidad notablemente mayor (0.7436) y mejor Balanced Accuracy (0.8590).

### Criterio de éxito

| Comprobación            |    Resultado |
| ----------------------- | -----------: |
| Balanced Accuracy > 0.5 |  Sí — 0.7885 |
| Sensibilidad > 0.5      |  Sí — 0.9872 |
| Especificidad > 0.5     |  Sí — 0.5897 |
| **Criterio de éxito**   | **Cumplido** |

Los resultados completos se encuentran en:

```text
models/model_results.json
```

---

## 8. Estructura del proyecto

```text
proyecto-neumonia/
├── data/
│   ├── interim/
│   │   └── stratified_split_train80_val20_test_original.csv
│   └── raw/
│       ├── chest_xray.dvc
│       └── chest_xray/
├── models/
│   ├── vgg16/best_model.keras
│   ├── resnet50/best_model.keras
│   ├── mobilenetv2/best_model.keras
│   ├── mobilenetv2_ajustado/best_model.keras
│   ├── mobilenetv2_desbalance_oversampling/best_model.keras
│   ├── model_results.json
│   ├── mobilenetv2_tuning_results.json
│   └── mobilenetv2_desbalance_results.json
├── notebooks/
│   └── 01_comprension_datos_eda.ipynb
├── references/
│   ├── documentacion_proyecto.md
│   └── guia_ejecucion.md
├── reports/
│   └── figures/
├── src/
│   ├── cli.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   │   ├── run_real_training.py
│   │   ├── tuning_mobilenetv2.py
│   │   └── tuning_desbalance_clases.py
│   ├── utils/
│   └── visualization/
├── tests/
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini
```

El entrenamiento real se encuentra en:

```text
src/training/run_real_training.py
```

El tuning de MobileNetV2 y los experimentos de desbalance se reproducen desde:

```text
src/training/tuning_mobilenetv2.py
src/training/tuning_desbalance_clases.py
```

Los archivos `build_features.py`, `train_model.py` y `predict_model.py` existen como parte de la estructura del proyecto, pero no son entry points del flujo real ejecutado.

---

## 9. Instalación

Se recomienda utilizar Python `3.10.11` y un entorno virtual.

En PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para recuperar el dataset mediante DVC:

```powershell
dvc pull data/raw/chest_xray.dvc
```

---

## 10. Ejecución

La interfaz recomendada del proyecto es la CLI `neumonia`.

### Análisis exploratorio

```powershell
neumonia eda
```

### Preparación de datos

```powershell
neumonia prepare
```

### Generación de ejemplos de augmentación

```powershell
neumonia augment
```

### Entrenamiento

```powershell
neumonia train
```

Este comando entrena y evalúa VGG16, ResNet50 y MobileNetV2 en su configuración original.

### Tuning de MobileNetV2

```powershell
neumonia tune
```

Este comando reproduce el tuning experimental de MobileNetV2 (learning rate, dropout y fine-tuning), selecciona la mejor configuración sobre validation y reporta los resultados sobre test. Genera o actualiza `models/mobilenetv2_tuning_results.json` y `models/mobilenetv2_ajustado/best_model.keras`.

### Experimentos de desbalance

```powershell
neumonia desbalance
```

Este comando reproduce los experimentos de desbalance (class weights y oversampling) sobre el ajustado `lr=3e-4`, selecciona la variante final sobre validation y reporta los resultados sobre test. Genera o actualiza `models/mobilenetv2_desbalance_results.json` y `models/mobilenetv2_desbalance_oversampling/best_model.keras`.

### Consultar resultados

```powershell
neumonia evaluate
```

Este comando consulta los resultados guardados sin volver a entrenar.

### Ejecutar pruebas

```powershell
neumonia test
```

### Ejecutar el flujo completo

```powershell
neumonia run
```

---

## 11. Testing

El proyecto cuenta con **34 pruebas automatizadas**, todas aprobadas en la ejecución final:

```text
34 passed
```

Las pruebas cubren, entre otros aspectos:

* carga y transformación de imágenes;
* estadísticas del dataset;
* distribución de clases;
* construcción de arquitecturas;
* métricas y matrices de confusión;
* selección de modelos;
* baseline mayoritaria;
* criterio de éxito;
* división experimental;
* estratificación;
* ausencia de duplicados por hash entre conjuntos;
* conservación del test original;
* exclusión del `val` original;
* ausencia de `RandomFlip("horizontal")`.

---

## 12. Reproducibilidad

La ejecución validada se realizó con:

* Python `3.10.11`
* TensorFlow CPU `2.15.0`
* NumPy
* Pandas
* Pillow
* Matplotlib
* scikit-learn
* pytest
* Click

El proyecto utiliza semilla `42` para la división experimental y configuración reproducible de los experimentos.

Los principales resultados y checkpoints se almacenan en:

```text
models/
```

y las figuras generadas se encuentran en:

```text
reports/figures/
```

La documentación técnica adicional se encuentra en:

```text
references/documentacion_proyecto.md
```

---

## 13. Limitaciones

* El `val` original contiene solamente 16 imágenes y no se utiliza en los experimentos.
* Existe desbalance entre las clases `NORMAL` y `PNEUMONIA`.
* Existen duplicados exactos dentro de algunos splits crudos.
* El oversampling de `NORMAL` en `train` (modelo final) duplica patrones existentes y no genera información nueva.
* El fine-tuning de las últimas capas de MobileNetV2 se descartó por degradar la especificidad (0.2388 en validation), posiblemente por el tamaño de lote pequeño y la base congelada.
* El dataset no representa necesariamente todas las poblaciones, equipos o condiciones de adquisición de radiografías.
* No se realizó validación clínica externa.
* No se realizó calibración de probabilidades.
* El proyecto no contempla despliegue productivo.
* Los resultados deben interpretarse dentro del contexto académico y experimental del dataset utilizado.

**Este proyecto no debe utilizarse como herramienta de diagnóstico médico.**

---

## 14. Estado del proyecto

| Etapa                 | Estado                  |
| --------------------- | ----------------------- |
| EDA                   | Completado              |
| Preparación de datos  | Completada              |
| Augmentación          | Completada              |
| Modelado              | Completado              |
| Entrenamiento         | Completado              |
| Tuning MobileNetV2    | Completado (`lr=3e-4`)  |
| Tratamiento del desbalance | Completado (oversampling) |
| Evaluación            | Completada              |
| Comparación           | Completada              |
| Selección final       | MobileNetV2 + oversampling |
| Testing               | 34/34 pruebas aprobadas |
| Documentación         | Actualizada             |
| Despliegue productivo | Fuera del alcance       |
