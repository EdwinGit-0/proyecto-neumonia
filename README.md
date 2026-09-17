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

No se utilizaron oversampling, undersampling ni class weights.

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
| Learning rate        | `1e-4`                  |
| Batch size           | `16`                    |
| Máximo de epochs     | `3`                     |
| Semilla              | `42`                    |
| Checkpoint           | `val_loss`              |
| Early Stopping       | paciencia 3             |
| ReduceLROnPlateau    | factor 0.5, paciencia 2 |
| Learning rate mínimo | `1e-6`                  |

---

## 5. Selección del modelo

La selección se realiza **exclusivamente sobre validation (1.043 imágenes)**.

El conjunto `test` de 624 imágenes permanece reservado para la evaluación final y no participa en:

* entrenamiento;
* selección del modelo;
* ajuste de hiperparámetros.

La regla de selección es jerárquica:

1. Balanced Accuracy
2. ROC-AUC
3. F1
4. Accuracy

Con esta regla, el modelo seleccionado fue **MobileNetV2**.

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

MobileNetV2 fue seleccionado por presentar el mejor desempeño según la regla jerárquica definida.

ResNet50 clasificó todos los casos de validation como `PNEUMONIA`, por lo que obtuvo sensibilidad de 1.0 pero especificidad de 0.0.

### Test final

El modelo seleccionado, **MobileNetV2**, fue evaluado posteriormente sobre las 624 imágenes del test original:

| Métrica              | Resultado |
| -------------------- | --------: |
| Accuracy             |    0.8381 |
| Balanced Accuracy    |    0.7885 |
| Precision            |    0.8004 |
| Recall / Sensitivity |    0.9872 |
| Specificity          |    0.5897 |
| F1                   |    0.8840 |
| ROC-AUC              |    0.9545 |

Matriz de confusión:

```text
[[138, 96],
 [  5, 385]]
```

Donde:

* TN = 138
* FP = 96
* FN = 5
* TP = 385

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
│   └── model_results.json
├── notebooks/
│   └── 01_comprension_datos_eda.ipynb
├── references/
│   ├── documentacion_proyecto.md
│   ├── guia_ejecucion.md
│   └── instrucciones_copilot.md
├── reports/
│   └── figures/
├── src/
│   ├── cli.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
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

Este comando entrena y evalúa VGG16, ResNet50 y MobileNetV2.

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
| Evaluación            | Completada              |
| Comparación           | Completada              |
| Selección             | MobileNetV2             |
| Testing               | 34/34 pruebas aprobadas |
| Documentación         | Actualizada             |
| Despliegue productivo | Fuera del alcance       |
