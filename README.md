# Proyecto de clasificación de neumonía

Clasificación de imágenes de rayos X de tórax en las clases `NORMAL` y `PNEUMONIA` mediante redes neuronales convolucionales con transferencia de aprendizaje.

## 1. Información general

**Título:** Clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA`.

**Objetivo:** entrenar MobileNetV2 con una estrategia única de tratamiento del desbalance (COMBINADO: oversampling 50/50 sobre `train` más `class_weight`), congelar su umbral de decisión sobre `validation`, reentrenarlo sobre `train + val` y evaluar el test final. Antes se analiza la sensibilidad de sus hiperparámetros. La arquitectura y la estrategia son **premisas fijadas**, no variables a comparar.

**Alcance:** análisis exploratorio, preparación de imágenes, sensibilidad de hiperparámetros, entrenamiento con COMBINADO medido en validación, ajuste de umbral, entrenamiento del modelo definitivo y evaluación del test. **Todas** las decisiones se toman con `train` y `validation`; el test se lee al final, con el umbral ya congelado, y no participa en ninguna decisión.

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

El `RandomFlip("horizontal")` no se usa porque una inversión horizontal puede alterar información de lateralidad anatómica y marcadores `L/R` presentes en algunas radiografías.

El tratamiento de desbalance del proyecto es **COMBINADO** y se aplica **únicamente sobre el `train`**: oversampling 50/50 de la minoritaria `NORMAL` más `class_weight` en la función de pérdida. `validation` y `test` conservan su distribución original y nunca se reponderan.

---

## 4. Modelos

Se evaluaron tres arquitecturas mediante transferencia de aprendizaje con pesos de ImageNet:

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

El flujo final usa **MobileNetV2**: la sensibilidad de hiperparámetros la eligió y su configuración
queda cerrada en `models/sensitivity_results.json`.

### Configuración de entrenamiento

| Parámetro            | Configuración           |
| -------------------- | ----------------------- |
| Pérdida              | Binary Crossentropy     |
| Optimizador          | Adam                    |
| Learning rate        | `1e-3` (cerrado por la sensibilidad) |
| Dropout              | `0.3` (cerrado por la sensibilidad) |
| Batch size           | `16`                    |
| Épocas               | `3` (cerrado por la sensibilidad) |
| Semilla              | `42`                    |
| Checkpoint           | `val_loss`              |
| Early Stopping       | paciencia 3             |
| ReduceLROnPlateau    | factor 0.5, paciencia 2 |
| Learning rate mínimo | `1e-6`                  |

El modelo definitivo se reentrena sobre `train + val` con el número de épocas ya decidido, así que en
esa etapa el checkpoint guarda los pesos del último epoch.

---

## 5. Selección del modelo

Todas las decisiones se realizan **exclusivamente sobre validation (1.043 imágenes)**.

El conjunto `test` de 624 imágenes permanece reservado y no participa en:

* entrenamiento;
* selección de modelo o arquitectura;
* ajuste de hiperparámetros;
* ajuste del umbral de decisión.

Se lee **una sola vez**, al final del flujo, en la etapa 5. El artefacto de sensibilidad registra
`test_utilizado: false`.

El aislamiento es **estructural**, no una convención: `construir_pipelines_datos()` excluye el split
`test` por defecto. La única etapa que **itera** sus imágenes es `evaluar_test()`, que lo pide
con `incluir_test=True`; `neumonia prepare` también lo pide, pero solo para comprobar que el split
es construible, y como el dataset es perezoso (`tf.data.Dataset.from_generator`) no lee ninguna
imagen.

La regla de selección es jerárquica y se aplicó en cada etapa:

1. Balanced Accuracy
2. ROC-AUC
3. F1
4. Accuracy

El proceso completo se dividió en etapas cerradas. Cada etapa se ejecuta **una sola vez**
y su resultado queda registrado en un artefacto JSON que las etapas posteriores leen, sin
volver a entrenar lo ya resuelto.

| # | Etapa                                        | Entrada                                    | Salida                                        | ¿Se repite?      |
| - | -------------------------------------------- | ------------------------------------------ | --------------------------------------------- | ---------------- |
| 1 | Sensibilidad de hiperparámetros               | 7 configuraciones × 3 arquitecturas = 21 entrenamientos | `models/sensitivity_results.json`             | No               |
| 2 | COMBINADO sobre `train`                       | mejor configuración de MobileNetV2 de la etapa 1 | `results/final/combinado_validacion.json`   | No               |
| 3 | Umbral de decisión                           | probabilidades de validation de la etapa 2 | `results/final/umbral_decision.json`        | No               |
| 4 | Modelo definitivo                            | decisión congelada, `train + val`         | `results/final/final_model_training.json`     | No               |
| 5 | Test final                                   | modelo definitivo + umbral congelado      | `results/final/final_test_report.json`        | No               |

La etapa 1 exploró learning rate, dropout y número de épocas sobre las tres arquitecturas
(7 configuraciones, 21 entrenamientos) y cerró el espacio de búsqueda. Sus ganadores por
arquitectura se leen de `models/sensitivity_results.json` **sin reentrenar nada**:

| Arquitectura | Learning rate | Dropout | Épocas | Balanced Accuracy (validation) |
| ------------ | -------------: | ------: | -----: | ------------------------------: |
| MobileNetV2  |            `1e-3` |    `0.3` |    `3` |                        0.959042 |
| ResNet50     |            `1e-3` |    `0.3` |    `3` |                        0.736625 |
| VGG16        |            `1e-3` |    `0.3` |    `3` |                        0.939913 |

**MobileNetV2 es la arquitectura del modelo final.** COMBINADO (oversampling 50/50 sobre `train`
más `class_weight` en la pérdida) es la única estrategia de desbalance del proyecto: es una premisa
metodológica cerrada, no el resultado de comparar variantes. Cualquier informe debe declararlo así.

La etapa 2 entrena **una sola** corrida, MobileNetV2 con COMBINADO:

| Paso | Qué hace |
| ---- | -------- |
| Oversampling | `NORMAL` se equipara con `PNEUMONIA` solo en `train` (1.073 → 3.100), hasta 6.200 filas efectivas |
| Class weights | `{0: 1.9445, 1: 0.6731}`, calculados sobre la distribución original de `train` |
| Ámbito | Solo `train`: oversampling al pipeline, pesos a la función de pérdida de `fit` |
| `validation` | Intacta: 268 NORMAL / 775 PNEUMONIA, ni oversampling ni pesos |

> **Doble corrección.** Como los pesos se calculan sobre la distribución original y no sobre el
> conjunto ya equilibrado, no se anulan con el oversampling: el refuerzo efectivo de `NORMAL` frente a
> `PNEUMONIA` es de **2.89x**. COMBINADO no solo corrige el desbalance, lo invierte parcialmente en el
> entrenamiento. Está registrado en `diagnostico_combinado.advertencia_doble_correccion`.

El resto de hiperparámetros (learning rate `1e-3`, dropout `0.3`, batch size `16`, 3 épocas,
semilla `42`) es el cerrado en la etapa 1. La etapa 3 recorre 91 umbrales sobre validation y
**congela** el mejor; la etapa 4 reentrena sobre `train + val` sin volver a mirar validation.

Los artefactos del flujo vigente son:

```text
models/sensitivity_results.json                 # 21 corridas de sensibilidad (cerradas, no se repiten)
results/final/combinado_validacion.json         # COMBINADO medido en validation
results/final/combinado_validacion.csv
results/final/validacion_y_{true,prob}.npy
results/final/umbral_decision.json              # barrido de 91 umbrales sobre validation
results/final/decision_modelo.json
results/final/final_model/best_model.keras
results/final/final_model_training.json
results/final/final_test_report.json            # informe del test final
results/final/final_test_metrics.csv
```

Las figuras se generan en `reports/figures` y se nombran según la etapa que las produce:

| Figura | Etapa |
| ------ | ----- |
| `dataset_distribution.png`, `file_size_distribution.png`, `image_dimensions.png`, `sample_normal.png`, `sample_pneumonia.png` | EDA |
| `data_augmentation_examples.png` | `augment` |
| `sensitivity_validation.png` | `sensibilidad` |
| `validation_combinado_confusion_matrix.png`, `validation_combinado_roc_curve.png` | `combinado` |
| `threshold_selection_validation.png` | `umbral` |
| `test_final_confusion_matrix.png`, `test_final_roc_curve.png`, `test_final_metrics.png` | `test` |

El informe de la etapa `sensibilidad` queda en
`experiments/sensibilidad_hiperparametros/INFORME_SENSIBILIDAD.md`; no contiene métricas de test.

### Configuración de entrenamiento

| Parámetro         | Valor                                    |
| ----------------- | ---------------------------------------- |
| Batch size        | `16`                                     |
| Semilla           | `42`                                     |
| Augmentación      | `RandomRotation(0.05, seed=42)` y `RandomZoom(0.05, seed=42)`, solo en `train` |
| Early Stopping    | paciencia 3, `restore_best_weights=True` |
| ModelCheckpoint   | `val_loss`, `mode="min"`, guarda solo el mejor |
| ReduceLROnPlateau | factor 0.5, paciencia 2, mínimo `1e-6`  |

---

## 6. Criterio de éxito

El criterio de éxito se evalúa únicamente sobre el **test original**, y se reporta junto con la
referencia `0.50` para poder cuantificar el efecto del umbral.

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

El modelo COMBINADO **cumple las tres** en el test final, con el umbral congelado: Balanced Accuracy
`0.9073`, recall `0.9256` y specificity `0.8889` (ver sección 7).

---

## 7. Resultados

### COMBINADO sobre validation (1.043 imágenes)

Corrida actual: MobileNetV2, `lr=1e-3`, dropout `0.3`, 3 épocas, umbral de medición 0.5.

| Balanced Acc | ROC-AUC |   F1 | Accuracy | Precision | Recall | Specificity |
| -----------: | ------: | ---: | -------: | --------: | -----: | ----------: |
| **0.9549**   | 0.9939 | 0.9542 | 0.9348 |    0.9986 | 0.9135 |      0.9963 |

Matriz de confusión: `[[267, 1], [67, 708]]` (TN 267, FP 1, FN 67, TP 708).

### Umbral de decisión (validation)

La etapa 3 recorrió 91 umbrales (0.05 a 0.95 en pasos de 0.01, más 0.50 explícito) sobre el modelo
COMBINADO, todavía sin reentrenar, y **congeló** el mejor por Balanced Accuracy:

| Umbral | Balanced Acc |   F1 | Accuracy | Precision | Specificity | Recall |
| -----: | -----------: | ---: | -------: | -------: | ----------: | -----: |
|  0.50 |       0.9549 | 0.9542 |  0.9348 |  0.9986 |      0.9963 | 0.9135 |
| **0.38** |   **0.9621** | **0.9647** | **0.9492** | **0.9959** | **0.9888** | **0.9355** |
| Δ      |     +0.0072 | +0.0105 | +0.0144 | -0.0027 |     -0.0075 | +0.0220 |

Con el umbral congelado, la matriz de confusión de validation es `[[265, 3], [50, 725]]`
(TN 265, FP 3, FN 50, TP 725).

Artefacto: `results/final/umbral_decision.json`. Figura: `threshold_selection_validation.png`.

> **El Δ también hay que leerlo con cuidado.** El umbral se eligió maximizando Balanced Accuracy sobre
> validation, y ese mismo conjunto es el que se reporta: el `0.9621` está **sesgado al alza**. El 0.38
> queda congelado por el criterio acordado, y el artefacto lleva esta advertencia en
> `advertencia_optimismo`. Un umbral más bajo compra recall a costa de specificity, que es exactamente
> lo que el balanced accuracy premia.

### Modelo definitivo

La etapa 4 reentrenó MobileNetV2 con COMBINADO sobre `train + val` (5.216 imágenes:
1.341 NORMAL / 3.875 PNEUMONIA) durante 3 épocas, con el umbral `0.38` ya congelado y sin volver a
consultar validation. Como no queda validación que monitorear, los pesos finales se guardan de forma
explícita. Artefactos: `results/final/final_model_training.json` y
`results/final/final_model/best_model.keras`.

## 7-bis. Test final (624 imágenes)

La etapa 5 evalúa el modelo definitivo COMBINADO sobre el test original, con el umbral congelado en
`0.38` y **sin** ningún tratamiento de desbalance: las imágenes y la distribución son las originales
(234 `NORMAL`, 390 `PNEUMONIA`).

| Balanced Acc | ROC-AUC |   F1 | Accuracy | Precision | Recall | Specificity |
| -----------: | ------: | ---: | -------: | --------: | -----: | ----------: |
| **0.9073**   | 0.9728 | 0.9292 | 0.9119 |    0.9328 | 0.9256 |      0.8889 |

Matriz de confusión: `[[208, 26], [29, 361]]` (TN 208, FP 26, FN 29, TP 361).

Con el umbral por defecto de 0.5 sobre las mismas probabilidades: Balanced Accuracy `0.9026`,
accuracy `0.9006`, precision `0.9432`, recall `0.8949`, specificity `0.9103`, F1 `0.9184`.

El criterio de éxito se cumple en las tres condiciones: supera el baseline de Balanced Accuracy
(`0.5000`), recall `0.9256 > 0.5` y specificity `0.8889 > 0.5`.

Artefactos: `results/final/final_test_report.json` y `final_test_metrics.csv`.
Figuras: `test_final_confusion_matrix.png`, `test_final_roc_curve.png`, `test_final_metrics.png`.

**Orden respecto al umbral.** El umbral se congeló en la etapa 3, antes de reentrenar el modelo
definitivo y antes de leer el test. Cuando la evaluación ocurre, la arquitectura, la estrategia, los
hiperparámetros y el punto de corte ya están cerrados: la evaluación no realimenta ninguna decisión.

> **Gap validation → test:** el Balanced Accuracy cae de `0.9621` a `0.9073` (~0.055), mientras que el
> ROC-AUC apenas se mueve (`0.9939` → `0.9728`). Eso apunta a una degradación del punto de corte y no
> del ordenamiento de las probabilidades, y la causa más probable es la transferencia del umbral:
> se eligió con las probabilidades del modelo entrenado solo con `train` y se aplicó al reentrenado
> con `train + val`, cuya escala de probabilidades es otra. Es una hipótesis, no una medición.

### Reproducibilidad de la etapa

`src/utils/reproducibility.py` centraliza el control: `SEMILLA = 42`,
`tf.config.experimental.enable_op_determinism()`, `seed=SEMILLA` en cada `shuffle` y en
`RandomRotation` / `RandomZoom`, y `map(..., deterministic=True)`. El orden en cada corrida es
`clear_session()` → `reiniciar_semilla()` → construir pipeline → construir modelo, para que las
corridas vean exactamente los mismos datos.

Verificado empíricamente: dos procesos independientes con `SEMILLA = 42` producen lotes de
entrenamiento **idénticos byte a byte**, y dos construcciones del modelo tras reiniciar la semilla dan
los mismos pesos. Cubierto por `tests/test_reproducibilidad.py`.

> COMBINADO sobre `train` tardó 445 s y el entrenamiento del definitivo 487 s, ambos en CPU; la
> evaluación del test, unos 20 s. No se ha ejecutado una segunda corrida completa para confirmar la
> igualdad exacta de las métricas: lo verificado es el determinismo del pipeline de datos y de la
> inicialización del modelo, no la identidad bit a bit de los pesos entrenados.

### Artefactos de resultados

| Etapa                                        | Artefacto                                        |
| -------------------------------------------- | ------------------------------------------------ |
| Sensibilidad (7 configs × 3 arquitecturas)   | `models/sensitivity_results.json`                 |
| COMBINADO sobre `train`                      | `results/final/combinado_validacion.json`        |
| Decisión de umbral                           | `results/final/umbral_decision.json`              |
| Entrenamiento del definitivo                 | `results/final/final_model_training.json`         |
| Test final                                   | `results/final/final_test_report.json`            |
| Informe de sensibilidad                      | `experiments/sensibilidad_hiperparametros/INFORME_SENSIBILIDAD.md` |

Los checkpoints (`.keras`) y las probabilidades (`.npy`) se generan en `results/` y están
excluidos de git por tamaño; se regeneran con `neumonia combinado` y `neumonia final`.

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
│   └── sensitivity_results.json     # 21 corridas de sensibilidad (único artefacto de models/)
├── results/
│   └── final/                          # COMBINADO, umbral, modelo definitivo y test
│       ├── combinado_validacion.json
│       ├── combinado_validacion.csv
│       ├── validacion_y_{true,prob}.npy
│       ├── combinado/best_model.keras   # checkpoint de la etapa COMBINADO
│       ├── umbral_decision.json
│       ├── decision_modelo.json
│       ├── final_model/best_model.keras
│       ├── final_model_training.json
│       ├── final_test_report.json
│       ├── final_test_metrics.csv
│       ├── confusion_matrix_test_final.png
│       ├── roc_curve_test_final.png
│       └── test_y_{true,prob}.npy
├── experiments/
│   └── sensibilidad_hiperparametros/   # barrido de 21 corridas + informe
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
│   │   ├── architectures.py         # construir_modelo_proyecto
│   │   ├── evaluation.py            # métricas, baseline, criterio de éxito
│   │   └── comparison.py
│   ├── training/
│   │   ├── pipeline_entrenamiento.py  # augmentation, callbacks, entrenamiento, figuras
│   │   ├── tratamiento_desbalance.py  # COMBINADO: oversampling y class weights
│   │   ├── sensitivity.py             # 21 corridas, sin acceso a test
│   │   └── flujo_final.py             # combinado → umbral → definitivo → test
│   ├── utils/
│   │   ├── paths.py
│   │   └── reproducibility.py          # única fuente de semilla
│   └── visualization/
├── tests/
│   ├── test_make_dataset.py
│   ├── test_pipeline_components.py
│   ├── test_reproducibilidad.py
│   ├── test_sensitivity.py
│   └── test_flujo_final.py
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini
```

Los entry points del flujo real son:

```text
src/training/pipeline_entrenamiento.py   # entrenamiento compartido por sensibilidad y flujo final
src/training/tratamiento_desbalance.py   # COMBINADO: oversampling y class weights (solo train)
src/training/sensitivity.py              # 21 corridas de sensibilidad
src/training/flujo_final.py              # flujo final: combinado → umbral → definitivo → test
```

`construir_modelo_proyecto()` en `src/models/architectures.py` es el **único** punto donde se
construye una arquitectura del proyecto, de modo que la única diferencia entre corridas sea el
hiperparámetro o la estrategia que se esté variando.

`construir_pipelines_datos()` en `src/data/datasets.py` excluye el split `test` por defecto;
`incluir_test=True` solo se usa en `evaluar_test()` —la última etapa del flujo— y en `neumonia
prepare` (que solo verifica que sea construible, sin leer imágenes).

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

### Sensibilidad de hiperparámetros (21 corridas, ya cerrada)

```powershell
neumonia sensibilidad
```

Explora 7 configuraciones de learning rate, dropout y número de épocas, repetidas en las tres
arquitecturas, y escribe `models/sensitivity_results.json`. Es la etapa más costosa del proyecto:
si el artefacto ya existe, el comando no repite los 21 entrenamientos. Para forzar el
reentrenamiento:

```powershell
neumonia sensibilidad --recalcular
```

> Las 21 corridas ya están completas y **no deben repetirse**: la etapa 2 parte de
> `models/sensitivity_results.json`.

### COMBINADO sobre train

```powershell
neumonia combinado
```

Entrena MobileNetV2 con COMBINADO (oversampling 50/50 sobre `train` más `class_weight`) y mide
exclusivamente sobre `validation`, que queda intacta. Es la única etapa de entrenamiento sobre `train`
del flujo: la estrategia es una premisa del proyecto, no el resultado de comparar variantes.

### Umbral de decisión

```powershell
neumonia umbral
```

Recorre 91 umbrales sobre `validation` con el modelo COMBINADO, **sin** mirar el test, y congela el
mejor por Balanced Accuracy en `results/final/umbral_decision.json`. Para la ejecución validada
quedó en `0.38`, con un Δ de `+0.0072` frente al 0.5 por defecto. Genera la figura
`threshold_selection_validation.png`.

### Modelo definitivo

```powershell
neumonia final
```

Reentrena COMBINADO sobre `train + val` con el umbral congelado y guarda
`results/final/final_model/best_model.keras`. No vuelve a mirar `validation`. Tardó 487 s en CPU.

### Test final

```powershell
neumonia test
```

Evalúa el modelo definitivo sobre el test original con el umbral congelado. Se ejecuta como cualquier
otra etapa: sin banderas de confirmación y sin consultar historial. Aplica el umbral, escribe
`results/final/final_test_report.json` y `final_test_metrics.csv`, y genera la matriz de confusión, la
curva ROC y el gráfico de métricas. No aplica oversampling, `class_weight` ni augmentation al test.
Ver la sección 7-bis.

### Consultar resultados

```powershell
neumonia evaluar
```

Consulta los artefactos guardados sin volver a entrenar.

### Ejecutar pruebas

```powershell
neumonia test-suite
```

### Ejecutar el flujo completo

```powershell
neumonia run
```

Encadena el EDA, la preparación de datos, la augmentación, la sensibilidad (sin repetirla si ya
existe), COMBINADO, el umbral, el modelo definitivo, la evaluación del test y la suite de pruebas.
El test es la última etapa: el umbral ya está congelado cuando se lee.

---

## 11. Testing

El proyecto cuenta con **95 pruebas automatizadas**, todas aprobadas:

```text
95 passed, 109 warnings
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
* ausencia de `RandomFlip("horizontal")`;
* reproducibilidad del pipeline: semilla única, `shuffle` sembrado, augmentation sembrada,
  `deterministic=True` e igualdad de lotes entre pipelines construidos por separado;
* orden `clear_session()` → `reiniciar_semilla()` → pipeline → modelo en cada corrida;
* que COMBINADO parta de la mejor configuración de MobileNetV2 de la sensibilidad y no vuelva a
  barrer learning rate, dropout ni epochs;
* **aislamiento estructural del test**: `construir_pipelines_datos()` no construye el pipeline de
  test salvo que se pida con `incluir_test=True`; solo lo hace `neumonia prepare` (que no lo itera) y
  `evaluar_test()` (la última etapa del flujo);
* que pedir `incluir_test=True` **no lee** ninguna imagen hasta que el pipeline se itera;
* rechazo de `test` en la etapa de sensibilidad y en la construcción del pipeline de COMBINADO;
* que el pipeline del test se pida con el manifiesto original y **sin** ninguna bandera de
  oversampling, `class_weight` o augmentation;
* que `evaluar_test()` no reciba parámetros y que `neumonia test` no exponga ninguna opción;
* que las figuras de cada etapa vivan en `reports/figures` con el nombre de la etapa que las produce;
* que las figuras del flujo actual existan en disco y no estén vacías;
* que la sensibilidad escriba un único artefacto, sin métricas de test;
* que los checkpoints de etapas distintas no se pisen entre sí, y que el entrenamiento definitivo
  guarde el modelo explícitamente al no haber `validation` que dispare el `ModelCheckpoint`;
* que cada `.format()` del flujo declare exactamente las claves que le pasa, ya que esos mensajes
  solo se ejecutan al correr el flujo real;
* que el umbral declare su propio sesgo optimista;
* las tres etapas del flujo final y sus rutas de artefactos.

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

`src/utils/reproducibility.py` es la **única fuente de semilla** del proyecto
(`SEMILLA = 42`) y se aplica en siete sitios:

| Elemento | Cómo se controla |
| -------- | ---------------- |
| División experimental | `random_state=42` en `crear_manifiesto_division_estratificada` |
| Orden de los datos | `shuffle(..., seed=SEMILLA, reshuffle_each_iteration=True)` |
| Augmentación | `RandomRotation(0.05, seed=SEMILLA)` y `RandomZoom(0.05, seed=SEMILLA)` |
| Augmentación en `tf.data` | `map(..., deterministic=True)` |
| Operaciones de TensorFlow | `tf.config.experimental.enable_op_determinism()` |
| Pesos iniciales | `tf.keras.utils.set_random_seed(SEMILLA)` antes de construir el modelo |
| Oversampling | `sample_from_datasets(..., seed=SEMILLA)` |

**Qué está verificado y qué no.** Sí está verificado que dos procesos independientes con la misma
semilla producen lotes de entrenamiento idénticos byte a byte, y que dos construcciones del modelo
dan los mismos pesos (véase `tests/test_reproducibilidad.py`). **No** se ha ejecutado una segunda
corrida completa de los entrenamientos, de modo que la igualdad exacta de las métricas finales entre
ejecuciones no está comprobada empíricamente. La sección 7 debe leerse como una ejecución concreta y
reproducible en su parte de datos, no como valores deterministas garantizados de principio a fin.

Los artefactos del flujo vigente se almacenan en:

```text
models/sensitivity_results.json     # resultados de las 21 corridas
results/final/                      # COMBINADO, umbral, modelo definitivo y test final
```

La documentación técnica adicional está en `references/documentacion_proyecto.md` y
`references/guia_ejecucion.md`.

---

## 13. Limitaciones

* El `val` original contiene solamente 16 imágenes y no se utiliza en los experimentos.
* Existe desbalance entre las clases `NORMAL` y `PNEUMONIA`.
* Existen duplicados exactos dentro de algunos splits crudos.
* El oversampling de `NORMAL` en `train` duplica patrones existentes y no genera información nueva.
* **COMBINADO dobla la corrección del desbalance.** Los `class_weight` se calculan sobre la
  distribución original de `train`, así que no se anulan con el oversampling: el refuerzo efectivo de
  `NORMAL` es de 2.89x. Si se quisieran pesos que no inviertan el balance habría que fijarlos a 1.0,
  lo que ya no sería COMBINADO.
* **COMBINADO es una premisa metodológica, no una comparación.** Es la única estrategia de desbalance
  del proyecto y no se mide frente a alternativas. Cualquier informe debe declararlo así.
* **El Balanced Accuracy de validation está sesgado al alza.** El umbral se maximiza sobre el mismo
  conjunto que se reporta, de modo que el `0.9621` es un techo de selección, no una estimación de
  desempeño.
* **El gap validation → test es de ~0.055 en Balanced Accuracy** (`0.9621` frente a `0.9073`). El
  ROC-AUC apenas cae (`0.9939` → `0.9728`), lo que sugiere que se degrada el punto de corte y no el
  ordenamiento de las probabilidades.
* El tamaño de validation (1 043 imágenes) y de test (624 imágenes, 234 NORMAL) introduce un
  intervalo de confianza amplio en todas las métricas: diferencias de menos de ~0.01 no son
  distinguibles con una sola ejecución.
* El umbral óptimo en validation (`0.38`) se ajustó sobre las probabilidades del modelo entrenado
  solo con `train`; el definitivo se reentrenó con `train + val` y sus probabilidades tienen otra
  escala. La transferencia del umbral es una suposición razonable, no verificada empíricamente, y es
  la explicación más probable del gap anterior.
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
| Sensibilidad de hiperparámetros | Completada (7 configs × 3 arquitecturas = 21 entrenamientos, todos con `lr=1e-3`) |
| COMBINADO sobre `train` | Completada (BA validation 0.9549, ROC-AUC 0.9939) |
| Umbral de decisión    | Completada (congelado en `0.38`, BA validation 0.9621) |
| Modelo definitivo     | Completada (`train + val`, 3 épocas, 487 s) |
| Test final            | Completada (BA 0.9073, ROC-AUC 0.9728, matriz `[[208, 26], [29, 361]]`) |
| Testing               | Automatizado (95 pruebas) |
| Documentación         | Actualizada             |
| Despliegue productivo | Fuera del alcance       |

El flujo se ejecuta de principio a fin: `combinado` → `umbral` → `final` → `test`.
