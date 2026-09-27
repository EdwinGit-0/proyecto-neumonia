# Proyecto de clasificación de neumonía

Clasificación de imágenes de rayos X de tórax en las clases `NORMAL` y `PNEUMONIA` mediante redes neuronales convolucionales con transferencia de aprendizaje.

## 1. Información general

**Título:** Clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA`.

**Objetivo:** entrenar, evaluar y comparar tres arquitecturas de redes convolucionales con una configuración comparable, seleccionar la mejor sobre el conjunto de validación, analizar la sensibilidad de sus hiperparámetros y optimizar el modelo elegido frente al desbalance de clases.

**Alcance:** análisis exploratorio, preparación de imágenes, entrenamiento experimental, evaluación y comparación de modelos, sensibilidad de hiperparámetros y optimización de cinco variantes. El modelo final se decide únicamente con el conjunto de validación.

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

El entrenamiento original de los tres modelos no utiliza oversampling, undersampling ni class weights. En la etapa de optimización se aplicó **oversampling de la clase minoritaria `NORMAL` únicamente sobre el `train`** del modelo final, y se probó también **class weights** como variante separada (ver sección 7). En ningún momento se modificaron `validation` ni `test`.

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
| Learning rate        | `1e-4` (flujo base) · `1e-3` (etapas 2 a 5) |
| Batch size           | `16`                    |
| Épocas               | `3` en el flujo base; la sensibilidad explora `3`, `5` y `10` |
| Semilla              | `42`                    |
| Checkpoint           | `val_loss`              |
| Early Stopping       | paciencia 3             |
| ReduceLROnPlateau    | factor 0.5, paciencia 2 |
| Learning rate mínimo | `1e-6`                  |

El learning rate es el único parámetro que cambia entre etapas: el flujo base de comparación de
arquitecturas usa `1e-4` y, una vez elegido el ganador de la sensibilidad, todas las etapas
posteriores parten de `1e-3`.

---

## 5. Selección del modelo

Todas las decisiones se realizan **exclusivamente sobre validation (1.043 imágenes)**.

El conjunto `test` de 624 imágenes permanece reservado y no participa en:

* entrenamiento;
* selección del modelo;
* ajuste de hiperparámetros.

Se consulta **dos veces y solo para medir**: el test inicial del checkpoint ganador de la
sensibilidad, antes de optimizar, y el test final del modelo ya elegido. En el artefacto oficial
`test_utilizado_para_seleccion` es `false`.

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
| 1 | Selección de arquitectura                    | VGG16, ResNet50, MobileNetV2               | `models/model_results.json`                  | No               |
| 2 | Sensibilidad de hiperparámetros               | 7 configuraciones × 3 arquitecturas = 21 entrenamientos | `models/sensitivity_results.json`             | No               |
| 3 | **Test inicial**                             | checkpoint del ganador de sensibilidad    | `models/optimization_results.json`           | No               |
| 4 | Optimización del desbalance                  | ganador fijo de la etapa 2                 | `models/optimization_results.json`           | No               |
| 5 | **Test final**                               | modelo elegido en la etapa 4               | `models/optimization_results.json`           | No               |

Después de la etapa 1, MobileNetV2 quedó como arquitectura y **no se volvió a comparar con
VGG16 ni ResNet50**. La etapa 2 exploró learning rate, dropout y número de épocas sobre las tres
arquitecturas (7 configuraciones, 21 entrenamientos en total, sin fine-tuning) y quedó cerrada con
este ganador:

| Parámetro    | Valor del ganador de sensibilidad        |
| ------------ | ----------------------------------------- |
| Arquitectura | MobileNetV2 (ImageNet)                    |
| Cabeza       | `GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)` |
| Learning rate | `1e-3`                                    |
| Dropout     | `0.3`                                     |
| Épocas       | `3`                                       |
| Entrenamiento | desde cero (sin fine-tuning)             |

Ese ganador es el **punto de partida fijo** de la etapa 4: su Balanced Accuracy en validation
es `0.959042` y se lee de `models/sensitivity_results.json`, sin reentrenar los 21 entrenamientos.

La etapa 4 evalúa cinco variantes sobre validation. Dos atacan el desbalance de clases y las tres
últimas el alcance del fine-tuning:

| Variante              | Qué cambia respecto al punto de partida                          |
| --------------------- | ----------------------------------------------------------------- |
| `oversampling_normal` | `NORMAL` se equipara con `PNEUMONIA` solo en `train` (1 073 → 3 100) |
| `class_weight`        | Pesos `{0: 1.9445, 1: 0.6731}` solo en `fit`                      |
| `finetune_block16`    | Se entrenan las capas del `block_16_expand` (15 de 158)          |
| `finetune_block13`    | Se entrenan las capas del `block_13_expand` (42 de 158)          |
| `finetune_block10`    | Se entrenan las capas del `block_10_expand` (68 de 158)          |

Cada variante cambia **un solo factor** respecto del punto de partida; el resto de
hiperparámetros (learning rate `1e-3`, dropout `0.3`, batch size `16`, 3 épocas, semilla `42`)
se mantiene constante. No se probaron learning rate, dropout ni epochs nuevos, porque la
etapa de sensibilidad ya había cerrado esa búsqueda.

El modelo final se elige aplicando la regla jerárquica acordada sobre **validation** —
Balanced Accuracy > ROC-AUC > F1 > Accuracy — sin ninguna tolerancia adicional: el recall se
reporta, pero no filtra candidatos. El test final se ejecuta **una sola vez**, sobre el modelo ya
elegido por validation.

Los artefactos del flujo vigente son:

```text
models/optimization_results.json          # test inicial, validation de las 5 variantes, selección y test final
models/mobilenetv2_optimizacion/           # checkpoint de cada variante
```

Los artefactos de las ejecuciones anteriores de esta etapa, no reproducibles, se conservan en:

```text
models/historico/
experiments/optimizacion_pendiente_mobilenetv2/
```

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

> **Nota sobre las figuras de VGG16 y ResNet50.** Esta tabla describe el **flujo base**, que es lo que
> declaran `models/model_results.json` → `models.VGG16.validation` y `models.ResNet50.validation`. Las
> imágenes `reports/figures/confusion_matrix_validation_vgg16.png`, `confusion_matrix_validation_resnet50.png`,
> `roc_curve_validation_vgg16.png` y `roc_curve_validation_resnet50.png` se regeneraron para mostrar la
> **mejor configuración por arquitectura** de la etapa de sensibilidad (`lr=1e-3`, dropout `0.3`, 3
> épocas) y por tanto **no** reproducen los números de esta tabla:
>
> | Figura | TN | FP | FN | TP | Balanced Accuracy | ROC-AUC |
> |---|---:|---:|---:|---:|---:|---:|
> | VGG16 | 260 | 8 | 70 | 705 | 0.9399 | 0.9850 |
> | ResNet50 | 150 | 118 | 67 | 708 | 0.7366 | 0.8931 |
>
> La afirmación de que ResNet50 clasificaba todo como `PNEUMONIA` sigue siendo válida para el flujo base
> (especificidad 0.0) y no describe la configuración de sensibilidad, donde su especificidad sube a
> 0.5597. Los detalles están en `references/guia_ejecucion.md`, sección 12.


### Test inicial

Primera medición sobre el test original, del checkpoint ganador de sensibilidad, antes de
optimizar:

| Métrica              | Test inicial (ganador de sensibilidad) |
| -------------------- | -------------------------------------: |
| Accuracy             |                               0.8413 |
| Balanced Accuracy    |                               0.7927 |
| Precision            |                               0.8038 |
| Recall / Sensitivity |                               0.9872 |
| Specificity          |                               0.5983 |
| F1                   |                               0.8861 |
| ROC-AUC              |                               0.9612 |

Matriz de confusión: `[[140, 94], [5, 385]]` (TN = 140, FP = 94, FN = 5, TP = 385).

### Optimización del desbalance (validation)

Decisión tomada **solo** sobre validation (1 043 imágenes), a partir del punto de partida
`MobileNetV2 lr_1e-3` (Balanced Accuracy 0.959042). Criterio aplicado:
Balanced Accuracy > ROC-AUC > F1 > Accuracy.

| Variante                          | Balanced Acc | ROC-AUC |   F1 | Accuracy | Recall | Specificity |
| --------------------------------- | -----------: | ------: | ---: | -------: | -----: | ----------: |
| `oversampling_normal` (seleccionada) | **0.9607** | 0.9949 | 0.9725 | 0.9597 | 0.9587 |      0.9627 |
| `class_weight`                    |       0.9602 | 0.9931 | 0.9746 |   0.9626 | 0.9652 |      0.9552 |
| Punto de partida (sensibilidad)   |       0.9590 | 0.9936 | 0.9760 |   0.9645 | 0.9703 |      0.9478 |
| `finetune_block16`                |       0.7344 | 0.9714 | 0.9154 |   0.8629 | 0.9987 |      0.4701 |
| `finetune_block10`                |       0.6922 | 0.9746 | 0.9038 |   0.8418 | 1.0000 |      0.3843 |
| `finetune_block13`                |       0.6119 | 0.9965 | 0.8817 |   0.8006 | 1.0000 |      0.2239 |

`oversampling_normal` gana por tener la mayor Balanced Accuracy (0.9607 frente a 0.9602 de
`class_weight` y 0.9590 del punto de partida). El criterio es jerárquico, así que basta con ganar la
primera métrica: no se exige ganar en las cuatro, y de hecho la variante seleccionada es peor que el
punto de partida en F1 (0.9725 frente a 0.9760) y en Accuracy (0.9597 frente a 0.9645). Las tres
variantes de fine-tuning muestran el patrón de colapso: recall 1.0 con especificidad entre 0.22 y 0.47.

### Test final

Ejecutado **una sola vez** sobre `oversampling_normal`, el modelo elegido por validation:

| Métrica              | Test inicial | **Test final (`oversampling_normal`)** |
| -------------------- | -----------: | ------------------------------------: |
| Accuracy             |      0.8413 |                             0.8734 |
| Balanced Accuracy    |      0.7927 |                             0.8372 |
| Precision            |      0.8038 |                             0.8418 |
| Recall / Sensitivity |      0.9872 |                             0.9821 |
| Specificity          |      0.5983 |                             0.6923 |
| F1                   |      0.8861 |                             0.9065 |
| ROC-AUC              |      0.9612 |                             0.9609 |

Matriz de confusión del test final: `[[162, 72], [7, 383]]` (TN = 162, FP = 72, FN = 7, TP = 383).

> **El modelo elegido por validation sí mejoró el test.** Balanced Accuracy pasa de 0.7927 a
> 0.8372 (+0.0444), con la mejora concentrada en especificidad (0.5983 → 0.6923) y F1
> (0.8861 → 0.9065). El recall se mantiene alto (0.9821) y el ROC-AUC es prácticamente igual
> (0.9612 → 0.9609).

### Reproducibilidad de la etapa

La etapa se corrigió para que sea reproducible. Antes, el `shuffle` del train y las capas de
augmentation se construían **sin semilla** y la semilla global se fijaba **después** de crear los
pipelines, de modo que dos ejecuciones del mismo código entrenaban sobre imágenes distintas.

Ahora `src/utils/reproducibility.py` centraliza el control: `SEMILLA = 42`,
`tf.config.experimental.enable_op_determinism()`, `seed=SEMILLA` en cada `shuffle` y en
`RandomRotation` / `RandomZoom`, y `map(..., deterministic=True)`. En
`entrenar_variante` el orden es `clear_session()` → `reiniciar_semilla()` → construir pipeline →
construir modelo, para que las cinco variantes vean exactamente los mismos datos.

Verificado empíricamente: dos procesos independientes con `SEMILLA = 42` producen lotes de
entrenamiento **idénticos byte a byte** (`ef090aacf796b100e2e1` en ambos, frente a huellas
distintas antes de la corrección), y dos construcciones del modelo tras reiniciar la semilla dan los
mismos pesos. Cubierto por `tests/test_reproducibilidad.py`.

> No se ha ejecutado una segunda corrida completa del entrenamiento para confirmar la igualdad
> exacta de las métricas finales: lo verificado es el determinismo del pipeline de datos y de la
> inicialización del modelo.

#### Comparación con las ejecuciones anteriores no controladas

Las dos ejecuciones previas de esta etapa se hicieron con el pipeline sin semilla, por lo que no
son comparables entre sí ni con la actual. Balanced Accuracy en validation:

| Variante              | Respaldo `experiments/` | Ejec. no reproducible | **Ejec. final reproducible** |
| --------------------- | ----------------------: | --------------------: | ---------------------------: |
| `oversampling_normal` |                 0.9555 |                0.9586 |                     **0.9607** |
| `class_weight`        |                 0.9646 |                0.9566 |                      0.9602 |
| `finetune_block16`    |                 0.6791 |                0.9674 |                      0.7344 |
| `finetune_block13`    |                 0.8221 |                0.8209 |                      0.6119 |
| `finetune_block10`    |                 0.7519 |                0.9516 |                      0.6922 |

El rango de `finetune_block16` entre las dos ejecuciones no controladas (0.6791 – 0.9674) es la medida del ruido que
introducía la ausencia de semilla. En la corrida controlada, el fine-tuning reproduce el mismo
patrón de colapso en las tres profundidades, lo que apunta a la actualización de las estadísticas
de Batch Normalization de los bloques descongelados (batch 16, learning rate `1e-3`, 3 épocas)
como causa, y no al azar del pipeline.

Los artefactos anteriores se conservan en `models/historico/` y en
`experiments/optimizacion_pendiente_mobilenetv2/`. No deben mezclarse con el resultado vigente.

### Resultados históricos (fuera del flujo vigente)

Las siguientes tablas corresponden a la experimentación anterior (`lr=3e-4` y oversampling) y se
conservan solo como registro histórico. **No son el resultado del flujo vigente** y no deben
usarse para justificar el modelo actual.

#### Tuning de MobileNetV2 (validation, histórico)

Sobre MobileNetV2 se probaron learning rate, dropout y fine-tuning. Resultados sobre validation:

| Configuración                     | Balanced Acc | ROC-AUC |   F1 | Accuracy | Recall | Specificity |
| --------------------------------- | -----------: | ------: | ---: | -------: | -----: | ----------: |
| Original (lr `1e-4`)              |       0.9412 |  0.9887 | 0.9613 |   0.9434 | 0.9458 |      0.9366 |
| lr `5e-5`                         |       0.9415 |  0.9862 | 0.9683 |   0.9530 | 0.9652 |      0.9179 |
| **lr `3e-4` (seleccionada)**       |     **0.9547** | **0.9928** | **0.9741** | **0.9616** | 0.9690 | 0.9403 |
| dropout `0.5`                      |       0.9423 |  0.9884 | 0.9703 |   0.9559 | 0.9703 |      0.9142 |
| fine-tuning `finetune_block13`          |       0.6194 |  0.9978 | 0.8837 |   0.8044 | 1.0000 |      0.2388 |

El fine-tuning de las últimas capas degradó gravemente la especificidad (0.2388) y se descartó. La configuración `lr=3e-4` (dropout `0.3`, base congelada) fue seleccionada como **MobileNetV2 ajustado**.

#### Tratamiento del desbalance (validation, histórico)

Sobre el ajustado (`lr=3e-4`) se probaron por separado **class weights** y **oversampling** de `NORMAL` en `train`:

| Configuración                         | Balanced Acc | ROC-AUC |   F1 | Accuracy | Recall | Specificity |
| ------------------------------------- | -----------: | ------: | ---: | -------: | -----: | ----------: |
| Ajustado `lr=3e-4` (referencia)       |       0.9555 |  0.9933 | 0.9774 |   0.9664 | 0.9781 |      0.9328 |
| Class weights (solo entrenamiento)    |       0.9530 |  0.9925 | 0.9644 |   0.9482 | 0.9432 |      0.9627 |
| **Oversampling `NORMAL` en train**     |     **0.9598** | **0.9936** | **0.9780** | **0.9674** | **0.9755** | **0.9440** |

El oversampling lideró según el criterio (Balanced Accuracy) y conservó una sensibilidad alta, por lo que fue seleccionado como modelo final de aquella etapa. Artefactos: `models/mobilenetv2_tuning_results.json` y `models/mobilenetv2_desbalance_results.json`.

> El resultado del oversampling es un hallazgo experimental de este dataset y no demuestra que el desbalance fuera la única causa de las diferencias observadas.

### Artefactos de resultados

| Etapa                        | Artefacto                                |
| ---------------------------- | ---------------------------------------- |
| Selección de arquitectura    | `models/model_results.json`              |
| Sensibilidad (7 configs × 3 arquitecturas) | `models/sensitivity_results.json` |
| Auditoría del test           | `models/sensitivity_test_audit.json`     |
| **Optimización y test final**| `models/optimization_results.json`      |

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
│   ├── mobilenetv2_tuning/<experimento>/best_model.keras   # histórico
│   ├── mobilenetv2_optimizacion/<variante>/best_model.keras
│   ├── historico/                       # ejecuciones no reproducibles
│   ├── model_results.json
│   ├── sensitivity_results.json
│   ├── sensitivity_test_audit.json
│   ├── optimization_results.json
│   ├── mobilenetv2_tuning_results.json
│   └── mobilenetv2_desbalance_results.json
├── experiments/
│   ├── sensibilidad_hiperparametros/   # 7 configuraciones x 3 arquitecturas = 21 checkpoints y resultados
│   └── optimizacion_pendiente_mobilenetv2/   # respaldo histórico (no forma parte del flujo)
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
│   │   ├── sensitivity.py
│   │   ├── optimizacion_mobilenetv2.py
│   │   ├── tuning_mobilenetv2.py        # construcción del modelo y criterio
│   │   └── tuning_desbalance_clases.py  # class weights y oversampling
│   ├── utils/
│   │   ├── paths.py
│   │   └── reproducibility.py          # única fuente de semilla
│   └── visualization/
├── tests/
│   ├── test_make_dataset.py
│   ├── test_pipeline_components.py
│   ├── test_reproducibilidad.py
│   ├── test_sensitivity.py
│   └── test_optimizacion_mobilenetv2.py
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini
```

El entrenamiento de la selección de arquitectura se encuentra en:

```text
src/training/run_real_training.py
```

La búsqueda de hiperparámetros de MobileNetV2 y el flujo de optimización vigente se reproducen
desde:

```text
src/training/sensitivity.py
src/training/optimizacion_mobilenetv2.py
```

Los módulos `tuning_mobilenetv2.py` y `tuning_desbalance_clases.py` ya no ejecutan barridos
propios: conservan el constructor del modelo, el criterio de selección, el cálculo de class weights
y el oversampling que usa el flujo vigente. Sus barridos históricos se replican solo desde
`experiments/`.

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

### Sensibilidad de hiperparámetros

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

### Test inicial

```powershell
neumonia test-inicial
```

Evalúa sobre el test original el checkpoint del ganador de sensibilidad, sin reentrenar los
21 entrenamientos. Es la primera medición sobre test del flujo y sirve de contraste con la
auditoría de sensibilidad.

### Optimización y test final

```powershell
neumonia optimizar
```

Ejecuta la etapa 4 completa: construye las cinco variantes sobre el ganador fijo de
sensibilidad, entrena cada una, selecciona el modelo sobre validation y ejecuta el test final
**una sola vez** sobre el modelo elegido. Genera o actualiza `models/optimization_results.json`
y `models/mobilenetv2_optimizacion/`.

Para reutilizar variantes ya entrenadas en lugar de repetirlas:

```powershell
neumonia optimizar --reusar
```

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

Encadena el EDA, la preparación de datos, la augmentación, la sensibilidad (sin repetirla si ya
existe), la optimización con su test inicial y su test final, y termina ejecutando la suite de
pruebas. No invoca `neumonia train`: la etapa de selección de arquitectura queda fuera porque la
sensibilidad ya entrena y evalúa las tres arquitecturas.

Los comandos `tune` y `desbalance` se han **eliminado de la CLI**: pertenecían a una
experimentación histórica (`lr=3e-4`) que quedó superada por la etapa de sensibilidad. Sus
resultados siguen disponibles como artefactos en `models/`, y el código que los generaba se conserva
en `src/training/tuning_mobilenetv2.py` y `src/training/tuning_desbalance_clases.py`. El checkpoint
que declara `models/mobilenetv2_desbalance_results.json`
(`models/mobilenetv2_desbalance_oversampling/best_model.keras`) ya no existe en el repositorio.

---

## 11. Testing

El proyecto cuenta con **73 pruebas automatizadas**, todas aprobadas:

```text
73 passed
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
* orden `clear_session()` → `reiniciar_semilla()` → pipeline → modelo en `entrenar_variante`;
* que el pipeline de optimización parte del ganador de sensibilidad y no vuelve a barrer
  learning rate, dropout ni epochs;
* aislamiento del `test` respecto de la decisión.

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
corrida completa del entrenamiento, de modo que la igualdad exacta de las métricas finales entre
ejecuciones no está comprobada empíricamente. La sección 7 debe leerse como una ejecución
concreta y reproducible en su parte de datos, no como valores deterministas garantizados de
principio a fin.

Los resultados de las dos ejecuciones **anteriores** de esta etapa no son reproducibles: se
obtuvieron con el pipeline sin semilla. Se conservan en `models/historico/` y en
`experiments/optimizacion_pendiente_mobilenetv2/` como registro histórico, y no deben mezclarse con
el resultado vigente.

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
* El oversampling de `NORMAL` en `train` duplica patrones existentes y no genera información nueva.
* El fine-tuning de las últimas capas de MobileNetV2 **empeora de forma consistente** el equilibrio
  entre clases: con learning rate `1e-3` y 3 épocas, las tres profundidades colapsan en validation
  (especificidad entre 0.22 y 0.47). No debe considerarse una técnica fiable en este dataset.
* La diferencia entre el punto de partida y la variante ganadora (Balanced Accuracy 0.9590 frente
  a 0.9607) es pequeña: con 1 043 imágenes de validation, diferencias de menos de ~0.01 no son
  distinguibles con una sola ejecución. La decisión se apoya en tener la mayor Balanced Accuracy,
  la primera métrica de un criterio jerárquico, y no exige ganar en las cuatro: la variante
  ganadora es en F1 y Accuracy peor que el punto de partida. El `test` se midió después de decidir
  y sirve para reportar el desempeño final, no para justificar la elección.
* El tamaño de validation (1 043 imágenes) introduce un intervalo de confianza amplio.
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
| Selección de arquitectura | Completado (MobileNetV2) |
| Sensibilidad de hiperparámetros | Completada (7 configs × 3 arquitecturas = 21 entrenamientos, ganador `lr=1e-3`) |
| Test inicial          | Completado (BA 0.7927)  |
| Optimización del desbalance | Completada y reproducible (gana `oversampling_normal`, BA val 0.9607) |
| Test final            | Completado (BA 0.8372)  |
| Selección final       | **Completada**: `MobileNetV2-oversampling_normal` |
| Testing               | Automatizado (73 pruebas) |
| Documentación         | Actualizada             |
| Despliegue productivo | Fuera del alcance       |
