# Documentación técnica del proyecto

## 1. Información general

**Título:** clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA` mediante modelos convolucionales con transferencia de aprendizaje.

**Objetivo:** comparar VGG16, ResNet50 y MobileNetV2 bajo condiciones experimentales idénticas (mismos datos, mismo preprocessing, misma augmentation, mismos hiperparámetros fijados por la sensibilidad, misma estrategia COMBINADO), seleccionar la mejor arquitectura en `validation`, congelar su umbral, reentrenar el modelo definitivo sobre `train + val` y evaluar el test final.

**Alcance:** análisis exploratorio, preparación de imágenes, sensibilidad de hiperparámetros (21 entrenamientos), COMBINADO para las tres arquitecturas, comparación en validation, selección de arquitectura, ajuste de umbral, entrenamiento del modelo definitivo y evaluación del test. **Todas** las decisiones se toman con `train` y `validation`; el test se lee una sola vez, al final, con el umbral ya congelado, y no participa en ninguna decisión.

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

El `val` original de 16 imágenes pertenece a la estructura cruda del dataset y **no se utiliza en ningún experimento** (ni entrenamiento, ni validación, ni selección de modelos).

El `test` original de **624 imágenes permanece completamente intacto**: no participa en entrenamiento, no participa en selección de modelos, de estrategia ni de umbral, y no se mezcla con `train` ni `validation`. Se consulta **una sola vez**, al final, sobre el modelo definitivo ya congelado.

### División experimental

Para el modelado, únicamente las **5.216 imágenes del train original** se dividen en aproximadamente 80% train y 20% validation, de forma estratificada por clase, con semilla 42 y agrupación por hash SHA-256 para evitar que imágenes con contenido idéntico queden repartidas entre `train` y `validation`. El manifiesto es `data/interim/stratified_split_train80_val20_test_original.csv`:

| Conjunto | NORMAL | PNEUMONIA | Total | Origen |
|---|---:|---:|---:|---|
| train | 1.073 | 3.100 | 4.173 | 80% aprox. del train original |
| validation | 268 | 775 | 1.043 | 20% aprox. del train original |
| test | 234 | 390 | 624 | test original íntegro |
| **Total** | **1.575** | **4.265** | **5.840** | excluye el val original de 16 |

Los totales (1.575 NORMAL + 4.265 PNEUMONIA = 5.840) corresponden a 5.856 imágenes del crudo menos las 16 del `val` original que no se usan.

El EDA verificó las imágenes y no detectó archivos corruptos, pero detectó duplicados por SHA-256 dentro de los splits crudos. El manifiesto experimental agrupa por hash y no existen grupos duplicados que crucen `train`/`validation`/`test`. El archivo `data/raw/chest_xray.dvc` registra 5.856 archivos y 1.236.482.806 bytes.

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
│   │   └── stratified_split_train80_val20_test_original.csv
│   └── raw/
│       ├── chest_xray.dvc
│       └── chest_xray/
├── models/
│   └── sensitivity_results.json          # 21 corridas de sensibilidad
├── results/
│   └── final/                              # COMBINADO, umbral, modelo definitivo y test
│       ├── combinado_validacion.json
│       ├── combinado_validacion.csv
│       ├── validacion_y_{true,prob}.npy
│       ├── combinado/best_model.keras      # checkpoint de la etapa COMBINADO
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
│   └── sensibilidad_hiperparametros/      # barrido de 21 corridas + informe
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
│   │   ├── architectures.py               # construir_modelo_proyecto
│   │   ├── comparison.py
│   │   └── evaluation.py                  # métricas, baseline, criterio
│   ├── training/
│   │   ├── pipeline_entrenamiento.py      # augmentation, callbacks, figuras
│   │   ├── tratamiento_desbalance.py      # COMBINADO: oversampling + class weights
│   │   ├── sensitivity.py                 # 21 corridas, sin acceso a test
│   │   └── flujo_final.py                 # combinado → umbral → final → test
│   ├── utils/
│   │   ├── paths.py
│   │   └── reproducibility.py            # única fuente de semilla
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

Los checkpoints (`.keras`) y las probabilidades (`.npy`) se excluyen de git por tamaño; los resultados en JSON y CSV sí se versionan.

## 5. CLI

| Comando | Función |
|---|---|
| `neumonia eda` | Análisis exploratorio del dataset. |
| `neumonia prepare` | Manifiesto estratificado 80/20 y verificación de pipelines. |
| `neumonia augment` | Figura de ejemplos de augmentación. |
| `neumonia sensibilidad` | 21 corridas de sensibilidad; no repite si el artefacto existe. |
| `neumonia combinado-arquitecturas` | COMBINADO para VGG16, ResNet50, MobileNetV2 sobre `train`, medido en `validation`. |
| `neumonia comparar` | Compara las 3 arquitecturas y selecciona la mejor (criterio jerárquico). |
| `neumonia combinado` | MobileNetV2 con COMBINADO (flujo original, compatibilidad). |
| `neumonia umbral` | Congela el umbral de decisión usando solo `validation`. |
| `neumonia final` | Reentrena el definitivo sobre `train + val`. |
| `neumonia test` | Evalúa el test original con el modelo definitivo y el umbral congelado. |
| `neumonia evaluar` | Muestra los resultados guardados sin entrenar. |
| `neumonia test-suite` | Ejecuta `pytest`. |
| `neumonia run` | Flujo completo nuevo: 3 arquitecturas → comparación → test final. |

`neumonia run` encadena: EDA → prepare → augment → sensibilidad → combinado-arquitecturas → comparar → umbral → final → test → evaluar → test-suite.

## 6. Preparación de datos y augmentación

El reparto experimental lo genera `crear_manifiesto_division_estratificada` (`src/data/splitting.py`). Lee el dataset crudo, divide únicamente las imágenes del `train` original en `train` (80%) y `validation` (20%) de forma estratificada por clase, con semilla 42, y asigna las imágenes del `test` original (624) directamente al conjunto `test` sin modificarlas. Agrupa por hash SHA-256 para que imágenes con contenido idéntico no queden repartidas entre conjuntos. Las 16 imágenes del `val` original no forman parte del manifiesto.

El preprocesamiento (`src/data/preprocessing.py`) carga cada imagen con Pillow y la convierte a RGB, la redimensiona a `224 x 224` con interpolación bilinear y la normaliza a `[0, 1]` (división por 255). El código no utiliza el `preprocess_input` específico de cada backbone.

`src/data/datasets.py` construye los `tf.data.Dataset` con tensores `224 x 224 x 3`, mezcla los ejemplos, agrupa en batches y usa `prefetch` automático. La augmentación se aplica únicamente sobre `train` con `RandomRotation(0.05)` y `RandomZoom(0.05)`; `validation` y `test` no reciben augmentation.

**Aislamiento estructural del test.** `construir_pipelines_datos()` recibe `incluir_test=False` por defecto, de modo que el pipeline de test no se construye en ninguna etapa salvo donde se pide explícitamente con `incluir_test=True`: la evaluación final (`evaluar_test()`), que lo itera para predecir, y la etapa `neumonia prepare`, que solo verifica que el split sea construible. El dataset se crea con `tf.data.Dataset.from_generator`, por lo que es perezoso: construir el objeto no lee ninguna imagen, y `prepare` no abre ni una sola del test.

El volteo horizontal (`RandomFlip("horizontal")`) no se usa: en radiografías de tórax puede existir información de lateralidad anatómica y marcadores L/R, y un volteo horizontal podría invertir artificialmente esa información. El tratamiento de desbalance COMBINADO se aplica **solo sobre `train`**, nunca sobre `validation` ni `test`. La figura de ejemplos se genera con `graficar_ejemplos_augmentation` (`src/visualization/visualize.py`) en `reports/figures/data_augmentation_examples.png`.

## 7. Configuración experimental de los modelos

Los tres modelos usan Keras Applications con pesos ImageNet, `include_top=False` y base convolucional **congelada**. Sobre la base se añade:

```text
GlobalAveragePooling2D
Dense(128, activation="relu")
Dropout(0.3)
Dense(1, activation="sigmoid")
```

El constructor único es `construir_modelo_proyecto()` (`src/models/architectures.py`), usado tanto por la sensibilidad como por el flujo final, de modo que la única diferencia entre corridas sea el hiperparámetro o la estrategia que se varíe.

| Parámetro | Valor vigente |
|---|---|
| Entrada | `224 x 224 x 3` |
| Pérdida | `binary_crossentropy` |
| Optimizador | Adam, `learning_rate=1e-3` |
| Dropout | `0.3` |
| Batch size | `16` |
| Épocas | `3` |
| Semilla | `42` |
| `ModelCheckpoint` | monitoriza `val_loss`, guarda solo el mejor |
| `EarlyStopping` | paciencia 3, restaura mejores pesos |
| `ReduceLROnPlateau` | factor 0.5, paciencia 2, mínimo `1e-6` |

Los valores `learning_rate=1e-3`, `dropout=0.3` y 3 épocas no son arbitrarios: los cerró la sensibilidad (sección 8.1). El modelo definitivo se entrena los mismos 3 épocas exactos, sin early stopping, porque ya no existe un conjunto de validación que monitorear.

## 8. Entrenamiento, evaluación y selección

Todo el flujo real está implementado en `src/training/`: `pipeline_entrenamiento.py` (entrenamiento compartido), `tratamiento_desbalance.py` (COMBINADO), `sensitivity.py` (21 corridas) y `flujo_final.py` (etapas 2 a 5).

El `test` original (624 imágenes) queda reservado y **nunca decide**: se lee una sola vez, al final del flujo, cuando la arquitectura, la estrategia y el umbral ya están congelados. Las predicciones binarias de las etapas de decisión usan umbral 0.5; el umbral final se ajusta en la etapa 3. La regla de selección es jerárquica y, en orden:

1. Balanced Accuracy;
2. ROC-AUC;
3. F1;
4. Accuracy.

Esta regla reemplazó a la anterior `Recall -> Accuracy -> F1`, que podía seleccionar un modelo que detectara todos los positivos mientras clasificaba mal todos los normales. Al ser jerárquica, basta con ganar la primera métrica: no se exige ganar en las cuatro, y el recall se reporta pero no filtra candidatos.

### Criterio de éxito

El criterio de éxito es independiente del criterio de selección y se evalúa únicamente sobre el test original. El modelo definitivo debe:

1. **Superar un baseline sencillo**: la *clase mayoritaria* (predecir siempre `PNEUMONIA`), con Balanced Accuracy 0.5.
2. **Demostrar equilibrio**: sensibilidad y especificidad ambas por encima de 0.5.

El baseline se calcula en `evaluar_baseline_mayoritaria` y el criterio en `evaluar_criterio_exito` (`src/models/evaluation.py`); ambos se persisten en `results/final/final_test_report.json` (`baseline_clase_mayoritaria` y `criterio_exito`).

### 8.1 Sensibilidad de hiperparámetros (21 corridas, cerrada)

Ejecutable en `src/training/sensitivity.py` (CLI: `neumonia sensibilidad`) y en
`experiments/sensibilidad_hiperparametros/run_sensibilidad.py`. Explora **7 configuraciones** de
hiperparámetros, cada una repetida en las **3 arquitecturas**, es decir **21 entrenamientos**
completos con la base convolucional congelada. La etapa **rechaza** cualquier split distinto de
`train` y `validation`.

| `id` | Grupo | Learning rate | Dropout | Épocas | Varía |
|---|---|---:|---:|---:|---|
| `ref_lr1e-4_do0.3_ep3` | referencia | 1e-4 | 0.3 | 3 | los tres |
| `lr_3e-4` | learning_rate | 3e-4 | 0.3 | 3 | learning rate |
| `lr_1e-3` | learning_rate | 1e-3 | 0.3 | 3 | learning rate |
| `do_0.2` | dropout | 1e-4 | 0.2 | 3 | dropout |
| `do_0.5` | dropout | 1e-4 | 0.5 | 3 | dropout |
| `ep_5` | epochs | 1e-4 | 0.3 | 5 | épocas |
| `ep_10` | epochs | 1e-4 | 0.3 | 10 | épocas |

Cada configuración varía **un solo factor** respecto de la referencia, lo que permite atribuir el
efecto a ese factor.

La mejor configuración por arquitectura (Balanced Accuracy en `validation`):

| Arquitectura | `id` ganadora | Balanced Accuracy | ROC-AUC | F1 | Accuracy | Δ vs referencia |
|---|---|---:|---:|---:|---:|---:|
| MobileNetV2 | `lr_1e-3` | 0.959042 | 0.993645 | 0.975990 | 0.964525 | +0.018796 |
| VGG16 | `lr_1e-3` | 0.939913 | 0.985031 | 0.947581 | 0.925216 | +0.064025 |
| ResNet50 | `lr_1e-3` | 0.736625 | 0.893067 | 0.884447 | 0.822627 | +0.236625 |

Las tres arquitecturas comparten `lr=1e-3`, `dropout=0.3` y 3 épocas, de modo que el learning rate es
el factor dominante y el resto de la búsqueda no aporta nada. En la configuración de referencia
(`1e-4`), ResNet50 colapsaba a Balanced Accuracy 0.5 (especificidad 0.0): clasificaba todo como
`PNEUMONIA`. Con `1e-3` su especificidad sube a 0.5597.

Artefactos: `models/sensitivity_results.json` (oficial, con `test_utilizado: false` y
`splits_utilizados: ["train", "val"]`) y
`experiments/sensibilidad_hiperparametros/INFORME_SENSIBILIDAD.md`. Las 21 corridas están
completadas y **no deben repetirse**: la etapa 2 parte de este JSON.

### 8.2 COMBINADO para 3 arquitecturas

Implementado en `tratamiento_desbalance.construir_train_combinado` y
`flujo_final.ejecutar_combinado_arquitecturas` (CLI: `neumonia combinado-arquitecturas`). Se toman
los hiperparámetros fijados por la sensibilidad (lr=1e-3, dropout=0.3, epochs=3) y se entrenan **tres**
corridas COMBINADO: una por arquitectura (VGG16, ResNet50, MobileNetV2):

| Paso | Detalle |
|---|---|
| Oversampling | `NORMAL` se equipara con `PNEUMONIA` en `train` (1.073 → 3.100), hasta 6.200 filas efectivas |
| Class weights | `{0: 1.9445, 1: 0.6731}`, sobre la distribución **original** de `train` |
| Ámbito | Solo `train`: oversampling al pipeline, pesos a la función de pérdida de `fit` |
| `validation` | Intacta: 268 NORMAL / 775 PNEUMONIA, sin oversampling ni pesos |

La función rechaza explícitamente cualquier solicitud que incluya el split `test`.

**Aviso de doble corrección.** Al calcularse los pesos sobre la distribución original y no sobre el
conjunto ya equilibrado, oversampling y pesos no se anulan: el refuerzo efectivo de `NORMAL` frente a
`PNEUMONIA` es de 2.89x. COMBINADO no solo corrige el desbalance, lo invierte parcialmente en el
entrenamiento. Queda registrado en `diagnostico_combinado.advertencia_doble_correccion`.

No hay comparación de alternativas de desbalance: COMBINADO es la única estrategia y se aplica
igualmente a las tres arquitecturas.

### 8.3 Comparación y selección de arquitectura

`flujo_final.comparar_arquitecturas` (CLI: `neumonia comparar`) carga los 3 resultados COMBINADO,
compara las arquitecturas usando el criterio jerárquico (Balanced Accuracy > ROC-AUC > F1 >
Accuracy) sobre validation y selecciona la mejor.

Resultados de la ejecución validada (umbral 0.5 de medición):

| Pos | Arquitectura | Balanced Acc | ROC-AUC | F1 | Accuracy | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **MobileNetV2** | 0.9620 | 0.9941 | 0.9619 | 0.9453 | 0.9277 | 0.9963 |
| 2 | VGG16 | 0.8904 | 0.9878 | 0.8786 | 0.8389 | 0.7845 | 0.9963 |
| 3 | ResNet50 | 0.7595 | 0.9002 | 0.6886 | 0.6462 | 0.5265 | 0.9925 |

**Arquitectura seleccionada: MobileNetV2**

Copia los resultados de MobileNetV2 a los archivos estándar del flujo
(`results/final/combinado_validacion.json`, `validacion_y_{true,prob}.npy`, etc.) para que las
etapas siguientes funcionen sin cambios.

Artefactos: `results/final/comparacion_arquitecturas.json`, `results/final/seleccion_arquitectura.json`.

### 8.4 Umbral de decisión

`flujo_final.ajustar_umbral` (CLI: `neumonia umbral`) recorre 91 umbrales (0.05 a 0.95 en pasos de
0.01, más 0.50 explícito) sobre las probabilidades de `validation` del modelo COMBINADO de la
arquitectura seleccionada, **sin** mirar el test, y **congela** el de mayor Balanced Accuracy.
Persiste la curva completa en `results/final/umbral_decision.json` con `test_utilizado: false` y
`congelado: true`.

El artefacto incluye `advertencia_optimismo`: como el umbral se maximiza sobre `validation` y ese
mismo conjunto se reporta, el Balanced Accuracy resultante está sesgado al alza. La diferencia frente
al 0.5 por defecto debe leerse con esa reserva, y la transferencia del umbral al modelo definitivo
—entrenado con otra distribución de probabilidades— es una suposición no verificada.

**Umbral congelado: 0.38** (Δ BA +0.0072 frente a 0.5).

### 8.5 COMBINADO MobileNetV2 en validation con umbral 0.38 (etapa de decisión)

Tras congelar el umbral en 0.38, las métricas de validation para la arquitectura seleccionada son:

| Métrica | Valor (umbral 0.38) |
|---|---:|
| **Balanced Accuracy** | **0.9621** |
| Recall | 0.9355 |
| Specificity | 0.9888 |
| Precision | 0.9959 |
| F1 | 0.9647 |
| Accuracy | 0.9492 |

Matriz de confusión: `[[265, 3], [50, 725]]` (TN 265, FP 3, FN 50, TP 725).

### 8.5 Modelo definitivo

`flujo_final.entrenar_modelo_definitivo` (CLI: `neumonia final`) reentrena COMBINADO sobre
`train + val` (5.216 imágenes) con el umbral ya congelado. Como no queda validación que monitorear,
entrena 3 épocas exactas y **guarda los pesos finales explícitamente**: sin `validation` no se añade
`ModelCheckpoint`, así que el `model.save` es manual. Persiste
`results/final/final_model_training.json` y `results/final/final_model/best_model.keras`.

### 8.6 Test final

`flujo_final.evaluar_test` (CLI: `neumonia test`) es la última etapa del flujo. No recibe parámetros,
no consulta ningún historial de ejecuciones anteriores y no expone banderas de confirmación: se
ejecuta como cualquier otra etapa.

Verifica que existan la decisión y el modelo definitivo, comprueba que el manifiesto declare las 624
filas del test original, lo carga con `incluir_test=True` **sin oversampling, sin `class_weight` y sin
augmentation**, aplica el umbral congelado y escribe `results/final/final_test_report.json`,
`final_test_metrics.csv`, la matriz de confusión, la curva ROC y las probabilidades.

`neumonia run` encadena esta etapa como última del flujo.

## 9. Resultados reales

### COMBINADO sobre `validation` (1.043 imágenes) - por arquitectura

Tres arquitecturas entrenadas con COMBINADO (lr=1e-3, dropout=0.3, epochs=3), umbral de medición 0.5:

| Arquitectura | Balanced Acc | ROC-AUC | F1 | Accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | **0.9620** | 0.9941 | 0.9619 | 0.9453 | 0.9986 | 0.9277 | 0.9963 |
| VGG16 | 0.8904 | 0.9878 | 0.8786 | 0.8389 | 0.9984 | 0.7845 | 0.9963 |
| ResNet50 | 0.7595 | 0.9002 | 0.6886 | 0.6462 | 0.9951 | 0.5265 | 0.9925 |

Matrices de confusión (umbral 0.5):
- MobileNetV2: TN 267, FP 1, FN 56, TP 719
- VGG16: TN 267, FP 1, FN 167, TP 608
- ResNet50: TN 266, FP 2, FN 367, TP 408

### Comparación y selección de arquitectura

Criterio jerárquico: **Balanced Accuracy > ROC-AUC > F1 > Accuracy** (sobre validation)

| Pos | Arquitectura | Balanced Acc | ROC-AUC | F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| 1 | **MobileNetV2** | 0.9620 | 0.9941 | 0.9619 | 0.9453 |
| 2 | VGG16 | 0.8904 | 0.9878 | 0.8786 | 0.8389 |
| 3 | ResNet50 | 0.7595 | 0.9002 | 0.6886 | 0.6462 |

**Arquitectura seleccionada: MobileNetV2**

### Umbral elegido

| Umbral | Balanced Accuracy | F1 | Accuracy | Precision | Specificity | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.9549 | 0.9542 | 0.9348 | 0.9986 | 0.9963 | 0.9135 |
| **0.38** | **0.9621** | 0.9647 | 0.9492 | 0.9959 | 0.9888 | 0.9355 |

El 0.38 queda congelado por el criterio acordado (máximo Balanced Accuracy, desempate por
cercanía a 0.5), con Δ +0.0072 frente al 0.5 por defecto. El número está sesgado al alza porque el
umbral se maximiza sobre el mismo conjunto que se reporta; ver `advertencia_optimismo` en el
artefacto.

### Baseline mayoritaria sobre test (referencia)

| Métrica | Baseline |
|---|---:|
| Accuracy | 0.6250 |
| Balanced Accuracy | 0.5000 |
| Recall | 1.0000 |
| Specificity | 0.0000 |

### Test final (624 imágenes: 234 NORMAL / 390 PNEUMONIA)

`neumonia test`, con el modelo definitivo `MobileNetV2/combinado` entrenado en `train + val` y el
umbral congelado en 0.38:

| Métrica | Valor (umbral 0.38) | Referencia (umbral 0.5) |
|---|---:|---:|
| **Balanced Accuracy** | **0.8996** | 0.8936 |
| Accuracy | 0.8990 | 0.8862 |
| Precision | 0.9383 | 0.9493 |
| Recall | 0.8974 | 0.8641 |
| Specificity | 0.9017 | 0.9231 |
| F1 | 0.9174 | 0.9047 |
| ROC-AUC | 0.9694 | 0.9694 |

Matriz de confusión: **TN 211 / FP 23 / FN 40 / TP 350** (umbral 0.38).

El criterio de éxito se cumple: supera el baseline de Balanced Accuracy (0.5000), recall 0.8974 > 0.5
y specificity 0.9017 > 0.5.

Artefactos: `results/final/final_test_report.json`, `final_test_metrics.csv`,
`test_final_confusion_matrix.png`, `test_final_roc_curve.png`, `test_final_metrics.png`.

**Orden de la evaluación de test**

El umbral se congela en la etapa 8.4, **antes** de que el modelo definitivo se reentrene y antes de
que el test se lea. En el momento de la evaluación, la arquitectura, la estrategia, los
hiperparámetros y el punto de corte ya están cerrados: la evaluación no realimenta ninguna decisión.

> **Gap validation → test:** el balanced accuracy pasa de 0.9621 (validation, sesgada por la selección
> del umbral) a 0.8996 (test), una caída de ~0.062. El ROC-AUC se mantiene alto (0.9694 frente a
> 0.9941), lo que sugiere que la degradación está en el ajuste del punto de corte y no en el
> ordenamiento de las probabilidades. La transferencia del umbral —elegido con las probabilidades del
> modelo entrenado solo con `train` y aplicado al reentrenado con `train + val`— es la causa más
> probable, y no está verificada empíricamente.

## 10. Testing

La suite se ejecuta con `pytest` o con `neumonia test-suite`. Contiene **95 pruebas** y la ejecución
final terminó con **95 passed, 109 warnings** (el tiempo de pared varía entre ejecuciones, en torno
a 17 s). Las advertencias provienen de dependencias de terceros (futuro fin de soporte de
`google.api_core` en Python 3.10 y `DeprecationWarning` de `random.randrange` en las pruebas de
reproducibilidad), no del código del proyecto.

Las pruebas cubren carga y transformación de imágenes, estadísticas y distribución del dataset,
construcción de arquitecturas, métricas y matrices de confusión, selección jerárquica, baseline
mayoritaria, criterio de éxito, la división experimental (test intacto, estratificación, ausencia de
duplicados por hash, exclusión del `val` original), la ausencia de `RandomFlip("horizontal")`, las
diez etapas del flujo final y sus rutas de artefactos, y en particular el aislamiento del test:

- `construir_pipelines_datos()` no construye el pipeline de test sin `incluir_test=True`;
- pedir `incluir_test=True` construye el pipeline pero no lee ninguna imagen hasta que se itera, que
  es lo que hace seguro que `neumonia prepare` valide el split original sin abrir radiografías;
- la etapa de sensibilidad y la construcción del pipeline de COMBINADO rechazan el split `test`;
- `evaluar_test()` no recibe parámetros y `neumonia test` no declara ninguna opción, porque la
  evaluación del test es un paso normal del flujo;
- el pipeline del test se pide con el manifiesto original y sin ninguna bandera de oversampling,
  `class_weight` o augmentation;
- las figuras de cada etapa se escriben en `reports/figures` con el nombre de la etapa que las
  produce, sin nombres de etapas retiradas, y existen en disco con contenido;
- la sensibilidad escribe un único artefacto, sin métricas de test;
- los checkpoints de etapas distintas no se pisan entre sí, y el entrenamiento definitivo guarda
  el modelo explícitamente al no haber `validation` que dispare el `ModelCheckpoint`;
- cada `.format()` del flujo declara exactamente las claves que le pasa, porque esos mensajes solo se
  ejecutan al correr el flujo real y un error ahí no lo detecta ninguna prueba de integración;
- el umbral declara su propio sesgo optimista.

`tests/test_reproducibilidad.py` cubre además la semilla única, `shuffle` sembrado, capas de
augmentación sembradas, `deterministic=True`, igualdad de lotes entre pipelines construidos por
separado, reproducibilidad del oversampling, igualdad de pesos iniciales y el orden
`clear_session()` → `reiniciar_semilla()` → pipeline → modelo.

## 11. Reproducibilidad

La ejecución validada utilizó Python 3.10.11 en el entorno `.venv` con las dependencias de
`requirements.txt` (TensorFlow CPU 2.15.0, NumPy, Pandas, Pillow, Matplotlib, scikit-learn, pytest,
click). Es una ejecución **en CPU**: COMBINADO VGG16 ~1500 s, ResNet50 ~1000 s, MobileNetV2 ~500 s,
modelo definitivo ~510 s, evaluación del test ~20 s (solo inferencia).

Flujo de ejecución (nuevo):

```powershell
dvc pull data/raw/chest_xray.dvc
neumonia eda
neumonia prepare
neumonia augment
neumonia sensibilidad
neumonia combinado-arquitecturas
neumonia comparar
neumonia umbral
neumonia final
neumonia test
neumonia evaluar
```

`neumonia run` encadena exactamente esa secuencia y añade la suite de pruebas al final.

`neumonia sensibilidad` y `neumonia combinado-arquitecturas` no repiten los entrenamientos si sus
artefactos existen. `neumonia umbral` y `neumonia test` son baratos; `neumonia umbral` lee las
probabilidades de validation ya guardadas y `neumonia test` solo hace inferencia. Cada comando puede
invocarse también como `python -m src.cli <comando>`.

`src/utils/reproducibility.py` es la única fuente de semilla del proyecto (`SEMILLA = 42`) y
controla siete puntos:

| Elemento | Mecanismo |
|---|---|
| División experimental | `random_state=42` |
| Orden de los datos | `shuffle(..., seed=SEMILLA, reshuffle_each_iteration=True)` |
| Augmentación | `RandomRotation(0.05, seed=SEMILLA)`, `RandomZoom(0.05, seed=SEMILLA)` |
| Augmentación en `tf.data` | `map(..., deterministic=True)` |
| Operaciones de TensorFlow | `tf.config.experimental.enable_op_determinism()` |
| Pesos iniciales | `tf.keras.utils.set_random_seed(SEMILLA)` antes de construir el modelo |
| Oversampling | `sample_from_datasets(..., seed=SEMILLA)` |

**Qué está verificado.** Dos procesos independientes con `SEMILLA = 42` producen lotes de
entrenamiento idénticos byte a byte. Dos construcciones del modelo tras `reiniciar_semilla(42)` dan
los mismos pesos.

**Qué no está verificado.** No se ejecutó una segunda corrida completa de los entrenamientos, por lo
que la igualdad exacta de las métricas finales entre ejecuciones no está comprobada empíricamente.
La sección 9 debe leerse como una ejecución concreta, reproducible en su parte de datos, no como
valores deterministas garantizados de principio a fin.

## 12. Limitaciones

- El `val` original del dataset crudo contiene solo 16 imágenes y no se utiliza en ningún experimento.
- Existe desbalance entre `NORMAL` y `PNEUMONIA`.
- Hay duplicados exactos dentro de algunos splits crudos, aunque no entre ellos.
- El oversampling duplica patrones `NORMAL` existentes en `train` y no genera información nueva; no debe confundirse con nuevas muestras clínicas.
- **COMBINADO dobla la corrección del desbalance.** Los `class_weight` se calculan sobre la distribución original de `train`, así que no se anulan con el oversampling: el refuerzo efectivo de `NORMAL` es de 2.89x. Si se quisieran pesos que no inviertan el balance habría que fijarlos a 1.0, lo que ya no sería COMBINADO.
- **COMBINADO es una premisa metodológica, no el resultado de una comparación.** Es la única estrategia de desbalance del proyecto y no se mide frente a alternativas. Cualquier informe debe declararlo así.
- **El Balanced Accuracy de validation está sesgado al alza.** El umbral se maximiza sobre el mismo conjunto que se reporta, de modo que el 0.9621 no es una estimación de desempeño sino un techo de selección.
- **El gap validation → test es de ~0.062 en Balanced Accuracy** (0.9621 frente a 0.8996). El ROC-AUC apenas cae (0.9941 → 0.9694), lo que sugiere que se degrada el punto de corte y no el ordenamiento de las probabilidades.
- El umbral `0.38` se ajustó sobre las probabilidades del modelo entrenado solo con `train`; el definitivo se reentrenó con `train + val` y sus probabilidades tienen otra escala. La transferencia del umbral es una suposición razonable, no verificada empíricamente, y es la explicación más probable del gap anterior.
- El tamaño de validation (1.043 imágenes) y de test (624 imágenes, 234 NORMAL) limita la resolución de las diferencias: cambios de Balanced Accuracy inferiores a ~0.01 no son distinguibles.
- **No se ejecutó una segunda corrida completa del entrenamiento**, por lo que la reproducibilidad está verificada a nivel de pipeline de datos e inicialización del modelo, no de métricas finales.
- La normalización es general a `[0, 1]` y no específica de cada backbone.
- El trabajo es académico y experimental; no hay validación clínica ni despliegue.
- Los modelos no deben usarse como herramientas de diagnóstico médico.

## 13. Estado final

| Etapa | Estado |
|---|---|
| EDA | Completado |
| Preparación | Completada |
| Augmentación | Completada |
| Modelado | Completado |
| Sensibilidad de hiperparámetros | Completada (7 configuraciones × 3 arquitecturas = 21 entrenamientos) |
| COMBINADO VGG16 | Completada (BA validation 0.8904, ROC-AUC 0.9878) |
| COMBINADO ResNet50 | Completada (BA validation 0.7595, ROC-AUC 0.9002) |
| COMBINADO MobileNetV2 | Completada (BA validation 0.9620, ROC-AUC 0.9941) |
| Comparación arquitecturas | Completada (seleccionado MobileNetV2) |
| Umbral de decisión | Completada (congelado en 0.38, BA validation 0.9621) |
| Modelo definitivo | Completada (`train + val`, 3 épocas) |
| Test final | Completada (BA 0.8996, ROC-AUC 0.9694, matriz `[[211, 23], [40, 350]]`) |
| Testing | 95/95 pruebas aprobadas |
| Documentación | Actualizada |
| Despliegue productivo | Fuera del alcance |

El flujo nuevo se ejecuta de principio a fin: `combinado-arquitecturas` → `comparar` → `umbral` → `final` → `test`.
