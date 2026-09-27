# Documentación técnica del proyecto

## 1. Información general

**Título:** clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA` mediante modelos convolucionales con transferencia de aprendizaje.

**Objetivo:** entrenar, evaluar y comparar VGG16, ResNet50 y MobileNetV2 con una configuración comparable, seleccionar la mejor arquitectura sobre el conjunto de validación con una regla reproducible, analizar la sensibilidad de hiperparámetros de MobileNetV2 y, sobre ese ganador fijo, comparar cinco variantes que atacan el desbalance de clases y el alcance del fine-tuning. El modelo final se elige únicamente con `validation` y se evalúa una última vez sobre el `test` original.

**Alcance:** análisis exploratorio, preparación de imágenes, entrenamiento experimental, evaluación y comparación de tres arquitecturas, análisis de sensibilidad de hiperparámetros de MobileNetV2 (learning rate, dropout y épocas) y optimización de cinco variantes (oversampling, class weights y tres profundidades de fine-tuning) para obtener el modelo final. El resultado es académico y experimental.

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

El `test` original de **624 imágenes permanece completamente intacto**: no participa en entrenamiento, no participa en selección de modelos ni en ajuste de hiperparámetros, y no se mezcla con `train` ni `validation`. Se consulta **dos veces y solo para medir, nunca para decidir**: la evaluación inicial del checkpoint ganador de la sensibilidad, antes de optimizar, y la evaluación final del modelo ya elegido. Ninguna de las dos entradas interviene en la decisión, que se toma solo con `validation`.

### División experimental

Para el modelado, únicamente las **5.216 imágenes del train original** se dividen en aproximadamente 80% train y 20% validation, de forma estratificada por clase, con semilla 42 y agrupación por hash SHA-256 para evitar que imágenes con contenido idéntico queden repartidas entre `train` y `validation`. El manifiesto es `data/interim/stratified_split_train80_val20_test_original.csv`:

| Conjunto | NORMAL | PNEUMONIA | Total | Origen |
|---|---:|---:|---:|---|
| train | 1.073 | 3.100 | 4.173 | 80% aprox. del train original |
| validation | 268 | 775 | 1.043 | 20% aprox. del train original |
| test | 234 | 390 | 624 | test original íntegro |
| **Total** | **1.575** | **4.265** | **5.840** | luego excluye el val original de 16 |

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
│   ├── vgg16/best_model.keras
│   ├── resnet50/best_model.keras
│   ├── mobilenetv2/best_model.keras              # arquitectura base, 1e-4
│   ├── mobilenetv2_ajustado/best_model.keras      # histórico, tuning lr=3e-4
│   ├── mobilenetv2_tuning/<experimento>/best_model.keras   # histórico, tuning + desbalance
│   ├── mobilenetv2_optimizacion/<variante>/best_model.keras # vigente, 5 variantes
│   ├── historico/                                 # ejecuciones no reproducibles
│   │   ├── optimization_results_ALEATORIO_NO_CONTROLADO.json
│   │   └── mobilenetv2_optimizacion_ALEATORIO_NO_CONTROLADO/
│   ├── model_results.json                         # etapa 1: selección de arquitectura
│   ├── sensitivity_results.json                   # etapa 2: 7 configs x 3 arquitecturas = 21 pruebas
│   ├── sensitivity_test_audit.json                # auditoría del test del ganador de la etapa 2
│   ├── optimization_results.json                  # vigente: test inicial, validación, selección y test final
│   ├── mobilenetv2_tuning_results.json            # histórico
│   └── mobilenetv2_desbalance_results.json        # histórico
├── experiments/
│   ├── sensibilidad_hiperparametros/              # 21 checkpoints, resultados, log e informe
│   └── optimizacion_pendiente_mobilenetv2/        # respaldo histórico (fuera del flujo vigente)
├── notebooks/
│   └── 01_comprension_datos_eda.ipynb
├── references/
│   ├── documentacion_proyecto.md
│   └── guia_ejecucion.md
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
│   │   ├── run_real_training.py            # etapa 1
│   │   ├── sensitivity.py                  # etapa 2
│   │   ├── optimizacion_mobilenetv2.py     # etapas 3, 4 y 5
│   │   ├── tuning_mobilenetv2.py           # histórico: constructor de modelo y criterio
│   │   └── tuning_desbalance_clases.py     # histórico: class weights y oversampling
│   ├── utils/
│   │   ├── paths.py
│   │   └── reproducibility.py              # única fuente de semilla
│   └── visualization/visualize.py
├── tests/
│   ├── test_make_dataset.py
│   ├── test_pipeline_components.py
│   ├── test_sensitivity.py
│   ├── test_optimizacion_mobilenetv2.py
│   └── test_reproducibilidad.py
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini
```

`build_features.py`, `train_model.py` y `predict_model.py` existen pero están vacíos y no son entry points del flujo ejecutado. El entrenamiento de la etapa 1 se invoca a través de la CLI (`neumonia train`), que llama a `entrenar_y_evaluar_modelos` en `src/training/run_real_training.py`. Las etapas 2 a 5 se ejecutan con `neumonia sensibilidad` (`sensitivity.py`) y `neumonia test-inicial` / `neumonia optimizar` (`optimizacion_mobilenetv2.py`).

## 5. CLI

La forma recomendada de ejecutar el flujo es la interfaz de línea de comandos `neumonia`, que invoca las funciones existentes sin duplicar lógica. Se registra con la instalación del proyecto (`pip install -r requirements.txt`, que incluye `-e .`; si falta, `pip install -e .`).

| Comando | Función |
|---|---|
| `neumonia eda` | Análisis exploratorio de datos existente. |
| `neumonia prepare` | Genera el manifiesto estratificado (80% train / 20% validation del train original, test original intacto) y verifica los pipelines. |
| `neumonia augment` | Genera la figura de ejemplos de augmentación. |
| `neumonia train` | Entrena VGG16, ResNet50 y MobileNetV2; evalúa, compara y selecciona. |
| `neumonia sensibilidad` | Registra los resultados de los 21 entrenamientos (7 configuraciones × 3 arquitecturas). Si `models/sensitivity_results.json` existe, no la repite. Acepta `--recalcular` para reentrenar las 21 pruebas. |
| `neumonia test-inicial` | Evalúa sobre el test original el checkpoint del ganador de sensibilidad, sin reentrenar. |
| `neumonia optimizar` | Entrena las 5 variantes sobre el ganador fijo de sensibilidad, selecciona sobre validation y ejecuta el test final una sola vez. Acepta `--reusar`. |
| `neumonia evaluate` | Muestra los resultados guardados sin volver a entrenar. |
| `neumonia test` | Ejecuta la suite de pruebas con pytest. Acepta `--verbose` / `-v`. |
| `neumonia run` | Encadena EDA, preparación, augmentación, sensibilidad, optimización (con test inicial y test final) y, al terminar, la suite de pruebas. No invoca `neumonia train`: la etapa 1 queda fuera porque la sensibilidad ya evalúa las tres arquitecturas. |

Los comandos `tune` y `desbalance` se han **eliminado de la CLI**. Correspondían a una
experimentación previa sobre `lr=3e-4` y quedaron superados por la etapa de sensibilidad. Sus
artefactos se conservan en `models/mobilenetv2_tuning_results.json` y
`models/mobilenetv2_desbalance_results.json`, y el código que los generaba se conserva únicamente
en `experiments/`. El flujo vigente no vuelve a explorar learning rate, dropout ni epochs: esa
búsqueda quedó cerrada en la etapa de sensibilidad.

## 6. Preparación de datos y augmentación

El reparto experimental lo genera `crear_manifiesto_division_estratificada` (`src/data/splitting.py`). Lee el dataset crudo, divide únicamente las imágenes del `train` original en `train` (80%) y `validation` (20%) de forma estratificada por clase, con semilla 42, y asigna las imágenes del `test` original (624) directamente al conjunto `test` sin modificarlas. Agrupa por hash SHA-256 para que imágenes con contenido idéntico no queden repartidas entre conjuntos. Las 16 imágenes del `val` original no forman parte del manifiesto.

El preprocesamiento (`src/data/preprocessing.py`) carga cada imagen con Pillow y la convierte a RGB, la redimensiona a `224 x 224` con interpolación bilinear y la normaliza a `[0, 1]` (división por 255). El código no utiliza el `preprocess_input` específico de cada backbone.

`src/data/datasets.py` construye los `tf.data.Dataset` con tensores `224 x 224 x 3`, mezcla los ejemplos, agrupa en batches y usa `prefetch` automático. La data augmentation se aplica únicamente sobre `train` con `RandomRotation(0.05)` y `RandomZoom(0.05)`; `validation` y `test` no reciben augmentation.

El volteo horizontal (`RandomFlip("horizontal")`) fue eliminado: en radiografías de tórax puede existir información de lateralidad anatómica y marcadores L/R, y un volteo horizontal podría invertir artificialmente esa información. El flujo base no aplica oversampling, undersampling ni class weights; el desbalance se reporta mediante métricas balanceadas. En la fase de mejora (sección 8.2) sí se probaron **class weights** y **oversampling** sobre `train` para el modelo final, siempre sin tocar `validation` ni `test`. La figura de ejemplos se genera con `graficar_ejemplos_augmentation` (`src/visualization/visualize.py`) en `reports/figures/data_augmentation_examples.png`.

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

Esta sección describe la configuración base compartida por los tres modelos. La búsqueda de
hiperparámetros de MobileNetV2 (sección 8.1) y la optimización del desbalance (sección 8.2)
conservan el mismo esquema de capas y callbacks.

## 8. Entrenamiento, evaluación y selección

El flujo real está implementado en `src/training/run_real_training.py` y se ejecuta con `neumonia train` (o `neumonia run`). Establece la semilla 42, regenera el manifiesto (train original → 80% train / 20% validation; test original intacto de 624), construye los datasets, aplica augmentation a `train`, entrena cada modelo y evalúa su checkpoint seleccionado por `val_loss`.

La comparación y la selección se realizan **exclusivamente** con las métricas sobre `validation` (1.043 imágenes). El `test` original (624 imágenes) queda reservado: se mide dos veces y **nunca decide** — antes de optimizar, sobre el ganador de la sensibilidad, y al final, sobre el modelo ya elegido. El `test` no participa en entrenamiento, selección ni ajuste. Las predicciones binarias usan umbral 0.5 y las curvas ROC se generan con probabilidades. La regla de selección es, en orden:

1. Balanced Accuracy;
2. ROC-AUC;
3. F1;
4. Accuracy.

Esta regla reemplazó a la anterior `Recall -> Accuracy -> F1`, que podía seleccionar un modelo que detectara todos los positivos mientras clasificaba mal todos los normales.

### Criterio de éxito

El criterio de éxito es independiente del criterio de selección. El modelo seleccionado, evaluado sobre el **test original independiente**, debe:

1. **Superar un baseline sencillo**: el baseline es la *clase mayoritaria* (predecir siempre `PNEUMONIA`). El modelo debe superar su Balanced Accuracy (0.5) sobre el test.
2. **Demostrar un equilibrio adecuado entre sensibilidad y especificidad**: ambas deben superar el nivel de azar (0.5).

El baseline se calcula en `evaluar_baseline_mayoritaria` y el criterio en `evaluar_criterio_exito` (`src/models/evaluation.py`); ambos se persisten en `models/model_results.json` (`baseline_test` y `criterio_exito`).

### 8.1 Sensibilidad de hiperparámetros de MobileNetV2

Ejecutable en `src/training/sensitivity.py` (CLI: `neumonia sensibilidad`). Explora **7
configuraciones** de hiperparámetros, cada una repetida en las **3 arquitecturas**, es decir
**21 entrenamientos** (3 × 7), siempre sobre el mismo reparto experimental, con el mismo criterio
de selección y los mismos callbacks que el flujo base. Se evalúan **solo** sobre `validation`.

| Eje | Valores explorados |
|---|---|
| Learning rate | `1e-4` (referencia), `3e-4`, `1e-3` |
| Dropout | `0.3` (referencia), `0.2`, `0.5` |
| Épocas | `3` (referencia), `5`, `10` |

La configuración de referencia es `lr=1e-4`, `dropout=0.3` y 3 épocas (`ref_lr1e-4_do0.3_ep3`); las
seis derivadas son `lr_3e-4`, `lr_1e-3`, `do_0.2`, `do_0.5`, `ep_5` y `ep_10`. Esta etapa **no**
incluye fine-tuning: las tres arquitecturas se entrenan desde cero en cada configuración. El
fine-tuning se evalúa exclusivamente en la etapa 8.2.

Resultados en `models/sensitivity_results.json`; artefactos completos (checkpoints, resultados
parciales y log) en `experiments/sensibilidad_hiperparametros/`. La ejecución es **idempotente**:
si el artefacto existe, el comando lo registra sin reentrenar los 21 experimentos; `--recalcular`
fuerza el reentrenamiento.

**Configuración ganadora: `MobileNetV2 lr_1e-3`** (learning rate `1e-3`, dropout `0.3`, 3 épocas,
entrenada desde cero), con Balanced Accuracy `0.959042` en validation. Esa configuración es el
**punto de partida fijo** de la sección 8.2 y no se vuelve a explorar.

### 8.2 Test inicial y optimización del desbalance

Ejecutable en `src/training/optimizacion_mobilenetv2.py` (CLI: `neumonia test-inicial` y
`neumonia optimizar`). A diferencia de 8.1, **no** explora learning rate, dropout ni epochs:
parte del ganador de sensibilidad y evalúa únicamente el tratamiento del desbalance y el alcance
del fine-tuning. Cada variante aísla **un solo factor** respecto del punto de partida.

| Variante | Factor aislado | Dataset de entrenamiento |
|---|---|---|
| `oversampling_normal` | `NORMAL` equipara a `PNEUMONIA` en `train` | 6 200 (1 073 → 3 100 `NORMAL`) |
| `class_weight` | Pesos `{0: 1.9445, 1: 0.6731}` solo en `fit` | 4 173 |
| `finetune_block16` | Capas de `block_16_expand` entrenables (15 de 158) | 4 173 |
| `finetune_block13` | Capas de `block_13_expand` entrenables (42 de 158) | 4 173 |
| `finetune_block10` | Capas de `block_10_expand` entrenables (68 de 158) | 4 173 |

El oversampling se aplica **solo** a `train`; `validation` y `test` no se duplican. El proyecto
no implementa factores de proporción configurables: `construir_train_oversampling` equipara
siempre `NORMAL` con `PNEUMONIA`. Los pesos de clase se calculan con
`sklearn.utils.class_weight.compute_class_weight` sobre el `train`; en TensorFlow 2.15 el
argumento `class_weight` respeta correctamente los batches de `tf.data`.

**Secuencia de decisiones:**

1. **Test inicial** (`registrar_test_inicial`): evalúa el checkpoint del ganador de sensibilidad
   sobre el test original, antes de decidir nada. Deja contraste con la auditoría de sensibilidad.
2. **Entrenamiento y validación** (`ejecutar_optimizacion`): entrena las cinco variantes con el
   learning rate `1e-3` del punto de partida y evalúa cada una sobre `validation`.
3. **Selección** (`seleccionar_ganador_optimizacion`): aplica la regla jerárquica
   (Balanced Accuracy, ROC-AUC, F1, Accuracy). No hay tolerancias adicionales: el recall se
   reporta como información, pero no filtra candidatos. `test` no participa de esta decisión.
4. **Test final** (`registrar_test_final`): se ejecuta **una sola vez**, sobre el modelo ya
   elegido por `validation`.

Resultados completos en `models/optimization_results.json` (`test_inicial`, `resultados`,
`seleccion`, `test_final`) y checkpoints en `models/mobilenetv2_optimizacion/<variante>/best_model.keras`.
La función `ejecutar_optimizacion` es independiente del test: se puede invocar para repetir solo
el entrenamiento y la selección sin volver a medir `test`.

> **Reproducibilidad.** La etapa se corrigió para que el pipeline de datos sea determinista. Antes
> de la corrección, `Dataset.shuffle` se creaba sin semilla, las capas `RandomRotation` y
> `RandomZoom` se creaban sin semilla, y `tf.data` corría el `map` con paralelismo automático; la
> semilla global se fijaba además *después* de construir los pipelines. El resultado era que dos
> ejecuciones del mismo código entrenaban sobre imágenes distintas. La corrección vive en
> `src/utils/reproducibility.py` (`SEMILLA = 42`, `enable_op_determinism()`,
> `reiniciar_semilla()`, `descripcion_reproducibilidad()`) y en el orden
> `clear_session()` → `reiniciar_semilla(SEMILLA)` → construir pipeline → construir modelo de
> `entrenar_variante`, de modo que las cinco variantes ven exactamente los mismos datos.
> Verificado: dos procesos independientes dan lotes de entrenamiento idénticos byte a byte
> (`ef090aacf796b100e2e1` en ambos) y dos construcciones del modelo dan los mismos pesos. No se ha
> ejecutado una segunda corrida completa, por lo que la igualdad de las métricas finales no está
> comprobada empíricamente. Las diferencias de Balanced Accuracy menores que ~0.01 en este tamaño
> de validation no son distinguibles con una sola corrida.

## 9. Resultados reales

Los valores de la comparación de arquitecturas están verificados contra `models/model_results.json`
tras el reentrenamiento con la nueva división. Cada subsection indica el artefacto del que proceden
sus cifras, porque el resultado del proyecto no es el de esta etapa: la comparación de
arquitecturas es la etapa 1 de un flujo de cinco.

### Baseline mayoritaria sobre test

| Métrica | Valor |
|---|---:|
| Balanced Accuracy | 0.5 |
| Recall/Sensitivity | 1.0 |
| Specificity | 0.0 |

### Resultados sobre `validation` (1.043 imágenes), usados para la selección

| Modelo | Balanced Accuracy | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.9412 | 0.9434 | 0.9773 | 0.9458 | 0.9366 | 0.9613 | 0.9887 |
| VGG16 | 0.8765 | 0.9108 | 0.9338 | 0.9471 | 0.8060 | 0.9404 | 0.9643 |
| ResNet50 | 0.5000 | 0.7430 | 0.7430 | 1.0000 | 0.0000 | 0.8526 | 0.8878 |

ResNet50 predijo todos los casos de `validation` como `PNEUMONIA` (Specificity 0.0 y Balanced Accuracy 0.5000): el Recall aislado no representa un desempeño global adecuado para este problema.

> **Nota: qué muestran las figuras de `validation` de VGG16 y ResNet50.** Esta tabla y la del README
> describen el **flujo base** y coinciden con `models/model_results.json`. Las cuatro imágenes
> `confusion_matrix_validation_vgg16.png`, `confusion_matrix_validation_resnet50.png`,
> `roc_curve_validation_vgg16.png` y `roc_curve_validation_resnet50.png` se regeneraron para mostrar la
> **mejor configuración por arquitectura** de la etapa de sensibilidad (`lr=1e-3`, dropout `0.3`,
> 3 épocas), por lo que no reproducen los números de esta tabla:
>
> | Figura | TN | FP | FN | TP | Balanced Accuracy | ROC-AUC |
> |---|---:|---:|---:|---:|---:|---:|
> | VGG16 | 260 | 8 | 70 | 705 | 0.9399 | 0.9850 |
> | ResNet50 | 150 | 118 | 67 | 708 | 0.7366 | 0.8931 |
>
> Con la configuración de sensibilidad, ResNet50 deja de clasificarlo todo como `PNEUMONIA` y su
> especificidad sube de 0.0 a 0.5597, aunque su Balanced Accuracy (0.7366) sigue siendo la peor de las
> tres. La observación sobre el recall aislado se mantiene para el flujo base. Ver
> `references/guia_ejecucion.md`, sección 12.


### Test del MobileNetV2 base (etapa 1, 624 imágenes originales intactas)

Estos valores son el `test` de la **comparación de arquitecturas**, no el resultado final del
proyecto: corresponden al MobileNetV2 entrenado con la configuración base (`lr=1e-4`), el
seleccionado en la etapa 1 y luego sustituido por el ganador de la sensibilidad (sección 8.1) y por
la variante ganadora de la optimización (sección 8.2).

| Métrica | Valor |
|---:|---:|
| Accuracy | 0.8381 |
| Balanced Accuracy | 0.7885 |
| Precision | 0.8004 |
| Recall/Sensitivity | 0.9872 |
| Specificity | 0.5897 |
| F1 | 0.8840 |
| ROC-AUC | 0.9545 |

Matriz de confusión en test del modelo base: `[[138, 96], [5, 385]]` (TN = 138, FP = 96, FN = 5, TP = 385).

### Criterio de éxito sobre test

| Comprobación | Resultado |
---|---:|
| Supera baseline (Balanced Accuracy > 0.5) | Sí (0.7885) |
| Sensibilidad sobre azar (> 0.5) | Sí (0.9872) |
| Especificidad sobre azar (> 0.5) | Sí (0.5897) |
| **Cumple el criterio de éxito** | **Sí** |

**Modelo seleccionado en el flujo base: MobileNetV2** (por Balanced Accuracy → ROC-AUC → F1 → Accuracy sobre `validation`).

### Ganador de la sensibilidad de hiperparámetros (validation)

Valores verificados contra `models/sensitivity_results.json`. La etapa evaluó 7 configuraciones
repetidas en 3 arquitecturas, es decir 21 entrenamientos, y se cerró con `MobileNetV2 lr_1e-3`, que
es el punto de partida fijo de la optimización.

| Configuración | Balanced Accuracy | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`MobileNetV2 lr_1e-3` (ganadora)** | **0.959042** | 0.964525 | 0.9817 | 0.9703 | 0.9478 | 0.9760 | 0.9936 |

### Test inicial: punto de partida de la optimización

Primera medición sobre el test original del checkpoint ganador de sensibilidad, antes de
optimizar. Verificada contra `models/optimization_results.json` (`test_inicial`).

| Métrica | Valor |
|---:|---:|
| Accuracy | 0.8413 |
| Balanced Accuracy | 0.7927 |
| Precision | 0.8038 |
| Recall/Sensitivity | 0.9872 |
| Specificity | 0.5983 |
| F1 | 0.8861 |
| ROC-AUC | 0.9612 |

Matriz de confusión: `[[140, 94], [5, 385]]` (TN = 140, FP = 94, FN = 5, TP = 385).

### Optimización del desbalance: resultados sobre `validation`

Valores verificados contra `models/optimization_results.json` (`resultados`), ejecución final
reproducible. Criterio: Balanced Accuracy > ROC-AUC > F1 > Accuracy.

| Variante | Balanced Accuracy | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oversampling_normal` (seleccionada) | 0.960698 | 0.9597 | 0.9867 | 0.9587 | 0.9627 | 0.9725 | 0.9949 |
| `class_weight` | 0.960193 | 0.9626 | 0.9842 | 0.9652 | 0.9552 | 0.9746 | 0.9931 |
| Punto de partida (sensibilidad) | 0.959042 | 0.9645 | 0.9817 | 0.9703 | 0.9478 | 0.9760 | 0.9936 |
| `finetune_block16` | 0.734429 | 0.8629 | 0.8450 | 0.9987 | 0.4701 | 0.9154 | 0.9714 |
| `finetune_block10` | 0.692164 | 0.8418 | 0.8245 | 1.0000 | 0.3843 | 0.9038 | 0.9746 |
| `finetune_block13` | 0.611940 | 0.8006 | 0.7884 | 1.0000 | 0.2239 | 0.8817 | 0.9965 |

La regla de selección eligió `oversampling_normal` por tener la mayor Balanced Accuracy (0.9607).
Conviene precisar qué significa eso: la regla es **jerárquica**, de modo que el desempate se resuelve
por la primera métrica y solo en caso de igualdad se pasarían a ROC-AUC, F1 y Accuracy. No se exige
ganar en las cuatro. De hecho, `oversampling_normal` **pierde** frente al punto de partida en F1
(0.9725 frente a 0.9760) y en Accuracy (0.9597 frente a 0.9645), y frente a `class_weight` pierde en
Accuracy, Recall y F1, aunque gane en Balanced Accuracy, Precision, Specificity y ROC-AUC.
`class_weight` quedó a solo 0.0005 de Balanced Accuracy, diferencia no distinguible con una sola
ejecución.

Las tres variantes de fine-tuning muestran el patrón de colapso: recall 1.0 con especificidad
entre 0.22 y 0.47.

### Test final: modelo elegido por `validation`

Ejecutado **una sola vez** sobre el modelo seleccionado.

| Métrica | Valor |
|---:|---:|
| Modelo | `MobileNetV2-oversampling_normal` |
| Accuracy | 0.8734 |
| Balanced Accuracy | 0.8372 |
| Precision | 0.8418 |
| Recall/Sensitivity | 0.9821 |
| Specificity | 0.6923 |
| F1 | 0.9065 |
| ROC-AUC | 0.9609 |

Matriz de confusión: `[[162, 72], [7, 383]]` (TN = 162, FP = 72, FN = 7, TP = 383).

> **El modelo seleccionado por `validation` mejoró el `test`.** El test inicial del punto de
> partida obtuvo Balanced Accuracy 0.7927 y el test final del modelo elegido 0.8372 (+0.0444). La
> ganancia se concentra en especificidad (0.5983 → 0.6923) y F1 (0.8861 → 0.9065); el recall se
> mantiene alto (0.9821) y el ROC-AUC es prácticamente idéntico (0.9612 → 0.9609).

### Ejecuciones anteriores de la etapa y su impacto en la selección

Antes de la corrección de reproducibilidad, la etapa se ejecutó varias veces con idéntico código,
datos y semilla declarada (42). Ninguna de ellas es reproducible, porque el `shuffle` y la
augmentation se creaban sin semilla. Balanced Accuracy en `validation`:

| Variante | Ejecución A | Ejecución B | **Ejecución final reproducible** |
|---|---:|---:|---:|
| `finetune_block16` | 0.6791 | 0.9674 | 0.7344 |
| `finetune_block10` | 0.7519 | 0.9516 | 0.6922 |
| `finetune_block13` | 0.8221 | 0.8209 | 0.6119 |
| `class_weight` | 0.9646 | 0.9566 | 0.9602 |
| `oversampling_normal` | 0.9555 | 0.9586 | **0.9607** |
| Punto de partida (cargado, no reentrenado) | 0.9590 | 0.9590 | 0.9590 |

Origen de cada ejecución: **A** = `experiments/optimizacion_pendiente_mobilenetv2/resultados/optimizacion_pendiente.json`;
**B** = `models/historico/optimization_results_ALEATORIO_NO_CONTROLADO.json`. Se conserva además el
respaldo `experiments/optimizacion_pendiente_mobilenetv2/NOTA_RESALDO.md`, que documenta por qué no
son reproducibles. Hubo también una tercera ejecución repetida de diagnóstico cuyos artefactos ya
no están en el repositorio; sus cifras se han retirado de esta tabla por no ser verificables. Las dos
ejecuciones que sí tienen artefacto no deben mezclarse con el resultado vigente.

El rango de `finetune_block16` entre las ejecuciones no controladas (0.6791 – 0.9674) es la medida
del ruido que introducía la ausencia de semilla. Las variantes con **base congelada** se mantenían
dentro de ~0.01; las variantes con **fine-tuning** llegaban a variar 0.29, y una de las ejecuciones
colapsó por completo.

En la ejecución reproducible, el fine-tuning **vuelve a colapsar en las tres profundidades**, con
especificidad entre 0.22 y 0.47. Esto refuerza la hipótesis de la actualización de las estadísticas
de **Batch Normalization** de los bloques descongelados: con batch size 16, learning rate `1e-3` y
solo 3 épocas, las medias y varianzas móviles de esos bloques no llegan a estabilizarse. El
comportamiento es consistente, no azar del pipeline.

**Consecuencia metodológica.** La victoria de `oversampling_normal` sobre el punto de partida es de
`+0.0017` de Balanced Accuracy en `validation`, una diferencia **muy pequeña** y no distinguible con
una sola ejecución. Conviene ser precisos sobre qué sostiene la decisión: la variante elegida tiene
la mayor Balanced Accuracy, que es la primera métrica de una regla **jerárquica**, pero **no supera
al punto de partida en las cuatro métricas** (gana BA 0.9607 > 0.9590 y ROC-AUC 0.9949 > 0.9936;
pierde F1 0.9725 < 0.9760 y Accuracy 0.9597 < 0.9645). El `test` no participa en este razonamiento:
es una medición posterior a una decisión ya tomada y, por diseño, no puede sostenerla. Lo que sí
puede afirmarse es un hecho descriptivo y no una jerarquía: ninguna de las cinco variantes colapsa
en `validation`, y el fine-tuning queda descartado de forma consistente en las tres profundidades.
Con una sola ejecución por variante no es posible ordenar el resto de candidatas por robustez.

### Resultados históricos (fuera del flujo vigente)

### Resultados del tuning de MobileNetV2 (validation)

Valores verificados contra `models/mobilenetv2_tuning_results.json`, salvo la fila de referencia
`Original (lr 1e-4)`, que procede de la comparación de arquitecturas (`models/model_results.json`)
porque ese artefacto de tuning no la incluye: sirve de línea base y coincide exactamente con el
MobileNetV2 del flujo base.

| Configuración | Balanced Accuracy | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original (lr `1e-4`) | 0.9412 | 0.9434 | 0.9773 | 0.9458 | 0.9366 | 0.9613 | 0.9887 |
| lr `5e-5` | 0.9415 | 0.9530 | 0.9714 | 0.9652 | 0.9179 | 0.9683 | 0.9862 |
| **lr `3e-4` (seleccionada)** | **0.9547** | **0.9616** | **0.9791** | 0.9690 | 0.9403 | **0.9741** | **0.9928** |
| dropout `0.5` | 0.9423 | 0.9559 | 0.9703 | 0.9703 | 0.9142 | 0.9703 | 0.9884 |
| fine-tuning `finetune_block13` | 0.6194 | 0.8044 | 0.7916 | 1.0000 | 0.2388 | 0.8837 | 0.9978 |

El ganador `lr_3e-4` se reentrenó con la misma configuración (final_validation: Balanced Accuracy 0.9555, ROC-AUC 0.9933, F1 0.9774, Accuracy 0.9664, Recall 0.9781, Specificity 0.9328) y se evaluó sobre `test`:

| Métrica | Valor |
---:|---:|
| Accuracy | 0.8478 |
| Balanced Accuracy | 0.7987 |
| Precision | 0.8067 |
| Recall/Sensitivity | 0.9949 |
| Specificity | 0.6026 |
| F1 | 0.8909 |
| ROC-AUC | 0.9633 |

### Resultados de los experimentos de desbalance (validation)

Valores verificados contra `models/mobilenetv2_desbalance_results.json`.

| Configuración | Balanced Accuracy | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ajustado `lr=3e-4` (referencia) | 0.9555 | 0.9664 | 0.9768 | 0.9781 | 0.9328 | 0.9774 | 0.9933 |
| Class weights (solo entrenamiento) | 0.9530 | 0.9482 | 0.9865 | 0.9432 | 0.9627 | 0.9644 | 0.9925 |
| **Oversampling `NORMAL` en train** | **0.9598** | **0.9674** | **0.9805** | **0.9755** | **0.9440** | **0.9780** | **0.9936** |

El oversampling lideró la Balanced Accuracy de validation manteniendo un recall alto (0.9755) y fue la variante elegida. Los class weights mejoraron la especificidad (0.9627) pero redujeron el recall (0.9432) y la Balanced Accuracy, por lo que se descartaron. El ganador se reentrenó (final_validation: Balanced Accuracy 0.9573, ROC-AUC 0.9940) y dio este resultado sobre `test`:

| Métrica | Valor |
---:|---:|
| Accuracy | **0.8878** |
| Balanced Accuracy | **0.8590** |
| Precision | 0.8636 |
| Recall/Sensitivity | 0.9744 |
| Specificity | **0.7436** |
| F1 | **0.9157** |
| ROC-AUC | 0.9612 |

Matriz de confusión en test de aquel modelo: `[[174, 60], [10, 380]]` (TN = 174, FP = 60, FN = 10, TP = 380).

> Nota: la mejora del oversampling es un hallazgo experimental sobre este dataset y no demuestra que el desbalance fuera la única causa de las diferencias observadas entre las configuraciones.

> **Este resultado ya no es el del proyecto.** Aquel modelo fue
> `MobileNetV2 + oversampling` sobre el ajuste `lr=3e-4` que la etapa de sensibilidad sustituyó por
> `lr=1e-3`. El artefacto `models/mobilenetv2_desbalance_results.json` conserva la métrica, pero la
> ruta de checkpoint que registra (`models/mobilenetv2_desbalance_oversampling/best_model.keras`)
> **ya no existe** en el repositorio: ese directorio se eliminó al consolidar los experimentos
> históricos bajo `models/mobilenetv2_tuning/`. Se conserva únicamente como registro histórico.
> El resultado vigente está en la sección 9: `models/optimization_results.json`.

## 10. Testing

La suite se ejecuta con `pytest` (también disponible con `neumonia test`). Contiene **73 pruebas** y la ejecución final terminó con **73 passed**.

Las pruebas cubren carga y transformación de imágenes, estadísticas y distribución del dataset, construcción de arquitecturas, validación de entradas, matrices de confusión, métricas, selección de modelos, baseline mayoritaria, criterio de éxito, la nueva división (test original intacto, estratificación, ausencia de duplicados por hash, no mezcla del test con train/validation, exclusión del `val` original), la ausencia de `RandomFlip("horizontal")` en la augmentación, la idempotencia de la etapa de sensibilidad, y el aislamiento del `test` en la optimización: el criterio de las cuatro métricas, el desempate jerárquico por ROC-AUC, F1 y Accuracy, el `test` fuera de la selección, que las variantes no repiten los 21 factores ya evaluados y que cada una aísla un solo factor. Las cinco variantes se entrenan **secuencialmente** en un bucle, y no hay ninguna prueba que exija paralelismo.

`tests/test_reproducibilidad.py` cubre específicamente la corrección: semilla única en todo el
proyecto, `shuffle` sembrado, capas de augmentación sembradas, `deterministic=True`, igualdad de
lotes entre pipelines construidos por separado, reproducibilidad del oversampling, igualdad de
pesos iniciales, orden `clear_session()` → `reiniciar_semilla()` → pipeline → modelo, y ausencia de
tolerancias en el criterio de selección.

## 11. Reproducibilidad

La ejecución validada utilizó Python 3.10.11 en el entorno `.venv` con las dependencias de `requirements.txt` (TensorFlow CPU 2.15.0, NumPy, Pandas, Pillow, Matplotlib, scikit-learn, pytest, click).

Flujo de ejecución:

```powershell
dvc pull data/raw/chest_xray.dvc
neumonia eda
neumonia prepare
neumonia augment
neumonia train
neumonia sensibilidad
neumonia test-inicial
neumonia optimizar
neumonia evaluate
neumonia test
```

`neumonia train` y `neumonia sensibilidad` consumen bastante tiempo y recursos. `neumonia sensibilidad`
no repite los 21 entrenamientos si `models/sensitivity_results.json` ya existe. `neumonia evaluate`
consulta los resultados guardados sin volver a entrenar. Como alternativa, cada comando puede
invocarse con `python -m src.cli <comando>`.

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
entrenamiento idénticos byte a byte (`ef090aacf796b100e2e1` en ambos). Antes de la corrección los
dos procesos producían huellas distintas (`38518e4aa6eb0ba5` y `18bf66fc790a21ba`) con las mismas
etiquetas, lo que aisló la causa en el pipeline y no en los datos. Dos construcciones del modelo
tras `reiniciar_semilla(42)` dan los mismos pesos.

**Qué no está verificado.** No se ejecutó una segunda corrida completa del entrenamiento, por lo
que la igualdad exacta de las métricas finales entre ejecuciones no está comprobada empíricamente.
El rango de Balanced Accuracy en `validation` entre las tres ejecuciones no controladas iba de
0.006 (base congelada) a 0.288 (fine-tuning); parte de esa dispersión era ruido del pipeline sin
semilla, pero la etapa de fine-tuning sigue mostrando un comportamiento inestable atribuible a las
estadísticas de Batch Normalization. Los artefactos de cada etapa quedan guardados para poder
auditar exactamente qué se evaluó.

## 12. Limitaciones

- El `val` original del dataset crudo contiene solo 16 imágenes y no se utiliza en ningún experimento.
- Existe desbalance entre `NORMAL` y `PNEUMONIA`.
- Hay duplicados exactos dentro de algunos splits crudos, aunque no entre ellos.
- El oversampling duplica patrones `NORMAL` existentes en `train` y no genera información nueva; no debe confundirse con nuevas muestras clínicas.
- **El fine-tuning de las últimas capas de MobileNetV2 empeora de forma consistente el equilibrio entre clases con la configuración actual** (learning rate `1e-3`, batch 16, 3 épocas). En la ejecución reproducible las tres profundidades colapsan en `validation` (Balanced Accuracy 0.7344, 0.6119 y 0.6922; especificidad entre 0.22 y 0.47). La causa probable es la actualización de las estadísticas de Batch Normalization de los bloques descongelados, que no llegan a estabilizarse. No debe considerarse una técnica fiable en este dataset.
- **La ventaja de la variante ganadora en `validation` es pequeña** (`+0.0017` de Balanced Accuracy sobre el punto de partida) y no es distinguible con una sola ejecución. La decisión se sostiene únicamente en que es la de mayor Balanced Accuracy, la primera métrica de un criterio jerárquico, y en que ninguna otra variante colapsa. Con una sola ejecución por variante no puede afirmarse que sea la más robusta. El `test` se midió después de decidir y sirve para reportar el desempeño final, no para justificar la elección.
- El tamaño de validation (1.043 imágenes) limita la resolución de las diferencias: cambios de Balanced Accuracy inferiores a ~0.01 no son distinguibles con una sola ejecución por variante.
- **No se ejecutó una segunda corrida completa del entrenamiento**, por lo que la reproducibilidad está verificada a nivel de pipeline de datos e inicialización del modelo, no de métricas finales.
- El resultado del tratamiento del desbalance es experimental y no demuestra que el desbalance fuera la única causa del comportamiento observado.
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
| Entrenamiento | Completado (reentrenado con la nueva división) |
| Selección de arquitectura | Completado (MobileNetV2) |
| Sensibilidad de hiperparámetros | Completada (7 configuraciones × 3 arquitecturas = 21 entrenamientos, ganador `lr=1e-3`) |
| Test inicial | Completado (Balanced Accuracy 0.7927) |
| Optimización del desbalance | Completada y reproducible (gana `oversampling_normal`, BA val 0.9607) |
| Test final | Completado (Balanced Accuracy 0.8372) |
| Evaluación | Completada |
| Comparación | Completada |
| Selección final | **Completada**: `MobileNetV2-oversampling_normal` |
| Testing | 73/73 pruebas aprobadas |
| Documentación | Actualizada |
| Despliegue productivo | Fuera del alcance |