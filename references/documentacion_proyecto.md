# Documentación técnica del proyecto

## 1. Información general

**Título:** clasificación de imágenes de rayos X de tórax para identificación de neumonía.

**Problema:** clasificar radiografías de tórax en las clases `NORMAL` y `PNEUMONIA` mediante modelos convolucionales con transferencia de aprendizaje.

**Objetivo:** entrenar, evaluar y comparar VGG16, ResNet50 y MobileNetV2 con una configuración comparable, seleccionar el mejor modelo sobre el conjunto de validación con una regla reproducible y, posteriormente, mejorar el modelo seleccionado mediante tuning de hiperparámetros y tratamiento del desbalance de clases.

**Alcance:** análisis exploratorio, preparación de imágenes, entrenamiento experimental, evaluación y comparación de tres arquitecturas, ajuste de hiperparámetros de MobileNetV2 y experimentos de desbalance (class weights y oversampling) para obtener el modelo final. El resultado es académico y experimental.

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

El `test` original de **624 imágenes permanece completamente intacto**: no participa en entrenamiento, no participa en selección de modelos ni en ajuste de hiperparámetros, y no se mezcla con `train` ni `validation`. Solo se utiliza para la evaluación final del modelo ganador.

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
│   │   ├── run_real_training.py
│   │   ├── tuning_mobilenetv2.py
│   │   └── tuning_desbalance_clases.py
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

`build_features.py`, `train_model.py` y `predict_model.py` existen pero están vacíos y no son entry points del flujo ejecutado. El entrenamiento real se invoca a través de la CLI (`neumonia train`), que llama a `entrenar_y_evaluar_modelos` en `src/training/run_real_training.py`. El tuning de MobileNetV2 y los experimentos de desbalance se reproducen con `neumonia tune` y `neumonia desbalance` (respectivamente `tuning_mobilenetv2.py` y `tuning_desbalance_clases.py`).

## 5. CLI

La forma recomendada de ejecutar el flujo es la interfaz de línea de comandos `neumonia`, que invoca las funciones existentes sin duplicar lógica. Se registra con la instalación del proyecto (`pip install -r requirements.txt`, que incluye `-e .`; si falta, `pip install -e .`).

| Comando | Función |
|---|---|
| `neumonia eda` | Análisis exploratorio de datos existente. |
| `neumonia prepare` | Genera el manifiesto estratificado (80% train / 20% validation del train original, test original intacto) y verifica los pipelines. |
| `neumonia augment` | Genera la figura de ejemplos de augmentación. |
| `neumonia train` | Entrena VGG16, ResNet50 y MobileNetV2; evalúa, compara y selecciona. |
| `neumonia tune` | Reproduce el tuning de MobileNetV2 (learning rate, dropout y fine-tuning) y selecciona la mejor configuración sobre validation. |
| `neumonia desbalance` | Reproduce los experimentos de desbalance (class weights y oversampling) sobre el ajustado `lr=3e-4` y selecciona la variante final sobre validation. |
| `neumonia evaluate` | Muestra los resultados guardados sin volver a entrenar. |
| `neumonia test` | Ejecuta la suite de pruebas con pytest. |
| `neumonia run` | Ejecuta la secuencia del flujo base: EDA, preparación, augmentación, entrenamiento, evaluación y tests. **No** incluye `tune` ni `desbalance`, que son etapas de mejora opcionales. |

`neumonia tune` y `neumonia desbalance` no forman parte del flujo base (`run`) porque son etapas de mejora posteriores a la selección inicial del modelo; se ejecutan de forma independiente y solo usan `train`/`validation` para decidir.

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

Esta sección describe la configuración base compartida por los tres modelos. El **modelo final** (MobileNetV2 + oversampling) conserva el mismo esquema de capas y callbacks, pero usa **learning rate `3e-4`** (sección 8.2).

## 8. Entrenamiento, evaluación y selección

El flujo real está implementado en `src/training/run_real_training.py` y se ejecuta con `neumonia train` (o `neumonia run`). Establece la semilla 42, regenera el manifiesto (train original → 80% train / 20% validation; test original intacto de 624), construye los datasets, aplica augmentation a `train`, entrena cada modelo y evalúa su checkpoint seleccionado por `val_loss`.

La comparación y la selección se realizan **exclusivamente** con las métricas sobre `validation` (1.043 imágenes); el `test` original (624 imágenes) queda reservado y solo se usa para la evaluación final del modelo ganador. El `test` no participa en entrenamiento, selección ni ajuste. Las predicciones binarias usan umbral 0.5 y las curvas ROC se generan con probabilidades. La regla de selección es, en orden:

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

### 8.1 Tuning de MobileNetV2

Ejecutable en `src/training/tuning_mobilenetv2.py` (CLI: `neumonia tune`). Reentrena MobileNetV2 desde los pesos ImageNet con varias configuraciones, siempre con el mismo reparto experimental, criterio de selección y callbacks que el flujo base:

| Configuración | Base | Learning rate | Dropout |
|---|---|---|---|
| `lr_5e-5` | congelada | `5e-5` | 0.3 |
| `lr_3e-4` | congelada | `3e-4` | 0.3 |
| `dropout_0.5` | congelada | `1e-4` | 0.5 |
| `finetune_block13` | `block_13`+ entrenable | `1e-4` | 0.3 |

Los experimentos se imprimen ordenados por la regla jerárquica; el ganador se reentrena y su checkpoint se guarda en `models/mobilenetv2_ajustado/best_model.keras`. Resultados en `models/mobilenetv2_tuning_results.json` con `baseline`, `tabla_ordenada`, `seleccion`, `final_validation` y `final_test`. El `test` solo se usa para el `final_test` del ganador.

La configuración elegida fue `lr_3e-4`. El fine-tuning (`finetune_block13`) se descartó: en validation alcanzó recall 1.0 pero especificidad 0.2388 (Balanced Accuracy 0.6194), probablemente por la base congelada y el batch pequeño que desestabiliza la normalización por lotes.

### 8.2 Tratamiento del desbalance

Ejecutable en `src/training/tuning_desbalance_clases.py` (CLI: `neumonia desbalance`). Sobre el ajustado `lr=3e-4` se evalúan por separado dos estrategias:

- **Class weights:** pesos `balanced` calculados con `sklearn.compute_class_weight` sobre el `train` (`{0: 1.9445, 1: 0.6731}`), aplicados únicamente en el entrenamiento; en TensorFlow 2.15 el argumento `class_weight` respeta correctamente los batches de `tf.data`.
- **Oversampling:** duplicación aleatoria de imágenes `NORMAL` en `train` (1 073 → 3 100; total train 6 200, +2 027 duplicadas). `validation` y `test` no se modifican.

Se elige la variante con mayor Balanced Accuracy sobre `validation`; el ganador se reentrena y su checkpoint final se guarda en `models/mobilenetv2_desbalance_oversampling/best_model.keras`. Resultados en `models/mobilenetv2_desbalance_results.json` con `pesos_clase_balanceados`, `oversampling_train`, `tabla_seleccion`, `motivo_seleccion`, `final_validation` y `final_test`.

La variante elegida fue el **oversampling**. El `model_name` del modelo final es `MobileNetV2-oversampling`.

## 9. Resultados reales

Valores verificados contra `models/model_results.json` tras el reentrenamiento con la nueva división.

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

### Resultados finales sobre `test` (624 imágenes originales intactas), solo el modelo ganador

| Métrica | Valor |
|---:|---:|
| Accuracy | 0.8381 |
| Balanced Accuracy | 0.7885 |
| Precision | 0.8004 |
| Recall/Sensitivity | 0.9872 |
| Specificity | 0.5897 |
| F1 | 0.8840 |
| ROC-AUC | 0.9545 |

Matriz de confusión en test del modelo ganador: `[[138, 96], [5, 385]]` (TN = 138, FP = 96, FN = 5, TP = 385).

### Criterio de éxito sobre test

| Comprobación | Resultado |
---|---:|
| Supera baseline (Balanced Accuracy > 0.5) | Sí (0.7885) |
| Sensibilidad sobre azar (> 0.5) | Sí (0.9872) |
| Especificidad sobre azar (> 0.5) | Sí (0.5897) |
| **Cumple el criterio de éxito** | **Sí** |

**Modelo seleccionado en el flujo base: MobileNetV2** (por Balanced Accuracy → ROC-AUC → F1 → Accuracy sobre `validation`).

### Resultados del tuning de MobileNetV2 (validation)

Valores verificados contra `models/mobilenetv2_tuning_results.json`.

| Configuración | Balanced Accuracy | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original (lr `1e-4`) | 0.9412 | 0.9434 | 0.9773 | 0.9458 | 0.9366 | 0.9613 | 0.9887 |
| lr `5e-5` | 0.9415 | 0.9530 | 0.9714 | 0.9652 | 0.9179 | 0.9683 | 0.9862 |
| **lr `3e-4` (seleccionada)** | **0.9547** | **0.9616** | **0.9791** | 0.9690 | 0.9403 | **0.9741** | **0.9928** |
| dropout `0.5` | 0.9423 | 0.9559 | 0.9703 | 0.9703 | 0.9142 | 0.9703 | 0.9884 |
| fine-tuning `block_13+` | 0.6194 | 0.8044 | 0.7916 | 1.0000 | 0.2388 | 0.8837 | 0.9978 |

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

El oversampling lideró la Balanced Accuracy de validation manteniendo un recall alto (0.9755) y fue la variante elegida. Los class weights mejoraron la especificidad (0.9627) pero redujeron el recall (0.9432) y la Balanced Accuracy, por lo que se descartaron. El ganador se reentrenó (final_validation: Balanced Accuracy 0.9573, ROC-AUC 0.9940) y dio el **resultado final del proyecto** sobre `test`:

| Métrica | Valor |
---:|---:|
| Accuracy | **0.8878** |
| Balanced Accuracy | **0.8590** |
| Precision | 0.8636 |
| Recall/Sensitivity | 0.9744 |
| Specificity | **0.7436** |
| F1 | **0.9157** |
| ROC-AUC | 0.9612 |

Matriz de confusión en test del modelo final: `[[174, 60], [10, 380]]` (TN = 174, FP = 60, FN = 10, TP = 380).

**Modelo final del proyecto: MobileNetV2 + oversampling** (`models/mobilenetv2_desbalance_oversampling/best_model.keras`).

> Nota: la mejora del oversampling es un hallazgo experimental sobre este dataset y no demuestra que el desbalance fuera la única causa de las diferencias observadas entre las configuraciones.

## 10. Testing

La suite se ejecuta con `pytest` (también disponible con `neumonia test`). Contiene **34 pruebas** y la ejecución final terminó con **34 passed**.

Las pruebas cubren carga y transformación de imágenes, estadísticas y distribución del dataset, construcción de arquitecturas, validación de entradas, matrices de confusión, métricas, selección de modelos, baseline mayoritaria, criterio de éxito, la nueva división (test original intacto, estratificación, ausencia de duplicados por hash, no mezcla del test con train/validation, exclusión del `val` original) y la ausencia de `RandomFlip("horizontal")` en la augmentación.

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

- El `val` original del dataset crudo contiene solo 16 imágenes y no se utiliza en ningún experimento.
- Existe desbalance entre `NORMAL` y `PNEUMONIA`.
- Hay duplicados exactos dentro de algunos splits crudos, aunque no entre ellos.
- El oversampling del modelo final duplica patrones `NORMAL` existentes en `train` y no genera información nueva; no debe confundirse con nuevas muestras clínicas.
- El fine-tuning de las últimas capas de MobileNetV2 se descartó por degradar la especificidad; no forma parte del modelo final.
- El resultado del oversampling es experimental y no demuestra que el desbalance fuera la única causa del comportamiento observado.
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
| Tuning de MobileNetV2 | Completado (`lr=3e-4`) |
| Tratamiento del desbalance | Completado (oversampling) |
| Evaluación | Completada |
| Comparación | Completada |
| Selección final | MobileNetV2 + oversampling |
| Testing | 34/34 pruebas aprobadas |
| Documentación | Actualizada |
| Despliegue productivo | Fuera del alcance |