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
| `neumonia prepare` | Genera el manifiesto estratificado (80% train / 20% validation del train original, test original intacto) y verifica los pipelines. |
| `neumonia augment` | Genera la figura de ejemplos de augmentación. |
| `neumonia train` | Entrena VGG16, ResNet50 y MobileNetV2; evalúa, compara y selecciona. |
| `neumonia evaluate` | Muestra los resultados guardados sin volver a entrenar. |
| `neumonia test` | Ejecuta la suite de pruebas con pytest. |
| `neumonia run` | Ejecuta el flujo completo: EDA, preparación, augmentación, entrenamiento, evaluación y tests. |

## 6. Preparación de datos y augmentación

El reparto experimental lo genera `crear_manifiesto_division_estratificada` (`src/data/splitting.py`). Lee el dataset crudo, divide únicamente las imágenes del `train` original en `train` (80%) y `validation` (20%) de forma estratificada por clase, con semilla 42, y asigna las imágenes del `test` original (624) directamente al conjunto `test` sin modificarlas. Agrupa por hash SHA-256 para que imágenes con contenido idéntico no queden repartidas entre conjuntos. Las 16 imágenes del `val` original no forman parte del manifiesto.

El preprocesamiento (`src/data/preprocessing.py`) carga cada imagen con Pillow y la convierte a RGB, la redimensiona a `224 x 224` con interpolación bilinear y la normaliza a `[0, 1]` (división por 255). El código no utiliza el `preprocess_input` específico de cada backbone.

`src/data/datasets.py` construye los `tf.data.Dataset` con tensores `224 x 224 x 3`, mezcla los ejemplos, agrupa en batches y usa `prefetch` automático. La data augmentation se aplica únicamente sobre `train` con `RandomRotation(0.05)` y `RandomZoom(0.05)`; `validation` y `test` no reciben augmentation.

El volteo horizontal (`RandomFlip("horizontal")`) fue eliminado: en radiografías de tórax puede existir información de lateralidad anatómica y marcadores L/R, y un volteo horizontal podría invertir artificialmente esa información. No se implementó oversampling, undersampling ni class weights porque el docente no lo solicitó; el desbalance se reporta mediante métricas balanceadas. La figura de ejemplos se genera con `graficar_ejemplos_augmentation` (`src/visualization/visualize.py`) en `reports/figures/data_augmentation_examples.png`.

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

**Modelo seleccionado: MobileNetV2** (por Balanced Accuracy → ROC-AUC → F1 → Accuracy sobre `validation`).

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
| Evaluación | Completada |
| Comparación | Completada |
| Selección | MobileNetV2 |
| Testing | 34/34 pruebas aprobadas |
| Documentación | Actualizada |
| Despliegue productivo | Fuera del alcance |