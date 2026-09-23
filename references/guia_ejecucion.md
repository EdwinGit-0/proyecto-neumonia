# Guía de ejecución

Esta guía describe cómo reproducir el flujo actual del proyecto en Windows PowerShell. Los comandos se basan en los módulos y rutas existentes.

## 1. Requisitos previos

- Windows PowerShell.
- Python 3.10 o compatible con TensorFlow CPU 2.15.0. La ejecución validada utilizó Python 3.10.11.
- Git para obtener el repositorio.
- Acceso al remoto Google Drive configurado en DVC.
- Espacio suficiente para el dataset y los checkpoints.

El entrenamiento de los tres modelos es costoso en CPU y requiere varios gigabytes para imágenes, dependencias y artefactos.

## 2. Abrir el proyecto

Después de clonar el repositorio, entrar en su carpeta:

```powershell
git clone <URL_DEL_REPOSITORIO>
Set-Location proyecto-neumonia
```

La URL concreta no está almacenada en este repositorio y debe sustituirse por la URL real del repositorio disponible para el usuario.

## 3. Crear y activar el entorno

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

También se puede ejecutar directamente el intérprete del entorno sin activarlo:

```powershell
.\.venv\Scripts\python.exe --version
```

## 4. Instalar dependencias

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

El archivo de requisitos incluye las dependencias de análisis, TensorFlow CPU y DVC con soporte para Google Drive.

## 5. Configurar DVC

La configuración del proyecto ya contiene un remoto llamado `google_drive`. Se puede comprobar con:

```powershell
dvc remote list
dvc status
```

No se deben ejecutar `dvc init` ni crear otro remoto. Si el acceso al remoto requiere autenticación, DVC solicitará la configuración correspondiente; no se deben guardar credenciales en el repositorio.

## 6. Recuperar el dataset

```powershell
dvc pull data/raw/chest_xray.dvc
```

El comando recupera `data/raw/chest_xray/` a partir del archivo `data/raw/chest_xray.dvc` y del remoto DVC configurado.

## 7. Verificar el dataset

```powershell
$images = Get-ChildItem data/raw/chest_xray -File -Recurse | Where-Object { $_.Extension -match '\.(jpe?g|png)$' }
$images.Count
Get-ChildItem data/raw/chest_xray -Directory -Recurse
```

La copia validada contiene 5.856 imágenes JPEG organizadas en `train`, `val` y `test`, con las clases `NORMAL` y `PNEUMONIA`. Esta es la estructura original del dataset crudo.

Para el modelado se utiliza un reparto experimental estratificado que divide el `train` original (5.216 imágenes) en aproximadamente 80% `train` y 20% `validation` (semilla 42, sin duplicados por contenido entre conjuntos). El `test` original de 624 imágenes permanece intacto y las 16 imágenes del `val` original no se utilizan:

- Manifiesto: `data/interim/stratified_split_train80_val20_test_original.csv`;
- train: 4.173 imágenes (1.073 NORMAL, 3.100 PNEUMONIA);
- validation: 1.043 imágenes (268 NORMAL, 775 PNEUMONIA);
- test: 624 imágenes (234 NORMAL, 390 PNEUMONIA), íntegro del dataset crudo.

## 8. Flujo rápido con la CLI `neumonia`

El proyecto incluye una interfaz de línea de comandos (CLI) que permite ejecutar cada etapa con un comando corto. La CLI solo invoca las funciones existentes del proyecto; no cambia la lógica científica ni los resultados.

Los comandos disponibles son:

```powershell
neumonia eda        # Análisis exploratorio de datos (sección 9)
neumonia prepare    # Preparación de datos: manifiesto 80/20 (train original) y verificación de pipelines (sección 10)
neumonia augment    # Figura de ejemplos de augmentación (sección 11.1)
neumonia train      # Entrenar VGG16, ResNet50 y MobileNetV2; evaluar, comparar y seleccionar (sección 11)
neumonia tune       # Tuning experimental de MobileNetV2: lr, dropout y fine-tuning (sección 11.2)
neumonia desbalance # Experimentos de desbalance: class weight y oversampling sobre el ajustado lr=3e-4 (sección 11.3)
neumonia evaluate   # Mostrar los resultados guardados sin volver a entrenar (sección 14)
neumonia test       # Ejecutar la suite de pruebas con pytest (sección 15)
neumonia run        # Flujo completo: EDA -> preparación -> augmentación -> entrenamiento -> evaluación -> test
```

Cada comando de la CLI equivale exactamente al entry point `python -m ...` de las secciones siguientes.

### 8.1 Instalar y dejar el comando disponible

La CLI se registra junto con el proyecto en el paso 4 mediante `pip install -r requirements.txt` (requisito `-e .`). Si las dependencias se instalaron antes de que existiera la CLI, registrar el ejecutable con:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Con el entorno activado, el comando queda disponible directamente:

```powershell
neumonia --help
neumonia eda --help
```

### 8.2 Flujo recomendado

```powershell
dvc pull data/raw/chest_xray.dvc
neumonia eda
neumonia prepare
neumonia augment
neumonia train
neumonia evaluate
neumonia test
```

Las etapas siguientes **no forman parte del flujo base** y solo son necesarias si se desea reproducir la mejora del modelo:

```powershell
neumonia tune        # regula learning rate, dropout y fine-tuning de MobileNetV2
neumonia desbalance  # decide entre class weight y oversampling sobre el ajustado lr=3e-4
neumonia evaluate
```

`neumonia train`, `neumonia tune` y `neumonia desbalance` consumen bastante tiempo y recursos en CPU. `neumonia run` encadena solo las etapas del flujo base en ese mismo orden (no ejecuta `dvc pull`; el dataset se administra con DVC) y **no** incluye `tune` ni `desbalance`.

Como alternativa, cada comando se puede invocar con el intérprete del entorno sin activarlo:

```powershell
.\.venv\Scripts\python.exe -m src.cli train
```

## 9. Ejecutar el EDA

El módulo ejecutable del EDA es:

```powershell
.\.venv\Scripts\python.exe -m src.data.make_dataset
```

El comando valida la estructura, inspecciona las imágenes y genera figuras en `reports/figures/`. El notebook asociado es `notebooks/01_comprension_datos_eda.ipynb`.

## 10. Ejecutar la preparación

La preparación genera el manifiesto experimental (80% train / 20% validation del train original; test original de 624 intacto; semilla 42; sin duplicados por contenido entre conjuntos) mediante `crear_manifiesto_division_estratificada`:

```powershell
.\.venv\Scripts\python.exe -c "from src.data.splitting import crear_manifiesto_division_estratificada; r=crear_manifiesto_division_estratificada('data/raw/chest_xray', 'data/interim/stratified_split_train80_val20_test_original.csv', random_state=42); print(r.groupby('split').size().to_dict())"
```

Después se construyen los datasets TensorFlow sobre ese reparto con `construir_pipelines_datos`:

```powershell
.\.venv\Scripts\python.exe -c "from src.data.datasets import construir_pipelines_datos; d=construir_pipelines_datos('data/raw/chest_xray', image_size=(224,224), batch_size=16, manifiesto_division='data/interim/stratified_split_train80_val20_test_original.csv'); print({k: v for k, v in d.items()})"
```

El pipeline carga imágenes RGB, las redimensiona a `224 x 224`, normaliza a `[0, 1]`, asigna las etiquetas binarias y crea datasets TensorFlow para `train` (4.173), `validation` (1.043) y `test` (624).

## 11. Entrenar los tres modelos

El entry point real del entrenamiento es:

```powershell
.\.venv\Scripts\python.exe -m src.training.run_real_training
```

Este comando sí vuelve a entrenar VGG16, ResNet50 y MobileNetV2. Produce los checkpoints, las curvas, las matrices y `models/model_results.json`. No debe ejecutarse para una simple consulta de resultados ya existentes.

Los tres modelos se entrenan sobre el subconjunto `train` del reparto experimental (4.173 imágenes) y se validan sobre `validation` (1.043 imágenes). El `test` original (624 imágenes) no participa en entrenamiento ni en selección; solo se evalúa el modelo ganador al final.

### 11.1 Generar la figura de ejemplos de augmentación

Para visualizar el efecto de la augmentation sobre una imagen real del `train` (la misma empleada en el entrenamiento), existe un entry point independiente que no reentrena modelos:

```powershell
.\.venv\Scripts\python.exe -m src.visualization.visualize
```

Genera `reports/figures/data_augmentation_examples.png`, una figura de 2x2 con la imagen original y variantes obtenidas únicamente con `RandomRotation(0.05)` y `RandomZoom(0.05)`. El volteo horizontal fue eliminado del pipeline porque en radiografías de tórax puede existir información de lateralidad anatómica (marcadores L/R) que un volteo horizontal podría invertir artificialmente.

### 11.2 Tuning de MobileNetV2

El tuning experimental de MobileNetV2 se ejecuta con:

```powershell
.\.venv\Scripts\python.exe -m src.training.tuning_mobilenetv2
```

Equivale a `neumonia tune`. Reentrena MobileNetV2 (partiendo de los pesos ImageNet) con varias configuraciones sobre el reparto experimental:

- lr `5e-5` (base congelada);
- lr `3e-4` (base congelada);
- dropout `0.5` (base congelada);
- fine-tuning de las últimas capas (`block_13+`) con lr `1e-4`.

La selección se hace **solo con métricas de `validation`** con el mismo criterio jerárquico (Balanced Accuracy, ROC-AUC, F1, Accuracy). El `test` original queda reservado y solo se evalúa la configuración ganadora al final.

La configuración ganadora fue **lr `3e-4`** con dropout `0.3` y base congelada. El fine-tuning `block_13+` se descartó por degradar la especificidad (0.2388 en validation). Al terminar se generan `models/mobilenetv2_tuning_results.json` y el checkpoint `models/mobilenetv2_ajustado/best_model.keras`.

### 11.3 Tratamiento del desbalance

Los experimentos de desbalance sobre el ajustado `lr=3e-4` se ejecutan con:

```powershell
.\.venv\Scripts\python.exe -m src.training.tuning_desbalance_clases
```

Equivale a `neumonia desbalance`. Reentrena el ajustado con dos estrategias tratadas por separado:

- **class weight** en el entrenamiento: pesos `balanced` calculados con `sklearn` sobre el `train` (NORMAL≈1.9445, PNEUMONIA≈0.6731);
- **oversampling**: se duplican aleatoriamente imágenes `NORMAL` en `train` (1 073 → 3 100; total train 6 200) para equilibrar las clases.

Ninguna estrategia modifica `validation` ni `test`. La selección usa el mismo criterio jerárquico sobre `validation`; el `test` solo se evalúa para el ganador al final.

El ganador fue el **oversampling**. Al terminar se generan `models/mobilenetv2_desbalance_results.json` y el checkpoint final `models/mobilenetv2_desbalance_oversampling/best_model.keras`.

> Nota: los resultados de estas dos secciones son experimentales sobre este dataset. Mejorar la Balanced Accuracy con oversampling no demuestra que el desbalance fuera la única causa de las diferencias observadas.

## 12. Evaluación

La evaluación está integrada en el comando anterior. Cada modelo se evalúa primero sobre el subconjunto `validation` (1.043 imágenes), se aplica umbral 0.5 sobre las probabilidades y se calculan métricas. Se generan:

- `reports/figures/confusion_matrix_validation_vgg16.png`;
- `reports/figures/confusion_matrix_validation_resnet50.png`;
- `reports/figures/confusion_matrix_validation_mobilenetv2.png`;
- `reports/figures/roc_curve_validation_vgg16.png`;
- `reports/figures/roc_curve_validation_resnet50.png`;
- `reports/figures/roc_curve_validation_mobilenetv2.png`.

El conjunto `test` original (624 imágenes) queda reservado íntegro y solo se evalúa el modelo ganador al final, lo que genera:

- `reports/figures/confusion_matrix_test_<modelo_ganador>.png`;
- `reports/figures/roc_curve_test_<modelo_ganador>.png`.

## 13. Comparación y criterio de éxito

La comparación también está integrada en `run_real_training.py` y se realiza únicamente sobre las métricas de `validation`; el conjunto `test` no participa en la selección. El código utiliza este orden:

1. Balanced Accuracy;
2. ROC-AUC;
3. F1;
4. Accuracy.

Sobre el `test` original se calcula además:

- `baseline_test`: baseline de clase mayoritaria (predecir siempre `PNEUMONIA`).
- `criterio_exito`: el modelo ganador debe superar al baseline en Balanced Accuracy y mostrar sensibilidad y especificidad por encima del nivel de azar (0.5), lo que indica un equilibrio adecuado entre ambas clases.

Todo se persiste en `models/model_results.json`.

> El tuning (sección 11.2) y los experimentos de desbalance (sección 11.3) reutilizan el mismo criterio jerárquico, pero persisten sus propios resultados en `models/mobilenetv2_tuning_results.json` y `models/mobilenetv2_desbalance_results.json`; `models/model_results.json` conserva intactos los resultados del flujo base.

## 14. Consultar el modelo seleccionado

Para consultar el ganador del flujo base guardado sin entrenar:

```powershell
$result = Get-Content -Raw models/model_results.json | ConvertFrom-Json
$result.validation_comparison.winner.model_name
$result.validation_comparison.results | Format-Table model_name, balanced_accuracy, accuracy, recall, specificity, f1, roc_auc
$result.baseline_test | Format-List
$result.criterio_exito | Format-List
$result.final_test | Format-List model_name, accuracy, balanced_accuracy, precision, recall, specificity, f1, roc_auc
```

El bloque `validation_comparison.winner` es el modelo seleccionado únicamente con métricas de `validation`. El bloque `final_test` contiene las métricas del ganador sobre el `test` original de 624 imágenes, única evaluación realizada sobre ese conjunto.

Para consultar la selección y el test del modelo final (MobileNetV2 + oversampling), usar los resultados experimentales:

```powershell
$desb = Get-Content -Raw models/mobilenetv2_desbalance_results.json | ConvertFrom-Json
$desb.tabla_seleccion | Format-Table experimento, balanced_accuracy, accuracy, recall, specificity, f1, roc_auc
$desb.motivo_seleccion
$desb.final_test | Format-List model_name, accuracy, balanced_accuracy, precision, recall, specificity, f1, roc_auc
$desb.final_test.confusion_matrix
```

Para consultar la selección del tuning:

```powershell
$tun = Get-Content -Raw models/mobilenetv2_tuning_results.json | ConvertFrom-Json
$tun.tabla_ordenada | Format-Table experimento, balanced_accuracy, accuracy, recall, specificity, f1, roc_auc
$tun.seleccion
```

## 15. Ejecutar los tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

La suite actual contiene 34 pruebas. Incluye verificaciones de la nueva división: el test original intacto, la estratificación, la ausencia de duplicados por hash entre conjuntos, la no mezcla del test con train/validation y la ausencia de `RandomFlip("horizontal")` en la augmentation.

## 16. Ubicación de artefactos

- Modelos del flujo base: `models/vgg16/best_model.keras`, `models/resnet50/best_model.keras` y `models/mobilenetv2/best_model.keras`.
- Modelo ajustado (`lr=3e-4`, experimental): `models/mobilenetv2_ajustado/best_model.keras`.
- **Modelo final (oversampling):** `models/mobilenetv2_desbalance_oversampling/best_model.keras`.
- Resultados del flujo base: `models/model_results.json`.
- Resultados del tuning: `models/mobilenetv2_tuning_results.json`.
- Resultados del desbalance: `models/mobilenetv2_desbalance_results.json`.
- Manifiesto del reparto experimental: `data/interim/stratified_split_train80_val20_test_original.csv`.
- Figuras EDA, evaluación, augmentación, tuning y desbalance: `reports/figures/`.
- Código de datos: `src/data/`.
- Código de modelos: `src/models/`.
- Código de entrenamiento: `src/training/` (`run_real_training.py`, `tuning_mobilenetv2.py`, `tuning_desbalance_clases.py`).
- Código de visualización: `src/visualization/`.
- Tests: `tests/`.

## 17. Solución de problemas comunes

### No se encuentra `dvc`

Instalar DVC con soporte para el remoto configurado:

```powershell
.\.venv\Scripts\python.exe -m pip install "dvc[gdrive]"
```

Después comprobar:

```powershell
dvc remote list
```

### No existe `data/raw/chest_xray`

Ejecutar:

```powershell
dvc pull data/raw/chest_xray.dvc
```

Verificar que el archivo `.dvc` y el remoto sean accesibles.

### TensorFlow no está disponible

Comprobar el entorno activo y reinstalar las dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__)"
```

### El entrenamiento tarda demasiado

El script entrena tres modelos y ejecuta tres epochs sobre CPU si no hay GPU configurada. Es un comportamiento esperado. No interrumpir una ejecución si se necesitan los tres resultados finales.

### Se desea consultar resultados sin reentrenar

No ejecutar `run_real_training.py`. Usar `neumonia evaluate` para consultar los resultados guardados, o leer `models/model_results.json` y revisar las figuras y los checkpoints ya existentes.

### Falla `make requirements`

El Makefile conserva reglas heredadas del template original y no es el entry point recomendado para este pipeline. Utilizar directamente los comandos de esta guía con el intérprete de `.venv`.