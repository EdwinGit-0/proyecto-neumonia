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

Para el modelado se utiliza un reparto experimental estratificado 70/15/15 (semilla 42) que se genera durante la preparación de datos y que no tiene duplicados entre conjuntos:

- `data/interim/stratified_split_70_15_15.csv`;
- train: 4.099 imágenes (1.108 NORMAL, 2.991 PNEUMONIA);
- validation: 879 imágenes (238 NORMAL, 641 PNEUMONIA);
- test: 878 imágenes (237 NORMAL, 641 PNEUMONIA).

## 8. Ejecutar el EDA

El módulo ejecutable del EDA es:

```powershell
.\.venv\Scripts\python.exe -m src.data.make_dataset
```

El comando valida la estructura, inspecciona las imágenes y genera figuras en `reports/figures/`. El notebook asociado es `notebooks/01_comprension_datos_eda.ipynb`.

## 9. Ejecutar la preparación

La preparación genera primero el manifiesto estratificado 70/15/15 (semilla 42, sin duplicados por contenido entre conjuntos) mediante `create_stratified_split_manifest`:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from src.utils.paths import DATA_DIR, PROJECT_ROOT; from src.data.splitting import create_stratified_split_manifest; p=PROJECT_ROOT/'data'/'interim'/'stratified_split_70_15_15.csv'; r=create_stratified_split_manifest(DATA_DIR, p, random_state=42); print(r.groupby('split').size().to_dict())"
```

Después se construyen los datasets TensorFlow sobre ese reparto con `build_data_pipelines`:

```powershell
.\.venv\Scripts\python.exe -c "from src.data.datasets import build_data_pipelines; d=build_data_pipelines('data/raw/chest_xray', image_size=(224,224), batch_size=16, split_manifest='data/interim/stratified_split_70_15_15.csv'); print({k: v for k, v in d.items()})"
```

El pipeline carga imágenes RGB, las redimensiona a `224 x 224`, normaliza a `[0, 1]`, asigna las etiquetas binarias y crea datasets TensorFlow para `train` (4.099), `validation` (879) y `test` (878).

## 10. Entrenar los tres modelos

El entry point real del entrenamiento es:

```powershell
.\.venv\Scripts\python.exe -m src.training.run_real_training
```

Este comando sí vuelve a entrenar VGG16, ResNet50 y MobileNetV2. Produce los checkpoints, las curvas, las matrices y `models/model_results.json`. No debe ejecutarse para una simple consulta de resultados ya existentes.

Los tres modelos se entrenan sobre el subconjunto `train` del reparto 70/15/15 (4.099 imágenes) y se validan sobre `validation` (879 imágenes).

## 11. Evaluación

La evaluación está integrada en el comando anterior. Cada modelo se evalúa primero sobre el subconjunto `validation` (879 imágenes), se aplica umbral 0.5 sobre las probabilidades y se calculan métricas. Se generan:

- `reports/figures/confusion_matrix_validation_vgg16.png`;
- `reports/figures/confusion_matrix_validation_resnet50.png`;
- `reports/figures/confusion_matrix_validation_mobilenetv2.png`;
- `reports/figures/roc_curve_validation_vgg16.png`;
- `reports/figures/roc_curve_validation_resnet50.png`;
- `reports/figures/roc_curve_validation_mobilenetv2.png`.

El conjunto `test` (878 imágenes) queda reservado y solo se evalúa el modelo ganador al final, lo que genera:

- `reports/figures/confusion_matrix_test_mobilenetv2.png`;
- `reports/figures/roc_curve_test_mobilenetv2.png`.

## 12. Comparación

La comparación también está integrada en `run_real_training.py` y se realiza únicamente sobre las métricas de `validation`; el conjunto `test` no participa en la selección. El código utiliza este orden:

1. Balanced Accuracy;
2. ROC-AUC;
3. F1;
4. Accuracy.

El resultado se persiste en `models/model_results.json`.

## 13. Consultar el modelo seleccionado

Para consultar el ganador guardado sin entrenar:

```powershell
$result = Get-Content -Raw models/model_results.json | ConvertFrom-Json
$result.validation_comparison.winner.model_name
$result.validation_comparison.results | Format-Table model_name, balanced_accuracy, accuracy, recall, specificity, f1, roc_auc
$result.final_test | Format-List model_name, accuracy, balanced_accuracy, precision, recall, specificity, f1, roc_auc
```

El resultado actual selecciona `MobileNetV2` (a partir de las métricas de `validation`). El bloque `final_test` contiene las métricas del ganador sobre `test`, única evaluación realizada sobre ese conjunto.

## 14. Ejecutar los tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

La suite actual contiene 23 pruebas.

## 15. Ubicación de artefactos

- Modelos: `models/vgg16/best_model.keras`, `models/resnet50/best_model.keras` y `models/mobilenetv2/best_model.keras`.
- Resultados: `models/model_results.json`.
- Manifiesto del reparto experimental: `data/interim/stratified_split_70_15_15.csv`.
- Figuras EDA y evaluación: `reports/figures/`.
- Código de datos: `src/data/`.
- Código de modelos: `src/models/`.
- Código de entrenamiento: `src/training/`.
- Tests: `tests/`.

## 16. Solución de problemas comunes

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

No ejecutar `run_real_training.py`. Leer `models/model_results.json` y consultar las figuras y checkpoints ya existentes.

### Falla `make requirements`

El Makefile conserva reglas heredadas del template original y no es el entry point recomendado para este pipeline. Utilizar directamente los comandos de esta guía con el intérprete de `.venv`.
