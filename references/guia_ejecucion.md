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

La copia validada contiene 5.856 imágenes JPEG organizadas en `train`, `val` y `test`, con las clases `NORMAL` y `PNEUMONIA`.

## 8. Ejecutar el EDA

El módulo ejecutable del EDA es:

```powershell
.\.venv\Scripts\python.exe -m src.data.make_dataset
```

El comando valida la estructura, inspecciona las imágenes y genera figuras en `reports/figures/`. El notebook asociado es `notebooks/01_comprension_datos_eda.ipynb`.

## 9. Ejecutar la preparación

No existe un script independiente de preparación con interfaz de línea de comandos. La preparación se ejecuta dentro del pipeline mediante `build_data_pipelines`:

```powershell
.\.venv\Scripts\python.exe -c "from src.data.datasets import build_data_pipelines; d=build_data_pipelines('data/raw/chest_xray', image_size=(224,224), batch_size=16); print({k: v for k, v in d.items()})"
```

El pipeline carga imágenes RGB, las redimensiona a `224 x 224`, normaliza a `[0, 1]`, asigna las etiquetas binarias y crea datasets TensorFlow.

## 10. Entrenar los tres modelos

El entry point real del entrenamiento es:

```powershell
.\.venv\Scripts\python.exe -m src.training.run_real_training
```

Este comando sí vuelve a entrenar VGG16, ResNet50 y MobileNetV2. Produce los checkpoints, las curvas, las matrices y `models/model_results.json`. No debe ejecutarse para una simple consulta de resultados ya existentes.

## 11. Evaluación

La evaluación está integrada en el comando anterior. Para cada modelo se ejecutan predicciones sobre `test`, se aplica umbral 0.5, se calculan métricas y se generan:

- `reports/figures/confusion_matrix_vgg16.png`;
- `reports/figures/confusion_matrix_resnet50.png`;
- `reports/figures/confusion_matrix_mobilenetv2.png`;
- `reports/figures/roc_curve_vgg16.png`;
- `reports/figures/roc_curve_resnet50.png`;
- `reports/figures/roc_curve_mobilenetv2.png`.

## 12. Comparación

La comparación también está integrada en `run_real_training.py`. El código utiliza este orden:

1. Balanced Accuracy;
2. ROC-AUC;
3. F1;
4. Accuracy.

El resultado se persiste en `models/model_results.json`.

## 13. Consultar el modelo seleccionado

Para consultar el ganador guardado sin entrenar:

```powershell
$result = Get-Content -Raw models/model_results.json | ConvertFrom-Json
$result.comparison.selection_criterion
$result.comparison.winner.model_name
$result.comparison.results | Format-Table model_name, balanced_accuracy, accuracy, recall, specificity, f1, roc_auc
```

El resultado actual selecciona `MobileNetV2`.

## 14. Ejecutar los tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

La suite actual contiene 22 pruebas.

## 15. Ubicación de artefactos

- Modelos: `models/vgg16/best_model.keras`, `models/resnet50/best_model.keras` y `models/mobilenetv2/best_model.keras`.
- Resultados: `models/model_results.json`.
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
