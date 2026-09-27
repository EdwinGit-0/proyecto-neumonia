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
neumonia eda          # Análisis exploratorio de datos (sección 9)
neumonia prepare      # Preparación de datos: manifiesto 80/20 (train original) y verificación de pipelines (sección 10)
neumonia augment      # Figura de ejemplos de augmentación (sección 11.1)
neumonia train        # Entrenar VGG16, ResNet50 y MobileNetV2; evaluar, comparar y seleccionar (sección 11)
neumonia sensibilidad # 7 configuraciones x 3 arquitecturas = 21 entrenamientos; no se repite si el artefacto existe (sección 11.2)
neumonia test-inicial # Test del checkpoint ganador de sensibilidad, antes de optimizar (sección 11.3)
neumonia optimizar    # 5 variantes + selección sobre validation + test final (sección 11.3)
neumonia evaluate     # Mostrar los resultados guardados sin volver a entrenar (sección 14)
neumonia test         # Ejecutar la suite de pruebas con pytest (sección 15)
neumonia run          # Flujo completo: EDA -> preparación -> augmentación -> sensibilidad -> optimización (con test inicial y test final) -> suite de pruebas
```

`neumonia sensibilidad` acepta `--recalcular` para reentrenar las 21 pruebas aunque el artefacto ya exista, y `neumonia test` acepta `--verbose` / `-v`.

Los comandos `tune` y `desbalance` se eliminaron de la CLI (ver sección 8.2).

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
neumonia sensibilidad
neumonia test-inicial
neumonia optimizar
neumonia evaluate
neumonia test
```

El orden importa: cada etapa lee el artefacto JSON de la etapa anterior y **no vuelve a
entrenar lo ya resuelto**.

| Orden | Comando            | Qué hace                                                          | Artefacto principal               |
| ----- | ------------------ | ----------------------------------------------------------------- | --------------------------------- |
| 1     | `neumonia eda`     | Inspección del dataset                                            | `reports/figures/`                |
| 2     | `neumonia prepare` | Manifiesto 80/20 + test original intacto                          | `data/interim/...csv`             |
| 3     | `neumonia augment` | Figura de ejemplos de augmentación                                | `reports/figures/`                |
| 4     | `neumonia train`   | VGG16, ResNet50 y MobileNetV2                                     | `models/model_results.json`       |
| 5     | `neumonia sensibilidad` | 7 configuraciones x 3 arquitecturas = 21 entrenamientos | `models/sensitivity_results.json` |
| 6     | `neumonia test-inicial` | Test del ganador de sensibilidad                               | `models/optimization_results.json`|
| 7     | `neumonia optimizar` | 5 variantes + selección sobre validation + test final          | `models/optimization_results.json`|
| 8     | `neumonia evaluate`| Consulta los resultados guardados sin entrenar                   | —                                 |
| 9     | `neumonia test`    | Pruebas automatizadas                                             | —                                 |

`neumonia sensibilidad` es la etapa más costosa (21 entrenamientos). Si `models/sensitivity_results.json`
ya existe, el comando **no la repite**: solo registra el resultado disponible. Esa es la razón por la
que las etapas 6 y 7 pueden ejecutarse sin haber entrenado los 21 experimentos en la sesión actual.

`neumonia train`, `neumonia sensibilidad` y `neumonia optimizar` consumen bastante tiempo y recursos
en CPU. `neumonia run` encadena todo ese recorrido, aunque **no** invoca `neumonia train`: la etapa 1
queda fuera porque la sensibilidad ya evalúa las tres arquitecturas. Para repetir solo la optimización
con variantes ya entrenadas:

```powershell
neumonia optimizar --reusar
```

Los comandos `tune` y `desbalance` se han **eliminado de la CLI**: correspondían a una
experimentación histórica anterior (`lr=3e-4` y oversampling) que quedó superada por la etapa de
sensibilidad. Sus artefactos siguen en `models/mobilenetv2_tuning_results.json` y
`models/mobilenetv2_desbalance_results.json`, y el código que los generaba se conserva únicamente
en `src/training/tuning_mobilenetv2.py` y `src/training/tuning_desbalance_clases.py`. El
checkpoint que el artefacto de desbalance declara
(`models/mobilenetv2_desbalance_oversampling/best_model.keras`) ya no existe en el repositorio. No
deben usarse para tomar decisiones nuevas.

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

Los tres modelos se entrenan sobre el subconjunto `train` del reparto experimental (4.173 imágenes) y se validan sobre `validation` (1.043 imágenes). El `test` original (624 imágenes) no participa en entrenamiento ni en selección; en esta etapa se evalúa únicamente el modelo ganador al final.

### 11.1 Generar la figura de ejemplos de augmentación

Para visualizar el efecto de la augmentation sobre una imagen real del `train` (la misma empleada en el entrenamiento), existe un entry point independiente que no reentrena modelos:

```powershell
.\.venv\Scripts\python.exe -m src.visualization.visualize
```

Genera `reports/figures/data_augmentation_examples.png`, una figura de 2x2 con la imagen original y variantes obtenidas únicamente con `RandomRotation(0.05)` y `RandomZoom(0.05)`. El volteo horizontal fue eliminado del pipeline porque en radiografías de tórax puede existir información de lateralidad anatómica (marcadores L/R) que un volteo horizontal podría invertir artificialmente.

### 11.2 Sensibilidad de hiperparámetros de MobileNetV2

La búsqueda de hiperparámetros de MobileNetV2 se ejecuta con:

```powershell
neumonia sensibilidad
```

Equivale a:

```powershell
.\.venv\Scripts\python.exe -m src.training.sensitivity
```

Explora **7 configuraciones** de hiperparámetros sobre el reparto experimental, variando learning
rate, dropout y número de épocas, y repite cada una en **las tres arquitecturas**: son 21
entrenamientos en total. El fine-tuning **no** se explora aquí; todas las arquitecturas se entrenan
desde cero. Los valores son:

| Eje | Valores |
|---|---|
| Learning rate | `1e-4` (referencia), `3e-4`, `1e-3` |
| Dropout | `0.3` (referencia), `0.2`, `0.5` |
| Épocas | `3` (referencia), `5`, `10` |

La selección es **exclusivamente sobre `validation`**, con el criterio jerárquico (Balanced Accuracy,
ROC-AUC, F1, Accuracy). El `test` original no se toca en esta etapa.

Esta etapa es la más costosa del proyecto. Si `models/sensitivity_results.json` ya existe, el
comando **no vuelve a ejecutarla**: solo registra y muestra el resultado disponible. Para
reentrenarla de todos modos, usar `neumonia sensibilidad --recalcular`. Los checkpoints de cada
configuración quedan en `experiments/sensibilidad_hiperparametros/checkpoints/` y los resultados en
`experiments/sensibilidad_hiperparametros/resultados/sensibilidad_resultados.json`.

**Configuración ganadora (`MobileNetV2 lr_1e-3`)**, que es el punto de partida fijo de la etapa
de optimización:

| Parámetro      | Valor    |
| -------------- | -------- |
| Learning rate  | `1e-3`   |
| Dropout        | `0.3`    |
| Épocas         | `3`      |
| Entrenamiento  | desde cero |
| Balanced Accuracy en validation | `0.959042` |

Sus resultados de `validation` están graficados en:

- `reports/figures/confusion_matrix_validation_mobilenetv2-sensibilidad.png` (TN 254, FP 14, FN 23, TP 752);
- `reports/figures/roc_curve_validation_mobilenetv2-sensibilidad.png` (ROC-AUC `0.9936`).

Estas dos figuras son distintas de las de `validation` del flujo base que lista la sección 12: las
base corresponden al punto de partida del ajuste fino, y las `-sensibilidad` al ganador de
hiperparámetros que arranca la optimización. La etapa de sensibilidad no persiste rutas de figura
en `models/sensitivity_results.json`, por lo que estas figuras no tienen vínculo artifact→figura.

Una vez elegida, esta configuración **no se vuelve a explorar**: la optimización posterior
únicamente evalúa el tratamiento del desbalance y el alcance del fine-tuning.

### 11.3 Test inicial y optimización del desbalance

El flujo vigente de optimización se ejecuta con:

```powershell
neumonia test-inicial   # evalúa el checkpoint ganador de sensibilidad sobre test
neumonia optimizar      # 5 variantes + selección sobre validation + test final
```

Equivale a:

```powershell
.\.venv\Scripts\python.exe -m src.cli test-inicial
.\.venv\Scripts\python.exe -m src.cli optimizar
```

El punto de partida es **fijo** y se lee de `models/sensitivity_results.json`; no se reentrena
ninguno de los 21 experimentos. A partir de ese punto de partida se entrenan cinco variantes,
cada una cambiando un único factor:

| Variante              | Factor aislado                                                        |
| --------------------- | --------------------------------------------------------------------- |
| `oversampling_normal` | `NORMAL` se equipara con `PNEUMONIA` solo en `train` (1 073 → 3 100; total 6 200) |
| `class_weight`        | Pesos `balanced` solo en `fit` (`NORMAL≈1.9445`, `PNEUMONIA≈0.6731`) |
| `finetune_block16`    | Se entrenan las capas de `block_16_expand` (15 de 158)               |
| `finetune_block13`    | Se entrenan las capas de `block_13_expand` (42 de 158)               |
| `finetune_block10`    | Se entrenan las capas de `block_10_expand` (68 de 158)               |

El oversampling se aplica únicamente a `train`: ni `validation` ni `test` se duplican. El
oversampling siempre equipara `NORMAL` con `PNEUMONIA`; el proyecto no implementa factores de
proporción configurables.

**Reglas de decisión:**

1. El test inicial se mide antes de decidir, para tener contraste con la auditoría de sensibilidad.
2. La comparación entre variantes es **solo sobre `validation`**, con el criterio jerárquico
   Balanced Accuracy > ROC-AUC > F1 > Accuracy.
3. No hay tolerancias adicionales: el recall se reporta como información, pero no filtra
   candidatos.
4. El test final se ejecuta **una sola vez**, sobre el modelo ya elegido por `validation`.

Al terminar se generan `models/optimization_results.json` (con el test inicial, las métricas de
`validation` de cada variante, la decisión y el test final) y los checkpoints en
`models/mobilenetv2_optimizacion/`.

> **Reproducibilidad.** El pipeline es determinista: `src/utils/reproducibility.py` fija
> `SEMILLA = 42`, activa `tf.config.experimental.enable_op_determinism()`, siembra cada `shuffle` y
> las capas `RandomRotation` / `RandomZoom`, usa `map(..., deterministic=True)` y reinicia la
> semilla antes de construir el pipeline de cada variante, de modo que las cinco ven los mismos
> datos. Verificado: dos procesos independientes dan lotes de entrenamiento idénticos byte a byte.
> Para consultar un resultado concreto sin volver a entrenar, lea
> `models/optimization_results.json`; para repetir solo la optimización, use
> `neumonia optimizar --reusar`.
>
> Las dos ejecuciones anteriores de esta etapa se hicieron **sin** ese control y no son
> reproducibles; están archivadas en `models/historico/` y en
> `experiments/optimizacion_pendiente_mobilenetv2/`.

### 11.4 Experimentación histórica (fuera del flujo vigente)

La experimentación anterior se hizo sobre el ajuste `lr=3e-4` y ya no se expone en la CLI. Se
reproduce desde los guiones de `experiments/`:

```text
experiments/optimizacion_pendiente_mobilenetv2/run_optimizacion.py
```

Sus artefactos (`models/mobilenetv2_tuning_results.json`,
`models/mobilenetv2_desbalance_results.json` y los checkpoints asociados) se conservan para
consultar resultados anteriores, pero el flujo vigente no los utiliza.

## 12. Evaluación

La evaluación está integrada en el comando anterior. Cada modelo se evalúa primero sobre el subconjunto `validation` (1.043 imágenes), se aplica umbral 0.5 sobre las probabilidades y se calculan métricas. Se generan:

- `reports/figures/confusion_matrix_validation_vgg16.png`;
- `reports/figures/confusion_matrix_validation_resnet50.png`;
- `reports/figures/confusion_matrix_validation_mobilenetv2.png`;
- `reports/figures/roc_curve_validation_vgg16.png`;
- `reports/figures/roc_curve_validation_resnet50.png`;
- `reports/figures/roc_curve_validation_mobilenetv2.png`.

> **Aviso: qué arquitectura representa cada figura de `validation` con nombre simple.** Las cuatro
> figuras de VGG16 y ResNet50 se regeneraron para mostrar la **mejor configuración por
> arquitectura** de la etapa de sensibilidad (`lr=1e-3`, dropout `0.3`, 3 épocas), no los resultados
> del flujo base:
>
> | Figura | Representa | TN | FP | FN | TP | BA | ROC-AUC |
> |---|---|---:|---:|---:|---:|---:|---:|
> | `confusion_matrix_validation_vgg16.png` | sensibilidad VGG16 | 260 | 8 | 70 | 705 | 0.9399 | 0.9850 |
> | `confusion_matrix_validation_resnet50.png` | sensibilidad ResNet50 | 150 | 118 | 67 | 708 | 0.7366 | 0.8931 |
> | `confusion_matrix_validation_mobilenetv2.png` | flujo base | 251 | 17 | 42 | 733 | 0.9412 | 0.9887 |
>
> Las dos primeras **no** coinciden con `models/model_results.json` → `models.VGG16.validation` y
> `models.ResNet50.validation`, que siguen declarando los valores del flujo base (VGG16 216/52/41/734,
> BA 0.8765; ResNet50 0/268/0/775, BA 0.5000). Los campos `confusion_plot` y `roc_plot` de esos dos
> bloques apuntan a las rutas de arriba, así que abrir la figura no reproduce la matriz declarada en
> el JSON. Las tablas de `validation` de la sección 7 y del README describen el flujo base, que
> sigue siendo el dato de esa sección; el cambio afecta solo a las imágenes. Para el ganador de
> sensibilidad de MobileNetV2, usar las figuras `-sensibilidad` de la sección 11.2, no la de nombre
> simple.

El conjunto `test` original (624 imágenes) queda reservado íntegro. En el flujo base se evalúa solo el modelo ganador al final, lo que genera:

- `reports/figures/confusion_matrix_test_<modelo_ganador>.png`;
- `reports/figures/roc_curve_test_<modelo_ganador>.png`.

En el flujo vigente ese mismo conjunto se evalúa dos veces, y el código guarda figuras en esos dos puntos concretos:

- Test inicial del ganador de la sensibilidad: `confusion_matrix_test_mobilenetv2-sensibilidad.png` y `roc_curve_test_mobilenetv2-sensibilidad.png`.
- Test final del modelo elegido: `confusion_matrix_test_mobilenetv2-<ganador>.png` y `roc_curve_test_mobilenetv2-<ganador>.png`, que en la ejecución actual son los de `oversampling_normal`.

Las variantes que no ganan no se evalúan sobre `test`, así que no tienen figura de test. Los archivos `confusion_matrix_test_mobilenetv2-ajustado.png` y `-oversampling.png` corresponden a las ejecuciones históricas, no al flujo vigente. El material de `finetune_block16` es de `validation` (`confusion_matrix_validation_mobilenetv2-finetune_block16.png` y `roc_curve_validation_mobilenetv2-finetune_block16.png`, TN 126, FP 142, FN 1, TP 774): la variante no ganó, nunca se evaluó sobre `test` y el nombre del archivo así lo indica. Ambas evaluaciones son mediciones: ninguna interviene en la selección.

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

> La sensibilidad (sección 11.2) y la optimización del desbalance (sección 11.3) reutilizan el mismo criterio jerárquico, pero persisten sus propios resultados en `models/sensitivity_results.json` y `models/optimization_results.json`; `models/model_results.json` conserva intactos los resultados del flujo base. El tuning y el desbalance anteriores son históricos y viven en `models/mobilenetv2_tuning_results.json` y `models/mobilenetv2_desbalance_results.json`.

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

El bloque `validation_comparison.winner` es el modelo seleccionado únicamente con métricas de `validation`. El bloque `final_test` contiene las métricas del ganador del flujo base sobre el `test` original de 624 imágenes. Es la única evaluación de `test` de **esa** etapa; en el flujo vigente el mismo conjunto se mide dos veces, como `test_inicial` y `test_final`, y en ningún caso decide.

Para consultar la selección y los test del flujo vigente (sensibilidad, test inicial,
optimización y test final):

```powershell
$opt = Get-Content -Raw models/optimization_results.json | ConvertFrom-Json
$opt.punto_de_partida | Format-List model_name, config_id
$opt.criterio_seleccion
$opt.test_utilizado_para_seleccion
$opt.test_inicial.test    | Format-List balanced_accuracy, accuracy, precision, recall, specificity, f1, roc_auc
$opt.seleccion.tabla_ordenada | Format-Table id, grupo, balanced_accuracy, roc_auc, f1, accuracy
$opt.seleccion           | Format-List ganador_id, ganador_grupo, ganador_checkpoint, supera_punto_de_partida
$opt.test_final.test     | Format-List balanced_accuracy, accuracy, precision, recall, specificity, f1, roc_auc
$opt.test_final.test.confusion_matrix
```

Cada elemento de `resultados` guarda sus métricas de `validation` anidadas bajo la propiedad
`validation`, así que `Format-Table id, balanced_accuracy, ...` sobre `resultados` mostraría columnas
vacías. Para la tabla plana y ya ordenada por el criterio, usar
`seleccion.tabla_ordenada`; para el detalle de cada variante,
`$opt.resultados | Format-List id, grupo, checkpoint, validation`.

El bloque `test_inicial` es la medición sobre `test` del punto de partida, anterior a decidir.
El bloque `resultados` solo contiene métricas de `validation`. El bloque `seleccion` documenta
el ganador según el criterio jerárquico de cuatro métricas, cuyo orden declara
`criterio_seleccion`, y deja constancia explícita en `test_utilizado_para_seleccion` de que el test
no intervino. El bloque `test_final` es la única evaluación del modelo elegido, ejecutada una sola
vez. El bloque `reproducibilidad` registra cómo se fijaron las semillas.

Para consultar los resultados históricos (fuera del flujo vigente):

```powershell
$desb = Get-Content -Raw models/mobilenetv2_desbalance_results.json | ConvertFrom-Json
$desb.tabla_seleccion | Format-Table experimento, balanced_accuracy, accuracy, recall, specificity, f1, roc_auc
$tun = Get-Content -Raw models/mobilenetv2_tuning_results.json | ConvertFrom-Json
$tun.tabla_ordenada | Format-Table experimento, balanced_accuracy, accuracy, recall, specificity, f1, roc_auc
```

## 15. Ejecutar los tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

La suite actual contiene 73 pruebas. Incluye verificaciones de la nueva división (test original
intacto, estratificación, ausencia de duplicados por hash entre conjuntos, no mezcla del test con
train/validation, ausencia de `RandomFlip("horizontal")` en la augmentation), la idempotencia de
la etapa de sensibilidad, el aislamiento del `test` en la optimización (criterio jerárquico de
cuatro métricas, desempate por ROC-AUC, F1 y Accuracy, `test` fuera de toda decisión, variantes que
no repiten los 21 factores ya evaluados y que cada una aísla un solo factor) y, en
`tests/test_reproducibilidad.py`, el control de semillas: `shuffle` y augmentation sembrados,
`deterministic=True`, igualdad de lotes entre pipelines construidos por separado, reproducibilidad
del oversampling, igualdad de pesos iniciales y orden `clear_session()` → `reiniciar_semilla()` →
pipeline → modelo. No hay ninguna prueba que exija paralelismo en el entrenamiento de las variantes.

## 16. Ubicación de artefactos

- Modelos del flujo base: `models/vgg16/best_model.keras`, `models/resnet50/best_model.keras` y `models/mobilenetv2/best_model.keras`.
- Checkpoints de las 5 variantes de optimización: `models/mobilenetv2_optimizacion/<variante>/best_model.keras`.
- Checkpoints de los 21 entrenamientos de sensibilidad (7 configuraciones × 3 arquitecturas): `experiments/sensibilidad_hiperparametros/checkpoints/`.
- Resultados de la selección de arquitectura: `models/model_results.json`.
- Resultados de la sensibilidad: `models/sensitivity_results.json` y `experiments/sensibilidad_hiperparametros/resultados/sensibilidad_resultados.json`.
- Auditoría del test del ganador de sensibilidad: `models/sensitivity_test_audit.json`.
- **Resultados del flujo vigente (test inicial, validation, selección y test final): `models/optimization_results.json`.**
- Resultados históricos: `models/mobilenetv2_tuning_results.json` y `models/mobilenetv2_desbalance_results.json`.
- Ejecuciones anteriores no reproducibles de la optimización: `models/historico/` y `experiments/optimizacion_pendiente_mobilenetv2/`.
- Manifiesto del reparto experimental: `data/interim/stratified_split_train80_val20_test_original.csv`.
- Figuras EDA, evaluación, augmentación y test: `reports/figures/`.
- Código de datos: `src/data/`.
- Código de modelos: `src/models/`.
- Código de entrenamiento: `src/training/` (`run_real_training.py`, `sensitivity.py`, `optimizacion_mobilenetv2.py`, `tuning_mobilenetv2.py`, `tuning_desbalance_clases.py`).
- Control de semillas: `src/utils/reproducibility.py`.
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