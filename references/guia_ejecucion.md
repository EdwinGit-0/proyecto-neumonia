# Guía de ejecución del proyecto

Guía paso a paso para ejecutar el proyecto de clasificación de neumonía en radiografías de tórax.
Cada etapa del flujo se documenta con su comando, su artefacto de salida, su tiempo aproximado y
las decisiones que toma. **El test original solo se consulta en la última etapa.**

Para el detalle técnico de cada componente ver `references/documentacion_proyecto.md`.

---

## 1. Requisitos

- Python `3.10.11` (validado con esa versión)
- Dataset restaurado con DVC en `data/raw/chest_xray/`
- CPU es suficiente; no se requiere GPU

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
dvc pull data/raw/chest_xray.dvc
```

Todos los comandos siguientes usan el intérprete del entorno virtual:

```powershell
.\.venv\Scripts\python.exe -m src.cli <comando>
```

La forma abreviada `neumonia <comando>` requiere que el paquete esté instalado en el entorno
(`.\.venv\Scripts\pip.exe install -e .`).

---

## 2. Orden de ejecución (nuevo flujo)

```powershell
neumonia eda                    # 1. Análisis exploratorio
neumonia prepare                # 2. Manifiesto estratificado 80/20
neumonia augment                # 3. Figura de ejemplos de augmentación
neumonia sensibilidad           # 4. 21 entrenamientos (cerrada, no repetir)
neumonia combinado-arquitecturas # 5. COMBINADO para VGG16, ResNet50, MobileNetV2
neumonia comparar               # 6. Compara y selecciona la mejor arquitectura
neumonia umbral                 # 7. Congela el umbral de decisión (0.38)
neumonia final                  # 8. Reentrena el definitivo sobre train + val
neumonia test                   # 9. Evaluación del test original con el umbral congelado
neumonia evaluar                # 10. Consulta de resultados
```

`neumonia run` encadena las diez etapas, incluida la evaluación del test, y cierra con la suite de
pruebas.

### Flujo original (compatibilidad)

Los comandos `neumonia combinado`, `neumonia umbral`, `neumonia final`, `neumonia test` siguen
disponibles para el flujo original de una sola arquitectura (MobileNetV2).

## 3. Etapas 1 a 3: preparación

### 3.1 Análisis exploratorio

```powershell
neumonia eda
```

Verifica la integridad de las imágenes, muestra la distribución de clases y detecta duplicados por
SHA-256 dentro de los splits crudos. No escribe artefactos de resultados.

### 3.2 Preparación de datos

```powershell
neumonia prepare
```

Genera `data/interim/stratified_split_train80_val20_test_original.csv` con `random_state=42`.
Divide **solo** las 5.216 imágenes del `train` original en `train` (4.173) y `validation` (1.043),
y asigna las 624 del `test` original sin modificarlas. Las 16 imágenes del `val` original quedan
excluidas. Agrupa por hash para que contenidos idénticos no se repartan entre conjuntos.

Después verifica que los tres pipelines sean construibles. Usa `incluir_test=True` únicamente para
comprobar que el split original existe; como el dataset es perezoso (`from_generator`), no se lee
ninguna imagen de test.

### 3.3 Ejemplos de augmentación

```powershell
neumonia augment
```

Escribe `reports/figures/data_augmentation_examples.png`. La augmentación es
`RandomRotation(0.05)` + `RandomZoom(0.05)` y se aplica **solo** a `train`. No se usa
`RandomFlip("horizontal")` por la lateralidad anatómica.

---

## 4. Etapa 4: sensibilidad de hiperparámetros (21 entrenamientos)

```powershell
neumonia sensibilidad
```

Explora 7 configuraciones × 3 arquitecturas = 21 entrenamientos con la base congelada, y escribe
`models/sensitivity_results.json`. Cada configuración varía **un solo factor** respecto de la
referencia (`lr=1e-4, dropout=0.3, 3` épocas):

| `id` | learning rate | dropout | epochs |
|---|---:|---:|---:|
| `ref_lr1e-4_do0.3_ep3` | 1e-4 | 0.3 | 3 |
| `lr_3e-4` | 3e-4 | 0.3 | 3 |
| `lr_1e-3` | 1e-3 | 0.3 | 3 |
| `do_0.2` | 1e-4 | 0.2 | 3 |
| `do_0.5` | 1e-4 | 0.5 | 3 |
| `ep_5` | 1e-4 | 0.3 | 5 |
| `ep_10` | 1e-4 | 0.3 | 10 |

Resultados (Balanced Accuracy en `validation`):

| `id` | MobileNetV2 | ResNet50 | VGG16 |
|---|---:|---:|---:|
| `ref_lr1e-4_do0.3_ep3` | 0.940246 | 0.500000 | 0.875888 |
| `lr_3e-4` | 0.955816 | 0.668938 | 0.930794 |
| **`lr_1e-3`** | **0.959042** | **0.736625** | **0.939913** |
| `do_0.2` | 0.944552 | 0.499355 | 0.880840 |
| `do_0.5` | 0.939670 | 0.500000 | 0.851129 |
| `ep_5` | 0.938659 | 0.522684 | 0.918955 |
| `ep_10` | 0.956026 | 0.688101 | 0.925407 |

**Las tres arquitecturas ganan con `lr=1e-3`, `dropout=0.3` y 3 épocas.** El learning rate es el
factor dominante; ninguna variante de dropout o de épocas supera a `1e-3`. Con la referencia `1e-4`,
ResNet50 colapsaba a BA 0.5 (especificidad 0.0).

> Esta etapa está **completa y no debe repetirse**: son 21 entrenamientos. La etapa 5 parte de este
> JSON. Para forzar el reentrenamiento existe `--recalcular`, pero no es necesario.

Equivalente desde el experimento, si se quiere regenerar el informe:

```powershell
.\.venv\Scripts\python.exe experiments\sensibilidad_hiperparametros\run_sensibilidad.py
.\.venv\Scripts\python.exe experiments\sensibilidad_hiperparametros\generar_informe.py
```

---

## 5. Etapa 5: COMBINADO para 3 arquitecturas (nuevo flujo)

```powershell
neumonia combinado-arquitecturas
```

Lee los hiperparámetros fijados por la sensibilidad (lr=1e-3, dropout=0.3, epochs=3) y entrena **tres**
corridas COMBINADO: una por arquitectura (VGG16, ResNet50, MobileNetV2), medida en `validation`.
Para rehacerla: `neumonia combinado-arquitecturas --recalcular`.

### 5.1 Qué hace COMBINADO (idéntico para las 3 arquitecturas)

| Paso | Detalle |
|---|---|
| Oversampling | `NORMAL` se equipara con `PNEUMONIA` en `train` (1.073 → 3.100), hasta 6.200 filas efectivas |
| Class weights | `{0: 1.9445, 1: 0.6731}`, calculados sobre la distribución **original** de `train` |
| Ámbito | Ambos se aplican **solo** a `train`: oversampling al pipeline, pesos a la función de pérdida de `fit` |
| Validation | Queda intacta: 268 NORMAL / 775 PNEUMONIA, ni oversampling ni pesos |

La función rechaza explícitamente cualquier solicitud que incluya el split `test`.

### 5.2 Aviso sobre la doble corrección

Como los pesos se calculan sobre la distribución original y no sobre el conjunto ya
equilibrado, no se anulan entre sí: el refuerzo total de `NORMAL` frente a `PNEUMONIA` es de
**2.89x** (1.00 por el oversampling × 2.89 por los pesos). Es decir, COMBINADO no solo corrige el
desbalance, lo invierte parcialmente en el entrenamiento.

Esto está registrado en `diagnostico_combinado.advertencia_doble_correccion` dentro de cada
`results/final/combinado_*_validacion.json`.

### 5.3 Resultados en validation (umbral 0.5)

| Arquitectura | Balanced Acc | ROC-AUC | F1 | Accuracy | Precision | Recall | Specificity | Matriz (TN/FP/FN/TP) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MobileNetV2 | 0.9620 | 0.9941 | 0.9619 | 0.9453 | 0.9986 | 0.9277 | 0.9963 | 267/1/56/719 |
| VGG16 | 0.8904 | 0.9878 | 0.8786 | 0.8389 | 0.9984 | 0.7845 | 0.9963 | 267/1/167/608 |
| ResNet50 | 0.7595 | 0.9002 | 0.6886 | 0.6462 | 0.9951 | 0.5265 | 0.9925 | 266/2/367/408 |

Artefactos (uno por arquitectura):

```text
results/final/combinado_vgg16_validacion.json
results/final/combinado_resnet50_validacion.json
results/final/combinado_mobilenetv2_validacion.json
results/final/combinado_vgg16_validacion.csv
results/final/combinado_resnet50_validacion.csv
results/final/combinado_mobilenetv2_validacion.csv
results/final/validacion_vgg16_y_{true,prob}.npy
results/final/validacion_resnet50_y_{true,prob}.npy
results/final/validacion_mobilenetv2_y_{true,prob}.npy
results/final/combinado/vgg16/best_model.keras
results/final/combinado/resnet50/best_model.keras
results/final/combinado/mobilenetv2/best_model.keras
```

Figuras en `reports/figures/`:

```text
validation_combinado_vgg16_confusion_matrix.png
validation_combinado_vgg16_roc_curve.png
validation_combinado_resnet50_confusion_matrix.png
validation_combinado_resnet50_roc_curve.png
validation_combinado_mobilenetv2_confusion_matrix.png
validation_combinado_mobilenetv2_roc_curve.png
```

### 5.4 Alcance de COMBINADO

COMBINADO (oversampling 50/50 sobre `train` más `class_weight` en la pérdida) es la única estrategia
de desbalance del proyecto: es una premisa metodológica, no el resultado de comparar alternativas.

El tratamiento se aplica **únicamente** a los splits de entrenamiento. `validation` y `test` conservan
su distribución original y nunca se reponderan.

---

## 6. Etapa 6: Comparación de arquitecturas y selección (nuevo flujo)

```powershell
neumonia comparar
```

Carga los 3 resultados COMBINADO, compara las arquitecturas usando el criterio jerárquico
(Balanced Accuracy > ROC-AUC > F1 > Accuracy) sobre validation y selecciona la mejor.

Resultados de la ejecución validada:

| Pos | Arquitectura | Balanced Acc | ROC-AUC | F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| 1 | **MobileNetV2** | 0.9620 | 0.9941 | 0.9619 | 0.9453 |
| 2 | VGG16 | 0.8904 | 0.9878 | 0.8786 | 0.8389 |
| 3 | ResNet50 | 0.7595 | 0.9002 | 0.6886 | 0.6462 |

**Arquitectura seleccionada: MobileNetV2**

Copia los resultados de MobileNetV2 a los archivos estándar del flujo
(`results/final/combinado_validacion.json`, `validacion_y_{true,prob}.npy`, etc.) para que las
etapas siguientes (umbral, final, test) funcionen sin cambios.

Artefactos:

```text
results/final/comparacion_arquitecturas.json    # tabla completa y ganador
results/final/seleccion_arquitectura.json       # arquitectura seleccionada
```

---

## 7. Etapas 7 a 8: umbral y modelo definitivo

### 6.1 Umbral de decisión

```powershell
neumonia umbral
```

Recorre 91 umbrales (0.05 a 0.95 en pasos de 0.01, más 0.50 explícito) sobre las probabilidades de
`validation` del modelo COMBINADO y **congela** el de mayor Balanced Accuracy. No toca el test.

| Umbral | Balanced Accuracy | Precision | Specificity | Recall |
|---:|---:|---:|---:|---:|
| 0.50 | 0.9549 | 0.9986 | 0.9963 | 0.9135 |
| **0.38** | **0.9621** | 0.9960 | 0.9776 | 0.9466 |

**Umbral congelado: `0.38`** (Δ BA +0.0072 frente a 0.50). Escribe
`results/final/umbral_decision.json` con la curva completa de 91 puntos y los campos
`test_utilizado: false` y `congelado: true`.

> **Cuidado con este número.** El umbral se eligió maximizando balanced accuracy sobre
> `validation`, y ese mismo conjunto es el que se reporta: el 0.9621 está sesgado al alza, y la
> diferencia frente al 0.5 por defecto se amplifica al recalcularla sobre las probabilidades del
> modelo definitivo. El artefacto incluye esta advertencia en `advertencia_optimismo`.

### 6.2 Modelo definitivo

```powershell
neumonia final
```

Reentrena `MobileNetV2/combinado` sobre `train + val` (5.216 imágenes: 1.341 NORMAL / 3.875
PNEUMONIA) con el umbral `0.38` ya congelado. Como no queda validación que monitorear, entrena 3
épocas exactas y guarda los pesos finales explícitamente (sin `validation` no hay `ModelCheckpoint`).
No vuelve a mirar `validation`.

Artefactos: `results/final/final_model_training.json` y
`results/final/final_model/best_model.keras`.

**Tiempo real: 163 s/época en CPU.**

---

## 7. Etapa 8: evaluación del test final

```powershell
neumonia test
```

Es la última etapa del flujo. Se ejecuta normalmente, sin banderas de confirmación y sin consultar
ningún historial de ejecuciones anteriores.

### Qué hace

1. Verifica que exista la decisión congelada (`results/final/decision_modelo.json`) y el modelo
   definitivo (`results/final/final_model/best_model.keras`).
2. Comprueba que el manifiesto declare las 624 filas del test original.
3. Carga el pipeline del test con `incluir_test=True` a partir del manifiesto original, **sin
   oversampling, sin `class_weight` y sin augmentation**: usa las imágenes y la distribución tal
   como están (234 `NORMAL`, 390 `PNEUMONIA`).
4. Aplica el umbral congelado en la etapa 6 y calcula las métricas.
5. Guarda `results/final/final_test_report.json`, `results/final/final_test_metrics.csv`, la matriz de
   confusión, la curva ROC y las probabilidades (`test_y_true.npy`, `test_y_prob.npy`).

### Orden respecto al umbral

El umbral se congela en la etapa 6, **antes** de entrenar el modelo definitivo y antes de leer el
test. La evaluación no realimenta ninguna decisión: el modelo, la arquitectura, la estrategia y el
punto de corte ya están cerrados cuando el test se lee.

### Resultados de la corrida

| Métrica (umbral congelado 0.38) | Valor | Referencia (umbral 0.5) |
|---|---:|---:|
| **Balanced Accuracy** | **0.8996** | 0.8936 |
| Accuracy | 0.8990 | 0.8862 |
| Precision | 0.9383 | 0.9493 |
| Recall | 0.8974 | 0.8641 |
| Specificity | 0.9017 | 0.9231 |
| F1 | 0.9174 | 0.9047 |
| ROC-AUC | 0.9694 | 0.9694 |

Matriz de confusión: **TN 211 / FP 23 / FN 40 / TP 350**.

El criterio de éxito se cumple: supera el baseline de Balanced Accuracy (0.5000), recall 0.8974 > 0.5
y specificity 0.9017 > 0.5.

---

## 8. Etapas 9 a 10: consulta y pruebas

### 8.1 Consultar resultados

```powershell
neumonia evaluar
```

Muestra los artefactos guardados sin volver a entrenar y sin tocar el test.

### 8.2 Suite de pruebas

```powershell
neumonia test-suite
```

O directamente:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Estado actual: **95 passed, 109 warnings** (en torno a 17 s; el tiempo varía entre ejecuciones).
Las advertencias provienen de dependencias de terceros, no del código del proyecto.

Las pruebas verifican, entre otras cosas:

- el aislamiento estructural del test: que `construir_pipelines_datos()` no construya el pipeline de
  test salvo que se pida, y que la etapa de sensibilidad y la construcción del pipeline de COMBINADO
  rechacen el split `test`;
- que la CLI exponga exactamente las etapas del flujo, sin opciones de desbalance adicionales;
- que cada etapa escriba su checkpoint en un directorio propio;
- que los checkpoints de etapas distintas no se pisen entre sí;
- que el entrenamiento definitivo guarde el modelo explícitamente, ya que sin `validation` no hay
  `ModelCheckpoint`;
- que cada `.format()` del flujo declare exactamente las claves que le pasa (estos mensajes solo se
  ejecutan al correr el flujo real);
- que el umbral declare su propio sesgo optimista;
- que `evaluar_test` no reciba parámetros ni consulte historial, y que `neumonia test` no exponga
  banderas de confirmación.

---

## 9. Resumen de artefactos

```text
models/sensitivity_results.json                 # 21 corridas de sensibilidad (cerradas)
experiments/sensibilidad_hiperparametros/       # script del barrido + INFORME_SENSIBILIDAD.md

# COMBINADO por arquitectura (etapa 5)
results/final/combinado_vgg16_validacion.json
results/final/combinado_resnet50_validacion.json
results/final/combinado_mobilenetv2_validacion.json
results/final/combinado_vgg16_validacion.csv
results/final/combinado_resnet50_validacion.csv
results/final/combinado_mobilenetv2_validacion.csv
results/final/validacion_vgg16_y_{true,prob}.npy
results/final/validacion_resnet50_y_{true,prob}.npy
results/final/validacion_mobilenetv2_y_{true,prob}.npy
results/final/combinado/vgg16/best_model.keras
results/final/combinado/resnet50/best_model.keras
results/final/combinado/mobilenetv2/best_model.keras

# Comparación y selección (etapa 6)
results/final/comparacion_arquitecturas.json
results/final/seleccion_arquitectura.json

# Flujo estándar con arquitectura seleccionada (etapas 7-9)
results/final/combinado_validacion.json         # copiado de la arquitectura seleccionada
results/final/combinado_validacion.csv
results/final/validacion_y_{true,prob}.npy      # copiados de la arquitectura seleccionada
results/final/umbral_decision.json              # 91 umbrales, elegido 0.38
results/final/decision_modelo.json              # estrategia + config + umbral congelados
results/final/final_model/best_model.keras      # modelo definitivo (no versionado)
results/final/final_model_training.json
results/final/final_test_report.json            # informe del test final
results/final/final_test_metrics.csv            # métricas del test en CSV
results/final/test_final_confusion_matrix.png   # matriz de confusión test
results/final/test_final_roc_curve.png           # curva ROC test
results/final/test_final_metrics.png             # gráfico métricas test
results/final/test_y_true.npy                   # probabilidades del test (no versionado)
results/final/test_y_prob.npy

reports/figures/                                # figuras de EDA, augmentación, sensibilidad,
                                                # combinado_arquitecturas (6), umbral, test
```

Los `.keras` y los `.npy` están excluidos de git por tamaño; los JSON, CSV y PNG sí se versionan.

---

## 10. Errores frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `KeyError: 'learning_rate'` en el log | Lectura de la configuración anidada incorrectamente | Corregido: la configuración se lee de `configuracion["config"]` de cada entrada. |
| `KeyError: 'val'` | La comparación solo construía el pipeline de `train` | Corregido en la etapa de COMBINADO, que sí usa `train` para entrenar y `val` para medir. |
| `NameError: name 'fig' is not defined` | `guardar_curva_roc` usaba una figura sin ejes | Corregido con `fig, ax = plt.subplots(...)`. |
| `IndexError: list index out of range` | `plt.figure()` no garantiza ejes | Corregido: siempre se usan `subplots`. |
| `No existe .../decision_modelo.json` en `neumonia test` | Faltan las etapas previas | Ejecute `neumonia umbral` antes de `neumonia final` y `neumonia test`. |
| `El manifiesto declara N filas de test; se esperaban 624` | El manifiesto cambió | No se evalúa un test que no sea el original: regenere el manifiesto con `neumonia prepare`. |
| Falta el dataset | DVC no restaurado | `dvc pull data/raw/chest_xray.dvc` |
| OOM o lentitud extrema | Batch 16 con imágenes 224×224 en CPU | Es el comportamiento esperado; ver tiempos en las secciones 4 a 7. |

---

## 11. Advertencia sobre el tiempo de cómputo

Las etapas de entrenamiento se validaron **en CPU**:

| Etapa | Tiempo |
|---|---|
| 21 corridas de sensibilidad | completadas previamente (cerrada, no repetir) |
| COMBINADO VGG16 (3 épocas) | ~1500 s |
| COMBINADO ResNet50 (3 épocas) | ~1000 s |
| COMBINADO MobileNetV2 (3 épocas) | ~500 s |
| Modelo definitivo train + val (3 épocas) | ~510 s |
| Búsqueda de umbral | segundos |
| Evaluación del test | ~20 s (solo inferencia) |

Para reejecutar el flujo completo desde cero (3 arquitecturas + definitivo), contar con ~1 hora de
cómputo en CPU. No es necesario para consultar resultados: use `neumonia evaluar`.
