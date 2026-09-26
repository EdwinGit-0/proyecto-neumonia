# Revisión experimental retrospectiva: sensibilidad de hiperparámetros

Generado: 2026-09-26 02:55

Experimento **aislado** realizado para responder una sola pregunta: si se hubiera hecho un análisis básico de sensibilidad de hiperparámetros (learning rate, dropout y número máximo de épocas) **antes** de las etapas de tuning y tratamiento del desbalance, ¿habría cambiado el comportamiento de los modelos o la selección final?

Este documento **no modifica** la monografía, ni los resultados existentes, ni el código del proyecto.

## 1. Aislamiento e integridad

Todo el experimento vive en `experiments/sensibilidad_hiperparametros/`:

```text
experiments/sensibilidad_hiperparametros/
|-- run_sensibilidad.py       # script del barrido
|-- generar_informe.py        # script de este informe
|-- INFORME_SENSIBILIDAD.md   # este informe
|-- resultados/
|   |-- sensibilidad_resultados.json  # 21 corridas de validación
|   |-- seleccion.json                # selección basada solo en validación
|   `-- auditoria_test.json           # auditoría retrospectiva sobre TEST
|-- checkpoints/              # checkpoints de las 21 corridas (no versionados)
`-- logs/sweep.log
```

Verificaciones de integridad ejecutadas por el script (hashes SHA-256 antes y después):

| Archivo del proyecto (solo lectura)                                              | Sin cambios |
|--------------------------------------------------------------------------------------|:---------:|
| `data/interim/stratified_split_train80_val20_test_original.csv`                    | sí |
| `models/model_results.json`                                                       | sí |

El manifiesto de división se **leyó**, nunca se regeneró. No se escribió en `src/`, `models/`,
`reports/` ni `data/`.

## 2. Metodología

Se conservaron **todos** los elementos del pipeline original, cambiando un solo hiperparámetro por
configuración:

| Elemento | Valor conservado |
|:---|:---|
| Dataset | Chest X-Ray (Pneumonia), 5.840 imágenes utilizadas |
| División | train 4.173 / validación 1.043 (80/20 estratificada, agrupada por hash) |
| Test | 624 imágenes del test original, intactas |
| Preprocesamiento | RGB, 224x224, interpolación bilinear, float32, división por 255 |
| Augmentación (solo train) | `RandomRotation(0.05)` y `RandomZoom(0.05)` |
| Semilla | 42, fijada de nuevo antes de cada corrida |
| Base | ImageNet, convolucional congelada |
| Cabeza | GAP -> Dense(128, ReLU) -> Dropout -> Dense(1, Sigmoid) |
| Pérdida y optimizador | Binary crossentropy y Adam |
| Batch size | 16 |
| Callbacks | `ModelCheckpoint(val_loss)`, `EarlyStopping(patience=3)`, `ReduceLROnPlateau(factor 0.5, patience 2, min_lr 1e-6)` |
| Criterio de selección | Balanced Accuracy > ROC-AUC > F1-Score > Accuracy |

Decisiones metodológicas relevantes para la interpretación:

* Los pipelines de datos se **reconstruyen en cada corrida** después de fijar la semilla 42, de modo que
  todas las configuraciones parten del mismo orden de datos y del mismo estado de augmentación. Así las
  diferencias se atribuyen al hiperparámetro y no al azar del orden de los datos.
* El modelo se construye con una réplica local de `src/models/architectures.py` (misma base congelada y
  misma cabeza) con `dropout` y `learning_rate` parametrizados. **No se modificó el archivo original.**
* La evaluación usa `src.models.evaluation.calcular_reporte_metricas`, la misma función del proyecto y
  el mismo umbral de 0.5.
* Solo se varía **un** hiperparámetro por configuración. No se combinan valores entre sí.

Entorno de ejecución: Python 3.10.11, TensorFlow 2.15.0, GPU: no disponible (solo CPU).

## 3. Configuraciones realmente ejecutadas

**21 corridas** (3 modelos x 7 configuraciones únicas). Todas completaron el entrenamiento
completo: ninguna configuración falló, quedó pendiente o se omitió.

La configuración `(lr=1e-4, dropout=0.3, 3 epochs)` es a la vez la **original del proyecto** y la
referencia común de los tres grupos, por lo que se ejecutó una sola vez por modelo.

| # | Modelo | ID | Grupo | LR | Dropout | Ep. máx. | Ep. ejec. | Early stop. | LR final | Min |
|--:|:---|:---|:---|--:|--:|--:|--:|:---:|--:|--:|
| 1 | VGG16 | ref_lr1e-4_do0.3_ep3 | referencia (= original) | 0.0001 | 0.3 | 3 | 3 | no | 1e-04 | 18.4 |
| 2 | VGG16 | lr_3e-4 | learning_rate | 0.0003 | 0.3 | 3 | 3 | no | 3e-04 | 17.6 |
| 3 | VGG16 | lr_1e-3 | learning_rate | 0.001 | 0.3 | 3 | 3 | no | 1e-03 | 16.6 |
| 4 | VGG16 | do_0.2 | dropout | 0.0001 | 0.2 | 3 | 3 | no | 1e-04 | 16.5 |
| 5 | VGG16 | do_0.5 | dropout | 0.0001 | 0.5 | 3 | 3 | no | 1e-04 | 16.7 |
| 6 | VGG16 | ep_5 | epochs | 0.0001 | 0.3 | 5 | 5 | no | 1e-04 | 28.9 |
| 7 | VGG16 | ep_10 | epochs | 0.0001 | 0.3 | 10 | 10 | no | 1e-04 | 58.8 |
| 8 | ResNet50 | ref_lr1e-4_do0.3_ep3 | referencia (= original) | 0.0001 | 0.3 | 3 | 3 | no | 1e-04 | 11.8 |
| 9 | ResNet50 | lr_3e-4 | learning_rate | 0.0003 | 0.3 | 3 | 3 | no | 3e-04 | 12.1 |
| 10 | ResNet50 | lr_1e-3 | learning_rate | 0.001 | 0.3 | 3 | 3 | no | 1e-03 | 11.7 |
| 11 | ResNet50 | do_0.2 | dropout | 0.0001 | 0.2 | 3 | 3 | no | 1e-04 | 11.5 |
| 12 | ResNet50 | do_0.5 | dropout | 0.0001 | 0.5 | 3 | 3 | no | 1e-04 | 11.4 |
| 13 | ResNet50 | ep_5 | epochs | 0.0001 | 0.3 | 5 | 5 | no | 1e-04 | 19.1 |
| 14 | ResNet50 | ep_10 | epochs | 0.0001 | 0.3 | 10 | 10 | no | 1e-04 | 40.4 |
| 15 | MobileNetV2 | ref_lr1e-4_do0.3_ep3 | referencia (= original) | 0.0001 | 0.3 | 3 | 3 | no | 1e-04 | 4.9 |
| 16 | MobileNetV2 | lr_3e-4 | learning_rate | 0.0003 | 0.3 | 3 | 3 | no | 3e-04 | 5.1 |
| 17 | MobileNetV2 | lr_1e-3 | learning_rate | 0.001 | 0.3 | 3 | 3 | no | 1e-03 | 4.9 |
| 18 | MobileNetV2 | do_0.2 | dropout | 0.0001 | 0.2 | 3 | 3 | no | 1e-04 | 4.9 |
| 19 | MobileNetV2 | do_0.5 | dropout | 0.0001 | 0.5 | 3 | 3 | no | 1e-04 | 5.0 |
| 20 | MobileNetV2 | ep_5 | epochs | 0.0001 | 0.3 | 5 | 5 | no | 1e-04 | 8.1 |
| 21 | MobileNetV2 | ep_10 | epochs | 0.0001 | 0.3 | 10 | 10 | no | 1e-04 | 15.8 |

Lecturas sobre la ejecución:

* Los 21 checkpoints se evaluaron sobre **validación** (1.043 imágenes: 268 NORMAL / 775 PNEUMONIA).
* `EarlyStopping` (paciencia 3) **no se activó** en ninguna corrida: el `val_loss` siguió mejorando hasta el
  último epoch permitido en los 21 casos.
* `ReduceLROnPlateau` **no redujo** el learning rate en ninguna corrida (columna *LR final*). Esto importa
  para leer el grupo de épocas: en las corridas de 5 y 10 epochs no hubo una segunda tasa de aprendizaje
  que contaminara la comparación.
* Tiempo total de entrenamiento: 5.7 h en CPU.

## 4. Resultados de validación

Las siete métricas se calculan sobre el conjunto de validación de 1.043 imágenes. En negrita, la mejor
configuración de cada modelo según el criterio jerárquico del proyecto.

### 4.1 VGG16

| Config | LR | Drop. | Ep. | Bal. Acc | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|:---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ref_lr1e-4_do0.3_ep3 | 0.0001 | 0.3 | 3 | 0.8759 | 0.9099 | 0.9338 | 0.9458 | 0.8060 | 0.9397 | 0.9643 |
| lr_3e-4 | 0.0003 | 0.3 | 3 | 0.9308 | 0.9262 | 0.9781 | 0.9213 | 0.9403 | 0.9488 | 0.9758 |
| **lr_1e-3** | 0.001 | 0.3 | 3 | **0.9399** | **0.9252** | **0.9888** | **0.9097** | **0.9701** | **0.9476** | **0.9850** |
| do_0.2 | 0.0001 | 0.2 | 3 | 0.8808 | 0.9118 | 0.9373 | 0.9445 | 0.8172 | 0.9409 | 0.9648 |
| do_0.5 | 0.0001 | 0.5 | 3 | 0.8511 | 0.9003 | 0.9168 | 0.9523 | 0.7500 | 0.9342 | 0.9634 |
| ep_5 | 0.0001 | 0.3 | 5 | 0.9190 | 0.9195 | 0.9701 | 0.9200 | 0.9179 | 0.9444 | 0.9700 |
| ep_10 | 0.0001 | 0.3 | 10 | 0.9254 | 0.9291 | 0.9705 | 0.9329 | 0.9179 | 0.9513 | 0.9768 |

### 4.2 ResNet50

| Config | LR | Drop. | Ep. | Bal. Acc | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|:---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ref_lr1e-4_do0.3_ep3 | 0.0001 | 0.3 | 3 | 0.5000 | 0.7430 | 0.7430 | 1.0000 | 0.0000 | 0.8526 | 0.8833 |
| lr_3e-4 | 0.0003 | 0.3 | 3 | 0.6689 | 0.8073 | 0.8175 | 0.9535 | 0.3843 | 0.8803 | 0.8894 |
| **lr_1e-3** | 0.001 | 0.3 | 3 | **0.7366** | **0.8226** | **0.8571** | **0.9135** | **0.5597** | **0.8844** | **0.8931** |
| do_0.2 | 0.0001 | 0.2 | 3 | 0.4994 | 0.7421 | 0.7428 | 0.9987 | 0.0000 | 0.8520 | 0.8842 |
| do_0.5 | 0.0001 | 0.5 | 3 | 0.5000 | 0.7430 | 0.7430 | 1.0000 | 0.0000 | 0.8526 | 0.8804 |
| ep_5 | 0.0001 | 0.3 | 5 | 0.5227 | 0.7459 | 0.7520 | 0.9819 | 0.0634 | 0.8517 | 0.8872 |
| ep_10 | 0.0001 | 0.3 | 10 | 0.6881 | 0.8140 | 0.8275 | 0.9471 | 0.4291 | 0.8833 | 0.8929 |

### 4.3 MobileNetV2

| Config | LR | Drop. | Ep. | Bal. Acc | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|:---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ref_lr1e-4_do0.3_ep3 | 0.0001 | 0.3 | 3 | 0.9402 | 0.9511 | 0.9714 | 0.9626 | 0.9179 | 0.9669 | 0.9890 |
| lr_3e-4 | 0.0003 | 0.3 | 3 | 0.9558 | 0.9597 | 0.9816 | 0.9639 | 0.9478 | 0.9727 | 0.9926 |
| **lr_1e-3** | 0.001 | 0.3 | 3 | **0.9590** | **0.9645** | **0.9817** | **0.9703** | **0.9478** | **0.9760** | **0.9936** |
| do_0.2 | 0.0001 | 0.2 | 3 | 0.9446 | 0.9521 | 0.9751 | 0.9600 | 0.9291 | 0.9675 | 0.9894 |
| do_0.5 | 0.0001 | 0.5 | 3 | 0.9397 | 0.9521 | 0.9702 | 0.9652 | 0.9142 | 0.9677 | 0.9874 |
| ep_5 | 0.0001 | 0.3 | 5 | 0.9387 | 0.9578 | 0.9656 | 0.9781 | 0.8993 | 0.9718 | 0.9907 |
| ep_10 | 0.0001 | 0.3 | 10 | 0.9560 | 0.9655 | 0.9780 | 0.9755 | 0.9366 | 0.9767 | 0.9940 |

### 4.4 Lectura por hiperparámetro (balanced accuracy en validación)

Cada columna cambia **solo** el factor indicado respecto de la configuración de referencia.

| Modelo | ref (1e-4/0.3/3) | lr=3e-4 | lr=1e-3 | dropout=0.2 | dropout=0.5 | 5 epochs | 10 epochs | Rango |
|:---|--:|--:|--:|--:|--:|--:|--:|--:|
| VGG16 | 0.8759 | 0.9308 | **0.9399** | 0.8808 | 0.8511 | 0.9190 | 0.9254 | 0.0888 |
| ResNet50 | 0.5000 | 0.6689 | **0.7366** | 0.4994 | 0.5000 | 0.5227 | 0.6881 | 0.2373 |
| MobileNetV2 | 0.9402 | 0.9558 | **0.9590** | 0.9446 | 0.9397 | 0.9387 | 0.9560 | 0.0204 |

## 5. Configuración seleccionada según validación

Selección hecha **exclusivamente con validación**, con el criterio de la monografía
(*Balanced Accuracy > ROC-AUC > F1-Score > Accuracy*). El conjunto de test no intervino.

| Modelo | Config. seleccionada | LR | Dropout | Ep. | Bal. Acc | ROC-AUC | F1 | Accuracy | Bal. Acc original | Δ |
|:---|:---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| VGG16 | **lr_1e-3** | 0.001 | 0.3 | 3 | **0.9399** | **0.9850** | **0.9476** | **0.9252** | 0.8759 | +0.0640 |
| ResNet50 | **lr_1e-3** | 0.001 | 0.3 | 3 | **0.7366** | **0.8931** | **0.8844** | **0.8226** | 0.5000 | +0.2366 |
| MobileNetV2 | **lr_1e-3** | 0.001 | 0.3 | 3 | **0.9590** | **0.9936** | **0.9760** | **0.9645** | 0.9402 | +0.0188 |

**Ganador global por validación: MobileNetV2 con la configuración `lr_1e-3`** (learning rate 0.001, dropout 0.3, 3 epochs; balanced accuracy 0.9590, ROC-AUC 0.9936, F1 0.9760).

Orden por balanced accuracy en validación:

1. MobileNetV2: 0.9590
2. VGG16: 0.9399
3. ResNet50: 0.7366

Margen entre el primero y el segundo: **0.0191**. Con la configuración original el margen era de
**0.0644** (MobileNetV2 0.9402 contra VGG16 0.8759). VGG16 mejora mucho con `lr=1e-3` (0.8759 -> 0.9399), de modo que la
ventaja de MobileNetV2 es real, pero mucho menor de lo que sugiere la tabla original.

## 6. Auditoría retrospectiva sobre TEST

> **Estas métricas no participaron en ninguna decisión.** Se calcularon una vez terminada la selección
> por validación, solo para comparar qué habría ocurrido. No deben usarse para elegir modelo ni
> hiperparámetros.

Test original: 624 imágenes (234 NORMAL / 390 PNEUMONIA).

| Modelo | Config. seleccionada en val. | Bal. Acc | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|:---|:---|--:|--:|--:|--:|--:|--:|--:|
| VGG16 | lr_1e-3 | 0.8620 | 0.8862 | 0.8718 | 0.9590 | 0.7650 | 0.9133 | 0.9469 |
| ResNet50 | lr_1e-3 | 0.6568 | 0.7228 | 0.7166 | 0.9205 | 0.3932 | 0.8058 | 0.8432 |
| MobileNetV2 | lr_1e-3 | 0.7927 | 0.8413 | 0.8038 | 0.9872 | 0.5983 | 0.8861 | 0.9612 |

Matrices de confusión (TN / FP / FN / TP):

* VGG16 (`lr_1e-3`): TN 179, FP 55, FN 16, TP 374
* ResNet50 (`lr_1e-3`): TN 92, FP 142, FN 31, TP 359
* MobileNetV2 (`lr_1e-3`): TN 140, FP 94, FN 5, TP 385

## 7. Comparación con los resultados de la monografía

### 7.1 ¿Se reproduce la configuración original?

La corrida de referencia `(lr=1e-4, dropout=0.3, 3 epochs)` es exactamente la configuración original del
proyecto. Repetirla mide cuánto de la diferencia con la monografía es ruido de ejecución.

| Modelo | Métrica | Monografía | Esta corrida | Δ |
|:---|:---|--:|--:|--:|
| VGG16 | Balanced Accuracy | 0.8765 | 0.8759 | -0.0006 |
| VGG16 | ROC-AUC | 0.9643 | 0.9643 | -0.0000 |
| VGG16 | F1-Score | 0.9404 | 0.9397 | -0.0007 |
| VGG16 | Accuracy | 0.9108 | 0.9099 | -0.0010 |
| ResNet50 | Balanced Accuracy | 0.5000 | 0.5000 | +0.0000 |
| ResNet50 | ROC-AUC | 0.8878 | 0.8833 | -0.0045 |
| ResNet50 | F1-Score | 0.8526 | 0.8526 | +0.0000 |
| ResNet50 | Accuracy | 0.7430 | 0.7430 | +0.0000 |
| MobileNetV2 | Balanced Accuracy | 0.9412 | 0.9402 | -0.0009 |
| MobileNetV2 | ROC-AUC | 0.9887 | 0.9890 | +0.0003 |
| MobileNetV2 | F1-Score | 0.9613 | 0.9669 | +0.0056 |
| MobileNetV2 | Accuracy | 0.9434 | 0.9511 | +0.0077 |

Desviación máxima respecto de la monografía en las métricas del criterio: **0.0077**;
en la métrica que decide la selección (balanced accuracy) la desviación máxima es **0.0009**.
La configuración original **sí es reproducible** en este entorno: las diferencias son del orden de la
cuarta decimal y los tres modelos mantienen exactamente el mismo orden. Las diferencias de las secciones
siguientes no son, por tanto, artefactos de máquina.

### 7.2 ¿Cambiaría la selección de modelo?

**No.** El orden por validación es el mismo en los tres escenarios:

| Escenario | 1º | 2º | 3º |
|:---|:---|:---|:---|
| Monografía (configuración original) | MobileNetV2 (0.9412) | VGG16 (0.8765) | ResNet50 (0.5000) |
| Esta corrida, configuración original | MobileNetV2 (0.9402) | VGG16 (0.8759) | ResNet50 (0.5000) |
| Esta corrida, mejor configuración por modelo | MobileNetV2 (0.9590) | VGG16 (0.9399) | ResNet50 (0.7366) |

MobileNetV2 sigue ganando por balanced accuracy en validación en todos los casos. **La selección de modelo
no se habría cambiado.** Lo que cambia es la distancia entre los dos primeros: se reduce de
0.0644 a 0.0191.

### 7.3 ¿Cambiaría el hiperparámetro elegido?

| Configuración de MobileNetV2 | Bal. Acc (esta corrida) | Bal. Acc (monografía) | Δ vs original |
|:---|--:|--:|--:|
| Original `lr=1e-4` | 0.9402 | 0.9412 | referencia |
| `lr=3e-4` | 0.9558 | 0.9547 | +0.0156 |
| `dropout=0.5` | 0.9397 | 0.9423 | -0.0006 |
| `lr=1e-3` (nuevo en este análisis) | **0.9590** | no evaluado | +0.0188 |

* Sobre MobileNetV2 el factor dominante es el **learning rate**: `1e-3` (+0.0188)
  supera a `3e-4` (+0.0156) y a `1e-4`. El valor `1e-3`, que la monografía no exploró, es el
  mejor de los tres explorados.
* El **dropout es casi irrelevante en MobileNetV2 y sí pesa en VGG16**: el rango de balanced accuracy entre
  `0.2`, `0.3` y `0.5` es de 0.0049 en MobileNetV2, 0.0297 en VGG16 y
  0.0006 en ResNet50. En VGG16 y MobileNetV2 el valor `0.5` es el peor de los tres,
  mientras que en ResNet50 la diferencia es despreciable (el peor es `0.2`, con 0.4994).
  En el modelo ganador, el dropout no es una dirección que merezca más búsqueda.
* El **número de épocas** es la segunda palanca y solo para los modelos que aprendían despacio: de 3 a 10
  epochs la balanced accuracy de VGG16 pasa de 0.8759 a 0.9254 y la de
  ResNet50 de 0.5000 a 0.6881; en MobileNetV2 pasa de 0.9402 a
  0.9560, sin superar el 0.9590 que consigue `lr=1e-3` con solo 3 epochs.
* **Sí habría cambiado el valor:** la monografía terminó en `lr=3e-4` porque era el mejor de
  `{5e-5, 1e-4, 3e-4}`. Al Incorporar `1e-3`, la mejor configuración de la rama original pasaría a ser
  `lr=1e-3` (balanced accuracy 0.9590).

### 7.4 Hallazgo sobre ResNet50

Con la configuración original, ResNet50 clasifica **todas** las imágenes de validación como `PNEUMONIA`
(recall 1.0000, specificity 0.0000, balanced accuracy 0.5000), tal como describe la monografía. Ese comportamiento **no es una
propiedad de la arquitectura**: con `lr=1e-3` deja de colapsar y alcanza balanced accuracy
0.7366 (recall 0.9135, specificity 0.5597).
La afirmación de la monografía sobre ResNet50 debería leerse como *"con learning rate 1e-4"* y no como
una limitación general del modelo.

### 7.5 Contraste con los resultados de test de la monografía

> Tabla de auditoría. **No se usó para seleccionar nada.** Se incluye para responder qué habría ocurrido.

| Origen | Modelo / configuración | Balanced Accuracy | Accuracy | Recall | Specificity | F1-Score | ROC-AUC |
|:---|:---|--:|--:|--:|--:|--:|--:|
| Monografía | MobileNetV2 original `lr=1e-4` | 0.7885 | 0.8381 | 0.9872 | 0.5897 | 0.8840 | 0.9545 |
| Monografía | MobileNetV2 ajustado `lr=3e-4` | 0.7987 | 0.8478 | 0.9949 | 0.6026 | 0.8909 | 0.9633 |
| Monografía | MobileNetV2 final (oversampling) | 0.8590 | 0.8878 | 0.9744 | 0.7436 | 0.9157 | 0.9612 |
| Este análisis | MobileNetV2 `lr=1e-3` (elegido en validación) | 0.7927 | 0.8413 | 0.9872 | 0.5983 | 0.8861 | 0.9612 |
| Este análisis | VGG16 `lr=1e-3` (elegido en validación) | 0.8620 | 0.8862 | 0.9590 | 0.7650 | 0.9133 | 0.9469 |

Lecturas, con las reservas del apartado 8:

* Para MobileNetV2, cambiar el learning rate de `1e-4` a `1e-3` **no mejora el test**: balanced accuracy
  0.7927 frente a 0.7885 del original (+0.0043). La ganancia observada en
  validación (+0.0188) no se transfiere al test.
* El modelo final de la monografía (MobileNetV2 con oversampling, balanced accuracy 0.8590) sigue por encima de la mejor configuración de MobileNetV2
  encontrada aquí (0.7927). El tratamiento del desbalance aporta más que el
  ajuste del learning rate.
* **Dato incómodo:** VGG16 con `lr=1e-3` alcanza en test balanced accuracy 0.8620, por encima del modelo final elegido en la monografía (0.8590), y una accuracy prácticamente igual (0.8862 frente a 0.8878), con una especificidad mayor (0.7650 frente a 0.7436) y un ROC-AUC menor (0.9469 frente a 0.9612). En validación, en cambio,
  VGG16 `lr=1e-3` pierde con claridad frente a MobileNetV2 `lr=1e-3` (0.9399
  frente a 0.9590). El test daría un orden
  distinto. Por protocolo ese resultado **no debe usarse para cambiar la selección**: sirve como
  advertencia sobre la distancia entre validación y test en este conjunto (624 imágenes, una sola semilla).
* ResNet50 con `lr=1e-3` obtiene en test balanced accuracy 0.6568 y specificity
  0.3932, por lo que **no** cumpliría el criterio de éxito del proyecto (especificidad
  superior a 0.5).

## 8. Limitaciones de este análisis

1. **Una sola semilla (42).** No hay replicaciones, así que no se puede estimar la variabilidad del
   entrenamiento. Las diferencias pequeñas, como `lr=1e-3` frente a `lr=3e-4` en MobileNetV2
   (+0.0032), pueden ser ruido.
2. **Un factor por vez.** No se exploran interacciones. En particular, `lr=1e-3` combinado con
   oversampling, que es la combinación que produjo el mejor resultado de la monografía, no se evaluó.
3. **Solo tres valores por hiperparámetro**, sin búsqueda en rejilla. No es una búsqueda exhaustiva.
4. **Rama original únicamente.** El barrido se hizo sobre los tres modelos en su configuración original
   (sin class weights ni oversampling). No se repitió sobre la rama con oversampling, que es la que
   termina siendo la configuración final de la monografía.
5. **Una sola partición de validación** de 1.043 imágenes con fuerte desbalance (268 NORMAL). El balanced
   accuracy es el criterio adecuado, pero un único split sigue siendo una fuente de incertidumbre.
6. **Umbral fijo de 0.5** en todas las métricas, sin ajuste de umbral.

## 9. Conclusión

**¿Aporta este análisis evidencia suficiente para incorporarlo después como análisis de sensibilidad?**

Sí, con un encuadre concreto: sirve como **evidencia de robustez y de alcance de las afirmaciones**,
no como motivo para cambiar el modelo final.

A favor de incorporarlo:

* Es barato de ejecutar y fácil de reproducir: 3 modelos por 7 configuraciones, un solo factor por vez,
  reutilizando todo el pipeline y el criterio de selección del proyecto.
* Deja constancia de que la elección de MobileNetV2 no es frágil: sigue ganando por validación con
  cualquiera de los tres valores de learning rate, de dropout y de épocas probados.
* Ajusta dos afirmaciones que hoy están más fuertes que la evidencia: la ventaja de MobileNetV2 sobre
  VGG16 es de 0.0191 y no de cerca de 0.06, y el colapso de ResNet50 es efecto del learning rate y
  no de la arquitectura.
* Identifica el learning rate como el único factor que realmente mueve el modelo, lo que justifica
  documentar con más detalle la decisión de tuning que ya existe en el proyecto.

En contra de tomarlo como cambio:

* La mejora de `lr=1e-3` sobre MobileNetV2 en validación no se traslada al test, y su margen sobre
  `lr=3e-4` es pequeño frente a lo que puede producir el azar de una sola semilla. No justifica
  reemplazar la configuración actual.
* Los resultados de test apuntan en sentido contrario a la validación en el caso VGG16 frente a
  MobileNetV2. Usar eso para cambiar la selección rompería el protocolo del proyecto.

**Recomendación concreta:** incorporar el barrido como tabla de sensibilidad y robustez con las 7
configuraciones por modelo, y **mantener la configuración final actual** (MobileNetV2 con oversampling y
`lr=3e-4`). Si se quisiera una conclusión firme sobre `lr=1e-3`, el paso siguiente, barato y correcto, sería
repetir las dos o tres configuraciones candidatas de MobileNetV2 con varias semillas y repetir el par
`lr in {3e-4, 1e-3}` sobre la rama con oversampling. Nada de eso cambia el modelo final por ahora.

Queda pendiente de decidir con usted si este análisis se incorpora a la monografía y en qué sección.

