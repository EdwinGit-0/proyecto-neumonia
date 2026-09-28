# Análisis de sensibilidad de hiperparámetros

Generado: 2026-09-28 00:47

Experimento **aislado** realizado para responder una sola pregunta: qué efecto tiene cada hiperparámetro (learning rate, dropout y número máximo de épocas) sobre el comportamiento de VGG16, ResNet50 y MobileNetV2.

> **TEST no se carga ni se evalúa en este experimento.** Las 21 corridas se entrenan con TRAIN y se
> comparan únicamente sobre VALIDACIÓN. La única evaluación sobre TEST en el proyecto es la
> evaluación final del modelo definitivo, ejecutada una sola vez y después de cerrar todas las
> decisiones (`src/training/flujo_final.py`).

## 1. Aislamiento e integridad

Todo el experimento vive en `experiments/sensibilidad_hiperparametros/`:

```text
experiments/sensibilidad_hiperparametros/
|-- run_sensibilidad.py       # script del barrido
|-- generar_informe.py        # script de este informe
|-- INFORME_SENSIBILIDAD.md   # este informe
|-- resultados/
|   |-- sensibilidad_resultados.json  # 21 corridas de validación
|   `-- seleccion.json                # selección basada solo en validación
|-- checkpoints/              # checkpoints de las 21 corridas (no versionados)
`-- logs/sweep.log
```

Verificación de integridad ejecutada por el script (hash SHA-256 antes y después del barrido):

| Archivo del proyecto (solo lectura)                          | Sin cambios |
|--------------------------------------------------------------|:-----------:|
| `data/interim/stratified_split_train80_val20_test_original.csv` | sí |

El manifiesto de división se **leyó**, nunca se regeneró. No se escribió en `src/`, `models/`,
`reports/` ni `data/`.

## 2. Metodología

Se conservaron **todos** los elementos del pipeline del proyecto, cambiando un solo hiperparámetro por
configuración:

| Elemento | Valor conservado |
|:---|:---|
| Dataset | Chest X-Ray (Pneumonia), 5.840 imágenes utilizadas |
| División | train 4.173 / validación 1.043 (80/20 estratificada) |
| Test | 624 imágenes del test original, **no cargadas ni evaluadas aquí** |
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
* La evaluación usa `src.models.evaluation.calcular_reporte_metricas`, la misma función del proyecto y
  el mismo umbral de 0.5.
* Solo se varía **un** hiperparámetro por configuración. No se combinan valores entre sí.
* El split `test` está bloqueado a nivel de código: `crear_pipelines` rechaza cualquier split distinto de
  `train` y `val`.

Entorno de ejecución: Python 3.10.11, TensorFlow 2.15.0, GPU: no disponible (solo CPU).

## 3. Configuraciones realmente ejecutadas

**21 corridas** (3 modelos x 7 configuraciones únicas). Todas completaron el entrenamiento
completo: ninguna configuración falló, quedó pendiente o se omitió.

La configuración `(lr=1e-4, dropout=0.3, 3 epochs)` es la referencia común de los tres grupos, por lo que
se ejecutó una sola vez por modelo.

| # | Modelo | ID | Grupo | LR | Dropout | Ep. máx. | Ep. ejec. | Early stop. | LR final | Min |
|--:|:---|:---|:---|--:|--:|--:|--:|:---:|--:|--:|
| 1 | VGG16 | ref_lr1e-4_do0.3_ep3 | referencia | 0.0001 | 0.3 | 3 | 3 | no | 1e-04 | 18.4 |
| 2 | VGG16 | lr_3e-4 | learning_rate | 0.0003 | 0.3 | 3 | 3 | no | 3e-04 | 17.6 |
| 3 | VGG16 | lr_1e-3 | learning_rate | 0.001 | 0.3 | 3 | 3 | no | 1e-03 | 16.6 |
| 4 | VGG16 | do_0.2 | dropout | 0.0001 | 0.2 | 3 | 3 | no | 1e-04 | 16.5 |
| 5 | VGG16 | do_0.5 | dropout | 0.0001 | 0.5 | 3 | 3 | no | 1e-04 | 16.7 |
| 6 | VGG16 | ep_5 | epochs | 0.0001 | 0.3 | 5 | 5 | no | 1e-04 | 28.9 |
| 7 | VGG16 | ep_10 | epochs | 0.0001 | 0.3 | 10 | 10 | no | 1e-04 | 58.8 |
| 8 | ResNet50 | ref_lr1e-4_do0.3_ep3 | referencia | 0.0001 | 0.3 | 3 | 3 | no | 1e-04 | 11.8 |
| 9 | ResNet50 | lr_3e-4 | learning_rate | 0.0003 | 0.3 | 3 | 3 | no | 3e-04 | 12.1 |
| 10 | ResNet50 | lr_1e-3 | learning_rate | 0.001 | 0.3 | 3 | 3 | no | 1e-03 | 11.7 |
| 11 | ResNet50 | do_0.2 | dropout | 0.0001 | 0.2 | 3 | 3 | no | 1e-04 | 11.5 |
| 12 | ResNet50 | do_0.5 | dropout | 0.0001 | 0.5 | 3 | 3 | no | 1e-04 | 11.4 |
| 13 | ResNet50 | ep_5 | epochs | 0.0001 | 0.3 | 5 | 5 | no | 1e-04 | 19.1 |
| 14 | ResNet50 | ep_10 | epochs | 0.0001 | 0.3 | 10 | 10 | no | 1e-04 | 40.4 |
| 15 | MobileNetV2 | ref_lr1e-4_do0.3_ep3 | referencia | 0.0001 | 0.3 | 3 | 3 | no | 1e-04 | 4.9 |
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

Selección hecha **exclusivamente con validación**, con el criterio jerárquico del proyecto
(*Balanced Accuracy > ROC-AUC > F1-Score > Accuracy*). El conjunto de test no intervino en ningún momento.

| Modelo | Config. seleccionada | LR | Dropout | Ep. | Bal. Acc | ROC-AUC | F1 | Accuracy | Bal. Acc ref. | Δ |
|:---|:---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| VGG16 | **lr_1e-3** | 0.001 | 0.3 | 3 | **0.9399** | **0.9850** | **0.9476** | **0.9252** | 0.8759 | +0.0640 |
| ResNet50 | **lr_1e-3** | 0.001 | 0.3 | 3 | **0.7366** | **0.8931** | **0.8844** | **0.8226** | 0.5000 | +0.2366 |
| MobileNetV2 | **lr_1e-3** | 0.001 | 0.3 | 3 | **0.9590** | **0.9936** | **0.9760** | **0.9645** | 0.9402 | +0.0188 |

**Ganador global por validación: MobileNetV2 con la configuración `lr_1e-3`** (learning rate 0.001, dropout 0.3, 3 epochs; balanced accuracy 0.9590, ROC-AUC 0.9936, F1 0.9760).

Orden por balanced accuracy en validación:

1. MobileNetV2: 0.9590
2. VGG16: 0.9399
3. ResNet50: 0.7366

Margen entre el primero y el segundo: **0.0191**. Con la configuración de referencia el margen
era de **0.0644** (MobileNetV2 0.9402 contra VGG16 0.8759). VGG16 mejora mucho con `lr=1e-3`
(0.8759 -> 0.9399), de modo que la
ventaja de MobileNetV2 es real, pero menor de lo que sugiere la configuración de referencia.

## 6. Qué hiperparámetro pesa más

| Configuración de MobileNetV2 | Bal. Acc (validación) | Δ vs referencia |
|:---|--:|--:|
| Referencia `lr=1e-4` | 0.9402 | referencia |
| `lr=3e-4` | 0.9558 | +0.0156 |
| `dropout=0.5` | 0.9397 | -0.0006 |
| `lr=1e-3` | **0.9590** | +0.0188 |

* Sobre MobileNetV2 el factor dominante es el **learning rate**: `1e-3` (+0.0188) supera a `3e-4` (+0.0156) y a `1e-4`.
* El **dropout es casi irrelevante en MobileNetV2 y sí pesa en VGG16**: el rango de balanced accuracy entre
  `0.2`, `0.3` y `0.5` es de 0.0049 en MobileNetV2, 0.0297 en VGG16 y
  0.0006 en ResNet50. En el modelo ganador el dropout no es una dirección que
  merezca más búsqueda.
* El **número de épocas** solo mueve a los modelos que aprendían despacio: de 3 a 10 epochs la balanced
  accuracy de VGG16 pasa de 0.8759 a 0.9254 y la de
  ResNet50 de 0.5000 a 0.6881; en MobileNetV2 pasa de 0.9402 a
  0.9560, sin superar el 0.9590 que
  consigue `lr=1e-3` con solo 3 epochs.

### 6.1 Hallazgo sobre ResNet50

Con la configuración de referencia, ResNet50 clasifica **todas** las imágenes de validación como
`PNEUMONIA` (recall 1.0000, specificity 0.0000, balanced accuracy
0.5000). Ese comportamiento **no es una propiedad de la arquitectura**: con
`lr=1e-3` deja de colapsar y alcanza balanced accuracy 0.7366 (recall
0.9135, specificity 0.5597). Cualquier afirmación sobre ResNet50
debe leerse como *"con learning rate 1e-4"* y no como una limitación general del modelo.

## 7. Limitaciones de este análisis

1. **Una sola semilla (42).** No hay replicaciones, así que no se puede estimar la variabilidad del
   entrenamiento. Las diferencias pequeñas, como `lr=1e-3` frente a `lr=3e-4` en MobileNetV2
   (+0.0032), pueden ser ruido.
2. **Un factor por vez.** No se exploran interacciones, ni la combinación de `lr=1e-3` con oversampling
   o class weights, que es justamente lo que evalúa la estrategia final COMBINADO.
3. **Solo tres valores por hiperparámetro**, sin búsqueda en rejilla. No es una búsqueda exhaustiva.
4. **Rama sin tratamiento de desbalance.** El barrido se hizo sobre los tres modelos en su
   configuración de referencia, sin class weights ni oversampling.
5. **Una sola partición de validación** de 1.043 imágenes con fuerte desbalance (268 NORMAL). El balanced
   accuracy es el criterio adecuado, pero un único split sigue siendo una fuente de incertidumbre.
6. **Umbral fijo de 0.5** en todas las métricas, sin ajuste de umbral.
7. **Ninguna evaluación sobre TEST.** Por protocolo del proyecto, TEST no se consulta antes de la
   evaluación final del modelo definitivo, así que este informe no permite afirmar nada sobre test.

## 8. Conclusión

El barrido deja tres conclusiones útiles para el proyecto:

* **El learning rate es el factor que más mueve el modelo**, y su valor óptimo (`1e-3`) es el mismo que
  gana en los tres modelos. Esa es la configuración base que hereda el entrenamiento con COMBINADO.
* **El dropout y las épocas son casi irrelevantes en MobileNetV2**, el modelo ganador, y sí tienen peso en
  VGG16 y ResNet50. Son la dirección de búsqueda equivocada si se busca mejorar el ganador.
* **La ventaja de MobileNetV2 es real pero moderada**: 0.0191 de balanced accuracy sobre VGG16 con las mejores configuraciones de cada uno, no el
  contraste amplio que se obtiene comparando contra la configuración de referencia.

El criterio de selección derivado de este análisis es el mismo del proyecto y se aplica, en el flujo
final, a la comparación de hiperparámetros por arquitectura. El tratamiento del desbalance no es una
variable a comparar: el flujo final usa una estrategia única, COMBINADO, que se toma como premisa del
proyecto y no se mide frente a sin tratamiento, oversampling o class weights por separado.

