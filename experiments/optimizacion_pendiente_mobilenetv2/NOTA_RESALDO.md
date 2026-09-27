# Respaldo histórico: optimización pendiente de MobileNetV2

> Este directorio es un **registro histórico**. No forma parte del flujo vigente del proyecto
> y no debe usarse para tomar decisiones nuevas.

## Qué contiene

Una ejecución anterior de la etapa de optimización del desbalance, anterior a la incorporation
de `src/training/optimizacion_mobilenetv2.py` al flujo principal. Conserva los checkpoints de las
cinco variantes, los resultados parciales, el log y el artefacto JSON de resultados.

## Por qué se conserva

Sirve como respaldo de una segunda ejecución independiente de la misma etapa, hecha **antes** de que
el proyecto corrigiera el control de semillas. Comparada con la ejecución no reproducible
archivada en `models/historico/optimization_results_ALEATORIO_NO_CONTROLADO.json`, permite medir
cuánto variaba cada variante entre ejecuciones del mismo código (ver la sección 7 del `README.md` y
la sección "Ejecuciones anteriores de la etapa" de `references/documentacion_proyecto.md`).

Al construirse el pipeline de `tf.data` sin semilla en el `shuffle` y en las capas de augmentation,
ninguna de las dos ejecuciones es reproducible; el artefacto vigente en
`models/optimization_results.json` corresponde a la ejecución final, ya controlada.

| Variante              | Este respaldo | Ejecución no reproducible | **Ejecución final reproducible** |
| --------------------- | ------------: | ------------------------: | ------------------------------: |
| `oversampling_normal` |        0.9555 |                   0.9586 |                      **0.9607** |
| `class_weight`        |        0.9646 |                   0.9566 |                       0.9602 |
| `finetune_block16`    |        0.6791 |                   0.9674 |                       0.7344 |
| `finetune_block13`    |        0.8221 |                   0.8209 |                       0.6119 |
| `finetune_block10`    |        0.7519 |                   0.9516 |                       0.6922 |

Valores de Balanced Accuracy sobre `validation`. Las variantes con base congelada se mantienen
dentro de ~0.01 entre las dos ejecuciones no controladas; las variantes con fine-tuning llegan a
variar 0.29. En este respaldo ganó `class_weight`; en la ejecución no reproducible ganó
`finetune_block16`; en la ejecución final reproducible ganó `oversampling_normal`.

En este respaldo el `class_weight` obtuvo sobre `test` Balanced Accuracy 0.864957
(ROC-AUC 0.962514, F1 0.917476, Accuracy 0.891026; matriz `[[178, 56], [12, 378]]`). Ese número
**no es comparable** con el test final vigente (0.8372), porque procede de una ejecución cuyo
pipeline de datos no estaba controlado.

## Estructura

```text
optimizacion_pendiente_mobilenetv2/
├── run_optimizacion.py
├── checkpoints/
│   ├── oversampling_normal/best_model.keras
│   ├── class_weight/best_model.keras
│   ├── finetune_block10/best_model.keras
│   ├── finetune_block13/best_model.keras
│   └── finetune_block16/best_model.keras
├── logs/
└── resultados/
    ├── optimizacion_pendiente.json
    └── parciales/
```

## Advertencia

Los resultados de este directorio **no** deben mezclarse con los del flujo vigente ni utilizarse
para seleccionar un modelo. El flujo vigente es:

```text
src/training/sensitivity.py             -> models/sensitivity_results.json
src/training/optimizacion_mobilenetv2.py -> models/optimization_results.json
```

Ambos artefactos deben leerse junto con la sección de reproducibilidad del `README.md`: el
resultado vigente es el primero que se ejecutó con el pipeline de datos controlado
(`src/utils/reproducibility.py`).

**Este proyecto no debe utilizarse como herramienta de diagnóstico médico.**
