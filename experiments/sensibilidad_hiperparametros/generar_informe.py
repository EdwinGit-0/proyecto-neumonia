"""Genera el informe de la revisión retrospectiva de sensibilidad.

Solo lee resultados ya calculados:
- resultados/sensibilidad_resultados.json
- resultados/seleccion.json
- resultados/auditoria_test.json
- models/*.json (referencia de la monografía, solo lectura)

Escribe únicamente INFORME_SENSIBILIDAD.md dentro del directorio del experimento.
No modifica src/, models/, reports/, data/ ni la monografía.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DIRECTORIO = Path(__file__).resolve().parent
RAIZ_PROYECTO = DIRECTORIO.parents[1]
RESULTADOS = DIRECTORIO / "resultados"
RUTA_INFORME = DIRECTORIO / "INFORME_SENSIBILIDAD.md"

MODELOS = ("VGG16", "ResNet50", "MobileNetV2")
ETIQUETAS = {
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1-Score",
    "roc_auc": "ROC-AUC",
}
ORDEN_CRITERIO = ("balanced_accuracy", "roc_auc", "f1", "accuracy")


def cargar(ruta: Path) -> Any:
    """Cargar un JSON."""
    return json.loads(ruta.read_text(encoding="utf-8"))


def f4(valor: float) -> str:
    """Formatear un número a cuatro decimales."""
    return f"{float(valor):.4f}"


def tabla(columnas: list[str], alineaciones: list[str]) -> list[str]:
    """Construir las dos primeras líneas de una tabla markdown."""
    return ["| " + " | ".join(columnas) + " |", "|" + "|".join(alineaciones) + "|"]


def fila(valores: list[str]) -> str:
    """Construir una fila de tabla markdown."""
    return "| " + " | ".join(valores) + " |"


def main() -> None:
    """Generar el informe markdown completo."""
    datos = cargar(RESULTADOS / "sensibilidad_resultados.json")
    seleccion = cargar(RESULTADOS / "seleccion.json")
    auditoria = cargar(RESULTADOS / "auditoria_test.json")

    monografia_modelos = cargar(RAIZ_PROYECTO / "models" / "model_results.json")
    monografia_tuning = cargar(RAIZ_PROYECTO / "models" / "mobilenetv2_tuning_results.json")
    monografia_desbalance = cargar(RAIZ_PROYECTO / "models" / "mobilenetv2_desbalance_results.json")

    ref_hash = next(iter(datos["referencia_monografia_sha256"].values()))
    configs = {c["id"]: c for c in datos["configuraciones"]}
    lineas: list[str] = []
    anadir = lineas.append

    def validacion(modelo: str, config_id: str) -> dict[str, Any]:
        """Métricas de validación de una configuración."""
        return datos["resultados"][f"{modelo}/{config_id}"]["validation"]

    # ------------------------------------------------------------------ 1
    anadir("# Revisión experimental retrospectiva: sensibilidad de hiperparámetros")
    anadir("")
    anadir(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    anadir("")
    anadir(
        "Experimento **aislado** realizado para responder una sola pregunta: si se hubiera hecho un análisis "
        "básico de sensibilidad de hiperparámetros (learning rate, dropout y número máximo de épocas) "
        "**antes** de las etapas de tuning y tratamiento del desbalance, ¿habría cambiado el comportamiento "
        "de los modelos o la selección final?"
    )
    anadir("")
    anadir("Este documento **no modifica** la monografía, ni los resultados existentes, ni el código del proyecto.")
    anadir("")

    # ------------------------------------------------------------------ 2
    anadir("## 1. Aislamiento e integridad")
    anadir("")
    anadir("Todo el experimento vive en `experiments/sensibilidad_hiperparametros/`:")
    anadir("")
    anadir("```text")
    anadir("experiments/sensibilidad_hiperparametros/")
    anadir("|-- run_sensibilidad.py       # script del barrido")
    anadir("|-- generar_informe.py        # script de este informe")
    anadir("|-- INFORME_SENSIBILIDAD.md   # este informe")
    anadir("|-- resultados/")
    anadir("|   |-- sensibilidad_resultados.json  # 21 corridas de validación")
    anadir("|   |-- seleccion.json                # selección basada solo en validación")
    anadir("|   `-- auditoria_test.json           # auditoría retrospectiva sobre TEST")
    anadir("|-- checkpoints/              # checkpoints de las 21 corridas (no versionados)")
    anadir("`-- logs/sweep.log")
    anadir("```")
    anadir("")
    anadir("Verificaciones de integridad ejecutadas por el script (hashes SHA-256 antes y después):")
    anadir("")
    anadir("| Archivo del proyecto (solo lectura)                                              | Sin cambios |")
    anadir("|--------------------------------------------------------------------------------------|:---------:|")
    anadir(f"| `data/interim/stratified_split_train80_val20_test_original.csv`                    | {'sí' if datos['manifiesto']['sin_modificar'] else 'NO'} |")
    anadir(f"| `models/model_results.json`                                                       | {'sí' if ref_hash['sin_modificar'] else 'NO'} |")
    anadir("")
    anadir("El manifiesto de división se **leyó**, nunca se regeneró. No se escribió en `src/`, `models/`,")
    anadir("`reports/` ni `data/`.")
    anadir("")

    # ------------------------------------------------------------------ 3
    entorno = datos["entorno"]
    anadir("## 2. Metodología")
    anadir("")
    anadir("Se conservaron **todos** los elementos del pipeline original, cambiando un solo hiperparámetro por")
    anadir("configuración:")
    anadir("")
    lineas.extend(tabla(
        ["Elemento", "Valor conservado"],
        [":---", ":---"],
    ))
    for etiqueta, valor in [
        ("Dataset", "Chest X-Ray (Pneumonia), 5.840 imágenes utilizadas"),
        ("División", "train 4.173 / validación 1.043 (80/20 estratificada, agrupada por hash)"),
        ("Test", "624 imágenes del test original, intactas"),
        ("Preprocesamiento", "RGB, 224x224, interpolación bilinear, float32, división por 255"),
        ("Augmentación (solo train)", "`RandomRotation(0.05)` y `RandomZoom(0.05)`"),
        ("Semilla", "42, fijada de nuevo antes de cada corrida"),
        ("Base", "ImageNet, convolucional congelada"),
        ("Cabeza", "GAP -> Dense(128, ReLU) -> Dropout -> Dense(1, Sigmoid)"),
        ("Pérdida y optimizador", "Binary crossentropy y Adam"),
        ("Batch size", "16"),
        ("Callbacks", "`ModelCheckpoint(val_loss)`, `EarlyStopping(patience=3)`, `ReduceLROnPlateau(factor 0.5, patience 2, min_lr 1e-6)`"),
        ("Criterio de selección", "Balanced Accuracy > ROC-AUC > F1-Score > Accuracy"),
    ]:
        anadir(fila([etiqueta, valor]))
    anadir("")
    anadir("Decisiones metodológicas relevantes para la interpretación:")
    anadir("")
    anadir("* Los pipelines de datos se **reconstruyen en cada corrida** después de fijar la semilla 42, de modo que")
    anadir("  todas las configuraciones parten del mismo orden de datos y del mismo estado de augmentación. Así las")
    anadir("  diferencias se atribuyen al hiperparámetro y no al azar del orden de los datos.")
    anadir("* El modelo se construye con una réplica local de `src/models/architectures.py` (misma base congelada y")
    anadir("  misma cabeza) con `dropout` y `learning_rate` parametrizados. **No se modificó el archivo original.**")
    anadir("* La evaluación usa `src.models.evaluation.calcular_reporte_metricas`, la misma función del proyecto y")
    anadir("  el mismo umbral de 0.5.")
    anadir("* Solo se varía **un** hiperparámetro por configuración. No se combinan valores entre sí.")
    anadir("")
    anadir(f"Entorno de ejecución: Python {entorno['python']}, TensorFlow {entorno['tensorflow']}, "
           f"GPU: {', '.join(entorno['gpus']) if entorno['gpus'] else 'no disponible (solo CPU)'}.")
    anadir("")

    # ------------------------------------------------------------------ 4
    total_corridas = len(datos["resultados"])
    anadir("## 3. Configuraciones realmente ejecutadas")
    anadir("")
    anadir(f"**{total_corridas} corridas** (3 modelos x 7 configuraciones únicas). Todas completaron el entrenamiento")
    anadir("completo: ninguna configuración falló, quedó pendiente o se omitió.")
    anadir("")
    anadir("La configuración `(lr=1e-4, dropout=0.3, 3 epochs)` es a la vez la **original del proyecto** y la")
    anadir("referencia común de los tres grupos, por lo que se ejecutó una sola vez por modelo.")
    anadir("")
    lineas.extend(tabla(
        ["#", "Modelo", "ID", "Grupo", "LR", "Dropout", "Ep. máx.", "Ep. ejec.", "Early stop.", "LR final", "Min"],
        ["--:", ":---", ":---", ":---", "--:", "--:", "--:", "--:", ":---:", "--:", "--:"],
    ))
    indice = 0
    for modelo in MODELOS:
        for config_id, config in configs.items():
            indice += 1
            registro = datos["resultados"][f"{modelo}/{config_id}"]
            grupo = "referencia (= original)" if registro["es_referencia"] else config["grupo"]
            anadir(fila([
                str(indice),
                modelo,
                config_id,
                grupo,
                f"{config['learning_rate']:g}",
                f"{config['dropout']:g}",
                str(config["epochs"]),
                str(registro["epochs_ejecutadas"]),
                "no" if not registro["early_stopping"] else "sí",
                f"{registro['learning_rate_final']:.0e}",
                f"{registro['segundos_entrenamiento'] / 60:.1f}",
            ]))
    anadir("")
    anadir("Lecturas sobre la ejecución:")
    anadir("")
    anadir("* Los 21 checkpoints se evaluaron sobre **validación** (1.043 imágenes: 268 NORMAL / 775 PNEUMONIA).")
    anadir("* `EarlyStopping` (paciencia 3) **no se activó** en ninguna corrida: el `val_loss` siguió mejorando hasta el")
    anadir("  último epoch permitido en los 21 casos.")
    anadir("* `ReduceLROnPlateau` **no redujo** el learning rate en ninguna corrida (columna *LR final*). Esto importa")
    anadir("  para leer el grupo de épocas: en las corridas de 5 y 10 epochs no hubo una segunda tasa de aprendizaje")
    anadir("  que contaminara la comparación.")
    anadir(f"* Tiempo total de entrenamiento: "
           f"{sum(r['segundos_entrenamiento'] for r in datos['resultados'].values()) / 3600:.1f} h en CPU.")
    anadir("")

    # ------------------------------------------------------------------ 5
    anadir("## 4. Resultados de validación")
    anadir("")
    anadir("Las siete métricas se calculan sobre el conjunto de validación de 1.043 imágenes. En negrita, la mejor")
    anadir("configuración de cada modelo según el criterio jerárquico del proyecto.")
    anadir("")
    for indice_modelo, modelo in enumerate(MODELOS, start=1):
        anadir(f"### 4.{indice_modelo} {modelo}")
        anadir("")
        ganador = seleccion[modelo]["ganador"]
        lineas.extend(tabla(
            ["Config", "LR", "Drop.", "Ep.", "Bal. Acc", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"],
            [":---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"],
        ))
        for config_id in configs:
            registro = datos["resultados"][f"{modelo}/{config_id}"]
            v = registro["validation"]
            c = registro["config"]
            negrita = "**" if config_id == ganador else ""
            anadir(fila([
                f"{negrita}{config_id}{negrita}",
                f"{c['learning_rate']:g}",
                f"{c['dropout']:g}",
                str(c["epochs_max"]),
                f"{negrita}{f4(v['balanced_accuracy'])}{negrita}",
                f"{negrita}{f4(v['accuracy'])}{negrita}",
                f"{negrita}{f4(v['precision'])}{negrita}",
                f"{negrita}{f4(v['recall'])}{negrita}",
                f"{negrita}{f4(v['specificity'])}{negrita}",
                f"{negrita}{f4(v['f1'])}{negrita}",
                f"{negrita}{f4(v['roc_auc'])}{negrita}",
            ]))
        anadir("")

    anadir("### 4.4 Lectura por hiperparámetro (balanced accuracy en validación)")
    anadir("")
    anadir("Cada columna cambia **solo** el factor indicado respecto de la configuración de referencia.")
    anadir("")
    columnas_lectura = ["ref (1e-4/0.3/3)", "lr=3e-4", "lr=1e-3", "dropout=0.2", "dropout=0.5", "5 epochs", "10 epochs"]
    claves_lectura = ["ref_lr1e-4_do0.3_ep3", "lr_3e-4", "lr_1e-3", "do_0.2", "do_0.5", "ep_5", "ep_10"]
    lineas.extend(tabla(
        ["Modelo"] + columnas_lectura + ["Rango"],
        [":---"] + ["--:"] * len(columnas_lectura) + ["--:"],
    ))
    for modelo in MODELOS:
        valores = [validacion(modelo, clave)["balanced_accuracy"] for clave in claves_lectura]
        mejor = max(valores)
        celdas = [modelo] + [
            f"**{f4(valor)}**" if abs(valor - mejor) < 1e-12 else f4(valor) for valor in valores
        ]
        celdas.append(f4(max(valores) - min(valores)))
        anadir(fila(celdas))
    anadir("")

    # ------------------------------------------------------------------ 6
    anadir("## 5. Configuración seleccionada según validación")
    anadir("")
    anadir("Selección hecha **exclusivamente con validación**, con el criterio de la monografía")
    anadir("(*Balanced Accuracy > ROC-AUC > F1-Score > Accuracy*). El conjunto de test no intervino.")
    anadir("")
    lineas.extend(tabla(
        ["Modelo", "Config. seleccionada", "LR", "Dropout", "Ep.", "Bal. Acc", "ROC-AUC", "F1", "Accuracy", "Bal. Acc original", "Δ"],
        [":---", ":---", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:", "--:"],
    ))
    for modelo in MODELOS:
        info = seleccion[modelo]
        cfg = info["config"]
        v = info["validation"]
        base = info["referencia_original"]["validation"]
        anadir(fila([
            modelo,
            f"**{info['ganador']}**",
            f"{cfg['learning_rate']:g}",
            f"{cfg['dropout']:g}",
            str(cfg["epochs_max"]),
            f"**{f4(v['balanced_accuracy'])}**",
            f"**{f4(v['roc_auc'])}**",
            f"**{f4(v['f1'])}**",
            f"**{f4(v['accuracy'])}**",
            f4(base["balanced_accuracy"]),
            f"{v['balanced_accuracy'] - base['balanced_accuracy']:+.4f}",
        ]))
    anadir("")
    ganador_global = seleccion["ganador_global"]
    vg = ganador_global["validation_por_modelo"]
    cfg_ganadora = datos["resultados"][f"{ganador_global['modelo']}/{ganador_global['config_id']}"]["config"]
    anadir(f"**Ganador global por validación: {ganador_global['modelo']} con la configuración "
           f"`{ganador_global['config_id']}`** (learning rate {cfg_ganadora['learning_rate']:g}, dropout "
           f"{cfg_ganadora['dropout']:g}, {cfg_ganadora['epochs_max']} epochs; balanced accuracy "
           f"{f4(vg['balanced_accuracy'])}, ROC-AUC {f4(vg['roc_auc'])}, F1 {f4(vg['f1'])}).")
    anadir("")
    ranking = sorted(((m, seleccion[m]["validation"]["balanced_accuracy"]) for m in MODELOS), key=lambda par: -par[1])
    ref_ranking = sorted(
        ((m, seleccion[m]["referencia_original"]["validation"]["balanced_accuracy"]) for m in MODELOS),
        key=lambda par: -par[1],
    )
    anadir("Orden por balanced accuracy en validación:")
    anadir("")
    for posicion, (modelo, valor) in enumerate(ranking, start=1):
        anadir(f"{posicion}. {modelo}: {f4(valor)}")
    anadir("")
    margen = ranking[0][1] - ranking[1][1]
    margen_original = ref_ranking[0][1] - ref_ranking[1][1]
    anadir(f"Margen entre el primero y el segundo: **{margen:.4f}**. Con la configuración original el margen era de")
    anadir(f"**{margen_original:.4f}** ({ref_ranking[0][0]} {f4(ref_ranking[0][1])} contra {ref_ranking[1][0]} "
           f"{f4(ref_ranking[1][1])}). VGG16 mejora mucho con `lr=1e-3` "
           f"({f4(ref_ranking[1][1])} -> {f4(seleccion['VGG16']['validation']['balanced_accuracy'])}), de modo que la")
    anadir("ventaja de MobileNetV2 es real, pero mucho menor de lo que sugiere la tabla original.")
    anadir("")

    # ------------------------------------------------------------------ 7
    anadir("## 6. Auditoría retrospectiva sobre TEST")
    anadir("")
    anadir("> **Estas métricas no participaron en ninguna decisión.** Se calcularon una vez terminada la selección")
    anadir("> por validación, solo para comparar qué habría ocurrido. No deben usarse para elegir modelo ni")
    anadir("> hiperparámetros.")
    anadir("")
    anadir(f"Test original: {auditoria['n_test']} imágenes "
           f"({auditoria['clases_test']['NORMAL']} NORMAL / {auditoria['clases_test']['PNEUMONIA']} PNEUMONIA).")
    anadir("")
    lineas.extend(tabla(
        ["Modelo", "Config. seleccionada en val.", "Bal. Acc", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"],
        [":---", ":---", "--:", "--:", "--:", "--:", "--:", "--:", "--:"],
    ))
    for modelo in MODELOS:
        entrada = auditoria["modelos"][modelo]
        t = entrada["test"]
        anadir(fila([
            modelo,
            entrada["config_id"],
            f4(t["balanced_accuracy"]),
            f4(t["accuracy"]),
            f4(t["precision"]),
            f4(t["recall"]),
            f4(t["specificity"]),
            f4(t["f1"]),
            f4(t["roc_auc"]),
        ]))
    anadir("")
    anadir("Matrices de confusión (TN / FP / FN / TP):")
    anadir("")
    for modelo in MODELOS:
        mc = auditoria["modelos"][modelo]["test"]["matriz_confusion"]
        anadir(f"* {modelo} (`{auditoria['modelos'][modelo]['config_id']}`): TN {mc['tn']}, FP {mc['fp']}, FN {mc['fn']}, TP {mc['tp']}")
    anadir("")

    # ------------------------------------------------------------------ 8
    anadir("## 7. Comparación con los resultados de la monografía")
    anadir("")

    anadir("### 7.1 ¿Se reproduce la configuración original?")
    anadir("")
    anadir("La corrida de referencia `(lr=1e-4, dropout=0.3, 3 epochs)` es exactamente la configuración original del")
    anadir("proyecto. Repetirla mide cuánto de la diferencia con la monografía es ruido de ejecución.")
    anadir("")
    lineas.extend(tabla(
        ["Modelo", "Métrica", "Monografía", "Esta corrida", "Δ"],
        [":---", ":---", "--:", "--:", "--:"],
    ))
    deltas = []
    for modelo in MODELOS:
        mono = monografia_modelos["models"][modelo]["validation"]
        ref = seleccion[modelo]["referencia_original"]["validation"]
        for metrica in ORDEN_CRITERIO:
            deltas.append(abs(ref[metrica] - mono[metrica]))
            anadir(fila([
                modelo,
                ETIQUETAS[metrica],
                f4(mono[metrica]),
                f4(ref[metrica]),
                f"{ref[metrica] - mono[metrica]:+.4f}",
            ]))
    anadir("")
    max_delta = max(deltas)
    deltas_bal = [
        abs(seleccion[m]["referencia_original"]["validation"]["balanced_accuracy"]
            - monografia_modelos["models"][m]["validation"]["balanced_accuracy"])
        for m in MODELOS
    ]
    anadir(f"Desviación máxima respecto de la monografía en las métricas del criterio: **{max_delta:.4f}**;")
    anadir(f"en la métrica que decide la selección (balanced accuracy) la desviación máxima es **{max(deltas_bal):.4f}**.")
    anadir("La configuración original **sí es reproducible** en este entorno: las diferencias son del orden de la")
    anadir("cuarta decimal y los tres modelos mantienen exactamente el mismo orden. Las diferencias de las secciones")
    anadir("siguientes no son, por tanto, artefactos de máquina.")
    anadir("")

    anadir("### 7.2 ¿Cambiaría la selección de modelo?")
    anadir("")
    anadir("**No.** El orden por validación es el mismo en los tres escenarios:")
    anadir("")
    lineas.extend(tabla(["Escenario", "1º", "2º", "3º"], [":---", ":---", ":---", ":---"]))
    mono_val = {m: monografia_modelos["models"][m]["validation"]["balanced_accuracy"] for m in MODELOS}
    mono_orden = sorted(MODELOS, key=lambda m: -mono_val[m])
    anadir(fila(["Monografía (configuración original)"] + [f"{m} ({f4(mono_val[m])})" for m in mono_orden]))
    anadir(fila(["Esta corrida, configuración original"] + [f"{m} ({f4(v)})" for m, v in ref_ranking]))
    anadir(fila(["Esta corrida, mejor configuración por modelo"] + [f"{m} ({f4(v)})" for m, v in ranking]))
    anadir("")
    anadir("MobileNetV2 sigue ganando por balanced accuracy en validación en todos los casos. **La selección de modelo")
    anadir("no se habría cambiado.** Lo que cambia es la distancia entre los dos primeros: se reduce de")
    anadir(f"{margen_original:.4f} a {margen:.4f}.")
    anadir("")

    anadir("### 7.3 ¿Cambiaría el hiperparámetro elegido?")
    anadir("")
    lineas.extend(tabla(
        ["Configuración de MobileNetV2", "Bal. Acc (esta corrida)", "Bal. Acc (monografía)", "Δ vs original"],
        [":---", "--:", "--:", "--:"],
    ))
    tuning = {r["experimento"]: r for r in monografia_tuning["experimentos"]}
    ref_mn = validacion("MobileNetV2", "ref_lr1e-4_do0.3_ep3")["balanced_accuracy"]
    val_3e4 = validacion("MobileNetV2", "lr_3e-4")["balanced_accuracy"]
    val_do5 = validacion("MobileNetV2", "do_0.5")["balanced_accuracy"]
    val_1e3 = validacion("MobileNetV2", "lr_1e-3")["balanced_accuracy"]
    anadir(fila(["Original `lr=1e-4`", f4(ref_mn), f4(monografia_tuning["baseline"]["balanced_accuracy"]), "referencia"]))
    anadir(fila(["`lr=3e-4`", f4(val_3e4), f4(tuning["lr_3e-4"]["balanced_accuracy"]), f"{val_3e4 - ref_mn:+.4f}"]))
    anadir(fila(["`dropout=0.5`", f4(val_do5), f4(tuning["dropout_0.5"]["balanced_accuracy"]), f"{val_do5 - ref_mn:+.4f}"]))
    anadir(fila(["`lr=1e-3` (nuevo en este análisis)", f"**{f4(val_1e3)}**", "no evaluado", f"{val_1e3 - ref_mn:+.4f}"]))
    anadir("")
    anadir("* Sobre MobileNetV2 el factor dominante es el **learning rate**: `1e-3` (" + f"{val_1e3 - ref_mn:+.4f}" + ")")
    anadir(f"  supera a `3e-4` ({val_3e4 - ref_mn:+.4f}) y a `1e-4`. El valor `1e-3`, que la monografía no exploró, es el")
    anadir("  mejor de los tres explorados.")
    rango_dropout = {
        m: (
            max(validacion(m, clave)["balanced_accuracy"] for clave in ("ref_lr1e-4_do0.3_ep3", "do_0.2", "do_0.5"))
            - min(validacion(m, clave)["balanced_accuracy"] for clave in ("ref_lr1e-4_do0.3_ep3", "do_0.2", "do_0.5"))
        )
        for m in MODELOS
    }
    anadir(f"* El **dropout es casi irrelevante en MobileNetV2 y sí pesa en VGG16**: el rango de balanced accuracy entre")
    anadir(f"  `0.2`, `0.3` y `0.5` es de {rango_dropout['MobileNetV2']:.4f} en MobileNetV2, "
           f"{rango_dropout['VGG16']:.4f} en VGG16 y")
    anadir(f"  {rango_dropout['ResNet50']:.4f} en ResNet50, donde el valor `0.5` es el peor de los tres, mientras que")
    anadir("  en ResNet50 la diferencia es despreciable.")
    anadir("  En el modelo ganador, el dropout no es una dirección que merezca más búsqueda.")
    anadir("* El **número de épocas** es la segunda palanca y solo para los modelos que aprendían despacio: de 3 a 10")
    anadir(f"  epochs la balanced accuracy de VGG16 pasa de {validacion('VGG16', 'ref_lr1e-4_do0.3_ep3')['balanced_accuracy']:.4f} a "
           f"{validacion('VGG16', 'ep_10')['balanced_accuracy']:.4f} y la de")
    anadir(f"  ResNet50 de {validacion('ResNet50', 'ref_lr1e-4_do0.3_ep3')['balanced_accuracy']:.4f} a "
           f"{validacion('ResNet50', 'ep_10')['balanced_accuracy']:.4f}; en MobileNetV2 pasa de "
           f"{ref_mn:.4f} a")
    anadir(f"  {validacion('MobileNetV2', 'ep_10')['balanced_accuracy']:.4f}, sin superar el {val_1e3:.4f} que consigue "
           f"`lr=1e-3` con solo 3 epochs.")
    anadir(f"* **Sí habría cambiado el valor:** la monografía terminó en `lr=3e-4` porque era el mejor de")
    anadir("  `{5e-5, 1e-4, 3e-4}`. Al Incorporar `1e-3`, la mejor configuración de la rama original pasaría a ser")
    anadir(f"  `lr=1e-3` (balanced accuracy {f4(val_1e3)}).")
    anadir("")

    anadir("### 7.4 Hallazgo sobre ResNet50")
    anadir("")
    ref_rn = validacion("ResNet50", "ref_lr1e-4_do0.3_ep3")
    sel_rn = seleccion["ResNet50"]["validation"]
    anadir("Con la configuración original, ResNet50 clasifica **todas** las imágenes de validación como `PNEUMONIA`")
    anadir(f"(recall {f4(ref_rn['recall'])}, specificity {f4(ref_rn['specificity'])}, balanced accuracy "
           f"{f4(ref_rn['balanced_accuracy'])}), tal como describe la monografía. Ese comportamiento **no es una")
    anadir("propiedad de la arquitectura**: con `lr=1e-3` deja de colapsar y alcanza balanced accuracy")
    anadir(f"{f4(sel_rn['balanced_accuracy'])} (recall {f4(sel_rn['recall'])}, specificity {f4(sel_rn['specificity'])}).")
    anadir("La afirmación de la monografía sobre ResNet50 debería leerse como *\"con learning rate 1e-4\"* y no como")
    anadir("una limitación general del modelo.")
    anadir("")

    anadir("### 7.5 Contraste con los resultados de test de la monografía")
    anadir("")
    anadir("> Tabla de auditoría. **No se usó para seleccionar nada.** Se incluye para responder qué habría ocurrido.")
    anadir("")
    mono_test_original = monografia_modelos["final_test"]
    mono_test_ajustado = monografia_tuning["final_test"]
    mono_test_final = monografia_desbalance["final_test"]
    aud_mn = auditoria["modelos"]["MobileNetV2"]["test"]
    aud_vgg = auditoria["modelos"]["VGG16"]["test"]
    aud_rn = auditoria["modelos"]["ResNet50"]["test"]
    columnas_test = ("balanced_accuracy", "accuracy", "recall", "specificity", "f1", "roc_auc")
    lineas.extend(tabla(
        ["Origen", "Modelo / configuración"] + [ETIQUETAS[m] for m in columnas_test],
        [":---", ":---"] + ["--:"] * len(columnas_test),
    ))
    comparativas_test = [
        ("Monografía", "MobileNetV2 original `lr=1e-4`", mono_test_original),
        ("Monografía", "MobileNetV2 ajustado `lr=3e-4`", mono_test_ajustado),
        ("Monografía", "MobileNetV2 final (oversampling)", mono_test_final),
        ("Este análisis", "MobileNetV2 `lr=1e-3` (elegido en validación)", aud_mn),
        ("Este análisis", "VGG16 `lr=1e-3` (elegido en validación)", aud_vgg),
    ]
    for etiqueta, descripcion, valores in comparativas_test:
        anadir(fila([etiqueta, descripcion] + [f4(valores[m]) for m in columnas_test]))
    anadir("")
    anadir("Lecturas, con las reservas del apartado 8:")
    anadir("")
    anadir(f"* Para MobileNetV2, cambiar el learning rate de `1e-4` a `1e-3` **no mejora el test**: balanced accuracy")
    anadir(f"  {f4(aud_mn['balanced_accuracy'])} frente a {f4(mono_test_original['balanced_accuracy'])} del original "
           f"({aud_mn['balanced_accuracy'] - mono_test_original['balanced_accuracy']:+.4f}). La ganancia observada en")
    anadir(f"  validación ({val_1e3 - ref_mn:+.4f}) no se transfiere al test.")
    anadir(f"* El modelo final de la monografía (MobileNetV2 con oversampling, balanced accuracy "
           f"{f4(mono_test_final['balanced_accuracy'])}) sigue por encima de la mejor configuración de MobileNetV2")
    anadir(f"  encontrada aquí ({f4(aud_mn['balanced_accuracy'])}). El tratamiento del desbalance aporta más que el")
    anadir("  ajuste del learning rate.")
    anadir(f"* **Dato incómodo:** VGG16 con `lr=1e-3` alcanza en test balanced accuracy {f4(aud_vgg['balanced_accuracy'])}, "
           f"por encima del modelo final elegido en la monografía ({f4(mono_test_final['balanced_accuracy'])}), y una "
           f"accuracy prácticamente igual ({f4(aud_vgg['accuracy'])} frente a {f4(mono_test_final['accuracy'])}), con una "
           f"especificidad mayor ({f4(aud_vgg['specificity'])} frente a {f4(mono_test_final['specificity'])}) y un ROC-AUC "
           f"menor ({f4(aud_vgg['roc_auc'])} frente a {f4(mono_test_final['roc_auc'])}). En validación, en cambio,")
    anadir(f"  VGG16 `lr=1e-3` pierde con claridad frente a MobileNetV2 `lr=1e-3` ({f4(seleccion['VGG16']['validation']['balanced_accuracy'])}")
    anadir(f"  frente a {f4(seleccion['MobileNetV2']['validation']['balanced_accuracy'])}). El test daría un orden")
    anadir("  distinto. Por protocolo ese resultado **no debe usarse para cambiar la selección**: sirve como")
    anadir("  advertencia sobre la distancia entre validación y test en este conjunto (624 imágenes, una sola semilla).")
    anadir(f"* ResNet50 con `lr=1e-3` obtiene en test balanced accuracy {f4(aud_rn['balanced_accuracy'])} y specificity")
    anadir(f"  {f4(aud_rn['specificity'])}, por lo que **no** cumpliría el criterio de éxito del proyecto (especificidad")
    anadir("  superior a 0.5).")
    anadir("")

    # ------------------------------------------------------------------ 9
    anadir("## 8. Limitaciones de este análisis")
    anadir("")
    anadir(f"1. **Una sola semilla (42).** No hay replicaciones, así que no se puede estimar la variabilidad del")
    anadir("   entrenamiento. Las diferencias pequeñas, como `lr=1e-3` frente a `lr=3e-4` en MobileNetV2")
    anadir(f"   ({val_1e3 - val_3e4:+.4f}), pueden ser ruido.")
    anadir("2. **Un factor por vez.** No se exploran interacciones. En particular, `lr=1e-3` combinado con")
    anadir("   oversampling, que es la combinación que produjo el mejor resultado de la monografía, no se evaluó.")
    anadir("3. **Solo tres valores por hiperparámetro**, sin búsqueda en rejilla. No es una búsqueda exhaustiva.")
    anadir("4. **Rama original únicamente.** El barrido se hizo sobre los tres modelos en su configuración original")
    anadir("   (sin class weights ni oversampling). No se repitió sobre la rama con oversampling, que es la que")
    anadir("   termina siendo la configuración final de la monografía.")
    anadir("5. **Una sola partición de validación** de 1.043 imágenes con fuerte desbalance (268 NORMAL). El balanced")
    anadir("   accuracy es el criterio adecuado, pero un único split sigue siendo una fuente de incertidumbre.")
    anadir("6. **Umbral fijo de 0.5** en todas las métricas, sin ajuste de umbral.")
    anadir("")

    # ------------------------------------------------------------------ 10
    anadir("## 9. Conclusión")
    anadir("")
    anadir("**¿Aporta este análisis evidencia suficiente para incorporarlo después como análisis de sensibilidad?**")
    anadir("")
    anadir("Sí, con un encuadre concreto: sirve como **evidencia de robustez y de alcance de las afirmaciones**,")
    anadir("no como motivo para cambiar el modelo final.")
    anadir("")
    anadir("A favor de incorporarlo:")
    anadir("")
    anadir("* Es barato de ejecutar y fácil de reproducir: 3 modelos por 7 configuraciones, un solo factor por vez,")
    anadir("  reutilizando todo el pipeline y el criterio de selección del proyecto.")
    anadir("* Deja constancia de que la elección de MobileNetV2 no es frágil: sigue ganando por validación con")
    anadir("  cualquiera de los tres valores de learning rate, de dropout y de épocas probados.")
    anadir(f"* Ajusta dos afirmaciones que hoy están más fuertes que la evidencia: la ventaja de MobileNetV2 sobre")
    anadir(f"  VGG16 es de {margen:.4f} y no de cerca de 0.06, y el colapso de ResNet50 es efecto del learning rate y")
    anadir("  no de la arquitectura.")
    anadir("* Identifica el learning rate como el único factor que realmente mueve el modelo, lo que justifica")
    anadir("  documentar con más detalle la decisión de tuning que ya existe en el proyecto.")
    anadir("")
    anadir("En contra de tomarlo como cambio:")
    anadir("")
    anadir("* La mejora de `lr=1e-3` sobre MobileNetV2 en validación no se traslada al test, y su margen sobre")
    anadir("  `lr=3e-4` es pequeño frente a lo que puede producir el azar de una sola semilla. No justifica")
    anadir("  reemplazar la configuración actual.")
    anadir("* Los resultados de test apuntan en sentido contrario a la validación en el caso VGG16 frente a")
    anadir("  MobileNetV2. Usar eso para cambiar la selección rompería el protocolo del proyecto.")
    anadir("")
    anadir("**Recomendación concreta:** incorporar el barrido como tabla de sensibilidad y robustez con las 7")
    anadir("configuraciones por modelo, y **mantener la configuración final actual** (MobileNetV2 con oversampling y")
    anadir("`lr=3e-4`). Si se quisiera una conclusión firme sobre `lr=1e-3`, el paso siguiente, barato y correcto, sería")
    anadir("repetir las dos o tres configuraciones candidatas de MobileNetV2 con varias semillas y repetir el par")
    anadir("`lr in {3e-4, 1e-3}` sobre la rama con oversampling. Nada de eso cambia el modelo final por ahora.")
    anadir("")
    anadir("Queda pendiente de decidir con usted si este análisis se incorpora a la monografía y en qué sección.")
    anadir("")

    RUTA_INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Informe escrito en {RUTA_INFORME}")
    print(f"Lineas generadas: {len(lineas)}")


if __name__ == "__main__":
    main()
