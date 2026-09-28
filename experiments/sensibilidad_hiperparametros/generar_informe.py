"""Genera el informe del análisis de sensibilidad de hiperparámetros.

Solo lee resultados ya calculados dentro del experimento:
- resultados/sensibilidad_resultados.json
- resultados/seleccion.json

No lee `models/*.json` (eliminados al retirar el flujo que usaba TEST) y no
necesita ninguna evaluación sobre TEST: la única evaluación sobre TEST del
proyecto es la evaluación final del modelo definitivo (`src/training/flujo_final.py`).

Escribe únicamente INFORME_SENSIBILIDAD.md dentro del directorio del experimento.
No modifica src/, models/, reports/, data/ ni la monografía.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DIRECTORIO = Path(__file__).resolve().parent
RESULTADOS = DIRECTORIO / "resultados"
RUTA_INFORME = DIRECTORIO / "INFORME_SENSIBILIDAD.md"

MODELOS = ("VGG16", "ResNet50", "MobileNetV2")
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

    configs = {c["id"]: c for c in datos["configuraciones"]}
    lineas: list[str] = []
    anadir = lineas.append

    def validacion(modelo: str, config_id: str) -> dict[str, Any]:
        """Métricas de validación de una configuración."""
        return datos["resultados"][f"{modelo}/{config_id}"]["validation"]

    # ------------------------------------------------------------------ 1
    anadir("# Análisis de sensibilidad de hiperparámetros")
    anadir("")
    anadir(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    anadir("")
    anadir(
        "Experimento **aislado** realizado para responder una sola pregunta: qué efecto tiene cada "
        "hiperparámetro (learning rate, dropout y número máximo de épocas) sobre el comportamiento de "
        "VGG16, ResNet50 y MobileNetV2."
    )
    anadir("")
    anadir("> **TEST no se carga ni se evalúa en este experimento.** Las 21 corridas se entrenan con TRAIN y se")
    anadir("> comparan únicamente sobre VALIDACIÓN. La única evaluación sobre TEST en el proyecto es la")
    anadir("> evaluación final del modelo definitivo, ejecutada una sola vez y después de cerrar todas las")
    anadir("> decisiones (`src/training/flujo_final.py`).")
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
    anadir("|   `-- seleccion.json                # selección basada solo en validación")
    anadir("|-- checkpoints/              # checkpoints de las 21 corridas (no versionados)")
    anadir("`-- logs/sweep.log")
    anadir("```")
    anadir("")
    anadir("Verificación de integridad ejecutada por el script (hash SHA-256 antes y después del barrido):")
    anadir("")
    anadir("| Archivo del proyecto (solo lectura)                          | Sin cambios |")
    anadir("|--------------------------------------------------------------|:-----------:|")
    anadir(
        "| `data/interim/stratified_split_train80_val20_test_original.csv` | "
        f"{'sí' if datos['manifiesto']['sin_modificar'] else 'NO'} |"
    )
    anadir("")
    anadir("El manifiesto de división se **leyó**, nunca se regeneró. No se escribió en `src/`, `models/`,")
    anadir("`reports/` ni `data/`.")
    anadir("")

    # ------------------------------------------------------------------ 3
    entorno = datos["entorno"]
    anadir("## 2. Metodología")
    anadir("")
    anadir("Se conservaron **todos** los elementos del pipeline del proyecto, cambiando un solo hiperparámetro por")
    anadir("configuración:")
    anadir("")
    lineas.extend(tabla(["Elemento", "Valor conservado"], [":---", ":---"]))
    for etiqueta, valor in [
        ("Dataset", "Chest X-Ray (Pneumonia), 5.840 imágenes utilizadas"),
        ("División", "train 4.173 / validación 1.043 (80/20 estratificada)"),
        ("Test", "624 imágenes del test original, **no cargadas ni evaluadas aquí**"),
        ("Preprocesamiento", "RGB, 224x224, interpolación bilinear, float32, división por 255"),
        ("Augmentación (solo train)", "`RandomRotation(0.05)` y `RandomZoom(0.05)`"),
        ("Semilla", "42, fijada de nuevo antes de cada corrida"),
        ("Base", "ImageNet, convolucional congelada"),
        ("Cabeza", "GAP -> Dense(128, ReLU) -> Dropout -> Dense(1, Sigmoid)"),
        ("Pérdida y optimizador", "Binary crossentropy y Adam"),
        ("Batch size", "16"),
        (
            "Callbacks",
            "`ModelCheckpoint(val_loss)`, `EarlyStopping(patience=3)`, "
            "`ReduceLROnPlateau(factor 0.5, patience 2, min_lr 1e-6)`",
        ),
        ("Criterio de selección", "Balanced Accuracy > ROC-AUC > F1-Score > Accuracy"),
    ]:
        anadir(fila([etiqueta, valor]))
    anadir("")
    anadir("Decisiones metodológicas relevantes para la interpretación:")
    anadir("")
    anadir("* Los pipelines de datos se **reconstruyen en cada corrida** después de fijar la semilla 42, de modo que")
    anadir("  todas las configuraciones parten del mismo orden de datos y del mismo estado de augmentación. Así las")
    anadir("  diferencias se atribuyen al hiperparámetro y no al azar del orden de los datos.")
    anadir("* La evaluación usa `src.models.evaluation.calcular_reporte_metricas`, la misma función del proyecto y")
    anadir("  el mismo umbral de 0.5.")
    anadir("* Solo se varía **un** hiperparámetro por configuración. No se combinan valores entre sí.")
    anadir("* El split `test` está bloqueado a nivel de código: `crear_pipelines` rechaza cualquier split distinto de")
    anadir("  `train` y `val`.")
    anadir("")
    anadir(
        f"Entorno de ejecución: Python {entorno['python']}, TensorFlow {entorno['tensorflow']}, "
        f"GPU: {', '.join(entorno['gpus']) if entorno['gpus'] else 'no disponible (solo CPU)'}."
    )
    anadir("")

    # ------------------------------------------------------------------ 4
    total_corridas = len(datos["resultados"])
    anadir("## 3. Configuraciones realmente ejecutadas")
    anadir("")
    anadir(f"**{total_corridas} corridas** (3 modelos x 7 configuraciones únicas). Todas completaron el entrenamiento")
    anadir("completo: ninguna configuración falló, quedó pendiente o se omitió.")
    anadir("")
    anadir("La configuración `(lr=1e-4, dropout=0.3, 3 epochs)` es la referencia común de los tres grupos, por lo que")
    anadir("se ejecutó una sola vez por modelo.")
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
            grupo = "referencia" if registro["es_referencia"] else config["grupo"]
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
    anadir(
        f"* Tiempo total de entrenamiento: "
        f"{sum(r['segundos_entrenamiento'] for r in datos['resultados'].values()) / 3600:.1f} h en CPU."
    )
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
    anadir("Selección hecha **exclusivamente con validación**, con el criterio jerárquico del proyecto")
    anadir("(*Balanced Accuracy > ROC-AUC > F1-Score > Accuracy*). El conjunto de test no intervino en ningún momento.")
    anadir("")
    lineas.extend(tabla(
        ["Modelo", "Config. seleccionada", "LR", "Dropout", "Ep.", "Bal. Acc", "ROC-AUC", "F1", "Accuracy", "Bal. Acc ref.", "Δ"],
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
    margen_ref = ref_ranking[0][1] - ref_ranking[1][1]
    anadir(f"Margen entre el primero y el segundo: **{margen:.4f}**. Con la configuración de referencia el margen")
    anadir(f"era de **{margen_ref:.4f}** ({ref_ranking[0][0]} {f4(ref_ranking[0][1])} contra {ref_ranking[1][0]} "
           f"{f4(ref_ranking[1][1])}). VGG16 mejora mucho con `lr=1e-3`")
    anadir(f"({f4(ref_ranking[1][1])} -> {f4(seleccion['VGG16']['validation']['balanced_accuracy'])}), de modo que la")
    anadir("ventaja de MobileNetV2 es real, pero menor de lo que sugiere la configuración de referencia.")
    anadir("")

    # ------------------------------------------------------------------ 7
    anadir("## 6. Qué hiperparámetro pesa más")
    anadir("")
    ref_mn = validacion("MobileNetV2", "ref_lr1e-4_do0.3_ep3")["balanced_accuracy"]
    val_3e4 = validacion("MobileNetV2", "lr_3e-4")["balanced_accuracy"]
    val_1e3 = validacion("MobileNetV2", "lr_1e-3")["balanced_accuracy"]
    val_do5 = validacion("MobileNetV2", "do_0.5")["balanced_accuracy"]
    lineas.extend(tabla(
        ["Configuración de MobileNetV2", "Bal. Acc (validación)", "Δ vs referencia"],
        [":---", "--:", "--:"],
    ))
    anadir(fila(["Referencia `lr=1e-4`", f4(ref_mn), "referencia"]))
    anadir(fila(["`lr=3e-4`", f4(val_3e4), f"{val_3e4 - ref_mn:+.4f}"]))
    anadir(fila(["`dropout=0.5`", f4(val_do5), f"{val_do5 - ref_mn:+.4f}"]))
    anadir(fila(["`lr=1e-3`", f"**{f4(val_1e3)}**", f"{val_1e3 - ref_mn:+.4f}"]))
    anadir("")
    rango_dropout = {
        m: (
            max(validacion(m, clave)["balanced_accuracy"] for clave in ("ref_lr1e-4_do0.3_ep3", "do_0.2", "do_0.5"))
            - min(validacion(m, clave)["balanced_accuracy"] for clave in ("ref_lr1e-4_do0.3_ep3", "do_0.2", "do_0.5"))
        )
        for m in MODELOS
    }
    anadir("* Sobre MobileNetV2 el factor dominante es el **learning rate**: `1e-3` "
           f"({val_1e3 - ref_mn:+.4f}) supera a `3e-4` ({val_3e4 - ref_mn:+.4f}) y a `1e-4`.")
    anadir("* El **dropout es casi irrelevante en MobileNetV2 y sí pesa en VGG16**: el rango de balanced accuracy entre")
    anadir(f"  `0.2`, `0.3` y `0.5` es de {rango_dropout['MobileNetV2']:.4f} en MobileNetV2, "
           f"{rango_dropout['VGG16']:.4f} en VGG16 y")
    anadir(f"  {rango_dropout['ResNet50']:.4f} en ResNet50. En el modelo ganador el dropout no es una dirección que")
    anadir("  merezca más búsqueda.")
    anadir("* El **número de épocas** solo mueve a los modelos que aprendían despacio: de 3 a 10 epochs la balanced")
    anadir(f"  accuracy de VGG16 pasa de {validacion('VGG16', 'ref_lr1e-4_do0.3_ep3')['balanced_accuracy']:.4f} a "
           f"{validacion('VGG16', 'ep_10')['balanced_accuracy']:.4f} y la de")
    anadir(f"  ResNet50 de {validacion('ResNet50', 'ref_lr1e-4_do0.3_ep3')['balanced_accuracy']:.4f} a "
           f"{validacion('ResNet50', 'ep_10')['balanced_accuracy']:.4f}; en MobileNetV2 pasa de {ref_mn:.4f} a")
    anadir(f"  {validacion('MobileNetV2', 'ep_10')['balanced_accuracy']:.4f}, sin superar el {val_1e3:.4f} que")
    anadir("  consigue `lr=1e-3` con solo 3 epochs.")
    anadir("")

    anadir("### 6.1 Hallazgo sobre ResNet50")
    anadir("")
    ref_rn = validacion("ResNet50", "ref_lr1e-4_do0.3_ep3")
    sel_rn = seleccion["ResNet50"]["validation"]
    anadir("Con la configuración de referencia, ResNet50 clasifica **todas** las imágenes de validación como")
    anadir(f"`PNEUMONIA` (recall {f4(ref_rn['recall'])}, specificity {f4(ref_rn['specificity'])}, balanced accuracy")
    anadir(f"{f4(ref_rn['balanced_accuracy'])}). Ese comportamiento **no es una propiedad de la arquitectura**: con")
    anadir(f"`lr=1e-3` deja de colapsar y alcanza balanced accuracy {f4(sel_rn['balanced_accuracy'])} (recall")
    anadir(f"{f4(sel_rn['recall'])}, specificity {f4(sel_rn['specificity'])}). Cualquier afirmación sobre ResNet50")
    anadir("debe leerse como *\"con learning rate 1e-4\"* y no como una limitación general del modelo.")
    anadir("")

    # ------------------------------------------------------------------ 8
    anadir("## 7. Limitaciones de este análisis")
    anadir("")
    anadir("1. **Una sola semilla (42).** No hay replicaciones, así que no se puede estimar la variabilidad del")
    anadir("   entrenamiento. Las diferencias pequeñas, como `lr=1e-3` frente a `lr=3e-4` en MobileNetV2")
    anadir(f"   ({val_1e3 - val_3e4:+.4f}), pueden ser ruido.")
    anadir("2. **Un factor por vez.** No se exploran interacciones, ni la combinación de `lr=1e-3` con oversampling")
    anadir("   o class weights, que es justamente lo que evalúa la estrategia final COMBINADO.")
    anadir("3. **Solo tres valores por hiperparámetro**, sin búsqueda en rejilla. No es una búsqueda exhaustiva.")
    anadir("4. **Rama sin tratamiento de desbalance.** El barrido se hizo sobre los tres modelos en su")
    anadir("   configuración de referencia, sin class weights ni oversampling.")
    anadir("5. **Una sola partición de validación** de 1.043 imágenes con fuerte desbalance (268 NORMAL). El balanced")
    anadir("   accuracy es el criterio adecuado, pero un único split sigue siendo una fuente de incertidumbre.")
    anadir("6. **Umbral fijo de 0.5** en todas las métricas, sin ajuste de umbral.")
    anadir("7. **Ninguna evaluación sobre TEST.** Por protocolo del proyecto, TEST no se consulta antes de la")
    anadir("   evaluación final del modelo definitivo, así que este informe no permite afirmar nada sobre test.")
    anadir("")

    # ------------------------------------------------------------------ 9
    anadir("## 8. Conclusión")
    anadir("")
    anadir("El barrido deja tres conclusiones útiles para el proyecto:")
    anadir("")
    anadir("* **El learning rate es el factor que más mueve el modelo**, y su valor óptimo (`1e-3`) es el mismo que")
    anadir("  gana en los tres modelos. Esa es la configuración base que hereda el entrenamiento con COMBINADO.")
    anadir("* **El dropout y las épocas son casi irrelevantes en MobileNetV2**, el modelo ganador, y sí tienen peso en")
    anadir("  VGG16 y ResNet50. Son la dirección de búsqueda equivocada si se busca mejorar el ganador.")
    anadir("* **La ventaja de MobileNetV2 es real pero moderada**: "
           f"{margen:.4f} de balanced accuracy sobre VGG16 con las mejores configuraciones de cada uno, no el")
    anadir("  contraste amplio que se obtiene comparando contra la configuración de referencia.")
    anadir("")
    anadir("El criterio de selección derivado de este análisis es el mismo del proyecto y se aplica, en el flujo")
    anadir("final, a la comparación de hiperparámetros por arquitectura. El tratamiento del desbalance no es una")
    anadir("variable a comparar: el flujo final usa una estrategia única, COMBINADO, que se toma como premisa del")
    anadir("proyecto y no se mide frente a sin tratamiento, oversampling o class weights por separado.")
    anadir("")

    RUTA_INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Informe escrito en {RUTA_INFORME}")
    print(f"Lineas generadas: {len(lineas)}")


if __name__ == "__main__":
    main()
