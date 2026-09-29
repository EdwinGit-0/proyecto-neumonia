"""Interfaz de línea de comandos (CLI) del proyecto de neumonía.

Comandos disponibles:

    neumonia eda                    Análisis exploratorio de datos
    neumonia prepare                Manifiesto de división 80/20 con test original intacto
    neumonia augment                Figura de ejemplos de augmentación
    neumonia sensibilidad           Sensibilidad de hiperparámetros (21 corridas, sin test)
    neumonia combinado_arquitecturas  COMBINADO para VGG16, ResNet50, MobileNetV2 (train -> val)
    neumonia comparar               Comparar las 3 arquitecturas y seleccionar la mejor
    neumonia combinado              MobileNetV2 con COMBINADO (flujo original, compatibilidad)
    neumonia umbral                 Umbral de decisión congelado sobre validación
    neumonia final                  Modelo definitivo reentrenado sobre train + val
    neumonia test                   Evaluación del test original con el umbral congelado
    neumonia evaluar                Muestra los artefactos de la última corrida
    neumonia test-suite             Suite de pruebas con pytest
    neumonia run                    Flujo completo nuevo (3 arquitecturas -> comparación -> test)

El orden del flujo nuevo:
  1. combinado_arquitecturas: entrena VGG16, ResNet50, MobileNetV2 con COMBINADO sobre train, mide en val
  2. comparar: compara las 3 arquitecturas (Balanced Accuracy > ROC-AUC > F1 > Accuracy), selecciona la mejor
  3. umbral: congela umbral usando validation de la arquitectura seleccionada
  4. final: reentrena modelo definitivo sobre train+val
  5. test: evalúa test original con umbral congelado

El flujo original (combinado -> umbral -> final -> test) se mantiene para compatibilidad.
"""

from __future__ import annotations

import json
import subprocess
import sys

import click

from src.utils.paths import DIRECTORIO_DATOS, RAIZ_PROYECTO

SEMILLA = 42
RUTA_MANIFIESTO = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"


@click.group()
def cli() -> None:
    """CLI del proyecto de clasificación de neumonía en radiografías de tórax."""


@cli.command()
def eda() -> None:
    """Ejecutar el análisis exploratorio de datos (EDA) existente."""
    from src.data.make_dataset import ejecutar_eda

    click.echo("Ejecutando el análisis exploratorio de datos (EDA)...")
    resumen = ejecutar_eda(DIRECTORIO_DATOS)
    summary = resumen["summary"]

    click.echo(f"Total de imágenes: {summary['total_images']:,}")
    click.echo("Distribución por conjunto y clase:")
    for split_name, counts in summary["counts_by_split_and_class"].items():
        partes = ", ".join(f"{clase}: {cantidad}" for clase, cantidad in counts.items())
        click.echo(f"  {split_name}: {partes}")
    if summary["corrupt_images"]:
        click.echo(f"Imágenes corruptas: {len(summary['corrupt_images'])}")
    if summary["duplicate_paths"]:
        click.echo(f"Imágenes duplicadas (por hash): {len(summary['duplicate_paths'])}")

    click.echo("Figuras generadas en el EDA:")
    for nombre, ruta in resumen["figures"].items():
        click.echo(f"  - {nombre}: {ruta}")
    click.echo("EDA completado.")


@cli.command()
def prepare() -> None:
    """Crear el manifiesto estratificado 80/20 (train/validation) con test original intacto y verificar pipelines."""
    from src.data.datasets import construir_pipelines_datos
    from src.data.splitting import crear_manifiesto_division_estratificada

    click.echo("Creando el manifiesto de división estratificada (80% train / 20% validation, test original intacto)...")
    manifiesto = crear_manifiesto_division_estratificada(DIRECTORIO_DATOS, RUTA_MANIFIESTO, random_state=SEMILLA)
    click.echo(f"Manifiesto guardado en: {RUTA_MANIFIESTO}")
    for split_name in ["train", "val", "test"]:
        split_records = manifiesto[manifiesto["split"] == split_name]
        total = int(len(split_records))
        normal = int((split_records["label"] == "NORMAL").sum())
        neumonia = int((split_records["label"] == "PNEUMONIA").sum())
        click.echo(f"  {split_name}: {total} imágenes (NORMAL: {normal}, PNEUMONIA: {neumonia})")

    click.echo("Verificando los pipelines de datos (imágenes de 224 x 224)...")
    # incluir_test solo comprueba que el split original sea construible. El dataset es perezoso
    # (from_generator), asi que aqui no se lee ninguna imagen de test.
    datasets = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=(224, 224),
        batch_size=16,
        manifiesto_division=RUTA_MANIFIESTO,
        incluir_test=True,
    )
    faltantes = [nombre for nombre in ["train", "val", "test"] if nombre not in datasets]
    if faltantes:
        raise click.ClickException(f"El pipeline del dataset está incompleto: {faltantes}")
    click.echo(f"Pipelines disponibles: {', '.join(sorted(datasets))}")
    click.echo("Preparación de datos completada.")


@cli.command()
def augment() -> None:
    """Generar la figura de ejemplos de augmentación de datos existente."""
    from src.visualization.visualize import graficar_ejemplos_augmentation

    click.echo("Generando la figura de ejemplos de augmentación de datos...")
    ruta_salida = graficar_ejemplos_augmentation()
    click.echo(f"Figura guardada en: {ruta_salida}")
    click.echo("Augmentación de datos completada.")


@cli.command()
@click.option(
    "--recalcular",
    is_flag=True,
    help="Reentrenar los 21 experimentos (3 arquitecturas x 7 configuraciones).",
)
def sensibilidad(recalcular: bool) -> None:
    """Analizar la sensibilidad de learning rate, dropout y épocas (solo TRAIN y VALIDATION)."""
    import pandas as pd

    from src.training.sensitivity import (
        METRICAS_REPORTE,
        RUTA_RESULTADOS as RUTA_SENSIBILIDAD,
        ejecutar_analisis_sensibilidad,
    )

    if recalcular:
        click.echo("Ejecutando el analisis de sensibilidad: 21 entrenamientos en CPU (puede tardar horas).")
    else:
        click.echo("Consultando el analisis de sensibilidad ya registrado...")

    try:
        payload = ejecutar_analisis_sensibilidad(recalcular=recalcular)
    except FileNotFoundError as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Artefacto: {RUTA_SENSIBILIDAD}")
    click.echo(f"Splits utilizados: {', '.join(payload['splits_utilizados'])} (el test no interviene)")

    columnas = ["model_name", "config_id", "learning_rate", "dropout", "epochs", *METRICAS_REPORTE]
    filas = []
    for item in payload["resultados"]:
        fila = {
            "model_name": item["model_name"],
            "config_id": item["config_id"],
            **item["config"],
        }
        fila.update({metrica: item["validation"][metrica] for metrica in METRICAS_REPORTE})
        filas.append(fila)
    tabla = pd.DataFrame(filas)[columnas]
    click.echo("\nResultados de validacion (21 experimentos):")
    click.echo(tabla.to_string(index=False))

    click.echo("\nMejor configuracion por arquitectura:")
    for model_name, info in payload["mejor_por_arquitectura"].items():
        config = info["config"]
        delta = info["delta_balanced_accuracy_vs_referencia"]
        click.echo(
            f"  {model_name}: lr={config['learning_rate']}, dropout={config['dropout']}, "
            f"epochs={config['epochs']} -> balanced_accuracy={info['validation']['balanced_accuracy']:.4f} "
            f"(delta vs base {delta:+.4f})"
        )

    ganador = payload["ganador_global"]
    click.echo(
        f"\nGanador global: {ganador['model_name']} con {ganador['config_id']} "
        f"(balanced_accuracy={ganador['validation']['balanced_accuracy']:.4f}, "
        f"roc_auc={ganador['validation']['roc_auc']:.4f})"
    )
    from src.utils.paths import DIRECTORIO_FIGURAS

    click.echo(f"Figura guardada en: {DIRECTORIO_FIGURAS / 'sensitivity_validation.png'}")
    click.echo("Analisis de sensibilidad completado.")


@cli.command()
@click.option(
    "--recalcular",
    is_flag=True,
    help="Rehacer los 3 entrenamientos aunque ya existan sus artefactos.",
)
def combinado_arquitecturas(recalcular: bool) -> None:
    """Ejecutar COMBINADO para VGG16, ResNet50 y MobileNetV2 (train -> validation).

    Cada arquitectura usa los hiperparámetros fijados por la sensibilidad:
    lr=0.001, dropout=0.3, epochs=3. Mismos datos, mismo preprocessing,
    misma augmentation, misma estrategia de desbalance. Solo cambia la arquitectura.
    """
    from src.training.flujo_final import (
        RUTA_COMBINADO_VGG16,
        RUTA_COMBINADO_RESNET50,
        RUTA_COMBINADO_MOBILENETV2,
        ejecutar_combinado_arquitecturas,
    )

    click.echo("=== COMBINADO para 3 arquitecturas ===")
    click.echo("Hiperparámetros fijos: lr=0.001, dropout=0.3, epochs=3")
    click.echo("Estrategia: oversampling 50/50 + class weights (solo train)")
    click.echo("Medición: validation intacta (sin oversampling ni class weights)")
    click.echo("Advertencia: 3 entrenamientos en CPU, no interrumpir.")

    try:
        resultados = ejecutar_combinado_arquitecturas(recalcular=recalcular)
    except (FileNotFoundError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    for arch in ("VGG16", "ResNet50", "MobileNetV2"):
        payload = resultados[arch]
        diagnostico = payload["diagnostico_combinado"]
        despues = diagnostico["composicion_tras_oversampling"]
        click.echo(f"\n--- {arch} ---")
        click.echo(
            f"Oversampling: {despues['NORMAL']} NORMAL + {despues['PNEUMONIA']} PNEUMONIA "
            f"= {despues['total']} filas efectivas ({diagnostico['filas_duplicadas']} duplicadas)"
        )
        click.echo(f"class_weight: {diagnostico['class_weight']}")
        click.echo(f"Refuerzo total minoritaria: {diagnostico['refuerzo_total_minoritaria']:.4f}x")
        click.echo(f"Validation ({payload['validation_n']} imagenes, sin tratar):")
        for metrica, valor in payload["validation"].items():
            click.echo(f"  {metrica}: {valor:.4f}")
        matriz = payload["validation_matriz_confusion"]
        click.echo(f"  TN {matriz['tn']} / FP {matriz['fp']} / FN {matriz['fn']} / TP {matriz['tp']}")

    click.echo(f"\nArtefactos generados:")
    click.echo(f"  VGG16: {RUTA_COMBINADO_VGG16}")
    click.echo(f"  ResNet50: {RUTA_COMBINADO_RESNET50}")
    click.echo(f"  MobileNetV2: {RUTA_COMBINADO_MOBILENETV2}")
    click.echo("Siguiente paso: neumonia comparar (comparar y seleccionar arquitectura).")


@cli.command()
def comparar() -> None:
    """Comparar las 3 arquitecturas y seleccionar la mejor según criterio jerárquico.

    Criterio: Balanced Accuracy > ROC-AUC > F1 > Accuracy (sobre validation).
    Las tres arquitecturas usan exactamente los mismos datos, preprocessing,
    augmentation, hiperparámetros y estrategia COMBINADO.
    """
    from src.training.flujo_final import RUTA_COMPARACION, RUTA_SELECCION, comparar_arquitecturas

    click.echo("=== Comparación de arquitecturas ===")
    click.echo("Criterio: Balanced Accuracy > ROC-AUC > F1 > Accuracy (validation)")

    try:
        comparacion = comparar_arquitecturas()
    except (FileNotFoundError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"\nTabla ordenada (mejor primero):")
    for i, res in enumerate(comparacion["resultados"], 1):
        click.echo(
            f"  {i}. {res['model_name']}: BA={res['balanced_accuracy']:.4f} "
            f"ROC-AUC={res['roc_auc']:.4f} F1={res['f1']:.4f} Acc={res['accuracy']:.4f}"
        )

    ganador = comparacion["ganador"]
    click.echo(f"\nArquitectura seleccionada: {ganador['model_name']}")
    click.echo(f"  Balanced Accuracy: {ganador['balanced_accuracy']:.4f}")
    click.echo(f"  ROC-AUC: {ganador['roc_auc']:.4f}")
    click.echo(f"  F1: {ganador['f1']:.4f}")
    click.echo(f"  Accuracy: {ganador['accuracy']:.4f}")

    click.echo(f"\nArtefactos:")
    click.echo(f"  Comparación completa: {RUTA_COMPARACION}")
    click.echo(f"  Selección: {RUTA_SELECCION}")
    click.echo("Siguiente paso: neumonia umbral (congelar umbral con la arquitectura seleccionada).")


@cli.command()
@click.option(
    "--recalcular",
    is_flag=True,
    help="Rehacer el entrenamiento aunque ya exista su artefacto.",
)
def combinado(recalcular: bool) -> None:
    """Entrenar MobileNetV2 con la estrategia unica COMBINADO (train -> validation)."""
    from src.training.flujo_final import RUTA_COMBINADO, ejecutar_combinado

    click.echo("=== Estrategia COMBINADO: oversampling 50/50 + class weights ===")
    click.echo("Se entrena sobre train y se mide solo sobre validation, que queda intacta.")
    click.echo("Advertencia: entrenamiento en CPU, no interrumpir.")

    try:
        payload = ejecutar_combinado(recalcular=recalcular)
    except (FileNotFoundError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    diagnostico = payload["diagnostico_combinado"]
    despues = diagnostico["composicion_tras_oversampling"]
    click.echo(
        f"\nOversampling: {despues['NORMAL']} NORMAL + {despues['PNEUMONIA']} PNEUMONIA "
        f"= {despues['total']} filas efectivas ({diagnostico['filas_duplicadas']} duplicadas)"
    )
    click.echo(f"class_weight: {diagnostico['class_weight']}")
    click.echo(f"Refuerzo total de la minoritaria: {diagnostico['refuerzo_total_minoritaria']:.4f}x")
    click.echo(f"  {diagnostico['advertencia_doble_correccion']}")

    click.echo(f"\nValidation ({payload['validation_n']} imagenes, sin tratar):")
    for metrica, valor in payload["validation"].items():
        click.echo(f"  {metrica}: {valor:.4f}")
    matriz = payload["validation_matriz_confusion"]
    click.echo(f"  TN {matriz['tn']} / FP {matriz['fp']} / FN {matriz['fn']} / TP {matriz['tp']}")

    click.echo(f"\nArtefacto: {RUTA_COMBINADO}")
    click.echo("Siguiente paso: neumonia umbral (congelar el umbral con las probabilidades de validation).")


@cli.command()
def umbral() -> None:
    """Congelar el umbral de decisión usando solo las probabilidades de validación."""
    from src.training.flujo_final import RUTA_DECISION, RUTA_UMBRAL, ajustar_umbral

    click.echo("Ajustando el umbral de decisión sobre validación...")
    try:
        decision = ajustar_umbral()
    except (FileNotFoundError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(
        f"Umbral congelado: {decision['umbral']} "
        f"({decision['model_name']}/{decision['estrategia']}, {decision['epochs_definitivos']} epochs)"
    )
    click.echo(f"Criterio: {decision['criterio_umbral']}")
    click.echo(f"Artefactos: {RUTA_UMBRAL}, {RUTA_DECISION}")
    click.echo("Siguiente paso: neumonia final (reentrenar sobre train + val).")


@cli.command()
def final() -> None:
    """Reentrenar el modelo definitivo sobre train + val con la decisión ya congelada."""
    from src.training.flujo_final import RUTA_MODELO_FINAL, RUTA_REPORTE_TEST, entrenar_modelo_definitivo

    if RUTA_REPORTE_TEST.exists():
        click.echo("Aviso: ya existe un informe de test. Reentrenar deja ese informe desactualizado;")
        click.echo("vuelva a ejecutar 'neumonia test' si quiere un informe del modelo nuevo.")

    click.echo("Advertencia: entrenamiento en CPU sobre train + val, no interrumpir.")
    try:
        registro = entrenar_modelo_definitivo()
    except (FileNotFoundError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    composicion = registro["composicion_entrenamiento"]
    click.echo(
        f"Modelo definitivo: {registro['model_name']}/{registro['estrategia']} "
        f"entrenado con {composicion['total']} imagenes "
        f"({composicion['NORMAL']} NORMAL / {composicion['PNEUMONIA']} PNEUMONIA)"
    )
    click.echo(f"Epochs: {registro['epochs_ejecutadas']} | umbral congelado: {registro['umbral']}")
    click.echo(f"Modelo guardado en: {RUTA_MODELO_FINAL}")
    click.echo("Siguiente paso: neumonia test")


@cli.command()
def test() -> None:
    """Evaluar el test original con el modelo definitivo COMBINADO."""
    from src.training.flujo_final import RUTA_REPORTE_TEST, evaluar_test

    click.echo("=== Evaluacion del test original ===")
    click.echo("Esta evaluacion no interviene en ninguna decision del modelo ni del umbral.")

    try:
        reporte = evaluar_test()
    except (FileNotFoundError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    composicion = reporte["test"]
    click.echo(
        f"Test: {composicion['total']} imagenes "
        f"({composicion['NORMAL']} NORMAL / {composicion['PNEUMONIA']} PNEUMONIA)"
    )
    click.echo(f"\nMetricas con el umbral congelado ({reporte['umbral']}):")
    for metrica, valor in reporte["metricas_umbral_congelado"].items():
        click.echo(f"  {metrica}: {valor:.4f}")
    click.echo("\nReferencia con umbral 0.5:")
    for metrica, valor in reporte["metricas_referencia_0.5"].items():
        click.echo(f"  {metrica}: {valor:.4f}")
    matriz = reporte["matriz_confusion"]
    click.echo(
        f"\nMatriz de confusion: TN {matriz['tn']} / FP {matriz['fp']} / "
        f"FN {matriz['fn']} / TP {matriz['tp']}"
    )
    criterio = reporte["criterio_exito"]["umbral_congelado"]
    click.echo(f"\nCriterio de exito (umbral congelado): {criterio['cumple']}")
    click.echo(f"  Supera baseline: {criterio['supera_baseline']}")
    click.echo(f"  Recall > 0.5: {criterio['sensibilidad_sobre_azar']}")
    click.echo(f"  Specificity > 0.5: {criterio['especificidad_sobre_azar']}")
    click.echo(f"\nInforme: {RUTA_REPORTE_TEST}")
    for nombre, ruta in reporte["figuras"].items():
        click.echo(f"Figura {nombre}: {ruta}")


@cli.command()
def evaluar() -> None:
    """Mostrar los resultados guardados sin volver a entrenar."""
    from src.training.flujo_final import (
        RUTA_COMBINADO,
        RUTA_DECISION,
        RUTA_ENTRENAMIENTO,
        RUTA_MODELO_FINAL,
        RUTA_REPORTE_TEST,
        RUTA_UMBRAL,
    )

    if RUTA_COMBINADO.exists():
        combinado = json.loads(RUTA_COMBINADO.read_text(encoding="utf-8"))
        diagnostico = combinado["diagnostico_combinado"]
        click.echo(f"Estrategia: {combinado['estrategia']['id']} (unica)")
        click.echo(f"Modelo: {combinado['model_name']} | config: {combinado['config']}")
        click.echo(f"Artefacto: {RUTA_COMBINADO}")
        click.echo(
            f"Oversampling: {diagnostico['composicion_tras_oversampling']['NORMAL']} NORMAL + "
            f"{diagnostico['composicion_tras_oversampling']['PNEUMONIA']} PNEUMONIA = "
            f"{diagnostico['composicion_tras_oversampling']['total']} filas"
        )
        click.echo(f"class_weight: {diagnostico['class_weight']} (refuerzo total {diagnostico['refuerzo_total_minoritaria']:.4f}x)")
        click.echo(f"\nValidation ({combinado['validation_n']} imagenes, sin tratar):")
        for metrica, valor in combinado["validation"].items():
            click.echo(f"  {metrica}: {valor:.4f}")
        matriz = combinado["validation_matriz_confusion"]
        click.echo(f"  TN {matriz['tn']} / FP {matriz['fp']} / FN {matriz['fn']} / TP {matriz['tp']}")
    else:
        click.echo("Todavia no hay corrida de COMBINADO: ejecute 'neumonia combinado'.")

    if RUTA_UMBRAL.exists():
        umbral = json.loads(RUTA_UMBRAL.read_text(encoding="utf-8"))
        click.echo(f"\nUmbral congelado: {umbral['umbral']}")
        click.echo(f"  Criterio: {umbral['criterio']}")
        click.echo(f"  Origen: {umbral['conjunto_origen']} ({umbral['validation_n']} imagenes)")
        click.echo(f"  Artefacto: {RUTA_UMBRAL}")
        click.echo(f"  Figura: {umbral['figura_seleccion_umbral']}")

    if RUTA_DECISION.exists():
        decision = json.loads(RUTA_DECISION.read_text(encoding="utf-8"))
        click.echo(f"\nDecision congelada: {RUTA_DECISION}")
        click.echo(f"  Entrenamiento definitivo: {'+'.join(decision['splits_entrenamiento_definitivo'])}")
        click.echo(f"  Epochs: {decision['epochs_definitivos']}")

    if RUTA_ENTRENAMIENTO.exists():
        final = json.loads(RUTA_ENTRENAMIENTO.read_text(encoding="utf-8"))
        click.echo(f"\nModelo definitivo: {RUTA_MODELO_FINAL}")
        click.echo(f"  {final['composicion_entrenamiento']}")

    if RUTA_REPORTE_TEST.exists():
        reporte = json.loads(RUTA_REPORTE_TEST.read_text(encoding="utf-8"))
        click.echo(f"\nInforme de test COMBINADO: {RUTA_REPORTE_TEST}")
        click.echo(
            f"Test: {reporte['test']['total']} imagenes | umbral congelado: {reporte['umbral']}"
        )
        for metrica, valor in reporte["metricas_umbral_congelado"].items():
            click.echo(f"    {metrica}: {valor:.4f}")
        matriz = reporte["matriz_confusion"]
        click.echo(
            f"    TN {matriz['tn']} / FP {matriz['fp']} / FN {matriz['fn']} / TP {matriz['tp']}"
        )
    else:
        click.echo("\nTodavia no hay informe de test: ejecute 'neumonia test'.")


@cli.command("test-suite")
@click.option("--verbose", "-v", is_flag=True, help="Mostrar el detalle de cada prueba.")
def test_suite(verbose: bool) -> None:
    """Ejecutar la suite de pruebas existente con pytest."""
    args = [sys.executable, "-m", "pytest"]
    if verbose:
        args.append("-v")
    click.echo("Ejecutando la suite de pruebas con pytest...")
    proceso = subprocess.run(args, cwd=str(RAIZ_PROYECTO))
    if proceso.returncode != 0:
        raise SystemExit(proceso.returncode)
    click.echo("Pruebas completadas correctamente.")


@cli.command()
@click.option("--recalcular-combinado", is_flag=True, help="Rehacer los 3 entrenamientos COMBINADO.")
def run(recalcular_combinado: bool) -> None:
    """Ejecutar el flujo completo nuevo: 3 arquitecturas COMBINADO -> comparar -> umbral -> definitivo -> test.

    El orden:
    1. EDA, preparación, augmentación, sensibilidad (solo consulta, no reentrena)
    2. combinado_arquitecturas: COMBINADO para VGG16, ResNet50, MobileNetV2
    3. comparar: compara y selecciona la mejor arquitectura
    4. umbral: congela umbral con la arquitectura seleccionada
    5. final: reentrena modelo definitivo sobre train+val
    6. test: evalúa test original
    7. evaluar: muestra resultados
    8. test-suite: ejecuta pruebas
    """
    click.echo("=== FLUJO COMPLETO NUEVO: 3 ARQUITECTURAS -> COMPARACIÓN -> MODELO DEFINITIVO -> TEST ===")
    eda()
    prepare()
    augment()
    sensibilidad(recalcular=False)  # Solo consulta, no reentrena las 21 configuraciones
    combinado_arquitecturas(recalcular=recalcular_combinado)
    comparar()
    umbral()
    final()
    test()
    evaluar()
    test_suite()
    click.echo("=== Flujo finalizado. Revise 'neumonia evaluar' para el informe de test. ===")


if __name__ == "__main__":
    cli()
