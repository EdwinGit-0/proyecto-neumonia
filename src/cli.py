"""Interfaz de línea de comandos (CLI) del proyecto de neumonía.

Permite ejecutar las distintas etapas del proyecto con comandos cortos en lugar
de invocar módulos completos:

    neumonia eda
    neumonia prepare
    neumonia augment
    neumonia sensibilidad
    neumonia train
    neumonia tune
    neumonia desbalance
    neumonia evaluate
    neumonia test
    neumonia run

La CLI solo invoca las funciones existentes del proyecto; no duplica ni modifica
la lógica científica de ninguna etapa.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click

from src.utils.paths import DIRECTORIO_DATOS, DIRECTORIO_MODELOS, RAIZ_PROYECTO

SEMILLA = 42
RUTA_MANIFIESTO = RAIZ_PROYECTO / "data" / "interim" / "stratified_split_train80_val20_test_original.csv"
RUTA_RESULTADOS = DIRECTORIO_MODELOS / "model_results.json"
COLUMNAS_METRICAS = [
    "model_name",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "f1",
    "roc_auc",
]


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
    datasets = construir_pipelines_datos(
        DIRECTORIO_DATOS,
        image_size=(224, 224),
        batch_size=16,
        manifiesto_division=RUTA_MANIFIESTO,
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
    click.echo(f"Comparacion actualizada en: {RUTA_RESULTADOS}")
    click.echo("Analisis de sensibilidad completado.")


@cli.command()
def train() -> None:
    """Entrenar VGG16, ResNet50 y MobileNetV2; evaluar, comparar y seleccionar el mejor."""
    from src.training.run_real_training import entrenar_y_evaluar_modelos

    click.echo("Este comando puede consumir bastante tiempo y recursos (no interrumpir).")
    resultado = entrenar_y_evaluar_modelos()
    click.echo("Resultados sobre validación:")
    click.echo(json.dumps(resultado["validation_comparison"], indent=2, default=str))
    click.echo("Resultados finales sobre test (solo el modelo ganador):")
    click.echo(json.dumps(resultado["final_test"], indent=2, default=str))
    click.echo("Entrenamiento completado.")


@cli.command()
def tune() -> None:
    """Ejecutar el tuning experimental de MobileNetV2 (learning rate, dropout y fine-tuning)."""
    from src.training.tuning_mobilenetv2 import entrenar_y_evaluar_tuning

    click.echo("Este comando reentrena MobileNetV2 con varias configuraciones y puede tardar bastante (no interrumpir).")
    click.echo("Solo se usan TRAIN y VALIDATION para decidir; el TEST queda reservado para el final.")
    resultado = entrenar_y_evaluar_tuning()
    click.echo("Configuración elegida:")
    click.echo(json.dumps(resultado["seleccion"], indent=2, default=str))
    click.echo("Resultados finales sobre test del modelo elegido:")
    click.echo(json.dumps(resultado["final_test"], indent=2, default=str))
    click.echo("Tuning completado. Resultados en models/mobilenetv2_tuning_results.json")


@cli.command()
def desbalance() -> None:
    """Ejecutar los experimentos de desbalance de clases (class_weight y oversampling) sobre el ajustado lr=3e-4."""
    from src.training.tuning_desbalance_clases import entrenar_y_evaluar_desbalance

    click.echo("Este comando reentrena variantes con class_weight y oversampling y puede tardar bastante (no interrumpir).")
    click.echo("Solo se usan TRAIN y VALIDATION para decidir; el TEST queda reservado para el final.")
    resultado = entrenar_y_evaluar_desbalance()
    click.echo("Tabla de selección sobre validation:")
    click.echo(json.dumps(resultado["tabla_seleccion"], indent=2, default=str))
    click.echo(f"Motivo de la selección: {resultado['motivo_seleccion']}")
    click.echo("Resultados finales sobre test del modelo elegido:")
    click.echo(json.dumps(resultado["final_test"], indent=2, default=str))
    click.echo("Experimentos completados. Resultados en models/mobilenetv2_desbalance_results.json")


@cli.command()
def evaluate() -> None:
    """Mostrar los resultados guardados de la evaluación sin volver a entrenar."""
    if not RUTA_RESULTADOS.exists():
        raise click.ClickException(
            f"No existe {RUTA_RESULTADOS}. Ejecute primero 'neumonia train' para generar los resultados."
        )

    import pandas as pd

    resultado = json.loads(RUTA_RESULTADOS.read_text(encoding="utf-8"))
    click.echo(f"Manifiesto de división: {resultado['split_manifest']}")
    click.echo(f"División: 80/20 con random_state={resultado['split_random_state']}")

    click.echo("\nDistribución del dataset:")
    distribucion = pd.DataFrame(resultado["split_distribution"]).T
    click.echo(distribucion.to_string())

    click.echo("\nResultados sobre validación (ordenados por criterio de selección):")
    tabla_validacion = pd.DataFrame(resultado["validation_comparison"]["results"])[COLUMNAS_METRICAS]
    click.echo(tabla_validacion.to_string(index=False))

    ganador = resultado["validation_comparison"]["winner"]["model_name"]
    click.echo(f"\nModelo ganador (selección por validación): {ganador}")

    click.echo("\nBaseline mayoritaria sobre test:")
    baseline = resultado.get("baseline_test", {})
    for clave, valor in baseline.items():
        click.echo(f"  {clave}: {valor}")

    click.echo("\nCriterio de éxito:")
    criterio = resultado.get("criterio_exito", {})
    click.echo(f"  ¿Cumple? {criterio.get('cumple')}")
    click.echo(f"  Supera baseline: {criterio.get('supera_baseline')}")
    click.echo(f"  Sensibilidad sobre azar: {criterio.get('sensibilidad_sobre_azar')}")
    click.echo(f"  Especificidad sobre azar: {criterio.get('especificidad_sobre_azar')}")

    click.echo("\nResultados finales sobre test (solo el modelo ganador):")
    tabla_test = pd.DataFrame([resultado["final_test"]])[COLUMNAS_METRICAS]
    click.echo(tabla_test.to_string(index=False))
    click.echo(f"Matriz de confusión en: {resultado['final_test']['confusion_plot']}")
    click.echo(f"Curva ROC en: {resultado['final_test']['roc_plot']}")


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Mostrar el detalle de cada prueba.")
def test(verbose: bool) -> None:
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
def run() -> None:
    """Ejecutar el flujo completo: EDA, preparación, augmentación, sensibilidad, entrenamiento, evaluación y test."""
    click.echo("=== Flujo completo del proyecto ===")
    eda()
    prepare()
    augment()
    sensibilidad(recalcular=False)
    train()
    evaluate()
    test()
    click.echo("=== Flujo completo finalizado ===")


if __name__ == "__main__":
    cli()