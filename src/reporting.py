from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def export_confusion_matrices(model_results, y_labels, report_dir):
    comparison_rows = []
    confusion_matrix_files = {}

    for model_name, (metrics, cm) in model_results.items():
        comparison_rows.append({"algoritmo": model_name, **metrics})

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=y_labels,
            yticklabels=y_labels,
        )
        plt.title(f"Matriz de confusion - {model_name}")
        plt.xlabel("Prediccion")
        plt.ylabel("Valor real")
        plt.tight_layout()

        model_key = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        image_path = report_dir / f"confusion_matrix_{model_key}.png"
        plt.savefig(image_path, dpi=200)
        plt.close()

        confusion_matrix_files[model_name] = image_path.name

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(report_dir / "metricas_comparacion.csv", index=False)
    return comparison_df, confusion_matrix_files


def export_metrics_chart(comparison_df, report_dir):
    metrics_plot_df = comparison_df.melt(
        id_vars="algoritmo",
        value_vars=["exactitud", "precision", "sensibilidad", "f1"],
        var_name="metrica",
        value_name="valor",
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=metrics_plot_df, x="metrica", y="valor", hue="algoritmo")
    plt.title("Comparacion de metricas por algoritmo")
    plt.ylim(0, 1)
    plt.tight_layout()

    chart_name = "comparacion_metricas.png"
    plt.savefig(report_dir / chart_name, dpi=200)
    plt.close()
    return chart_name


def summarize_success(comparison_df, success_metric, success_threshold):
    best_row = comparison_df.sort_values(by=success_metric, ascending=False).iloc[0]
    best_model_name = best_row["algoritmo"]
    best_model_score = best_row[success_metric]
    success_message = "Cumple" if best_model_score >= success_threshold else "No cumple"
    return best_model_name, best_model_score, success_message


def write_report(
    report_dir,
    comparison_df,
    confusion_matrix_files,
    comparison_chart_name,
    success_metric,
    success_threshold,
    best_model_name,
    best_model_score,
    success_message,
    rf_best_params,
    rf_cv_score,
):
    report_path = report_dir / "informe.md"
    table_md = comparison_df.to_markdown(index=False)

    report_content = f"""# Informe de Modelos de Machine Learning

## 1. Descripcion del problema
Se desarrolla un modelo de clasificacion para predecir niveles de ingresos diarios (Low, Medium, High)
de una cafeteria a partir de variables operativas.

Este informe esta escrito de forma sencilla para explicar rapidamente el flujo de Machine Learning.

## 2. Medida de exito
- Metrica principal: {success_metric.upper()}
- Umbral de aceptacion: {success_threshold:.2f}
- Mejor resultado observado: {best_model_score:.4f} ({best_model_name})
- Estado: {success_message} el criterio de exito

## 3. Protocolo de evaluacion
- Hold-out con 80% entrenamiento y 20% prueba
- Muestreo estratificado por clase (Low, Medium, High)
- Semilla fija: 42

## 4. Preparacion de datos
- Carga de dataset desde data/
- Eliminacion de valores nulos
- Seleccion de variables predictoras y variable objetivo
- Discretizacion de ingresos en tres clases

## 5. Punto de referencia del modelo
- Logistic Regression se usa como baseline

## 6. Buen modelo y ajuste fino
- Random Forest se optimiza con GridSearchCV
- Mejores hiperparametros RF: {rf_best_params}
- Mejor F1 ponderado en validacion cruzada (RF): {rf_cv_score:.4f}

## 7. Algoritmos evaluados
- Logistic Regression (Baseline)
- Random Forest (Tuned)

## 8. Resultados (faciles de explicar)
Para cada algoritmo se reportan las metricas pedidas:
- Exactitud
- Precision
- Sensibilidad (Recall)
- F1-score

### Tabla comparativa de metricas
{table_md}

### Visualizaciones
- Matriz de confusion Logistic Regression: {confusion_matrix_files['Logistic Regression (Baseline)']}
- Matriz de confusion Random Forest: {confusion_matrix_files['Random Forest (Tuned)']}
- Comparacion de metricas: {comparison_chart_name}

## 9. Programa(s) que aplica el algoritmo
- main.py
- src/models/logistic_model.py
- src/models/random_forest_model.py
- src/evaluation.py

## 10. Archivos generados en report/
- metricas_comparacion.csv
- {confusion_matrix_files['Logistic Regression (Baseline)']}
- {confusion_matrix_files['Random Forest (Tuned)']}
- {comparison_chart_name}
- informe.md
"""

    report_path.write_text(report_content, encoding="utf-8")
    return report_path.name


def print_model_metrics(model_results):
    for model_name, (metrics, cm) in model_results.items():
        print(f"\n{model_name.upper()}")
        print("Matriz de confusion:")
        print(cm)
        print(
            "Exactitud: {exactitud:.4f} | Precision: {precision:.4f} | "
            "Sensibilidad: {sensibilidad:.4f} | F1: {f1:.4f}".format(**metrics)
        )


def print_generated_files(confusion_matrix_files, comparison_chart_name, report_file_name):
    print("\nArchivos de reporte generados en report/:")
    print("- metricas_comparacion.csv")
    print(f"- {confusion_matrix_files['Logistic Regression (Baseline)']}")
    print(f"- {confusion_matrix_files['Random Forest (Tuned)']}")
    print(f"- {comparison_chart_name}")
    print(f"- {report_file_name}")


def print_success_summary(best_model_name, best_model_score, success_threshold, success_message):
    print("\nResumen simple:")
    print(f"- Mejor modelo por F1: {best_model_name} ({best_model_score:.4f})")
    print(f"- Estado frente al umbral de exito ({success_threshold:.2f}): {success_message}")