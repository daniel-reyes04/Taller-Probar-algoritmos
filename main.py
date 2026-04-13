from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data_preparation import load_and_prepare_data
from src.feature_engineering import categorize_revenue
from src.model_deploy import save_model
from src.models.logistic_model import train_logistic
from src.models.random_forest_model import train_rf
from src.models.run_peer_pipelines import run_peer_pipelines
from src.evaluation import evaluate_multiclass

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "report"
MODEL_DIR = BASE_DIR / "models"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data_path_candidates = [
        BASE_DIR / "data" / "coffee.csv",
        BASE_DIR / "data" / "coffee_shop_revenue.csv",
    ]
    data_path = next((path for path in data_path_candidates if path.exists()), None)

    if data_path is None:
        raise FileNotFoundError(
            "No se encontro un archivo de datos valido en la carpeta data/."
        )

    # Paso 1: Recopilacion de datos
    X, y = load_and_prepare_data(str(data_path))

    # Paso 4: Preparacion de datos (multiclase: Bajo/Medio/Alto).
    y_classes = categorize_revenue(y)
    labels = list(y_classes.cat.categories)

    # Paso 3: Protocolo de evaluacion
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_classes,
        test_size=0.2,
        random_state=42,
        stratify=y_classes,
    )

    # Paso 5: Modelo baseline
    log_model = train_logistic(X_train, y_train)

    # Paso 6: Buen modelo y ajuste fino
    rf_model, rf_best_params, rf_cv_score = train_rf(X_train, y_train)

    # Guardar modelos entrenados
    save_model(log_model, "logistic_regression", MODEL_DIR)
    save_model(rf_model, "random_forest", MODEL_DIR)

    model_results = {
        "Logistic Regression": evaluate_multiclass(log_model, X_test, y_test, labels),
        "Random Forest": evaluate_multiclass(rf_model, X_test, y_test, labels),
    }

    confusion_image_files = []
    for model_name, (metrics, cm) in model_results.items():
        print(f"\n{model_name}")
        print("Matriz de confusion (filas = real, columnas = prediccion):")
        cm_df = pd.DataFrame(
            cm,
            index=labels,
            columns=labels,
        )
        print(cm_df)

        # Grafico de matriz de confusion por algoritmo.
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar_kws={"label": "Cantidad de casos"},
        )
        plt.title(f"Matriz de confusión - {model_name}")
        plt.xlabel("Valores predichos")
        plt.ylabel("Valores reales")
        plt.tight_layout()
        model_key = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")

        # Crear subcarpeta por algoritmo
        algo_dir = REPORT_DIR / model_key
        algo_dir.mkdir(parents=True, exist_ok=True)

        image_name = "confusion_matrix.png"
        plt.savefig(algo_dir / image_name, dpi=180)
        plt.close()
        confusion_image_files.append(f"{model_key}/{image_name}")

        summary = (
            "Exactitud={exactitud:.4f}, Precision(Macro)={precision_macro:.4f}, "
            "Sensibilidad(Macro)={sensibilidad_macro:.4f}, "
            "F1(Macro)={f1_macro:.4f}, F1(Weighted)={f1_weighted:.4f}"
        ).format(**metrics)
        print(summary)

    # Ejecutar pipelines de los otros algoritmos
    df_data = pd.read_csv(data_path)
    run_peer_pipelines(df_data, REPORT_DIR, MODEL_DIR)

    report_text = f"""# Informe de clasificacion

## Variable objetivo y
- y original: `Daily_Revenue`

## Definicion de clases
- Bajo (0)
- Medio (1)
- Alto (2)
- Criterio de division: cuantiles (terciles)

## Como leer la matriz facilmente
- Filas: valores reales
- Columnas: valores predichos
- La diagonal principal son aciertos por clase

## Matriz de confusion usada
Filas = etiqueta real (conocida)
Columnas = prediccion del algoritmo

Dimension de la matriz: 3x3

## Metricas
- Cada carpeta de algoritmo incluye su archivo `metricas.csv`.
- En consola se imprime el detalle de cada modelo al ejecutar `python main.py`.

## Graficos generados
- Matrices de confusion por algoritmo:
{chr(10).join([f"- {name}" for name in confusion_image_files])}
- Salidas adicionales por algoritmo:
- arbol_decision/confusion_matrix.png
- arbol_decision/arbol_decision.png
- arbol_decision/importancias.png
- svm/confusion_matrix.png
- red_neuronal/confusion_matrix.png
- red_neuronal/loss_curve.png

## Ajuste de Random Forest
- Mejores hiperparametros: {rf_best_params}
- Mejor F1 en CV: {rf_cv_score:.4f}
"""

    (REPORT_DIR / "informe.md").write_text(report_text, encoding="utf-8")

    print("\nArchivos generados:")
    for name in confusion_image_files:
        print(f"- report/{name}")
    print("- report/arbol_decision/confusion_matrix.png")
    print("- report/svm/confusion_matrix.png")
    print("- report/red_neuronal/confusion_matrix.png")
    print("- report/informe.md")


if __name__ == "__main__":
    main()
