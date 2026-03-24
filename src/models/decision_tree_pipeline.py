from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from src.evaluation import evaluate_multiclass


FEATURE_COLUMNS = [
    "Number_of_Customers_Per_Day",
    "Average_Order_Value",
    "Location_Foot_Traffic",
    "Marketing_Spend_Per_Day",
    "Number_of_Employees",
    "Operating_Hours_Per_Day",
]


def run_decision_tree_pipeline(df, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS]
    y = pd.qcut(df["Daily_Revenue"], q=3, labels=[0, 1, 2])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeClassifier(max_depth=4, criterion="entropy", ccp_alpha=0.0, random_state=42)
    model.fit(X_train, y_train)

    eval_labels = [0, 1, 2]
    display_labels = ["Bajo (0)", "Medio (1)", "Alto (2)"]
    metrics, cm = evaluate_multiclass(model, X_test, y_test, eval_labels)

    print("\n=== REPORTE DEL ÁRBOL DE DECISIÓN ===")
    print(f"Precisión Total: {metrics['exactitud']:.2f}")
    print("\nInforme de Clasificación:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusion - Arbol de Decision")
    plt.ylabel("Valores Reales")
    plt.xlabel("Valores Predichos")
    plt.tight_layout()
    cm_file = report_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=180)
    plt.close()

    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["Bajo", "Medio", "Alto"],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Estructura de Decision para los Ingresos de la Cafeteria")
    plt.tight_layout()
    tree_file = report_dir / "arbol_decision.png"
    plt.savefig(tree_file, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    importancias = pd.Series(model.feature_importances_, index=X.columns)
    importancias.nlargest(6).sort_values().plot(kind="barh", color="orange")
    plt.title("Variables mas influyentes segun el Arbol")
    plt.xlabel("Importancia Relativa")
    plt.tight_layout()
    imp_file = report_dir / "importancias.png"
    plt.savefig(imp_file, dpi=180)
    plt.close()

    pd.DataFrame([metrics]).to_csv(report_dir / "metricas.csv", index=False)

    return {
        "modelo": "Arbol de Decision",
        "metricas": metrics,
        "archivos": [cm_file.name, tree_file.name, imp_file.name, "metricas.csv"],
    }
