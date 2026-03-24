from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.evaluation import evaluate_multiclass


FEATURE_COLUMNS = [
    "Number_of_Customers_Per_Day",
    "Average_Order_Value",
    "Location_Foot_Traffic",
    "Marketing_Spend_Per_Day",
    "Number_of_Employees",
    "Operating_Hours_Per_Day",
]


def run_svm_pipeline(df, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS]
    y = pd.qcut(df["Daily_Revenue"], q=3, labels=[0, 1, 2])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="linear", C=1.0, gamma="scale", random_state=42)
    model.fit(x_train_scaled, y_train)

    eval_labels = [0, 1, 2]
    display_labels = ["Bajo (0)", "Medio (1)", "Alto (2)"]
    metrics, cm = evaluate_multiclass(model, x_test_scaled, y_test, eval_labels)

    print("\n=== REPORTE DE MÁQUINAS DE VECTOR DE SOPORTE (SVM) ===")
    print(f"Precisión Total: {metrics['exactitud']:.2f}")
    print("\nInforme de Clasificación:")
    y_pred = model.predict(x_test_scaled)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusion - SVM")
    plt.ylabel("Valores Reales")
    plt.xlabel("Valores Predichos")
    plt.tight_layout()
    cm_file = report_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=180)
    plt.close()

    pd.DataFrame([metrics]).to_csv(report_dir / "metricas.csv", index=False)

    return {
        "modelo": "SVM",
        "metricas": metrics,
        "archivos": [cm_file.name, "metricas.csv"],
    }
