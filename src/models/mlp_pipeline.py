from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.evaluation import evaluate_multiclass


FEATURE_COLUMNS = [
    "Number_of_Customers_Per_Day",
    "Average_Order_Value",
    "Location_Foot_Traffic",
    "Marketing_Spend_Per_Day",
    "Number_of_Employees",
    "Operating_Hours_Per_Day",
]


def run_mlp_pipeline(df, report_dir: Path, model_dir: Path = None):
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

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=1000,
        random_state=42,
        activation="relu",
        solver="adam",
    )
    model.fit(x_train_scaled, y_train)

    eval_labels = [0, 1, 2]
    display_labels = ["Bajo (0)", "Medio (1)", "Alto (2)"]
    metrics, cm = evaluate_multiclass(model, x_test_scaled, y_test, eval_labels)

    print("\n=== REPORTE DE LA RED NEURONAL ===")
    print(f"Precisión Total (Accuracy): {metrics['exactitud']:.2f}")
    print("\nInforme Detallado:")
    y_pred = model.predict(x_test_scaled)
    print(classification_report(y_test, y_pred))
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Matriz de Confusion - Red Neuronal")
    plt.ylabel("Valores Reales")
    plt.xlabel("Valores Predichos")
    plt.tight_layout()
    cm_file = report_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, color="orange", linewidth=2)
    plt.title("Curva de Aprendizaje - Red Neuronal")
    plt.xlabel("Iteraciones / Epocas")
    plt.ylabel("Perdida (Loss)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    loss_file = report_dir / "loss_curve.png"
    plt.savefig(loss_file, dpi=180)
    plt.close()

    pd.DataFrame([metrics]).to_csv(report_dir / "metricas.csv", index=False)

    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "neural_network.joblib")
        joblib.dump(scaler, model_dir / "neural_network_scaler.joblib")

    return {
        "modelo": "Red Neuronal (MLP)",
        "metricas": metrics,
        "archivos": [cm_file.name, loss_file.name, "metricas.csv"],
    }
