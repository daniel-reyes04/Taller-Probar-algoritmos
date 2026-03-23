from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data_preparation import load_and_prepare_data
from src.feature_engineering import categorize_revenue
from src.models.logistic_model import train_logistic
from src.models.random_forest_model import train_rf
from src.evaluation import evaluate_multiclass

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "report"
SUCCESS_METRIC = "f1_macro"
SUCCESS_THRESHOLD = 0.80

def main():
	REPORT_DIR.mkdir(parents=True, exist_ok=True)

	data_path_candidates = [
		BASE_DIR / "data" / "coffee.csv",
		BASE_DIR / "data" / "coffee_shop_revenue.csv",
	]
	data_path = next((path for path in data_path_candidates if path.exists()), None)

	if data_path is None:
		raise FileNotFoundError("No se encontro un archivo de datos valido en la carpeta data/.")

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

	model_results = {
		"Logistic Regression (Baseline)": evaluate_multiclass(log_model, X_test, y_test, labels),
		"Random Forest (Tuned)": evaluate_multiclass(rf_model, X_test, y_test, labels),
	}

	rows = []
	confusion_image_files = []
	per_algorithm_metric_files = []
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
		model_report_dir = REPORT_DIR / model_key
		model_report_dir.mkdir(parents=True, exist_ok=True)
		image_name = "confusion_matrix.png"
		plt.savefig(model_report_dir / image_name, dpi=180)
		plt.close()
		confusion_image_files.append(f"{model_key}/{image_name}")

		summary = (
			"Exactitud={exactitud:.4f}, Precision(Macro)={precision_macro:.4f}, "
			"Sensibilidad(Macro)={sensibilidad_macro:.4f}, "
			"F1(Macro)={f1_macro:.4f}, F1(Weighted)={f1_weighted:.4f}"
		).format(**metrics)
		print(summary)

		metrics_df = pd.DataFrame([metrics])
		metric_file_name = "metricas.csv"
		metrics_df.to_csv(model_report_dir / metric_file_name, index=False)
		per_algorithm_metric_files.append(f"{model_key}/{metric_file_name}")

		rows.append({"algoritmo": model_name, **metrics})

	comparison_df = pd.DataFrame(rows)
	comparison_path = REPORT_DIR / "metricas_comparacion.csv"
	comparison_df.to_csv(comparison_path, index=False)

	metrics_plot_df = comparison_df.melt(
		id_vars="algoritmo",
		value_vars=["exactitud", "precision_macro", "sensibilidad_macro", "f1_macro", "f1_weighted"],
		var_name="metrica",
		value_name="valor",
	)

	plt.figure(figsize=(9, 5))
	sns.barplot(data=metrics_plot_df, x="metrica", y="valor", hue="algoritmo")
	plt.ylim(0, 1)
	plt.title("Comparación de métricas por algoritmo")
	plt.xlabel("Métrica")
	plt.ylabel("Valor")
	plt.legend(title="Algoritmo")
	plt.tight_layout()
	comparison_chart_name = "comparacion_metricas.png"
	plt.savefig(REPORT_DIR / comparison_chart_name, dpi=180)
	plt.close()

	best_row = comparison_df.sort_values(by=SUCCESS_METRIC, ascending=False).iloc[0]
	best_model_name = best_row["algoritmo"]
	best_model_score = best_row[SUCCESS_METRIC]
	success_message = "Cumple" if best_model_score >= SUCCESS_THRESHOLD else "No cumple"

	report_text = f"""# Informe de clasificacion

## Variable objetivo y
- y original: `Daily_Revenue`

## Definicion de clases (alineado con tu companero)
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

## Metricas priorizadas por algoritmo
{comparison_df.to_markdown(index=False)}

## Graficos generados
- Matrices de confusion por algoritmo:
{chr(10).join([f"- {name}" for name in confusion_image_files])}
- Metricas por algoritmo:
{chr(10).join([f"- {name}" for name in per_algorithm_metric_files])}
- Grafica comparativa de metricas:
- {comparison_chart_name}

## Criterio de exito
- Metrica principal: F1(Macro)
- Umbral: {SUCCESS_THRESHOLD:.2f}
- Mejor modelo: {best_model_name} ({best_model_score:.4f})
- Estado: {success_message}

## Ajuste de Random Forest
- Mejores hiperparametros: {rf_best_params}
- Mejor F1 en CV: {rf_cv_score:.4f}
"""

	(REPORT_DIR / "informe.md").write_text(report_text, encoding="utf-8")

	print("\nArchivos generados:")
	print("- report/metricas_comparacion.csv")
	for name in confusion_image_files:
		print(f"- report/{name}")
	for name in per_algorithm_metric_files:
		print(f"- report/{name}")
	print(f"- report/{comparison_chart_name}")
	print("- report/informe.md")


if __name__ == "__main__":
	main()