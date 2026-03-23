# Coffee ML Pipeline

Pipeline de clasificacion multiclase para `Daily_Revenue`, alineado con el enfoque Bajo/Medio/Alto.

## 1) Variables del modelo

`X`:
- `Number_of_Customers_Per_Day`
- `Average_Order_Value`
- `Location_Foot_Traffic`
- `Marketing_Spend_Per_Day`
- `Number_of_Employees`
- `Operating_Hours_Per_Day`

`y`:
- `Daily_Revenue`

Transformacion de `y` para clasificacion:
- `Bajo (0)`
- `Medio (1)`
- `Alto (2)`

La separacion se hace por terciles (cuantiles), para mantener equilibrio entre clases.

## 2) Metricas priorizadas

El proyecto prioriza estas metricas para comparar algoritmos:
1. `exactitud`
2. `precision_macro`
3. `sensibilidad_macro`
4. `f1_macro`

## 3) Matriz de confusion

Formato usado (estandar academico):
- Filas: valores reales
- Columnas: valores predichos

Para 3 clases, la matriz es `3x3`.

## 4) Algoritmos comparados

- Logistic Regression (baseline)
- Random Forest (tuned con GridSearchCV)

## 5) Como ejecutar

```bash
pip install -r requirements.txt
python main.py
```

## 6) Salidas en report/

- Subdirectorios por algoritmo (uno por cada modelo evaluado), por ejemplo:
	- `report/logistic_regression_baseline/`
	- `report/random_forest_tuned/`

- En cada subdirectorio:
	- `confusion_matrix.png`
	- `metricas.csv`

- `metricas_comparacion.csv`
- `comparacion_metricas.png`
- `informe.md`

## 7) Archivos principales

- `main.py`: orquesta pipeline, entrenamiento, evaluacion y reporte.
- `src/data_preparation.py`: carga y limpieza de datos.
- `src/feature_engineering.py`: construccion de clases Bajo/Medio/Alto.
- `src/evaluation.py`: matriz de confusion y metricas principales.
- `src/models/logistic_model.py`: baseline.
- `src/models/random_forest_model.py`: modelo ajustado.

## 8) Scripts individuales incorporados del remoto

Adicionalmente, el repositorio incluye scripts separados para otros algoritmos:

- `RED NEURONAL.py`
- `ARBOLES DE DESICION.py`
- `Máquinas de Vector de Soporte.py`

Dataset usado por estos scripts:

- `coffee_shop_revenue.csv`

Ejecucion manual (si deseas correrlos por separado):

```bash
python "RED NEURONAL.py"
python "ARBOLES DE DESICION.py"
python "Máquinas de Vector de Soporte.py"
```

Nota: algunos de estos scripts abren figuras y pueden pausar la ejecucion hasta cerrar la ventana.
