# Taller - Probar Algoritmos (Ingresos de Cafeterias)

Repositorio para comparar algoritmos de Machine Learning en la prediccion del ingreso diario de cafeterias.

## 1) Objetivo

Comparar el desempeno de 5 algoritmos de clasificacion con el mismo conjunto de datos:

1. Regresion Logistica
2. Maquinas de Vector de Soporte (SVM)
3. Arboles de Decision
4. Random Forest
5. Red Neuronal Artificial (MLP)

## 2) Datos y variables

Variables de entrada `X`:

- `Number_of_Customers_Per_Day`
- `Average_Order_Value`
- `Location_Foot_Traffic`
- `Marketing_Spend_Per_Day`
- `Number_of_Employees`
- `Operating_Hours_Per_Day`

Variable objetivo original `y`:

- `Daily_Revenue`

Transformacion de `y` para clasificacion multiclase:

- `Bajo (0)`
- `Medio (1)`
- `Alto (2)`

La separacion se realiza por terciles (cuantiles).

## 3) Estructura del proyecto

```text
coffe_ML/
|- data/
|  |- coffee_shop_revenue.csv
|- src/
|  |- data_preparation.py
|  |- feature_engineering.py
|  |- evaluation.py
|  |- models/
|     |- logistic_model.py
|     |- random_forest_model.py
|- main.py
|- ARBOLES DE DESICION.py
|- Máquinas de Vector de Soporte.py
|- RED NEURONAL.py
|- report/
|- README.md
|- requirements.txt
```

## 4) Organizacion sin solapamientos

Para evitar solapamiento de responsabilidades:

- `main.py` ejecuta el flujo modular de `src/` y compara Logistic Regression + Random Forest.
- Los scripts independientes en raiz ejecutan cada algoritmo adicional de forma autonoma:
	- `ARBOLES DE DESICION.py`
	- `Máquinas de Vector de Soporte.py`
	- `RED NEURONAL.py`
- Las salidas del flujo modular se guardan en `report/`, con subdirectorios por algoritmo.

## 5) Metricas de evaluacion

Metricas usadas para comparacion (multiclase):

1. `exactitud`
2. `precision_macro`
3. `sensibilidad_macro`
4. `f1_macro`
5. `f1_weighted`

Matriz de confusion:

- Filas: valores reales
- Columnas: valores predichos

## 6) Ejecucion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

### 6.1 Flujo principal (modular)

```bash
python main.py
```

### 6.2 Scripts individuales por algoritmo

```bash
python "ARBOLES DE DESICION.py"
python "Máquinas de Vector de Soporte.py"
python "RED NEURONAL.py"
```

## 7) Salidas en report/

Salida principal:

- `report/metricas_comparacion.csv`
- `report/comparacion_metricas.png`
- `report/informe.md`

Subdirectorios por algoritmo del flujo modular:

- `report/logistic_regression_baseline/`
	- `confusion_matrix.png`
	- `metricas.csv`
- `report/random_forest_tuned/`
	- `confusion_matrix.png`
	- `metricas.csv`

## 8) Nota de reproducibilidad

Se usa `random_state=42` en entrenamiento y division de datos para mantener comparabilidad entre ejecuciones.
