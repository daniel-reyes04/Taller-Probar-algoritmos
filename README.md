# Coffee ML Pipeline

Proyecto de clasificacion multiclase de ingresos diarios de cafeteria.

## Resumen

El proyecto ejecuta 5 algoritmos desde un unico punto de entrada:
- Logistic Regression
- Random Forest
- Arbol de Decision
- SVM
- Red Neuronal (MLP)

La ejecucion principal esta en main.py y usa una sola fuente de datos en data/coffee_shop_revenue.csv.

## Estructura

- main.py
- README.md
- requirements.txt
- data/coffee_shop_revenue.csv
- src/data_preparation.py
- src/feature_engineering.py
- src/evaluation.py
- src/models/logistic_model.py
- src/models/random_forest_model.py
- src/models/decision_tree_pipeline.py
- src/models/svm_pipeline.py
- src/models/mlp_pipeline.py
- src/models/run_peer_pipelines.py
- report/

## Ejecucion

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecutar pipeline:

```bash
python main.py
```

## Flujo de main.py

1. Carga datos desde data/coffee_shop_revenue.csv.
2. Prepara y clasifica Daily_Revenue en 3 clases: 0, 1, 2.
3. Entrena Logistic Regression y Random Forest.
4. Ejecuta Arbol, SVM y MLP desde src/models/run_peer_pipelines.py.
5. Genera reportes por algoritmo en report/.

## Reportes generados

Por cada algoritmo se generan salidas en su carpeta:

- report/logistic_regression/
- report/random_forest/
- report/arbol_decision/
- report/svm/
- report/red_neuronal/

Ademas, se genera:

- report/informe.md

Nota: la comparativa global en codigo y graficas fue removida para simplificar la ejecucion y evitar errores.

## Variables

Features:
- Number_of_Customers_Per_Day
- Average_Order_Value
- Location_Foot_Traffic
- Marketing_Spend_Per_Day
- Number_of_Employees
- Operating_Hours_Per_Day

Target:
- Daily_Revenue, transformado a 3 clases con terciles.

## Troubleshooting

- Si falta el dataset, verificar data/coffee_shop_revenue.csv.
- Si hay warning de convergencia en Logistic Regression, no bloquea la ejecucion.
- Si VS Code muestra errores antiguos de archivos borrados, recargar la ventana.

## Estado actual

- Punto de entrada unico: main.py
- Scripts legacy en raiz: eliminados
- Pipeline modular en src/models: activo
- Reportes por algoritmo: activos
