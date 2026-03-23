# Informe de clasificacion

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
| algoritmo                      |   exactitud |   precision_macro |   sensibilidad_macro |   f1_macro |   f1_weighted |
|:-------------------------------|------------:|------------------:|---------------------:|-----------:|--------------:|
| Logistic Regression (Baseline) |      0.8475 |          0.848961 |             0.84738  |   0.848028 |      0.848163 |
| Random Forest (Tuned)          |      0.82   |          0.821296 |             0.819848 |   0.82041  |      0.820553 |

## Graficos generados
- Matrices de confusion por algoritmo:
- logistic_regression_baseline/confusion_matrix.png
- random_forest_tuned/confusion_matrix.png
- Metricas por algoritmo:
- logistic_regression_baseline/metricas.csv
- random_forest_tuned/metricas.csv
- Grafica comparativa de metricas:
- comparacion_metricas.png

## Criterio de exito
- Metrica principal: F1(Macro)
- Umbral: 0.80
- Mejor modelo: Logistic Regression (Baseline) (0.8480)
- Estado: Cumple

## Ajuste de Random Forest
- Mejores hiperparametros: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
- Mejor F1 en CV: 0.8510
