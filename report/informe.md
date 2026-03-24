# Informe de clasificacion

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
- logistic_regression/confusion_matrix.png
- random_forest/confusion_matrix.png
- Salidas adicionales por algoritmo:
- arbol_decision/confusion_matrix.png
- arbol_decision/arbol_decision.png
- arbol_decision/importancias.png
- svm/confusion_matrix.png
- red_neuronal/confusion_matrix.png
- red_neuronal/loss_curve.png

## Ajuste de Random Forest
- Mejores hiperparametros: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
- Mejor F1 en CV: 0.8510
