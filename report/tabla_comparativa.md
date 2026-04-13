# Tabla Comparativa de Métricas - Algoritmos de Machine Learning

| Algoritmo | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
|-----------|----------|-------------------|----------------|------------------|---------------------|
| **SVM** | **0.90** | **0.90** | **0.89** | **0.89** | **0.90** |
| Red Neuronal | 0.89 | 0.89 | 0.89 | 0.89 | 0.89 |
| Regresión Logística | 0.8475 | 0.8490 | 0.8474 | 0.8480 | 0.8482 |
| Random Forest | 0.82 | 0.8213 | 0.8198 | 0.8204 | 0.8206 |
| Árbol de Decisión | 0.78 | 0.78 | 0.77 | 0.78 | 0.78 |

## Análisis de Resultados

### Mejor modelo: **SVM (Máquinas de Vector de Soporte)**
- Accuracy: 90%
- Mejor rendimiento en todas las métricas
- Adecuado para datasets con clases linealmente separables

### Segundo mejor: **Red Neuronal (MLP)**
- Accuracy: 89%
- Buena generalización
- Curva de aprendizaje converge adecuadamente

### Recomendación
Para este dataset de ingresos de cafeterías, **SVM es el algoritmo recomendado** debido a su mayor accuracy y estabilidad en todas las métricas.
