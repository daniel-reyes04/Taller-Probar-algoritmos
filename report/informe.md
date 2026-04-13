# Informe de Clasificación - Predicción de Ingresos Diarios de Cafeterías

## 1. Descripción del Proyecto

**Objetivo:** Predecir el nivel de ingresos diarios de una cafetería (Bajo, Medio, Alto) basándose en variables operacionales.

**Dataset:** Coffee Shop Daily Revenue Prediction Dataset (Kaggle)

**Variables de entrada:**
- Number_of_Customers_Per_Day
- Average_Order_Value
- Location_Foot_Traffic
- Marketing_Spend_Per_Day
- Number_of_Employees
- Operating_Hours_Per_Day

**Variable objetivo:** Daily_Revenue (categorizada en 3 clases)

---

## 2. Ciclo de Vida de Machine Learning

| Paso | Descripción | Estado | Archivo/Función |
|------|-------------|--------|-----------------|
| 1. Planificación | Definición del problema y objetivos | ✅ Completado | Planteamiento del proyecto |
| 2. Preparación de datos | Carga, limpieza, selección de features | ✅ Completado | `src/data_preparation.py` |
| 3. Ingeniería de modelos | Entrenamiento de 5 algoritmos | ✅ Completado | `src/models/*.py` |
| 4. Evaluación del modelo | Métricas y matrices de confusión | ✅ Completado | `src/evaluation.py` |
| 5. Implementación del modelo | Guardado de modelos entrenados | ✅ Completado | `src/model_deploy.py`, carpeta `models/` |
| 6. Supervisión y mantenimiento | No aplica (trabajo académico) | N/A | - |

---

## 3. Metodología

### 3.1 Preparación de Datos
- División de datos: 80% entrenamiento, 20% prueba
- Estratificación por clase
- Manejo de valores nulos
- Categorización de la variable objetivo usando cuantiles (terciles)

### 3.2 Protocolo de Evaluación
- Métricas: Accuracy, Precision, Recall, F1-Score
- Validación cruzada (5-fold) para Random Forest
- Matriz de confusión 3x3

---

## 4. Resultados

### 4.1 Tabla Comparativa de Métricas

| Algoritmo | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
|-----------|----------|-------------------|----------------|------------------|---------------------|
| **SVM** | **0.90** | **0.90** | **0.89** | **0.89** | **0.90** |
| Red Neuronal | 0.89 | 0.89 | 0.89 | 0.89 | 0.89 |
| Regresión Logística | 0.8475 | 0.8490 | 0.8474 | 0.8480 | 0.8482 |
| Random Forest | 0.82 | 0.8213 | 0.8198 | 0.8204 | 0.8206 |
| Árbol de Decisión | 0.78 | 0.78 | 0.77 | 0.78 | 0.78 |

### 4.2 Mejor Modelo: **SVM (Máquinas de Vector de Soporte)**
- Accuracy: 90%
- Precision clase Bajo: 0.94
- Precision clase Medio: 0.81
- Precision clase Alto: 0.93

---

## 5. Conclusiones

1. **SVM es el mejor algoritmo** para este dataset con un 90% de accuracy
2. **Red Neuronal** también muestra buen rendimiento (89%)
3. **Random Forest** con ajuste de hiperparámetros no supera a SVM
4. **Árbol de Decisión** tiene el menor rendimiento pero permite interpretación

### Recomendación
Para implementar en producción, se recomienda **SVM** por su mayor accuracy y estabilidad.

---

## 6. Archivos Generados

### Modelos guardados (carpeta `models/`):
- logistic_regression.joblib
- random_forest.joblib
- decision_tree.joblib
- svm.joblib + svm_scaler.joblib
- neural_network.joblib + neural_network_scaler.joblib

### Reportes (carpeta `report/`):
- Informe general (este archivo)
- Tabla comparativa de métricas
- Matrices de confusión por algoritmo
- Árbol de decisión visual
- Importancia de variables
- Curva de aprendizaje (Red Neuronal)
- Métricas CSV por algoritmo

---

## 7. Ejecución del Proyecto

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo
python main.py

# Los modelos se guardan en la carpeta models/
# Los reportes se guardan en la carpeta report/
```

---

*Proyecto desarrollado como parte del taller de predicción de Machine Learning*
