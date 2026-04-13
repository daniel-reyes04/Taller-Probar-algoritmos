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

## 2. Ciclo de Vida de Machine Learning (CRISP-ML(Q))

### 2.1 Planificación ✅

**Evaluación del alcance y viabilidad:**
- ¿Necesitamos machine learning? Sí, para predecir niveles de ingresos
- ¿Es viable? El dataset tiene 2000 ejemplos suficientes

**Métricas de éxito definidas:**
- Negocio: Clasificación correcta de ingresos (Bajo/Medio/Alto)
- ML: Accuracy ≥ 80%, F1-Score ≥ 0.75

**Análisis de disponibilidad de datos:**
- ✅ Datos disponibles en Kaggle
- ✅ Distribución de clases balanceada (terciles)
- ✅ Variables relevantes para el problema

**Limitaciones consideradas:**
- Limitaciones legales: Datos públicos y éticos
- Robustez: Validación con train/test split

---

### 2.2 Preparación de Datos ✅

**Recopilación de datos:**
- Carga del dataset desde CSV
- Eliminación de valores nulos

**Limpieza de datos:**
- dropna() para valores faltantes
- Verificación de tipos de datos

**Tratamiento de datos:**
- Selección de 6 features relevantes
- Categorización de variable objetivo (cuantiles/terciles)
- Normalización para SVM y Red Neuronal

**Gestión de datos:**
- División 80% entrenamiento, 20% prueba
- Estratificación por clase

---

### 2.3 Ingeniería de Modelos ✅

**Algoritmos implementados (5 modelos):**
1. Regresión Logística (baseline)
2. Máquinas de Vector de Soporte (SVM)
3. Árbol de Decisión
4. Random Forest (con GridSearchCV)
5. Red Neuronal (MLP)

**Definición de métricas:**
- Accuracy (exactitud)
- Precision
- Recall
- F1-Score

**Entrenamiento y validación:**
- Validación cruzada 5-fold para Random Forest
- Train/test split con random_state=42

---

### 2.4 Evaluación del Modelo ✅

**Pruebas realizadas:**
- Evaluación en conjunto de datos de prueba (20%)
- Matriz de confusión 3x3
- Reportes de clasificación por clase

**Métricas obtenidas:**

| Algoritmo | Accuracy | F1-Score (Macro) |
|-----------|----------|------------------|
| **SVM** | **0.90** | **0.89** |
| Red Neuronal | 0.89 | 0.89 |
| Regresión Logística | 0.8475 | 0.8480 |
| Random Forest | 0.82 | 0.8204 |
| Árbol de Decisión | 0.78 | 0.78 |

**Análisis de resultados:**
- ✅ Mejor modelo: SVM con 90% accuracy
- ✅ Cumplimiento de métricas de éxito (≥80%)
- ✅ Modelo listo para implementación

---

### 2.5 Implementación del Modelo ✅

**Modelos guardados (carpeta `models/`):**
- logistic_regression.joblib
- random_forest.joblib
- decision_tree.joblib
- svm.joblib + svm_scaler.joblib
- neural_network.joblib + neural_network_scaler.joblib

**Formato de despliegue:**
- Formato joblib (serialización)
- Listo para carga con joblib.load()

**Acceso a predicciones:**
- Mediante API o aplicación web
- Función load_model() disponible en model_deploy.py

---

### 2.6 Supervisión y Mantenimiento ⚠️

**Estado:** No implementado (trabajo académico)

**En producción real se requeriría:**
- Monitoreo de métricas del modelo
- Alertas automáticas por degradación
- Reentrenamiento periódico con nuevos datos
- Control de versiones de modelos

---

## 3. Conclusiones

1. **SVM es el mejor algoritmo** para este dataset con 90% de accuracy
2. **Red Neuronal** también muestra buen rendimiento (89%)
3. Todos los modelos superan el umbral de 80% establecido
4. Los modelos están guardados y listos para implementación

### Recomendación Final
Para implementar en producción, se recomienda **SVM** por su mayor accuracy, velocidad de inferencia y estabilidad.

---

## 4. Archivos Generados

### Modelos (carpeta `models/`): 7 archivos
### Reportes (carpeta `report/`): 
- informe.md (este archivo)
- tabla_comparativa.md
- metricas.csv por algoritmo
- Visualizaciones: matrices de confusión, árbol, curva de aprendizaje

---

## 5. Ejecución

```bash
# Ejecutar pipeline
python main.py

# Cargar modelo para predicciones
import joblib
model = joblib.load("models/svm.joblib")
```

---

*Informe basado en metodología CRISP-ML(Q)*
*Proyecto desarrollado para el taller de Machine Learning*
