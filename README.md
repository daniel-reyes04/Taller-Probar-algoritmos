# Coffee ML Pipeline - Guia Tecnica para macOS

Proyecto de clasificacion multiclase para ingresos diarios de cafeteria, orientado a ejecucion local en macOS.

## Objetivo tecnico

Ejecutar desde un solo entry point (`main.py`) un pipeline de 5 algoritmos:
1. Logistic Regression
2. Random Forest
3. Arbol de Decision
4. SVM
5. Red Neuronal (MLP)

Fuente unica de datos:
- `data/coffee_shop_revenue.csv`

## Requisitos para macOS

Versiones recomendadas:
1. macOS 12+
2. Python 3.10 a 3.12
3. `pip` actualizado
4. Terminal (`zsh`) o iTerm2

Dependencias del sistema:
1. Xcode Command Line Tools
2. (Opcional) Homebrew

Instalar command line tools:

```bash
xcode-select --install
```

## Setup rapido en macOS

Desde la raiz del proyecto:

```bash
cd /ruta/a/coffe_ML
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Validacion de entorno:

```bash
python --version
pip --version
```

## Estructura del proyecto

Archivos principales:
1. `main.py`
2. `requirements.txt`
3. `data/coffee_shop_revenue.csv`

Modulos de codigo:
1. `src/data_preparation.py`
2. `src/feature_engineering.py`
3. `src/evaluation.py`
4. `src/models/logistic_model.py`
5. `src/models/random_forest_model.py`
6. `src/models/decision_tree_pipeline.py`
7. `src/models/svm_pipeline.py`
8. `src/models/mlp_pipeline.py`
9. `src/models/run_peer_pipelines.py`

Salidas:
1. `report/logistic_regression/`
2. `report/random_forest/`
3. `report/arbol_decision/`
4. `report/svm/`
5. `report/red_neuronal/`
6. `report/informe.md`

## Ejecucion en macOS

Con entorno virtual activo:

```bash
python main.py
```

El flujo ejecuta:
1. Carga y preparacion de datos
2. Clasificacion del target por terciles (`0`, `1`, `2`)
3. Entrenamiento y evaluacion por algoritmo
4. Generacion de matrices de confusion e imagenes
5. Escritura de informe tecnico en `report/informe.md`

## Modelo de datos

Features de entrada:
1. `Number_of_Customers_Per_Day`
2. `Average_Order_Value`
3. `Location_Foot_Traffic`
4. `Marketing_Spend_Per_Day`
5. `Number_of_Employees`
6. `Operating_Hours_Per_Day`

Target:
1. `Daily_Revenue`
2. Transformado a 3 clases por terciles

## Reportes y artefactos

Por cada algoritmo se espera al menos:
1. `confusion_matrix.png`
2. `metricas.csv` (si aplica en ese pipeline)

Adicionales por modelo:
1. Arbol: `arbol_decision.png`, `importancias.png`
2. MLP: `loss_curve.png`

Informe general:
1. `report/informe.md`

## Troubleshooting macOS

### Error: comando `python` no encontrado

Usar `python3`:

```bash
python3 main.py
```

### Error: entorno virtual no activo

Activar:

```bash
source .venv/bin/activate
```

### Error: dataset no encontrado

Verificar existencia de:
1. `data/coffee_shop_revenue.csv`

### Warning de convergencia en Logistic Regression

No bloquea la ejecucion. Es warning, no error fatal.

### Problemas con dependencias en Mac ARM (M1/M2/M3)

Actualizar `pip` y reinstalar dependencias:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

## Operacion recomendada

1. Mantener solo `main.py` como entry point
2. No usar scripts legacy fuera de `src/models`
3. Ejecutar siempre con entorno virtual activo
4. Versionar cambios con commits pequenos y descriptivos

## Estado actual

1. Pipeline modular activo en `src/models`
2. Reportes por algoritmo activos en `report/`
3. Entry point unico: `main.py`
