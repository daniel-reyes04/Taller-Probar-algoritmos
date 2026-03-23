# Taller-Probar-algoritmoGuía de Configuración y Ejecución de Modelos de Inteligencia Computacional
Estudiante: Daniel Alejandro Reyes León

Entorno de Pruebas: macOS (Apple Silicon / Intel)

Proyecto: Clasificación de Ingresos de Cafetería mediante Redes Neuronales, Árboles de Decisión y SVM.

1. Requisitos del Sistema y Dependencias
Para la correcta ejecución de los scripts, el sistema debe contar con Python 3.9 o superior. Los modelos utilizan librerías de terceros que deben ser instaladas previamente para evitar errores de importación (ModuleNotFoundError).

Instalación de Librerías
Abra la Terminal de su Mac y ejecute el siguiente comando para instalar las dependencias necesarias:

Bash
pip3 install pandas scikit-learn matplotlib seaborn
Nota: Se recomienda el uso de un entorno virtual para mantener la integridad de las dependencias.

2. Estructura de Archivos
Es estrictamente necesario que todos los archivos se encuentren en el mismo directorio (carpeta) para que el lector de datos de los scripts pueda localizar el dataset:

📂 RED NEURONAL.py

📂 ARBOLES DE DESICION.py

📂 Máquinas de Vector de Soporte.py

📂 coffee_shop_revenue.csv (Dataset fuente)

3. Instrucciones de Compilación y Ejecución
Los scripts están optimizados para ejecutarse mediante el intérprete de Python 3. Siga los pasos a continuación:

Navegación: En la Terminal, sitúese en la carpeta que contiene los archivos:

Bash
cd [RUTA_DE_LA_CARPETA]
Ejecución de Scripts: Ejecute el modelo deseado mediante los siguientes comandos:

python3 "RED NEURONAL.py"

python3 "ARBOLES DE DESICION.py"

python3 "Máquinas de Vector de Soporte.py"

4. Observaciones en el Entorno macOS
Visualización de Gráficos: Cada script genera visualizaciones (Matriz de Confusión, Curva de Pérdida o Árbol). El proceso de ejecución se pausará hasta que la ventana del gráfico sea cerrada manualmente por el usuario.

Escalamiento de Datos: Tanto la Red Neuronal como el SVM incluyen internamente el proceso de StandardScaler. No es necesario pre-procesar el archivo CSV.

Gestión de Memoria: Los modelos han sido configurados con un random_state fijo para garantizar la reproducibilidad de los resultados mostrados en el reporte escrito.
