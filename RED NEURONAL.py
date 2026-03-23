import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('coffee_shop_revenue.csv')

X = df[['Number_of_Customers_Per_Day', 'Average_Order_Value', 'Location_Foot_Traffic', 
        'Marketing_Spend_Per_Day', 'Number_of_Employees', 'Operating_Hours_Per_Day']]

df['Revenue_Category'] = pd.qcut(df['Daily_Revenue'], q=3, labels=[0, 1, 2])
y = df['Revenue_Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(16, 8), 
                    max_iter=1000, 
                    random_state=42, 
                    activation='relu', 
                    solver='adam')

mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)

print("=== REPORTE DE LA RED NEURONAL ===")
print(f"Precisión Total (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print("\nInforme Detallado:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
labels = ['Bajo (0)', 'Medio (1)', 'Alto (2)']
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Matriz de Confusión - Ingresos Cafetería')
plt.ylabel('Valores Reales (Actual)')
plt.xlabel('Valores Predichos (Predicted)')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, color='orange', linewidth=2)
plt.title('Curva de Aprendizaje - Red Neuronal')
plt.xlabel('Iteraciones / Épocas')
plt.ylabel('Pérdida (Loss)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()