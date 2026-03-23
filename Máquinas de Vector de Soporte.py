import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv('coffee_shop_revenue.csv')
X = df[['Number_of_Customers_Per_Day', 'Average_Order_Value', 'Location_Foot_Traffic', 
        'Marketing_Spend_Per_Day', 'Number_of_Employees', 'Operating_Hours_Per_Day']]

df['Revenue_Category'] = pd.qcut(df['Daily_Revenue'], q=3, labels=[0, 1, 2])
y = df['Revenue_Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)

print("=== REPORTE DE MÁQUINAS DE VECTOR DE SOPORTE (SVM) ===")
print(f"Precisión Total: {accuracy_score(y_test, y_pred):.2f}")
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
labels = ['Bajo (0)', 'Medio (1)', 'Alto (2)']
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - SVM (Máquinas de Vector de Soporte)')
plt.ylabel('Valores Reales')
plt.xlabel('Valores Predichos')
plt.show()