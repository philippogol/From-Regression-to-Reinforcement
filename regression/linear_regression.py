import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos reales
data = pd.read_csv('dataset/salary_data.csv')

# Variables independientes (años de experiencia) y dependientes (salario)
X = data['YearsExperience'].values.reshape(-1, 1)  # Convertir a matriz 2D
y = data['Salary'].values

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Coeficiente (Pendiente): {model.coef_[0]:.2f}")
print(f"Intercepto: {model.intercept_:.2f}")

# Visualización de los resultados
plt.scatter(X, y, color="blue", label="Datos reales")
plt.plot(X_test, y_pred, color="red", label="Línea de regresión")
plt.title("Regresión Lineal: Años de Experiencia vs Salario")
plt.xlabel("Años de Experiencia")
plt.ylabel("Salario Anual (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
