import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generar datos sintéticos
np.random.seed(42)
X = np.random.uniform(0, 10, 100).reshape(-1, 1)  # Variable independiente (ej: concentración de CO2)
y = 2 * X ** 3 - 10 * X ** 2 + 5 * X + np.random.normal(0, 10, 100).reshape(-1, 1)  # Relación no lineal con ruido

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.ravel())  # Ajustar modelo (y_train.ravel() aplana el array)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Datos reales")
plt.scatter(X_test, y_pred, color="red", label="Predicciones")
plt.title("Random Forest Regression: Relación No Lineal")
plt.xlabel("Variable Independiente")
plt.ylabel("Variable Dependiente")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
