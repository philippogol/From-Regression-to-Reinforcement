import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generar datos sintéticos
np.random.seed(42)
age = np.random.uniform(0, 15, 100)  # Edad del automóvil (en años)
mileage = np.random.uniform(0, 200000, 100)  # Kilometraje (en km)
price = 30000 - (age * 1000) - (mileage * 0.05) + np.random.normal(0, 2000, 100)  # Precio con ruido

# Crear matriz de características y variable objetivo
X = np.column_stack((age, mileage))  # Características: edad y kilometraje
y = price  # Precio del automóvil

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
model = DecisionTreeRegressor(max_depth=4, random_state=42)  # Limitar profundidad para interpretabilidad
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", label="Predicciones vs Reales")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", label="Línea ideal")
plt.title("Árbol de Decisión: Precio del Automóvil")
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
