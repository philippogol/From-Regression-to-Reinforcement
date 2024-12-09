#Polynomial regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generar datos sintéticos (publicidad vs ingresos)
np.random.seed(42)
X = np.random.uniform(0, 10, 100).reshape(-1, 1)  # Inversión en publicidad
y = 5 * X**2 + 10 * X + 7 + np.random.normal(0, 10, 100).reshape(-1, 1)  # Relación cuadrática con ruido

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear características polinómicas
poly_features = PolynomialFeatures(degree=2)  # Grado del polinomio
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Ajustar modelo de regresión lineal sobre características polinómicas
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predicciones
y_pred = model.predict(X_test_poly)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualizar resultados
plt.scatter(X, y, color="blue", label="Datos reales")
plt.scatter(X_test, y_pred, color="red", label="Predicciones", alpha=0.7)
plt.plot(
    np.sort(X, axis=0), 
    model.predict(poly_features.transform(np.sort(X, axis=0))),
    color="green", label="Regresión Polinómica"
)
plt.title("Regresión Polinómica: Publicidad vs Ingresos")
plt.xlabel("Inversión en Publicidad (x1000 USD)")
plt.ylabel("Ingresos (x1000 USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
