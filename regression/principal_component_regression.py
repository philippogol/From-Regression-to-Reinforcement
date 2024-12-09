# principal_component_regression.py

# Implementation of Principal Component Regression (PCR) using dataset/salary_data.csv
# Use PCR when dealing with high-dimensional data or multicollinearity among predictors.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

def principal_component_regression(X, y, n_components):
    """
    Perform Principal Component Regression (PCR) by reducing dimensions with PCA 
    and fitting a Linear Regression model on the principal components.

    Parameters:
    - X: np.array, Independent variables (Years of Experience or expanded features)
    - y: np.array, Dependent variable (Salary)
    - n_components: int, Number of principal components to retain

    Returns:
    - model: Trained Linear Regression model on the principal components
    - pca: PCA object for feature transformation
    """
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Train a linear regression model on the principal components
    model = LinearRegression()
    model.fit(X_pca, y)

    return model, pca


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('dataset/salary_data.csv')
    X = data['YearsExperience'].values.reshape(-1, 1)  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Expand features for demonstration
    poly_degree = 3  # Expand Years of Experience to polynomial terms
    poly = PolynomialFeatures(degree=poly_degree)
    X_expanded = poly.fit_transform(X)

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X_expanded, y, test_size=0.2, random_state=42)

    # Principal Component Regression with 2 components
    n_components = 2
    model, pca = principal_component_regression(X_train, y_train, n_components)
    print(f"Explained Variance Ratios: {pca.explained_variance_ratio_}")

    # Predictions on the test set
    X_test_pca = pca.transform(X_test)
    predictions = model.predict(X_test_pca)
    mse = np.mean((y_test - predictions) ** 2)
    print(f"\nMean Squared Error on Test Set: {mse:.2f}")

    # Predict new values
    new_X = np.array([6, 7, 8]).reshape(-1, 1)
    new_X_expanded = poly.transform(new_X)
    new_X_pca = pca.transform(new_X_expanded)
    predictions_new = model.predict(new_X_pca)
    print("\nPredictions for New Data:")
    for exp, pred in zip(new_X.flatten(), predictions_new):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    plt.scatter(y_test, predictions, color="blue", label="Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", label="Perfect Prediction")
    plt.title("Principal Component Regression: Actual vs Predicted")
    plt.xlabel("Actual Salaries")
    plt.ylabel("Predicted Salaries")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
