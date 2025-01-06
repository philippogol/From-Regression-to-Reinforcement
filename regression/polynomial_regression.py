# polynomial_regression.py

# This script implements Polynomial Regression using the dataset 'salary_data.csv'.
# Polynomial Regression extends Linear Regression by modeling the relationship
# between variables as an nth-degree polynomial, allowing for the capture of 
# non-linear trends while retaining interpretability.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_regression(X, y, degree):
    """
    Perform Polynomial Regression by transforming features to polynomial terms.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)
    - degree: int, Degree of the polynomial terms to include

    Returns:
    - model: Trained Polynomial Regression model
    - poly: PolynomialFeatures object for feature transformation
    """
    # Transform input features into polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.reshape(-1, 1))

    # Train a linear regression model on the polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)

    return model, poly


if __name__ == "__main__":
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
    dataset_path = os.path.join(script_dir, '../dataset/salary_data.csv')
    data = pd.read_csv(dataset_path)
    
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Polynomial Regression with a chosen degree
    degree = 3  # Degree of the polynomial
    model, poly = polynomial_regression(X, y, degree)
    print(f"Polynomial Regression coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Predict new values
    new_X = np.array([6, 7, 8]).reshape(-1, 1)
    new_X_poly = poly.transform(new_X)
    predictions = model.predict(new_X_poly)
    print("\nPredictions:")
    for exp, pred in zip(new_X.flatten(), predictions):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    X_poly = poly.transform(X.reshape(-1, 1))
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, model.predict(X_poly), color="orange", label="Polynomial Regression Line")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title(f"Polynomial Regression (Degree = {degree})")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
