# lasso_regression.py

# Implementation of Lasso Regression using dataset/salary_data.csv
# Lasso Regression is useful for feature selection by shrinking some coefficients to zero.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def lasso_regression(X, y, alpha):
    """
    Perform Lasso Regression using scikit-learn's Lasso implementation.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)
    - alpha: float, Regularization strength (higher value = more regularization)

    Returns:
    - model: Trained Lasso regression model
    """
    # Reshape X for scikit-learn compatibility
    X = X.reshape(-1, 1)

    # Create and fit the Lasso model
    model = Lasso(alpha=alpha)
    model.fit(X, y)

    return model


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('../dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Lasso Regression with a chosen regularization strength
    alpha = 1000  # Regularization strength
    model = lasso_regression(X, y, alpha)
    m, b = model.coef_[0], model.intercept_
    print(f"Parameters from Lasso Regression: m = {m:.4f}, b = {b:.4f}")

    # Predict new values
    new_X = np.array([6, 7, 8]).reshape(-1, 1)
    predictions = model.predict(new_X)
    print("\nPredictions:")
    for exp, pred in zip(new_X.flatten(), predictions):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="purple", label="Regression Line (Lasso Regression)")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Lasso Regression with L1 Regularization")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
