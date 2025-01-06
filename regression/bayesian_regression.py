# bayesian_regression.py

# Implementation of Bayesian Linear Regression using dataset/salary_data.csv
# Bayesian Regression is useful for capturing uncertainty in predictions and working with small datasets.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

def bayesian_regression(X, y):
    """
    Perform Bayesian Linear Regression using scikit-learn's BayesianRidge implementation.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)

    Returns:
    - model: Trained Bayesian Ridge regression model
    """
    # Reshape X for scikit-learn compatibility
    X = X.reshape(-1, 1)

    # Create and fit the Bayesian Ridge regression model
    model = BayesianRidge()
    model.fit(X, y)

    return model


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('../dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Train Bayesian Regression model
    model = bayesian_regression(X, y)
    m, b = model.coef_[0], model.intercept_
    print(f"Parameters from Bayesian Regression: m = {m:.4f}, b = {b:.4f}")

    # Predict new values with uncertainty estimates
    new_X = np.array([6, 7, 8]).reshape(-1, 1)
    predictions, std_devs = model.predict(new_X, return_std=True)
    print("\nPredictions with Uncertainty:")
    for exp, pred, std in zip(new_X.flatten(), predictions, std_devs):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f} ± ${std:.2f}")

    # Visualization
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred, y_std = model.predict(X_plot, return_std=True)

    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X_plot, y_pred, color="orange", label="Mean Prediction")
    plt.fill_between(
        X_plot.flatten(),
        y_pred - 2 * y_std,
        y_pred + 2 * y_std,
        color="orange",
        alpha=0.2,
        label="Confidence Interval (95%)"
    )
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Bayesian Linear Regression")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
