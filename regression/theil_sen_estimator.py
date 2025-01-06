# theil_sen_estimator.py

# Implementation of Theil-Sen Estimator using dataset/salary_data.csv
# Theil-Sen Estimator is robust to outliers and works well for small datasets.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor

def theil_sen_regression(X, y):
    """
    Perform regression using the Theil-Sen Estimator.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)

    Returns:
    - model: Trained Theil-Sen regression model
    """
    # Reshape X for scikit-learn compatibility
    X = X.reshape(-1, 1)

    # Create and fit the Theil-Sen Regression model
    model = TheilSenRegressor()
    model.fit(X, y)

    return model


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('../dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Train Theil-Sen Regression model
    model = theil_sen_regression(X, y)
    m, b = model.coef_[0], model.intercept_
    print(f"Parameters from Theil-Sen Regression: m = {m:.4f}, b = {b:.4f}")

    # Predict new values
    new_X = np.array([6, 7, 8]).reshape(-1, 1)
    predictions = model.predict(new_X)
    print("\nPredictions:")
    for exp, pred in zip(new_X.flatten(), predictions):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_plot)

    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X_plot, y_pred, color="orange", label="Theil-Sen Regression Line")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Theil-Sen Regression (Outlier-Resistant)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
