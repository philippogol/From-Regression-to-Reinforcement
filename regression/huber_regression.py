# huber_regression.py

# Implementation of Huber Regression using dataset/salary_data.csv
# Huber Regression is useful for handling datasets with outliers by combining L1 and L2 loss functions.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

def huber_regression(X, y, epsilon):
    """
    Perform Huber Regression using scikit-learn's HuberRegressor implementation.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)
    - epsilon: float, The threshold at which the loss function transitions from L2 to L1

    Returns:
    - model: Trained Huber regression model
    """
    # Reshape X for scikit-learn compatibility
    X = X.reshape(-1, 1)

    # Create and fit the Huber Regression model
    model = HuberRegressor(epsilon=epsilon)
    model.fit(X, y)

    return model


if __name__ == "__main__":
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
    dataset_path = os.path.join(script_dir, '../dataset/salary_data.csv')
    data = pd.read_csv(dataset_path)
    
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Train Huber Regression model
    epsilon = 1.35  # Default epsilon for Huber loss
    model = huber_regression(X, y, epsilon)
    m, b = model.coef_[0], model.intercept_
    print(f"Parameters from Huber Regression: m = {m:.4f}, b = {b:.4f}")

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
    plt.plot(X_plot, y_pred, color="green", label="Huber Regression Line")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Huber Regression (Outlier-Resistant)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
