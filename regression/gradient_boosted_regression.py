# gradient_boosted_regression.py

# Implementation of Gradient-Boosted Regression using dataset/salary_data.csv
# Gradient-Boosted Regression combines weak learners to minimize error iteratively.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

def gradient_boosted_regression(X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Perform Gradient-Boosted Regression.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)
    - n_estimators: int, The number of boosting stages
    - learning_rate: float, Learning rate shrinks the contribution of each tree
    - max_depth: int, The maximum depth of the individual regression estimators

    Returns:
    - model: Trained Gradient-Boosted Regression model
    """
    # Reshape X for scikit-learn compatibility
    X = X.reshape(-1, 1)

    # Create and fit the Gradient-Boosted Regression model
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    model.fit(X, y)

    return model


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Train Gradient-Boosted Regression model
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3
    model = gradient_boosted_regression(X, y, n_estimators, learning_rate, max_depth)

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
    plt.plot(X_plot, y_pred, color="orange", label="Gradient-Boosted Regression Line")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Gradient-Boosted Regression")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
