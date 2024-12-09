# decision_tree_regression.py

# Implementation of Decision Tree Regression using dataset/salary_data.csv
# Decision Tree Regression is useful for capturing non-linear relationships and handling complex data distributions.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def decision_tree_regression(X, y, max_depth=None):
    """
    Perform Decision Tree Regression.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)
    - max_depth: int or None, The maximum depth of the tree

    Returns:
    - model: Trained Decision Tree Regression model
    """
    # Reshape X for scikit-learn compatibility
    X = X.reshape(-1, 1)

    # Create and fit the Decision Tree Regression model
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X, y)

    return model


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Train Decision Tree Regression model
    max_depth = 5
    model = decision_tree_regression(X, y, max_depth)

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
    plt.plot(X_plot, y_pred, color="green", label="Decision Tree Regression Line")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Decision Tree Regression")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
