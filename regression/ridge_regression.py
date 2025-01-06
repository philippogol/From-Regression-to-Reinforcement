# ridge_regression.py

# Implementation of Ridge Regression using dataset/salary_data.csv
# Ridge Regression is useful when features are highly correlated (multicollinearity) or to prevent overfitting.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ridge_regression(X, y, alpha):
    """
    Compute the optimal parameters for Ridge Regression using the Normal Equation with L2 regularization.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)
    - alpha: float, Regularization strength (higher value = more regularization)

    Returns:
    - theta: np.array, Optimal parameters [slope (m), intercept (b)]
    """
    # Add a column of ones to X for the intercept (b)
    X_b = np.hstack([X.reshape(-1, 1), np.ones((X.shape[0], 1))])

    # Identity matrix for regularization (exclude intercept term)
    identity = np.eye(X_b.shape[1])
    identity[-1, -1] = 0  # Don't regularize the intercept term

    # Compute the optimal parameters using Ridge Regression formula
    theta = np.linalg.inv(X_b.T.dot(X_b) + alpha * identity).dot(X_b.T).dot(y)

    return theta


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('../dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Ridge Regression with a chosen regularization strength
    alpha = 10  # Regularization strength
    theta = ridge_regression(X, y, alpha)
    m, b = theta[0], theta[1]
    print(f"Parameters from Ridge Regression: m = {m:.4f}, b = {b:.4f}")

    # Predict new values
    new_X = np.array([6, 7, 8])
    new_X_b = np.hstack([new_X.reshape(-1, 1), np.ones((new_X.shape[0], 1))])
    predictions = new_X_b.dot(theta)
    print("\nPredictions:")
    for exp, pred in zip(new_X, predictions):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, m * X + b, color="orange", label="Regression Line (Ridge Regression)")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Ridge Regression with L2 Regularization")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
