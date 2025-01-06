# normal_equation.py

# Implementation of the Normal Equation for Linear Regression using dataset/salary_data.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normal_equation(X, y):
    """
    Compute the optimal parameters for Linear Regression using the Normal Equation.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)

    Returns:
    - theta: np.array, Optimal parameters [slope (m), intercept (b)]
    """
    # Add a column of ones to X for the intercept (b)
    X_b = np.hstack([X.reshape(-1, 1), np.ones((X.shape[0], 1))])

    # Compute the optimal parameters using the Normal Equation
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return theta


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('../dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Compute optimal parameters
    theta = normal_equation(X, y)
    m, b = theta[0], theta[1]
    print(f"Parameters from Normal Equation: m = {m:.4f}, b = {b:.4f}")

    # Predict new values
    new_X = np.array([6, 7, 8])
    new_X_b = np.hstack([new_X.reshape(-1, 1), np.ones((new_X.shape[0], 1))])
    predictions = new_X_b.dot(theta)
    print("\nPredictions:")
    for exp, pred in zip(new_X, predictions):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, m * X + b, color="green", label="Regression Line (Normal Equation)")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Linear Regression using Normal Equation")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
