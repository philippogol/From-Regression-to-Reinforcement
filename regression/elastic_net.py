# elastic_net.py

# Implementation of Elastic Net Regression using dataset/salary_data.csv
# Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization, balancing feature selection and model stability.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

def elastic_net_regression(X, y, alpha, l1_ratio):
    """
    Perform Elastic Net Regression using scikit-learn's ElasticNet implementation.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)
    - alpha: float, Regularization strength (higher value = more regularization)
    - l1_ratio: float, Mixing parameter between Lasso (L1) and Ridge (L2) regularization (0 = Ridge, 1 = Lasso)

    Returns:
    - model: Trained Elastic Net regression model
    """
    # Reshape X for scikit-learn compatibility
    X = X.reshape(-1, 1)

    # Create and fit the Elastic Net model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X, y)

    return model


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('../dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Elastic Net Regression with chosen parameters
    alpha = 1000  # Regularization strength
    l1_ratio = 0.5  # Balance between L1 and L2 regularization
    model = elastic_net_regression(X, y, alpha, l1_ratio)
    m, b = model.coef_[0], model.intercept_
    print(f"Parameters from Elastic Net Regression: m = {m:.4f}, b = {b:.4f}")

    # Predict new values
    new_X = np.array([6, 7, 8]).reshape(-1, 1)
    predictions = model.predict(new_X)
    print("\nPredictions:")
    for exp, pred in zip(new_X.flatten(), predictions):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="green", label="Regression Line (Elastic Net)")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Elastic Net Regression (L1 + L2 Regularization)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
