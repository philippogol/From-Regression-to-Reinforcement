# support_vector_regression.py

# Implementation of Support Vector Regression (SVR) using dataset/salary_data.csv
# SVR is useful for handling non-linear relationships and is robust to outliers.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def support_vector_regression(X, y, kernel="rbf", C=1.0, epsilon=0.1):
    """
    Perform Support Vector Regression (SVR) using scikit-learn's SVR implementation.

    Parameters:
    - X: np.array, Independent variable (Years of Experience)
    - y: np.array, Dependent variable (Salary)
    - kernel: str, Kernel type to be used in the algorithm ('linear', 'poly', 'rbf')
    - C: float, Regularization parameter
    - epsilon: float, Insensitivity parameter defining margin of tolerance

    Returns:
    - model: Trained SVR model
    - scaler_X: Scaler for feature normalization
    - scaler_y: Scaler for target normalization
    """
    # Standardize the features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X.reshape(-1, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the SVR model
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_scaled, y_scaled)

    return model, scaler_X, scaler_y


if __name__ == "__main__":
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
    dataset_path = os.path.join(script_dir, '../dataset/salary_data.csv')
    data = pd.read_csv(dataset_path)
    
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Train Support Vector Regression model
    kernel = "rbf"  # Radial basis function kernel for non-linear regression
    C = 100  # Regularization strength
    epsilon = 0.1  # Margin of tolerance
    model, scaler_X, scaler_y = support_vector_regression(X, y, kernel, C, epsilon)

    # Predict new values
    new_X = np.array([6, 7, 8]).reshape(-1, 1)
    new_X_scaled = scaler_X.transform(new_X)
    predictions_scaled = model.predict(new_X_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    print("\nPredictions:")
    for exp, pred in zip(new_X.flatten(), predictions):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    X_scaled = scaler_X.transform(X.reshape(-1, 1))
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_scaled = scaler_X.transform(X_plot)
    y_pred_scaled = model.predict(X_plot_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X_plot, y_pred, color="green", label="SVR Prediction Line")
    plt.scatter(new_X, predictions, color="red", label="Predictions", marker='x', s=100)
    plt.title("Support Vector Regression (SVR)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
