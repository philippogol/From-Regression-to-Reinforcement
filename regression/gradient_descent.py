# gradient_descent.py

# Gradient Descent implementation for Linear Regression using dataset/salary_data.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Perform Gradient Descent to optimize the slope (m) and intercept (b).
    
    Parameters:
    - X: np.array, Independent variable
    - y: np.array, Dependent variable
    - learning_rate: float, Step size for updates
    - iterations: int, Number of iterations

    Returns:
    - m: float, Optimized slope
    - b: float, Optimized intercept
    - loss_history: list, Loss (MSE) at each iteration
    """
    m, b = 0, 0  # Initialize slope and intercept
    n = len(X)  # Number of data points
    loss_history = []  # To store the loss at each iteration

    for i in range(iterations):
        # Calculate predictions
        y_pred = m * X + b

        # Calculate gradients
        dm = -(2/n) * np.sum(X * (y - y_pred))  # Gradient w.r.t. m
        db = -(2/n) * np.sum(y - y_pred)        # Gradient w.r.t. b

        # Update parameters
        m -= learning_rate * dm
        b -= learning_rate * db

        # Calculate loss (Mean Squared Error)
        loss = (1/n) * np.sum((y - y_pred) ** 2)
        loss_history.append(loss)

        # Debug output every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")

    return m, b, loss_history


if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('dataset/salary_data.csv')
    X = data['YearsExperience'].values  # Independent variable: Years of Experience
    y = data['Salary'].values  # Dependent variable: Salary

    # Perform Gradient Descent
    learning_rate = 0.01
    iterations = 1000
    m, b, loss_history = gradient_descent(X, y, learning_rate, iterations)

    # Display final parameters
    print(f"Final Parameters: m = {m:.4f}, b = {b:.4f}")

    # Predict new values
    new_X = np.array([6, 7, 8])
    predictions = m * new_X + b
    print("\nPredictions:")
    for exp, pred in zip(new_X, predictions):
        print(f"Years of Experience: {exp}, Predicted Salary: ${pred:.2f}")

    # Visualization
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, m * X + b, color="red", label="Regression Line")
    plt.scatter(new_X, predictions, color="green", label="Predictions", marker='x', s=100)
    plt.title("Linear Regression using Gradient Descent")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Loss over iterations
    plt.plot(range(iterations), loss_history, color="purple", label="Loss (MSE)")
    plt.title("Loss over Iterations (Gradient Descent)")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
