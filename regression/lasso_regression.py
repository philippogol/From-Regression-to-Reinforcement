# lasso_regression.py
# Implementation of Lasso Regression using dataset/salary_data.csv
# Lasso Regression is useful for feature selection by shrinking some coefficients to zero.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

def preprocess_data(data, preprocessor=None, is_training=True):
    """
    Preprocess data: encode categorical variables and scale features.

    Parameters:
    - data: pd.DataFrame, Input dataset
    - preprocessor: ColumnTransformer or None, Pre-fitted preprocessor for prediction phase
    - is_training: bool, Whether the dataset includes the dependent variable (Salary)

    Returns:
    - X: np.array, Processed independent variables
    - y: np.array (optional), Dependent variable (Salary), only if is_training=True
    - preprocessor: ColumnTransformer, Fitted preprocessor
    - feature_names: List of feature names after preprocessing
    """
    if is_training:
        # Separate independent and dependent variables
        y = data['Salary'].values
        X = data.drop(columns=['Salary'])
    else:
        # For new prediction data, there's no 'Salary' column
        y = None
        X = data

    # Define preprocessing steps
    categorical_features = ['EducationLevel', 'Industry', 'CitySize']
    numeric_features = ['YearsExperience', 'YearsInCompany']

    if is_training:
        # Create and fit the preprocessor during training
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )
        X_processed = preprocessor.fit_transform(X)
        feature_names = (
            numeric_features +
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
        )
    else:
        # Use the pre-fitted preprocessor during prediction
        X_processed = preprocessor.transform(X)
        feature_names = (
            numeric_features +
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
        )

    return (X_processed, y, preprocessor, feature_names) if is_training else (X_processed, feature_names)

def lasso_regression(X_train, y_train, alpha):
    """
    Perform Lasso Regression using scikit-learn's Lasso implementation.

    Parameters:
    - X_train: np.array, Independent variable (training data)
    - y_train: np.array, Dependent variable (training data)
    - alpha: float, Regularization strength (higher value = more regularization)

    Returns:
    - model: Trained Lasso regression model
    """
    # Create and fit the Lasso model
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
    dataset_path = os.path.join(script_dir, '../dataset/salary_data.csv')
    data = pd.read_csv(dataset_path)

    # Preprocess the data
    X, y, preprocessor, feature_names = preprocess_data(data, is_training=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lasso Regression with a chosen regularization strength
    alpha = 1000  # Regularization strength
    model = lasso_regression(X_train, y_train, alpha)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Display feature importance
    print("\nFeature Importances (Lasso Coefficients):")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"{feature}: {coef:.4f}")

    # Predict new values
    new_data = pd.DataFrame({
        'YearsExperience': [6, 7, 8],
        'EducationLevel': ['Master', 'PhD', 'Bachelor'],
        'Industry': ['Tech', 'Finance', 'Healthcare'],
        'CitySize': ['Large', 'Medium', 'Small'],
        'YearsInCompany': [4.5, 5.5, 6.0]
    })
    new_X, feature_names = preprocess_data(new_data, preprocessor=preprocessor, is_training=False)
    predictions = model.predict(new_X)

    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"Data {i + 1}: Predicted Salary = ${pred:.2f}")

    # Visualization: Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", label="Perfect Fit Line")
    plt.title("Lasso Regression: Predicted vs Actual")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
