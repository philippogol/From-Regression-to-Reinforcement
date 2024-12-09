
# Advanced Linear Regression Algorithms

This repository contains implementations of various linear regression algorithms and related techniques. 
Each algorithm is categorized based on its use case, key features, and suitability for specific types of data.

---

## 1. Table Categorization

### Categories of Regression Algorithms
| **Category**                 | **Algorithm**                                    | **Linear Model?**         |
|-------------------------------|------------------------------------------------|---------------------------|
| **Linear Regression**         | Gradient Descent, Normal Equation, Ridge, Lasso, Elastic Net | Yes                       |
| **Extended Linear Models**    | Polynomial Regression, Principal Component Regression (PCR), Bayesian Linear Regression | Yes, after feature engineering |
| **Robust Linear Models**      | Huber Regression, Theil-Sen Estimator          | Yes                       |
| **Non-Linear Models**         | Decision Tree, Random Forest, SVR, Gradient Boosted Models | No                        |
| **Deep Learning Models**      | Neural Networks                                | No, but flexible          |

---

## 2. Expanded Table: Comparative Overview

| **Approach**                  | **Best For**                                      | **Key Feature**                                                                 |
|-------------------------------|--------------------------------------------------|---------------------------------------------------------------------------------|
| **Gradient Descent**           | Large datasets, standard regression problems     | Iterative optimization                                                         |
| **Closed-Form Solution**       | Small datasets, exact results                    | Direct solution using the Normal Equation                                       |
| **Stochastic Gradient Descent (SGD)** | Streaming or large datasets                       | Updates parameters based on single data points                                 |
| **Mini-Batch Gradient Descent**| Large datasets, balance between batch and SGD    | Updates parameters in small batches                                            |
| **Ridge Regression** (L2)      | Multicollinearity, preventing overfitting         | Penalizes large coefficients to stabilize the model                            |
| **Lasso Regression** (L1)      | Feature selection, sparse models                 | Shrinks some coefficients to zero, effectively selecting features              |
| **Elastic Net**                | Combining L1 and L2 penalties                    | Balances Ridge and Lasso regression for flexibility                            |
| **Principal Component Regression (PCR)** | High-dimensional data                             | Combines PCA for dimensionality reduction with regression                      |
| **Bayesian Regression**        | Small datasets, uncertainty quantification       | Treats parameters as probability distributions                                 |
| **Support Vector Regression (SVR)** | Noisy data, robust predictions                     | Fits data within a margin of tolerance                                         |
| **Gradient-Boosted Regression**| Complex data, improved accuracy                  | Combines weak learners iteratively to minimize error                           |
| **Neural Networks**            | Highly non-linear relationships                  | Flexible and powerful, uses hidden layers to capture complex patterns          |
| **Decision Tree Regression**   | Data with non-linear relationships               | Splits data into hierarchical rules to predict outcomes                        |
| **Random Forest Regression**   | Reducing overfitting, ensemble methods           | Combines multiple decision trees to improve accuracy and stability             |
| **Polynomial Regression**      | Data with polynomial trends                      | Extends linear regression to fit polynomial relationships                      |
| **Theil-Sen Estimator**        | Outlier-resistant linear regression              | Robust method that uses medians rather than means to compute the line          |
| **Huber Regression**           | Handling outliers                                | Combines squared loss for small errors and absolute loss for large errors      |

---

## 3. Index of Algorithms

### Linear Regression
1. **Gradient Descent**: [`gradient_descent.py`](./gradient_descent.py)
2. **Normal Equation (Closed-Form Solution)**: [`normal_equation.py`](./normal_equation.py)

### Regularized Regression
3. **Ridge Regression (L2)**: [`ridge_regression.py`](./ridge_regression.py)
4. **Lasso Regression (L1)**: [`lasso_regression.py`](./lasso_regression.py)
5. **Elastic Net**: [`elastic_net.py`](./elastic_net.py)

### Extended Linear Models
6. **Polynomial Regression**: [`polynomial_regression.py`](./polynomial_regression.py)
7. **Principal Component Regression (PCR)**: [`principal_component_regression.py`](./principal_component_regression.py)
8. **Bayesian Regression**: [`bayesian_regression.py`](./bayesian_regression.py)

### Robust Regression
9. **Huber Regression**: [`huber_regression.py`](./huber_regression.py)
10. **Theil-Sen Estimator**: [`theil_sen_estimator.py`](./theil_sen_estimator.py)

### Non-Linear Models
11. **Support Vector Regression (SVR)**: [`support_vector_regression.py`](./support_vector_regression.py)
12. **Decision Tree Regression**: [`decision_tree_regression.py`](./decision_tree_regression.py)
13. **Random Forest Regression**: [`random_forest_regression.py`](./random_forest_regression.py)
14. **Gradient-Boosted Regression**: [`gradient_boosted_regression.py`](./gradient_boosted_regression.py)

### Deep Learning Models
15. **Neural Networks for Regression**: [`neural_network_regression.py`](./neural_network_regression.py)
