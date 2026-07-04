"""
Template: Linear Regression via Maximum Likelihood Estimation

Task: Implement a linear regression model using MLE principles.

Key concepts to implement:
1. Fit the model using normal equations: β = (X^T X)^(-1) X^T y
2. Estimate error variance: σ² = RSS / n
3. Compute standard errors using Fisher Information
4. Generate confidence intervals for coefficients
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any


class LinearRegressionMLE:
    """
    Linear Regression using Maximum Likelihood Estimation.

    Assumes: y_i = β₀ + β₁x_{i1} + ... + β_p x_{ip} + ε_i
    where ε_i ~ N(0, σ²)
    """

    def __init__(self):
        self.coefficients = None
        self.sigma_sq = None
        # TODO: Add other attributes as needed
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Fit linear regression using MLE.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Returns:
            Dictionary with 'coefficients' and 'sigma_squared'
        """
        # TODO: Implement normal equations
        # 1. Compute X^T X and X^T y
        # 2. Solve for β using matrix inversion
        # 3. Compute residuals
        # 4. Estimate σ²
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        # TODO: Implement prediction using fitted coefficients
        pass

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute log-likelihood: -n/2 log(2π) - n log(σ) - RSS/(2σ²)"""
        # TODO: Implement log-likelihood calculation
        pass

    def standard_errors(self) -> np.ndarray:
        """
        Compute standard error for each coefficient.

        SE(β) = sqrt(diag(σ² (X^T X)^(-1)))
        """
        # TODO: Implement using Fisher Information
        pass

    def confidence_intervals(self, alpha: float = 0.05) -> np.ndarray:
        """
        Compute confidence intervals using t-distribution.

        Returns array of shape (n_features, 2) with [lower, upper] bounds
        """
        # TODO: Implement confidence interval calculation
        # Use t-distribution with df = n - p - 1
        pass

    def r_squared(self) -> float:
        """Compute R² coefficient of determination."""
        # TODO: Implement R² calculation
        pass


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n = 50
    X_raw = np.random.randn(n, 2)
    X = np.column_stack([np.ones(n), X_raw])  # Add intercept column
    true_beta = np.array([5, 2, -1])
    y = X @ true_beta + np.random.normal(0, 1, n)

    # Fit model
    model = LinearRegressionMLE()
    model.fit(X, y)

    # Print results
    print("Fitted coefficients:", model.coefficients)
    print("Standard errors:", model.standard_errors())
    print("R²:", model.r_squared())
    print("95% Confidence Intervals:\n", model.confidence_intervals())
