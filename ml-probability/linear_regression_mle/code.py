"""
Linear Regression via Maximum Likelihood Estimation Implementation
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class LinearRegressionMLE:
    """
    Linear Regression using Maximum Likelihood Estimation.

    Assumes: y_i = β₀ + β₁x_{i1} + ... + β_p x_{ip} + ε_i
    where ε_i ~ N(0, σ²)

    This demonstrates that OLS is equivalent to MLE under Gaussian error assumption.
    """

    def __init__(self):
        self.coefficients = None
        self.sigma_sq = None
        self.X = None
        self.y = None
        self.n_samples = None
        self.n_features = None
        self.predictions = None
        self.residuals = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Fit linear regression using MLE (normal equations).

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)

        Returns:
            Dictionary with coefficients and error variance
        """
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

        # Solve normal equations: β = (X^T X)^(-1) X^T y
        XTX = X.T @ X
        XTy = X.T @ y

        # Add small regularization for numerical stability (optional)
        XTX_inv = np.linalg.inv(XTX)
        self.coefficients = XTX_inv @ XTy

        # Compute residuals and error variance
        self.predictions = X @ self.coefficients
        self.residuals = y - self.predictions

        # MLE of variance: sum of squared residuals / n (note: divides by n, not n-p-1)
        self.sigma_sq = np.sum(self.residuals ** 2) / self.n_samples

        return {
            'coefficients': self.coefficients,
            'sigma_squared': self.sigma_sq
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model must be fit before making predictions")
        return X @ self.coefficients

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute log-likelihood for given data and current parameters.

        Returns:
            Log-likelihood value
        """
        if self.coefficients is None or self.sigma_sq is None:
            raise ValueError("Model must be fit first")

        residuals = y - X @ self.coefficients
        n = len(y)

        # log L = -n/2 * log(2π) - n * log(σ) - 1/(2σ²) * sum(residuals²)
        ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(np.sqrt(self.sigma_sq)) - \
             np.sum(residuals ** 2) / (2 * self.sigma_sq)

        return ll

    def fisher_information(self) -> np.ndarray:
        """
        Compute Fisher Information Matrix: I(β) = 1/σ² * X^T X

        Returns:
            Fisher Information matrix of shape (n_features, n_features)
        """
        if self.X is None or self.sigma_sq is None:
            raise ValueError("Model must be fit first")

        return (self.X.T @ self.X) / self.sigma_sq

    def coefficient_variances(self) -> np.ndarray:
        """
        Compute variance of each coefficient estimate.

        Var(β) = σ² (X^T X)^(-1)

        Returns:
            Variance for each coefficient
        """
        if self.X is None or self.sigma_sq is None:
            raise ValueError("Model must be fit first")

        XTX_inv = np.linalg.inv(self.X.T @ self.X)
        cov_matrix = self.sigma_sq * XTX_inv

        return np.diag(cov_matrix)

    def standard_errors(self) -> np.ndarray:
        """
        Compute standard error for each coefficient.

        Returns:
            Standard errors (sqrt of variances)
        """
        return np.sqrt(self.coefficient_variances())

    def confidence_intervals(self, alpha: float = 0.05) -> np.ndarray:
        """
        Compute confidence intervals for coefficients.

        Uses t-distribution with n-p-1 degrees of freedom.

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            Array of shape (n_features, 2) with [lower, upper] bounds
        """
        if self.coefficients is None:
            raise ValueError("Model must be fit first")

        se = self.standard_errors()

        # t-critical value with n-p-1 degrees of freedom
        df = self.n_samples - self.n_features
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        ci_lower = self.coefficients - t_crit * se
        ci_upper = self.coefficients + t_crit * se

        return np.column_stack([ci_lower, ci_upper])

    def r_squared(self) -> float:
        """
        Compute R² (coefficient of determination).

        Returns:
            R² value between 0 and 1
        """
        if self.residuals is None or self.y is None:
            raise ValueError("Model must be fit first")

        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the fitted model.

        Returns:
            Dictionary with various model statistics
        """
        if self.coefficients is None:
            raise ValueError("Model must be fit first")

        se = self.standard_errors()
        ci = self.confidence_intervals()

        summary_dict = {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'coefficients': self.coefficients,
            'standard_errors': se,
            'confidence_intervals': ci,
            'sigma_squared': self.sigma_sq,
            'sigma': np.sqrt(self.sigma_sq),
            'r_squared': self.r_squared(),
            'log_likelihood': self.log_likelihood(self.X, self.y),
            'fisher_information': self.fisher_information()
        }

        return summary_dict


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("LINEAR REGRESSION VIA MAXIMUM LIKELIHOOD ESTIMATION")
    print("=" * 80)

    # Example 1: Simple 1D regression
    print("\n" + "=" * 80)
    print("Example 1: Simple 1D Linear Regression (House Prices)")
    print("=" * 80)

    np.random.seed(42)

    # Generate synthetic data: Price = 50000 + 100*Sqft + noise
    n_samples = 50
    sqft = np.random.uniform(1500, 3500, n_samples)
    true_intercept = 50000
    true_slope = 100
    noise = np.random.normal(0, 10000, n_samples)
    price = true_intercept + true_slope * sqft + noise

    # Create feature matrix (add intercept column)
    X = np.column_stack([np.ones(n_samples), sqft])
    y = price

    # Fit model
    model = LinearRegressionMLE()
    model.fit(X, y)

    summary = model.summary()

    print(f"\nTrue model: Price = {true_intercept} + {true_slope}*Sqft + ε")
    print(f"\nMLE Estimates:")
    print(f"  β₀ (intercept) = {summary['coefficients'][0]:.2f}")
    print(f"  β₁ (slope)     = {summary['coefficients'][1]:.2f}")
    print(f"\nStandard Errors:")
    print(f"  SE(β₀) = {summary['standard_errors'][0]:.2f}")
    print(f"  SE(β₁) = {summary['standard_errors'][1]:.6f}")
    print(f"\n95% Confidence Intervals:")
    print(f"  β₀ ∈ [{summary['confidence_intervals'][0][0]:.2f}, {summary['confidence_intervals'][0][1]:.2f}]")
    print(f"  β₁ ∈ [{summary['confidence_intervals'][1][0]:.6f}, {summary['confidence_intervals'][1][1]:.6f}]")
    print(f"\nError Variance Estimate:")
    print(f"  σ̂² = {summary['sigma_squared']:.2f}")
    print(f"  σ̂  = {summary['sigma']:.2f}")
    print(f"\nModel Fit:")
    print(f"  R² = {summary['r_squared']:.4f}")
    print(f"  Log-Likelihood = {summary['log_likelihood']:.2f}")

    # Example 2: Multiple features
    print("\n" + "=" * 80)
    print("Example 2: Multiple Linear Regression (Multivariate)")
    print("=" * 80)

    np.random.seed(123)

    # Generate multivariate data
    n_samples = 100
    X_raw = np.random.normal(0, 1, (n_samples, 3))
    X = np.column_stack([np.ones(n_samples), X_raw])  # Add intercept

    # True parameters
    true_beta = np.array([10, 2, -1.5, 3])  # [intercept, β₁, β₂, β₃]
    y = X @ true_beta + np.random.normal(0, 2, n_samples)

    # Fit model
    model2 = LinearRegressionMLE()
    model2.fit(X, y)

    summary2 = model2.summary()

    print(f"\nTrue coefficients: {true_beta}")
    print(f"MLE estimates:     {summary2['coefficients']}")
    print(f"\nStandard Errors: {summary2['standard_errors']}")
    print(f"\nR² = {summary2['r_squared']:.4f}")

    # Example 3: Demonstrate OLS equivalence
    print("\n" + "=" * 80)
    print("Example 3: Demonstrating OLS Equivalence")
    print("=" * 80)

    try:
        from sklearn.linear_model import LinearRegression

        # Fit using scikit-learn (OLS)
        sklearn_model = LinearRegression(fit_intercept=True)
        sklearn_model.fit(X_raw, y)

        sklearn_coef = np.concatenate([[sklearn_model.intercept_], sklearn_model.coef_])
        print(f"\nOLS Coefficients (sklearn): {sklearn_coef}")
        print(f"MLE Coefficients (ours):    {summary2['coefficients']}")
        print(f"\nDifference: {np.max(np.abs(sklearn_coef - summary2['coefficients'])):.2e}")
        print("(Should be ~0, demonstrating equivalence)")
    except ImportError:
        print("\nSkip: scikit-learn not installed")
        print(f"MLE Coefficients: {summary2['coefficients']}")
