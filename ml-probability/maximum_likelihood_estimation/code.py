"""
Maximum Likelihood Estimation Implementation
"""

import numpy as np
from scipy import stats, optimize
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class MaximumLikelihoodEstimator:
    """Base class for MLE implementations."""

    def __init__(self):
        self.params = {}
        self.log_likelihood = None
        self.fisher_information = None

    def fit(self, X: np.ndarray) -> Dict[str, float]:
        """Fit the model and return parameter estimates."""
        raise NotImplementedError

    def log_likelihood_func(self, X: np.ndarray, **params) -> float:
        """Compute log-likelihood for given parameters."""
        raise NotImplementedError

    def confidence_interval(self, X: np.ndarray, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for parameters."""
        raise NotImplementedError


class NormalMLE(MaximumLikelihoodEstimator):
    """MLE for Normal (Gaussian) distribution."""

    def fit(self, X: np.ndarray) -> Dict[str, float]:
        """
        Estimate mean and std dev of normal distribution.

        Returns:
            Dictionary with 'mean' and 'std' estimates
        """
        self.params['mean'] = np.mean(X)
        self.params['std'] = np.std(X, ddof=0)  # MLE uses n, not n-1

        return self.params

    def log_likelihood_func(self, X: np.ndarray, mean: float, std: float) -> float:
        """Compute log-likelihood for normal distribution."""
        if std <= 0:
            return -np.inf
        return np.sum(stats.norm.logpdf(X, loc=mean, scale=std))

    def confidence_interval(self, X: np.ndarray, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals using asymptotic normality."""
        n = len(X)
        mean = self.params['mean']
        std = self.params['std']

        # Fisher Information for mean
        fisher_mean = n / (std ** 2)
        se_mean = 1 / np.sqrt(fisher_mean)

        # Critical value
        z_crit = stats.norm.ppf(1 - alpha / 2)

        ci_mean = (mean - z_crit * se_mean, mean + z_crit * se_mean)

        return {
            'mean': ci_mean,
            'std': (std, std)  # Placeholder for std (more complex)
        }


class BernoulliMLE(MaximumLikelihoodEstimator):
    """MLE for Bernoulli distribution."""

    def fit(self, X: np.ndarray) -> Dict[str, float]:
        """
        Estimate success probability.

        Args:
            X: Binary observations (0 or 1)

        Returns:
            Dictionary with 'p' (success probability)
        """
        self.params['p'] = np.mean(X)
        return self.params

    def log_likelihood_func(self, X: np.ndarray, p: float) -> float:
        """Compute log-likelihood for Bernoulli distribution."""
        if p <= 0 or p >= 1:
            return -np.inf
        return np.sum(X * np.log(p) + (1 - X) * np.log(1 - p))

    def confidence_interval(self, X: np.ndarray, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Compute confidence interval for success probability."""
        n = len(X)
        p = self.params['p']

        # Fisher Information
        fisher_p = n / (p * (1 - p))
        se_p = 1 / np.sqrt(fisher_p)

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci = (p - z_crit * se_p, p + z_crit * se_p)

        return {'p': ci}


class ExponentialMLE(MaximumLikelihoodEstimator):
    """MLE for Exponential distribution."""

    def fit(self, X: np.ndarray) -> Dict[str, float]:
        """
        Estimate rate parameter.

        Args:
            X: Positive observations

        Returns:
            Dictionary with 'lambda' (rate parameter)
        """
        self.params['lambda'] = 1 / np.mean(X)
        return self.params

    def log_likelihood_func(self, X: np.ndarray, lam: float) -> float:
        """Compute log-likelihood for exponential distribution."""
        if lam <= 0:
            return -np.inf
        return np.sum(stats.expon.logpdf(X, scale=1/lam))

    def confidence_interval(self, X: np.ndarray, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Compute confidence interval for rate parameter."""
        n = len(X)
        lam = self.params['lambda']

        # Fisher Information (note: Var[lambda_mle] ≠ 1/I due to bias)
        fisher_lambda = n * (lam ** 2)
        se_lambda = 1 / np.sqrt(fisher_lambda)

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci = (lam - z_crit * se_lambda, lam + z_crit * se_lambda)

        return {'lambda': ci}


class PoissonMLE(MaximumLikelihoodEstimator):
    """MLE for Poisson distribution."""

    def fit(self, X: np.ndarray) -> Dict[str, float]:
        """
        Estimate rate parameter.

        Args:
            X: Non-negative integer observations

        Returns:
            Dictionary with 'lambda' (rate parameter)
        """
        self.params['lambda'] = np.mean(X)
        return self.params

    def log_likelihood_func(self, X: np.ndarray, lam: float) -> float:
        """Compute log-likelihood for Poisson distribution."""
        if lam <= 0:
            return -np.inf
        return np.sum(stats.poisson.logpmf(X, mu=lam))

    def confidence_interval(self, X: np.ndarray, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Compute confidence interval for rate parameter."""
        n = len(X)
        lam = self.params['lambda']

        # Fisher Information
        fisher_lambda = n / lam
        se_lambda = 1 / np.sqrt(fisher_lambda)

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci = (lam - z_crit * se_lambda, lam + z_crit * se_lambda)

        return {'lambda': ci}


def compare_distributions(X: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Fit multiple distributions to data and compare.

    Args:
        X: Observations (assumed positive for all)

    Returns:
        Dictionary with results for each distribution
    """
    results = {}

    # Normal
    normal_mle = NormalMLE()
    normal_mle.fit(X)
    results['Normal'] = {
        'params': normal_mle.params,
        'ci': normal_mle.confidence_interval(X)
    }

    # Exponential (requires positive data)
    if np.all(X > 0):
        exp_mle = ExponentialMLE()
        exp_mle.fit(X)
        results['Exponential'] = {
            'params': exp_mle.params,
            'ci': exp_mle.confidence_interval(X)
        }

    # Poisson (requires non-negative integers)
    if np.all(X >= 0) and np.all(X == X.astype(int)):
        poisson_mle = PoissonMLE()
        poisson_mle.fit(X)
        results['Poisson'] = {
            'params': poisson_mle.params,
            'ci': poisson_mle.confidence_interval(X)
        }

    return results


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("MAXIMUM LIKELIHOOD ESTIMATION EXAMPLES")
    print("=" * 70)

    # Example 1: Normal Distribution
    print("\n" + "=" * 70)
    print("Example 1: Normal Distribution")
    print("=" * 70)

    np.random.seed(42)
    data_normal = np.random.normal(loc=2.5, scale=0.8, size=100)

    normal_estimator = NormalMLE()
    params = normal_estimator.fit(data_normal)
    ci = normal_estimator.confidence_interval(data_normal)

    print(f"Data generated from: N(μ=2.5, σ=0.8)")
    print(f"\nMLEs:")
    print(f"  μ̂ = {params['mean']:.4f}")
    print(f"  σ̂ = {params['std']:.4f}")
    print(f"\n95% Confidence Intervals:")
    print(f"  μ̂ ∈ [{ci['mean'][0]:.4f}, {ci['mean'][1]:.4f}]")

    # Example 2: Bernoulli Distribution
    print("\n" + "=" * 70)
    print("Example 2: Bernoulli Distribution")
    print("=" * 70)

    data_bernoulli = np.random.binomial(n=1, p=0.7, size=100)

    bernoulli_estimator = BernoulliMLE()
    params = bernoulli_estimator.fit(data_bernoulli)
    ci = bernoulli_estimator.confidence_interval(data_bernoulli)

    print(f"Data generated from: Bernoulli(p=0.7)")
    print(f"\nMLEs:")
    print(f"  p̂ = {params['p']:.4f}")
    print(f"\n95% Confidence Intervals:")
    print(f"  p̂ ∈ [{ci['p'][0]:.4f}, {ci['p'][1]:.4f}]")

    # Example 3: Poisson Distribution
    print("\n" + "=" * 70)
    print("Example 3: Poisson Distribution")
    print("=" * 70)

    data_poisson = np.random.poisson(lam=3.5, size=100)

    poisson_estimator = PoissonMLE()
    params = poisson_estimator.fit(data_poisson)
    ci = poisson_estimator.confidence_interval(data_poisson)

    print(f"Data generated from: Poisson(λ=3.5)")
    print(f"\nMLEs:")
    print(f"  λ̂ = {params['lambda']:.4f}")
    print(f"\n95% Confidence Intervals:")
    print(f"  λ̂ ∈ [{ci['lambda'][0]:.4f}, {ci['lambda'][1]:.4f}]")

    # Example 4: Exponential Distribution
    print("\n" + "=" * 70)
    print("Example 4: Exponential Distribution")
    print("=" * 70)

    data_exponential = np.random.exponential(scale=2.0, size=100)

    exp_estimator = ExponentialMLE()
    params = exp_estimator.fit(data_exponential)
    ci = exp_estimator.confidence_interval(data_exponential)

    print(f"Data generated from: Exponential(λ=0.5, scale=2.0)")
    print(f"\nMLEs:")
    print(f"  λ̂ = {params['lambda']:.4f}")
    print(f"\n95% Confidence Intervals:")
    print(f"  λ̂ ∈ [{ci['lambda'][0]:.4f}, {ci['lambda'][1]:.4f}]")
