"""
Bayesian Credible Intervals Implementation
"""

import numpy as np
from scipy import stats, special
from typing import Tuple, Dict, Any


class BayesianAnalysis:
    """Bayesian inference for Beta-Binomial model."""

    def __init__(self, prior_alpha: float, prior_beta: float):
        """
        Initialize with Beta prior.

        Args:
            prior_alpha: Prior shape parameter α
            prior_beta: Prior shape parameter β
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.posterior_alpha = prior_alpha
        self.posterior_beta = prior_beta

    def update(self, successes: int, trials: int) -> Dict[str, float]:
        """
        Update posterior with observed data.

        Args:
            successes: Number of observed successes
            trials: Total number of trials

        Returns:
            Dictionary with posterior parameters
        """
        failures = trials - successes

        # Conjugate update for Beta-Binomial
        self.posterior_alpha = self.prior_alpha + successes
        self.posterior_beta = self.prior_beta + failures

        return {
            'alpha': self.posterior_alpha,
            'beta': self.posterior_beta,
            'successes': successes,
            'trials': trials,
            'failures': failures
        }

    def posterior_mean(self) -> float:
        """Compute posterior mean."""
        return self.posterior_alpha / (self.posterior_alpha + self.posterior_beta)

    def posterior_variance(self) -> float:
        """Compute posterior variance."""
        a, b = self.posterior_alpha, self.posterior_beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def posterior_std(self) -> float:
        """Compute posterior standard deviation."""
        return np.sqrt(self.posterior_variance())

    def credible_interval(self, credibility: float = 0.95) -> Tuple[float, float]:
        """
        Compute credible interval (quantile-based).

        Args:
            credibility: Credibility level (default 0.95 for 95%)

        Returns:
            (lower, upper) bounds of credible interval
        """
        alpha = (1 - credibility) / 2
        lower = stats.beta.ppf(alpha, self.posterior_alpha, self.posterior_beta)
        upper = stats.beta.ppf(1 - alpha, self.posterior_alpha, self.posterior_beta)
        return lower, upper

    def highest_density_interval(self, credibility: float = 0.95) -> Tuple[float, float]:
        """
        Compute highest density interval (shortest interval).

        For Beta distribution, HDI and quantile-based CI are similar.
        This uses a numerical approximation.

        Args:
            credibility: Credibility level

        Returns:
            (lower, upper) bounds of highest density interval
        """
        # Use grid search to find HDI
        x = np.linspace(0, 1, 10000)
        density = stats.beta.pdf(x, self.posterior_alpha, self.posterior_beta)

        # Threshold for credibility region
        threshold = np.percentile(density, (1 - credibility) * 100)
        in_region = density >= threshold

        # Find contiguous regions
        diffs = np.diff(in_region.astype(int))
        starts = x[np.where(diffs == 1)[0] + 1]
        ends = x[np.where(diffs == -1)[0]]

        if len(starts) > 0 and len(ends) > 0:
            # Return the longest/densest region
            widths = ends - starts
            if len(widths) > 0:
                idx = np.argmax(density[np.searchsorted(x, starts)])
                return starts[idx], ends[idx]

        return self.credible_interval(credibility)

    def predictive_distribution(self) -> Dict[str, float]:
        """
        Compute posterior predictive distribution parameters.

        For next trial, E[X_new] and Var[X_new].

        Returns:
            Dictionary with predictive probability and variance
        """
        mean = self.posterior_mean()
        # Predictive variance includes parameter uncertainty
        var = self.posterior_variance() + mean * (1 - mean)
        return {'mean': mean, 'variance': var}

    def summary(self) -> Dict[str, Any]:
        """Compute comprehensive summary statistics."""
        return {
            'posterior_alpha': self.posterior_alpha,
            'posterior_beta': self.posterior_beta,
            'posterior_mean': self.posterior_mean(),
            'posterior_variance': self.posterior_variance(),
            'posterior_std': self.posterior_std(),
            'credible_interval_95': self.credible_interval(0.95),
            'credible_interval_90': self.credible_interval(0.90),
            'credible_interval_99': self.credible_interval(0.99),
        }


def frequentist_confidence_interval(successes: int, trials: int,
                                   confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute frequentist confidence interval (Wald).

    Args:
        successes: Number of successes
        trials: Total trials
        confidence: Confidence level

    Returns:
        (lower, upper) bounds
    """
    p_hat = successes / trials
    se = np.sqrt(p_hat * (1 - p_hat) / trials)
    z = stats.norm.ppf((1 + confidence) / 2)
    lower = p_hat - z * se
    upper = p_hat + z * se
    return max(0, lower), min(1, upper)


def compare_bayesian_frequentist(
    successes: int,
    trials: int,
    prior_alpha: float = 1,
    prior_beta: float = 1
) -> Dict[str, Any]:
    """
    Compare Bayesian and frequentist analyses.

    Args:
        successes: Observed successes
        trials: Total trials
        prior_alpha: Beta prior α
        prior_beta: Beta prior β

    Returns:
        Dictionary with both analyses
    """
    # Bayesian analysis
    bayesian = BayesianAnalysis(prior_alpha, prior_beta)
    bayesian.update(successes, trials)

    # Frequentist analysis
    p_hat = successes / trials
    freq_ci = frequentist_confidence_interval(successes, trials)

    return {
        'sample_size': trials,
        'observed_successes': successes,
        'sample_proportion': p_hat,
        'bayesian': {
            'prior': f"Beta({prior_alpha}, {prior_beta})",
            'prior_mean': prior_alpha / (prior_alpha + prior_beta),
            'posterior': f"Beta({bayesian.posterior_alpha}, {bayesian.posterior_beta})",
            'posterior_mean': bayesian.posterior_mean(),
            'posterior_std': bayesian.posterior_std(),
            'credible_interval_95': bayesian.credible_interval(0.95),
        },
        'frequentist': {
            'point_estimate': p_hat,
            'standard_error': np.sqrt(p_hat * (1 - p_hat) / trials),
            'confidence_interval_95': freq_ci,
        }
    }


def sensitivity_analysis(successes: int, trials: int,
                        prior_specs: list) -> Dict[str, Any]:
    """
    Perform sensitivity analysis on different priors.

    Args:
        successes: Observed successes
        trials: Total trials
        prior_specs: List of (alpha, beta) tuples for different priors

    Returns:
        Results for each prior specification
    """
    results = {}

    for i, (alpha, beta) in enumerate(prior_specs):
        bayesian = BayesianAnalysis(alpha, beta)
        bayesian.update(successes, trials)
        prior_mean = alpha / (alpha + beta)

        results[f"Prior_{i}_Beta({alpha},{beta})"] = {
            'prior_mean': prior_mean,
            'posterior_mean': bayesian.posterior_mean(),
            'posterior_std': bayesian.posterior_std(),
            'credible_interval': bayesian.credible_interval(0.95),
        }

    return results


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("BAYESIAN CREDIBLE INTERVALS")
    print("=" * 80)

    # Problem: CTR estimation
    successes = 45  # clicks
    trials = 500    # impressions

    print(f"\nObserved Data:")
    print(f"  Successes (clicks): {successes}")
    print(f"  Trials (impressions): {trials}")
    print(f"  Sample proportion: {successes/trials:.4f}")

    # Bayesian analysis with informative prior
    print(f"\n" + "=" * 80)
    print(f"BAYESIAN ANALYSIS: Beta(10, 100) Prior")
    print("=" * 80)

    bayesian = BayesianAnalysis(prior_alpha=10, prior_beta=100)
    update_info = bayesian.update(successes, trials)

    print(f"\nPrior:")
    print(f"  Distribution: Beta(10, 100)")
    print(f"  Prior mean: {10/(10+100):.4f}")

    print(f"\nPosterior:")
    print(f"  Distribution: Beta({bayesian.posterior_alpha}, {bayesian.posterior_beta})")
    print(f"  Posterior mean: {bayesian.posterior_mean():.4f}")
    print(f"  Posterior std: {bayesian.posterior_std():.4f}")

    print(f"\nCredible Intervals:")
    ci_95 = bayesian.credible_interval(0.95)
    ci_90 = bayesian.credible_interval(0.90)
    ci_99 = bayesian.credible_interval(0.99)

    print(f"  95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"  90% CI: [{ci_90[0]:.4f}, {ci_90[1]:.4f}]")
    print(f"  99% CI: [{ci_99[0]:.4f}, {ci_99[1]:.4f}]")

    # Frequentist analysis
    print(f"\n" + "=" * 80)
    print(f"FREQUENTIST ANALYSIS")
    print("=" * 80)

    freq_ci = frequentist_confidence_interval(successes, trials)
    se = np.sqrt((successes/trials) * (1 - successes/trials) / trials)

    print(f"\nPoint estimate: {successes/trials:.4f}")
    print(f"Standard error: {se:.4f}")
    print(f"95% Confidence Interval: [{freq_ci[0]:.4f}, {freq_ci[1]:.4f}]")

    # Comparison
    print(f"\n" + "=" * 80)
    print(f"COMPARISON: Bayesian vs Frequentist")
    print("=" * 80)

    comparison = compare_bayesian_frequentist(successes, trials, prior_alpha=10, prior_beta=100)

    print(f"\nBayesian:")
    print(f"  Posterior mean: {comparison['bayesian']['posterior_mean']:.4f}")
    print(f"  95% Credible interval: [{comparison['bayesian']['credible_interval_95'][0]:.4f}, "
          f"{comparison['bayesian']['credible_interval_95'][1]:.4f}]")

    print(f"\nFrequentist:")
    print(f"  Point estimate: {comparison['frequentist']['point_estimate']:.4f}")
    print(f"  95% Confidence interval: [{comparison['frequentist']['confidence_interval_95'][0]:.4f}, "
          f"{comparison['frequentist']['confidence_interval_95'][1]:.4f}]")

    # Sensitivity analysis
    print(f"\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS: Different Prior Choices")
    print("=" * 80)

    priors = [
        (1, 1),        # Uniform prior
        (10, 100),     # Informative (our example)
        (50, 50),      # Strong optimistic
        (5, 150),      # Strong pessimistic
    ]

    sensitivity = sensitivity_analysis(successes, trials, priors)

    for prior_name, results in sensitivity.items():
        print(f"\n{prior_name}:")
        print(f"  Prior mean: {results['prior_mean']:.4f}")
        print(f"  Posterior mean: {results['posterior_mean']:.4f}")
        print(f"  Posterior std: {results['posterior_std']:.4f}")
        print(f"  95% CI: [{results['credible_interval'][0]:.4f}, {results['credible_interval'][1]:.4f}]")

    # Sequential updating
    print(f"\n" + "=" * 80)
    print(f"SEQUENTIAL BAYESIAN UPDATING")
    print("=" * 80)

    print(f"\nDay 1: Observe 10 clicks out of 100")
    day1 = BayesianAnalysis(10, 100)
    day1.update(10, 100)
    print(f"  Posterior after Day 1: Beta({day1.posterior_alpha}, {day1.posterior_beta})")
    print(f"  Mean: {day1.posterior_mean():.4f}")

    print(f"\nDay 2: Observe 35 clicks out of 400")
    day2 = BayesianAnalysis(day1.posterior_alpha, day1.posterior_beta)  # Previous posterior as new prior
    day2.update(35, 400)
    print(f"  Posterior after Day 2: Beta({day2.posterior_alpha}, {day2.posterior_beta})")
    print(f"  Mean: {day2.posterior_mean():.4f}")

    print(f"\nComparison:")
    print(f"  Batch (all 45/500): Beta(55, 555), mean = {bayesian.posterior_mean():.4f}")
    print(f"  Sequential (day by day): Beta({day2.posterior_alpha}, {day2.posterior_beta}), "
          f"mean = {day2.posterior_mean():.4f}")
