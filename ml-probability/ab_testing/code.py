import numpy as np
from scipy import stats


def two_proportion_ztest(x1, n1, x2, n2):
    """
    Perform two-proportion Z-test.

    Args:
        x1: Number of successes in group 1
        n1: Total observations in group 1
        x2: Number of successes in group 2
        n2: Total observations in group 2

    Returns:
        Dictionary with test results
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    # Z-statistic
    z = (p2 - p1) / se

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% Confidence interval
    ci_lower = (p2 - p1) - 1.96 * se
    ci_upper = (p2 - p1) + 1.96 * se

    return {
        'p1': p1,
        'p2': p2,
        'difference': p2 - p1,
        'relative_lift': (p2 - p1) / p1,
        'z_statistic': z,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant_at_5pct': p_value < 0.05
    }


def sample_size_for_power(baseline_rate, effect_size, alpha=0.05, power=0.80):
    """
    Calculate required sample size per group.

    Args:
        baseline_rate: Control group conversion rate
        effect_size: Absolute difference in rates to detect
        alpha: Significance level (two-tailed)
        power: Statistical power (1 - beta)

    Returns:
        Required sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = stats.norm.ppf(power)

    p = baseline_rate
    delta = effect_size

    n = 2 * ((z_alpha + z_beta) / (2 * np.arcsin(np.sqrt(p)) -
                                    2 * np.arcsin(np.sqrt(p - delta/2)))) ** 2

    return int(np.ceil(n))


def min_detectable_effect(n, baseline_rate, alpha=0.05, power=0.80):
    """
    Calculate minimum detectable effect (MDE).

    Args:
        n: Sample size per group
        baseline_rate: Control group conversion rate
        alpha: Significance level
        power: Statistical power

    Returns:
        Minimum detectable effect size
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    p = baseline_rate
    se = np.sqrt(2 * p * (1 - p) / n)

    mde = (z_alpha + z_beta) * se

    return mde


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("A/B Test Analysis")
    print("=" * 60)

    # Test data
    n_control = 10_000
    conversions_control = 850

    n_test = 10_000
    conversions_test = 920

    # Perform test
    results = two_proportion_ztest(
        conversions_control, n_control,
        conversions_test, n_test
    )

    print(f"\nControl Group (A):")
    print(f"  Sample size: {n_control:,}")
    print(f"  Conversions: {conversions_control:,}")
    print(f"  Conversion rate: {results['p1']:.4f} ({results['p1']*100:.2f}%)")

    print(f"\nTest Group (B):")
    print(f"  Sample size: {n_test:,}")
    print(f"  Conversions: {conversions_test:,}")
    print(f"  Conversion rate: {results['p2']:.4f} ({results['p2']*100:.2f}%)")

    print(f"\nTest Results:")
    print(f"  Absolute difference: {results['difference']:.4f} ({results['difference']*100:.2f}%)")
    print(f"  Relative lift: {results['relative_lift']:.4f} ({results['relative_lift']*100:.2f}%)")
    print(f"  Z-statistic: {results['z_statistic']:.4f}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    print(f"  Statistically significant (α=0.05)? {results['significant_at_5pct']}")

    # Sample size analysis
    print(f"\n" + "=" * 60)
    print("Sample Size Planning")
    print("=" * 60)

    baseline = results['p1']
    mde_needed = 0.007  # Minimum effect to detect

    required_n = sample_size_for_power(baseline, mde_needed)
    print(f"\nTo detect {mde_needed*100:.2f}% absolute difference:")
    print(f"  Required sample size per group: {required_n:,}")
    print(f"  Total required: {2*required_n:,}")

    # MDE analysis
    actual_mde = min_detectable_effect(n_control, baseline)
    print(f"\nWith current sample size ({n_control:,} per group):")
    print(f"  Minimum detectable effect: {actual_mde*100:.2f}%")
