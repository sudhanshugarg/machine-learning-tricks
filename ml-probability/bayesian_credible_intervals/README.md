# Bayesian Credible Intervals

## Problem Statement

You're estimating the click-through rate (CTR) of an online advertisement. You've observed:
- 45 clicks out of 500 impressions
- Prior belief: CTR follows Beta(10, 100) distribution (belief that CTR is around 9%)

Compute:
1. The posterior distribution after observing the data
2. A 95% credible interval for the CTR
3. Posterior mean and variance
4. Compare with frequentist confidence intervals

## Key Concepts

**Bayesian Inference** combines prior beliefs with observed data:

$$P(\theta | \text{data}) = \frac{P(\text{data} | \theta) \cdot P(\theta)}{P(\text{data})}$$

Where:
- $P(\theta)$ = prior (beliefs before data)
- $P(\text{data} | \theta)$ = likelihood (data model)
- $P(\theta | \text{data})$ = posterior (beliefs after data)
- $P(\text{data})$ = marginal likelihood (normalization constant)

**Credible Interval** (Bayesian):
- A range of parameter values with high probability under the posterior
- Direct probability interpretation: "95% of the probability mass"
- Depends on prior specification

**Confidence Interval** (Frequentist):
- A range that would contain the true parameter 95% of the time if experiment repeated
- Different interpretation: coverage property, not probability of parameter

## Challenge

1. Derive the posterior distribution for conjugate priors
2. Compute credible intervals (analytic and numerical)
3. Compare Bayesian vs frequentist intervals
4. Understand how prior affects conclusions
5. Perform sensitivity analysis on prior choice

## Applications

- A/B testing with Bayesian updates
- Estimate conversion rates, click-through rates
- Sequential decision making
- Clinical trials (adaptive designs)
- Machine learning model parameters
