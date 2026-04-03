# Maximum Likelihood Estimation (MLE)

## Problem Statement

You have a dataset of 100 observations from an unknown probability distribution. Your goal is to estimate the parameters of the distribution using Maximum Likelihood Estimation (MLE).

Given observations that appear to follow a normal distribution, estimate:
1. Mean ($\mu$)
2. Standard deviation ($\sigma$)

Also solve MLE problems for other common distributions:
- Bernoulli distribution (success probability $p$)
- Exponential distribution (rate $\lambda$)
- Poisson distribution (rate $\lambda$)

## Key Concepts

**Maximum Likelihood Estimation** is a method to find parameter estimates that make the observed data most likely. The idea is to maximize the **likelihood function**:

$$L(\theta; X) = \prod_{i=1}^{n} P(x_i | \theta)$$

Or equivalently, maximize the **log-likelihood** (easier to compute):

$$\ell(\theta; X) = \sum_{i=1}^{n} \log P(x_i | \theta)$$

## Challenge

1. Derive MLE estimators for different distributions
2. Implement numerical optimization to find MLEs
3. Compute confidence intervals for estimates
4. Compare MLE with other estimation methods (method of moments)
5. Analyze bias and consistency of estimators

## Applications

- Parameter estimation in machine learning models
- Distribution fitting for data analysis
- Model selection and comparison
- Bayesian inference (MLE as point estimate)
- Maximum entropy models
