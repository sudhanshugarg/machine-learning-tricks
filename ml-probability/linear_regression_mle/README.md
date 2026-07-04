# Linear Regression via Maximum Likelihood Estimation

## Problem Overview

This problem demonstrates a fundamental insight in statistics: **Ordinary Least Squares (OLS) regression is equivalent to Maximum Likelihood Estimation** when we assume that errors are normally distributed.

**Interview Context:** This is a classic problem that appears in ML interviews and shows deep understanding of the connection between optimization and probability.

---

## Files

- **solution.md** - Complete mathematical derivation and theory
- **code.py** - Full implementation with examples
- **template.py** - Starter template for implementation

---

## Quick Summary

Given data $(X, y)$ where $y_i = \mathbf{x}_i^T \boldsymbol{\beta} + \epsilon_i$ and $\epsilon_i \sim N(0, \sigma^2)$:

**MLE for coefficients:** $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ (Normal equations)

**MLE for variance:** $\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$

This is identical to minimizing mean squared error!

---

## Key Concepts

1. **Log-likelihood function** - Deriving and maximizing it leads to OLS
2. **Normal equations** - Closed-form solution for coefficients
3. **Fisher Information** - Computing standard errors and confidence intervals
4. **Bias-variance** - Understanding MLE variance (biased, divides by $n$)

---

## Interview Questions

- Why is OLS equivalent to MLE?
- How do you compute confidence intervals?
- What assumptions are required?
- What if errors aren't Gaussian?
- How does regularization relate to priors?

---

## Complexity

- **Time:** $O(np^2 + p^3)$ for fitting
- **Space:** $O(np + p^2)$

---

## Related Concepts

- Bayesian linear regression
- Ridge/Lasso regression
- Generalized Linear Models (GLM)
- Maximum Likelihood Estimation fundamentals
