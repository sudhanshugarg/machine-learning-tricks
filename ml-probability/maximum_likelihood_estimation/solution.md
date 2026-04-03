# Maximum Likelihood Estimation Solutions

## Overview

MLE finds parameter values that maximize the probability of observing the given data. The procedure:

1. Write the likelihood function $L(\theta) = \prod_i P(x_i | \theta)$
2. Take the log: $\ell(\theta) = \sum_i \log P(x_i | \theta)$
3. Take derivatives: $\frac{\partial \ell}{\partial \theta} = 0$
4. Solve for $\theta$ (may require numerical methods)

---

## 1. Normal Distribution

**Likelihood for Normal Distribution:**
$$\ell(\mu, \sigma) = \sum_{i=1}^{n} \log \left( \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right) \right)$$

**Simplified:**
$$\ell(\mu, \sigma) = -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

### Finding the MLE

**For mean $\mu$:**
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

**Result:** MLE of mean is the sample mean.

**For standard deviation $\sigma$:**
$$\frac{\partial \ell}{\partial \sigma} = -\frac{n}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^{n}(x_i - \mu)^2 = 0$$

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2$$

**Result:** MLE of variance is the average squared deviation (note: biased, divides by $n$ not $n-1$).

---

## 2. Bernoulli Distribution

**Distribution:** $P(x|p) = p^x(1-p)^{1-x}$ for $x \in \{0,1\}$

**Likelihood:**
$$\ell(p) = \sum_{i=1}^{n}[x_i \log p + (1-x_i)\log(1-p)]$$

$$\ell(p) = \log(p)\sum_{i=1}^{n}x_i + \log(1-p)\sum_{i=1}^{n}(1-x_i)$$

**Finding MLE:**
$$\frac{\partial \ell}{\partial p} = \frac{1}{p}\sum_{i=1}^{n}x_i - \frac{1}{1-p}\sum_{i=1}^{n}(1-x_i) = 0$$

$$\hat{p} = \frac{1}{n}\sum_{i=1}^{n}x_i = \bar{x}$$

**Result:** MLE of success probability is the sample proportion.

---

## 3. Exponential Distribution

**Distribution:** $P(x|\lambda) = \lambda e^{-\lambda x}$ for $x > 0$

**Likelihood:**
$$\ell(\lambda) = n\log(\lambda) - \lambda\sum_{i=1}^{n}x_i$$

**Finding MLE:**
$$\frac{\partial \ell}{\partial \lambda} = \frac{n}{\lambda} - \sum_{i=1}^{n}x_i = 0$$

$$\hat{\lambda} = \frac{n}{\sum_{i=1}^{n}x_i} = \frac{1}{\bar{x}}$$

**Result:** MLE of rate is inverse of sample mean.

---

## 4. Poisson Distribution

**Distribution:** $P(x|\lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$ for $x \in \{0,1,2,...\}$

**Likelihood:**
$$\ell(\lambda) = \sum_{i=1}^{n}[x_i \log(\lambda) - \lambda - \log(x_i!)]$$

$$\ell(\lambda) = \log(\lambda)\sum_{i=1}^{n}x_i - n\lambda - \sum_{i=1}^{n}\log(x_i!)$$

**Finding MLE:**
$$\frac{\partial \ell}{\partial \lambda} = \frac{1}{\lambda}\sum_{i=1}^{n}x_i - n = 0$$

$$\hat{\lambda} = \frac{1}{n}\sum_{i=1}^{n}x_i = \bar{x}$$

**Result:** MLE of rate is the sample mean.

---

## Comparison Table

| Distribution | Parameter | MLE |
|---|---|---|
| Normal | $\mu$ | $\bar{x}$ |
| Normal | $\sigma^2$ | $\frac{1}{n}\sum(x_i - \bar{x})^2$ |
| Bernoulli | $p$ | $\bar{x}$ |
| Exponential | $\lambda$ | $1/\bar{x}$ |
| Poisson | $\lambda$ | $\bar{x}$ |

---

## Properties of MLEs

### 1. Consistency
As $n \to \infty$, $\hat{\theta}_{MLE} \xrightarrow{p} \theta^*$ (converges to true parameter)

### 2. Asymptotic Normality
For large $n$:
$$\sqrt{n}(\hat{\theta}_{MLE} - \theta^*) \xrightarrow{d} N(0, \mathcal{I}^{-1})$$

Where $\mathcal{I}$ is the Fisher Information Matrix.

**Fisher Information:**
$$\mathcal{I}(\theta) = -E\left[\frac{\partial^2 \ell}{\partial \theta^2}\right] = E\left[\left(\frac{\partial \ell}{\partial \theta}\right)^2\right]$$

### 3. Asymptotic Efficiency
MLE achieves the Cramér-Rao lower bound asymptotically (most efficient unbiased estimator).

### 4. Bias
Some MLEs are biased:
- Sample mean: **unbiased** for Normal $\mu$
- Sample variance: **biased** for Normal $\sigma^2$ (divides by $n$, should divide by $n-1$)
- Exponential rate: **biased** (Jensen's inequality: $E[1/\bar{X}] \neq 1/\lambda$)

---

## Confidence Intervals from MLEs

Using asymptotic normality, approximate confidence interval for single parameter:

$$\hat{\theta} \pm z_{\alpha/2} \sqrt{\frac{1}{\mathcal{I}(\hat{\theta})}}$$

Where $z_{\alpha/2}$ is the critical value (1.96 for 95% CI).

---

## Example: Normal Distribution MLEs

**Data:** $X = [2.1, 2.3, 1.9, 2.4, 2.0]$ (5 observations)

**Compute MLEs:**
- $\hat{\mu} = (2.1 + 2.3 + 1.9 + 2.4 + 2.0) / 5 = 10.7 / 5 = 2.14$
- $\hat{\sigma}^2 = \frac{1}{5}[(2.1-2.14)^2 + (2.3-2.14)^2 + ... ] = 0.0304$
- $\hat{\sigma} = 0.174$

**Asymptotic distribution:**
$$\hat{\mu} \approx N(2.14, \frac{\sigma^2}{n}) = N(2.14, 0.0061)$$

**95% CI for mean:**
$$2.14 \pm 1.96 \sqrt{0.0061} = 2.14 \pm 0.15 = [1.99, 2.29]$$

---

## Numerical Optimization

For complex likelihoods, use numerical methods:

1. **Newton-Raphson**: Uses gradient and Hessian
2. **Gradient Descent**: Follows gradient of likelihood
3. **Expectation-Maximization (EM)**: For missing data
4. **Stochastic Gradient Descent**: For large datasets

Most ML libraries use automatic differentiation (JAX, PyTorch) to compute gradients automatically.
