# Bayesian Credible Intervals Solution

## Problem Setup

**Observed Data:**
- Clicks: 45 (successes)
- Impressions: 500 (trials)
- Sample CTR: $\hat{p} = 45/500 = 0.09$

**Prior:**
- Distribution: Beta(α=10, β=100)
- Mean: $\alpha/(\alpha+\beta) = 10/110 ≈ 0.0909$
- Interpretation: Prior belief that CTR ≈ 9%

**Likelihood:**
- Distribution: Bernoulli/Binomial
- Model: $P(\text{data} | p) = \binom{n}{x} p^x (1-p)^{n-x}$

---

## Posterior Distribution

### Conjugate Prior Analysis

For Binomial likelihood and Beta prior, the posterior is also Beta (conjugate pair).

**Posterior distribution:**
$$P(p | \text{data}) = \text{Beta}(\alpha + x, \beta + n - x)$$

Where:
- $\alpha$ = prior successes = 10
- $\beta$ = prior failures = 100
- $x$ = observed successes = 45
- $n$ = total trials = 500
- $n - x$ = observed failures = 455

**Posterior parameters:**
$$\alpha' = 10 + 45 = 55$$
$$\beta' = 100 + 455 = 555$$

**Posterior distribution:**
$$P(p | \text{data}) = \text{Beta}(55, 555)$$

---

## Posterior Mean and Variance

**Posterior mean:**
$$E[p | \text{data}] = \frac{\alpha'}{\alpha' + \beta'} = \frac{55}{55 + 555} = \frac{55}{610} ≈ 0.0902$$

**Posterior variance:**
$$\text{Var}[p | \text{data}] = \frac{\alpha' \beta'}{(\alpha' + \beta')^2(\alpha' + \beta' + 1)}$$
$$= \frac{55 \times 555}{610^2 \times 611} ≈ 0.000147$$

**Posterior standard deviation:**
$$\text{SD} ≈ \sqrt{0.000147} ≈ 0.0121$$

---

## Credible Intervals

### Method 1: Highest Density Interval (HDI)

The HDI is the shortest interval containing a given probability mass.

For Beta distribution, compute numerically or use quantiles (often similar).

### Method 2: Quantile-Based Interval

For 95% credible interval:
- Lower quantile: $Q_{0.025}$ (2.5th percentile)
- Upper quantile: $Q_{0.975}$ (97.5th percentile)

For Beta(55, 555):
$$\text{95% Credible Interval} = [0.0662, 0.1154]$$

**Interpretation:** Given the data and prior, there's 95% probability that the true CTR is between 6.62% and 11.54%.

---

## Comparison: Bayesian vs Frequentist

### Frequentist Confidence Interval (Wald/Score)

**Standard error:**
$$SE = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.09 \times 0.91}{500}} ≈ 0.0131$$

**95% Confidence Interval:**
$$\hat{p} \pm 1.96 \times SE = 0.09 \pm 0.0257 = [0.0643, 0.1157]$$

**Interpretation:** If we repeated the experiment infinitely, 95% of such intervals would contain the true parameter.

### Comparison Table

| Aspect | Bayesian | Frequentist |
|---|---|---|
| **Distribution** | Beta(55, 555) | N(0.09, 0.0131²) |
| **Point Estimate** | 0.0902 | 0.0900 |
| **95% Interval** | [0.0662, 0.1154] | [0.0643, 0.1157] |
| **Interpretation** | Probability of parameter | Coverage property |
| **Prior** | Required | Not used |
| **Decision making** | Direct | Indirect (p-values) |

---

## Influence of Prior

### Scenario 1: Weak Prior
Beta(1, 1) - Uniform prior (no prior knowledge)

**Posterior:** Beta(45+1, 455+1) = Beta(46, 456)
- **Mean:** 46/502 ≈ 0.0916
- **95% CI:** ≈ [0.0662, 0.1186]

### Scenario 2: Strong Prior (optimistic)
Beta(50, 50) - Prior belief that CTR ≈ 50%

**Posterior:** Beta(95, 505)
- **Mean:** 95/600 ≈ 0.1583
- **95% CI:** ≈ [0.1314, 0.1869]

### Scenario 3: Strong Prior (pessimistic)
Beta(5, 150) - Prior belief that CTR ≈ 3%

**Posterior:** Beta(50, 605)
- **Mean:** 50/655 ≈ 0.0763
- **95% CI:** ≈ [0.0548, 0.0996]

**Observation:** Stronger priors pull the posterior toward the prior, shrinking credible intervals.

---

## Sequential Bayesian Updating

**Day 1:** Observe 10 clicks out of 100

**Posterior becomes new prior:** Beta(10+10, 100+90) = Beta(20, 190)

**Day 2:** Observe 35 clicks out of 400

**Updated posterior:** Beta(20+35, 190+365) = Beta(55, 555)

**Key advantage:** Sequential learning without batch reprocessing.

---

## Summary of Key Differences

### Bayesian Credible Interval
- **Setup:** Posterior distribution after incorporating prior
- **Meaning:** "95% probability parameter is in this range"
- **Action:** Directly use for decision making
- **Prior:** Must be specified
- **Interpretation:** Straightforward probabilistic

### Frequentist Confidence Interval
- **Setup:** Derived from repeated sampling distribution
- **Meaning:** "If we repeat experiment, 95% of intervals contain parameter"
- **Action:** Indirect (hypothesis tests, p-values)
- **Prior:** Not used
- **Interpretation:** Subtle (long-run property)

Both give similar intervals in this example, but differ in:
- Computational approach
- Philosophical interpretation
- How they handle priors and sequential data
