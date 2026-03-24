# A/B Testing and Statistical Significance

## Problem Statement

You want to test whether a new feature increases user engagement. You run an A/B test where:
- Control group (A): 10,000 users, 850 conversions
- Test group (B): 10,000 users, 920 conversions

Determine if the difference is statistically significant and estimate the true effect size.

## Solution

### 1. Problem Setup

**Null Hypothesis (H₀):** $p_A = p_B$ (no difference in conversion rates)
**Alternative Hypothesis (H₁):** $p_A \neq p_B$ (there is a difference)

**Observed Data:**
- Group A: $n_A = 10,000$, $x_A = 850$, $\hat{p}_A = 0.085$
- Group B: $n_B = 10,000$, $x_B = 920$, $\hat{p}_B = 0.092$
- Observed difference: $\Delta = \hat{p}_B - \hat{p}_A = 0.007$

### 2. Hypothesis Test

#### Option 1: Two-Proportion Z-Test

Assumes both samples are large (np > 5 for both groups - ✓ satisfied)

**Pooled proportion:**
$$\hat{p}_{pool} = \frac{x_A + x_B}{n_A + n_B} = \frac{850 + 920}{20,000} = 0.0885$$

**Standard error:**
$$SE = \sqrt{\hat{p}_{pool}(1-\hat{p}_{pool})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}$$
$$SE = \sqrt{0.0885 \times 0.9115 \times \frac{2}{10,000}} = 0.00412$$

**Z-statistic:**
$$Z = \frac{\hat{p}_B - \hat{p}_A}{SE} = \frac{0.007}{0.00412} = 1.70$$

**P-value (two-tailed):**
For Z = 1.70, p-value ≈ 0.089

#### Option 2: Confidence Interval

**95% Confidence Interval for difference:**
$$\Delta \pm Z_{0.025} \times SE = 0.007 \pm 1.96 \times 0.00412$$
$$= 0.007 \pm 0.0081 = [-0.0011, 0.0151]$$

Since the CI includes 0, we cannot reject H₀ at the 5% level.

### 3. Interpretation

**Statistical Significance (α = 0.05):**
- p-value = 0.089 > 0.05
- **Not statistically significant** at 5% level
- Cannot reject null hypothesis

**Effect Size:**
- Relative lift: $(920 - 850) / 850 = 8.24\%$
- Absolute difference: $0.7\%$ (7 basis points)

**Practical Significance:**
- While the relative lift seems large (8.24%), it's not statistically significant
- Need to consider:
  - Sample size: May need larger samples for this effect size
  - Cost of false positive vs false negative
  - Minimum detectable effect (MDE)

### 4. Sample Size Planning

To detect this effect size with 80% power and 5% significance:

**Using formula:**
$$n = 2 \times \left(\frac{Z_{\alpha/2} + Z_{\beta}}{\Delta/\sqrt{p(1-p)}}\right)^2$$

Where:
- $Z_{0.025} = 1.96$ (5% significance, two-tailed)
- $Z_{0.20} = 0.84$ (80% power)
- $\Delta = 0.007$ (0.7 percentage point difference)
- $p = 0.0885$ (pooled proportion)

$$n \approx 25,000 \text{ per group}$$

**Recommendation:** Need ~25,000 users per group (~50 days at current traffic) to reliably detect this effect.

### 5. Key Considerations

1. **Multiple Testing:** If running many tests, apply multiple comparison correction (e.g., Bonferroni)
2. **Traffic Allocation:** Consider 50-50 split vs other allocations
3. **Seasonality:** Run test long enough to account for daily/weekly patterns
4. **Heterogeneous Effects:** Check if effect varies by user segment
5. **Peeking Problem:** Don't stop test early based on intermediate results
6. **Intent to Treat:** Analyze by assignment, not by actual treatment received
