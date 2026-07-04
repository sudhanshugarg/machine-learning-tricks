# Linear Regression via Maximum Likelihood Estimation

## Problem Statement

Given a dataset of features $X = \{x_1, x_2, ..., x_n\}$ and corresponding targets $y = \{y_1, y_2, ..., y_n\}$, estimate the regression coefficients $\beta = [\beta_0, \beta_1, ..., \beta_p]$ using Maximum Likelihood Estimation.

**Key Assumption:** The errors are normally distributed:
$$y_i = \beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip} + \epsilon_i$$

where $\epsilon_i \sim N(0, \sigma^2)$ (errors are independent, normally distributed with constant variance).

**What this implies:** Given x_i, the target y_i follows a conditional normal distribution:
$$y_i | x_i \sim N(\mu_i, \sigma^2) \quad \text{where} \quad \mu_i = \beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip}$$

**Important:** We assume y is **conditionally Gaussian** given x, NOT that y is marginally Gaussian. The relationship between x and y is deterministic; only the deviations from this relationship (the errors) are random.

---

## Key Insight

This problem demonstrates a fundamental connection: **Ordinary Least Squares (OLS) regression is equivalent to Maximum Likelihood Estimation under the assumption of Gaussian errors**.

---

## Clarification: Conditional vs. Marginal Gaussianity

**Important distinction for interviews:**

- **What we assume:** Errors are Gaussian: $\epsilon_i \sim N(0, \sigma^2)$
- **What this means:** y is **conditionally** Gaussian given x: $y_i | x_i \sim N(f(x_i), \sigma^2)$
- **What we DON'T assume:** y is marginally Gaussian or that (x, y) are jointly Gaussian

**Example:** If x ranges from 0 to 10 and y = 2x + ε where ε ~ N(0, 1):
- y values near x=0 cluster around 0 (approximately)
- y values near x=10 cluster around 20 (approximately)
- The marginal distribution of y is NOT Gaussian - it has multiple modes!
- But given any specific x, the conditional distribution of y is Gaussian

This distinction matters because it clarifies that we're modeling uncertainty in predictions (the error term), not claiming the raw data is Gaussian.

---

## Mathematical Derivation

### 1. Likelihood Function

For a single observation, the **conditional distribution** of y given x is:
$$y_i | \mathbf{x}_i \sim N(\mu_i, \sigma^2) \quad \text{where} \quad \mu_i = \mathbf{x}_i^T \boldsymbol{\beta}$$

This comes from the model: $y_i = \mathbf{x}_i^T \boldsymbol{\beta} + \epsilon_i$ with $\epsilon_i \sim N(0, \sigma^2)$.

The conditional probability density is:
$$P(y_i | \mathbf{x}_i, \boldsymbol{\beta}, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2}{2\sigma^2}\right)$$

For all $n$ observations (assuming independence):
$$L(\boldsymbol{\beta}, \sigma^2 | X, y) = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2}{2\sigma^2}\right)$$

### 2. Log-Likelihood

Taking the log of the joint likelihood:

$$\ell(\boldsymbol{\beta}, \sigma^2) = \sum_{i=1}^{n} \log P(y_i | \mathbf{x}_i, \boldsymbol{\beta}, \sigma^2)$$

$$= -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2$$

$$= -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \frac{1}{2\sigma^2}\text{RSS}(\boldsymbol{\beta})$$

where RSS is the Residual Sum of Squares (sum of squared errors).

### 3. Finding MLE for $\boldsymbol{\beta}$

Taking the derivative with respect to $\boldsymbol{\beta}$:

$$\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \frac{1}{\sigma^2} \sum_{i=1}^{n} \mathbf{x}_i (y_i - \mathbf{x}_i^T \boldsymbol{\beta}) = 0$$

$$\frac{1}{\sigma^2} \mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0$$

$$\mathbf{X}^T \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^T \mathbf{y}$$

$$\boxed{\hat{\boldsymbol{\beta}}_{MLE} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}}$$

**This is exactly the OLS solution!**

### 4. Finding MLE for $\sigma^2$

Taking the derivative with respect to $\sigma^2$:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^T \hat{\boldsymbol{\beta}})^2 = 0$$

$$\boxed{\hat{\sigma}^2_{MLE} = \frac{1}{n}\text{RSS}(\hat{\boldsymbol{\beta}}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{x}_i^T \hat{\boldsymbol{\beta}})^2}$$

**Note:** This divides by $n$, not $n-1$ (MLE is biased; unbiased estimator divides by $n-p-1$).

---

## Algorithm

**Input:** Feature matrix $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$, target vector $\mathbf{y} \in \mathbb{R}^n$

**Output:** Estimated coefficients $\hat{\boldsymbol{\beta}}$, error variance $\hat{\sigma}^2$

1. Compute the normal equation solution: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
2. Compute residuals: $\mathbf{r} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$
3. Estimate error variance: $\hat{\sigma}^2 = \frac{1}{n}\|\mathbf{r}\|^2$
4. Compute standard errors and confidence intervals using Fisher Information

---

## Complexity Analysis

- **Time Complexity:** $O(n p^2 + p^3)$ for solving the normal equations (matrix inversion scales as $O(p^3)$)
- **Space Complexity:** $O(np + p^2)$ to store feature matrix and Gram matrix $\mathbf{X}^T \mathbf{X}$

---

## Fisher Information Matrix

For linear regression under Gaussian errors:

$$\mathcal{I}(\boldsymbol{\beta}) = \frac{1}{\sigma^2} \mathbf{X}^T \mathbf{X}$$

**Asymptotic distribution of $\hat{\boldsymbol{\beta}}$:**
$$\hat{\boldsymbol{\beta}} \approx N\left(\boldsymbol{\beta}^*, \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}\right)$$

**Standard errors:** The diagonal elements of $\sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$ give the variance of each coefficient.

---

## Confidence Intervals

For a 95% confidence interval on coefficient $\beta_j$:

$$\hat{\beta}_j \pm t_{n-p-1, 0.025} \cdot \text{SE}(\hat{\beta}_j)$$

where $\text{SE}(\hat{\beta}_j) = \sqrt{\text{Var}(\hat{\beta}_j)}$ and $t_{n-p-1}$ is the Student's t-distribution critical value.

---

## Example: House Price Prediction

**Dataset:** 10 houses with features (square footage, bedrooms) and prices

| Sqft | Beds | Price |
|------|------|-------|
| 2000 | 3 | 300k |
| 2500 | 4 | 350k |
| 1800 | 3 | 280k |
| ... | ... | ... |

**Model:** $\text{Price} = \beta_0 + \beta_1 \cdot \text{Sqft} + \beta_2 \cdot \text{Beds} + \epsilon$

**MLE yields:**
- $\hat{\beta}_0 = 50,000$ (base price)
- $\hat{\beta}_1 = 100$ (price per sqft)
- $\hat{\beta}_2 = 20,000$ (price per bedroom)
- $\hat{\sigma}^2 = 5,000,000$ (variance of prediction errors)

---

## Interview Questions to Expect

1. **Why is OLS equivalent to MLE?**
   - Answer: Both minimize the sum of squared errors when assuming Gaussian errors.

2. **What assumptions must hold for MLE to be valid?**
   - Linear relationship between features and target
   - Errors are normally distributed
   - Constant variance (homoscedasticity)
   - Independence of observations

3. **What if errors aren't normally distributed?**
   - MLE may not be efficient
   - Could use other loss functions (e.g., Laplace for heavy tails)

4. **How do you compute confidence intervals?**
   - Use Fisher Information to get standard errors
   - Construct intervals using t-distribution

5. **What about multicollinearity?**
   - $\mathbf{X}^T \mathbf{X}$ becomes singular or ill-conditioned
   - Need regularization (Ridge/Lasso regression)

---

## Extension: Regularized Linear Regression

If we add a prior on $\boldsymbol{\beta}$ (e.g., $\boldsymbol{\beta} \sim N(0, \frac{\sigma^2}{\lambda} \mathbf{I})$), the MLE becomes:

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

This connects MLE with Bayesian inference and Ridge regression!

