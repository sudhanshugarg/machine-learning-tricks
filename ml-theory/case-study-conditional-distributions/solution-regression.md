# Solution: Understanding Conditional Distributions and Regression

## Initial Question: How Would You Model x and y?

**Strong Answer:**

I would frame this as a **regression problem using conditional expectation**:

1. **Estimate the conditional expectation:** Learn $E[y|x]$ from the data
   - This tells us: "Given feature value $x$, what is the expected value of $y$?"
   - It's the best single prediction for any value of $x$

2. **Estimate the conditional distribution:** Understand $P(y|x)$
   - Beyond just the mean, we care about the variance and shape of the distribution
   - In some applications, quantiles or full distribution matter

3. **Make predictions with uncertainty:**
   $$\hat{y} = E[y|x] \text{ (point prediction)}$$
   $$\text{Confidence interval: } \hat{y} \pm 1.96 \cdot \text{SE}[y|x]$$
   where $\text{SE}[y|x]$ is the standard error of the prediction

**Why this framing matters:**
- It's theoretically grounded in probability and statistics
- It separates the problem of learning from the problem of uncertainty quantification
- The choice of loss function becomes a knob we can turn based on business constraints
- It naturally leads to discussing irreducible error and heteroscedasticity

---

## Question 1: Can You Assume a Linear Relationship?

**Answer: It depends on the data, but linearity is worth trying first. Here's why:**

### The Case for Linear Models

Linear models are reasonable when:
- The expected value $E[y|x]$ appears to follow a straight line (on average)
- The relationship is monotonic (consistently increasing or decreasing)
- Computational simplicity is valued
- Interpretability matters

**Visual example:**
```
     y
     ▲
     │       ╱  ← True relationship (noisy)
     │      ╱
  E[y|x]  ╱
     │   ╱
     │  ╱      ± noise
     │ ╱
     │╱________─────► x
```

### Why Non-Linearity Matters

But linear may be insufficient if:
- Scatter plot shows clear curve (non-monotonic, accelerating, etc.)
- Residuals show systematic patterns (heteroscedasticity, non-constant variance)
- The relationship changes slope dramatically at certain values

**Visual example:**
```
     y
     ▲
     │           ╱╱  ← True relationship is curved
     │         ╱╱
     │       ╱╱
     │      ╱
     │    ╱
     │   ╱
     │  ╱
     │ ╱
     │╱________─────► x

Linear model would systematically under/over-predict
```

### Practical Approach

1. **Start simple:** Fit a linear model
2. **Check residuals:** Do they show patterns? Heteroscedasticity?
3. **Consider non-linear extensions:**
   - Polynomial features: $x, x^2, x^3, ...$
   - Splines: Piecewise polynomials
   - Tree-based methods: Non-parametric flexibility
   - Neural networks: High flexibility but prone to overfitting

### Key Insight
- **Linear is not bad:** It's interpretable and has low variance. If it fits reasonably, use it.
- **Linear is not magic:** Sometimes the relationship genuinely requires non-linearity
- **Validate empirically:** Use cross-validation to see if non-linear models generalize better

---

## Question 2: Is a Single Model Sufficient?

**Answer: Yes, if it estimates the conditional mean. But you should be aware of its limitations.**

### What a Single Model Captures

A traditional regression model (linear, polynomial, tree-based) learns:
$$\hat{y} = \hat{E}[y|x]$$

This is the **best single prediction** under squared error loss. It minimizes MSE.

**Advantages:**
- Simple, interpretable, computationally efficient
- Well-understood theory and evaluation metrics
- Works well for most applications

**Visual:**
```
     y
     ▲
     │                    ●●
     │                  ●●  ← Data points (with noise)
  E[y|x] │  ────────────      ← Regression fit (mean trajectory)
     │  ●●●
     │●●
     │╱________─────► x
```

### What a Single Model Misses

However, a single model **cannot capture:**
- The full conditional distribution $P(y|x)$
- The conditional variance $\text{Var}(y|x)$ (may vary with $x$)
- Quantiles like 25th percentile, 75th percentile
- Prediction intervals or uncertainty estimates

**Problem example:**
```
     y
     ▲
     │          ││        ← Wide spread (high uncertainty)
     │        ││  ││
  E[y|x] │  ──│────│──      ← Single model can't capture this variation
     │  │││││││
     │ ││││  │     ← Tight spread (low uncertainty)
     │╱────┼────► x
```

### When You Need More

1. **For uncertainty quantification:**
   - Quantile regression: Learn $\hat{y}_{0.25}(x)$ and $\hat{y}_{0.75}(x)$ separately
   - Bayesian regression: Get posterior distributions over predictions
   - Ensemble methods: Use variance across predictions

2. **For heteroscedasticity (non-constant variance):**
   - Separate model for variance: Learn both $\hat{E}[y|x]$ and $\hat{\text{Var}}(y|x)$
   - Mixture models: Different distributions in different regions

3. **For special objectives:**
   - Risk-averse applications: Optimize for worst-case or high percentiles
   - Safety-critical: Want tight confidence intervals

---

## Question 3: Irreducible Error and Bayes Optimal Rate

**Answer: This is fundamental. Here's the full explanation:**

### What is Irreducible Error?

**Definition:** The minimum prediction error achievable by *any* regression model, given the data distribution.

Also called:
- **Bayes error** (in the regression context)
- **Aleatoric uncertainty** (inherent randomness)
- **Noise** in the classical sense

It's a **fundamental limit** set by nature, not by your choice of algorithm or model capacity.

### Mathematical Definition

The irreducible error is the variance of $y$ around its conditional mean:

$$\sigma^2_{\text{irred}} = E[(y - E[y|x])^2] = E[\text{Var}(y|x)]$$

This is the expected squared deviation even with perfect knowledge of $E[y|x]$.

### Intuition

Even if you know the true $E[y|x]$ perfectly, individual observations $y$ vary randomly around this mean.

**Example: House price prediction**
- Feature $x$ = square footage
- You learn that 2000 sq-ft houses average $500k (i.e., $E[y|x=2000] = 500k$)
- But individual 2000 sq-ft houses vary: some sell for $480k, others for $520k
- **This variation is irreducible** — no amount of data or model complexity eliminates it
- It comes from unobserved factors (location, condition, market fluctuations)

### Visualizing Irreducible Error

```
     y
     ▲
     │  ╱╲ ╱╲ ╱╲
     │ ╱  ╲  ╱  ╲
  E[y|x] │────────────────  ← True conditional mean
     │ ╲  ╱  ╲  ╱  ╲ ╱╲
     │  ╱╲ ╱╲ ╱╲
     │╱────────────────► x

     Vertical spread = Irreducible error (variance around mean)
     Even perfect model can't eliminate this variation
```

### Key Mathematical Fact

For any regression model $\hat{y} = f(x)$:

$$\text{Expected MSE} = (E[\hat{y}] - y)^2 + \text{Var}(\hat{y}) + \sigma^2_{\text{irred}}$$

Which decomposes as:
$$\text{Expected MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

The irreducible error is the **third term** — nothing can reduce it.

### Practical Implications

1. **More overlap in conditional distributions = Higher irreducible error**
   - Tight conditional distributions → Low irreducible error → Easy prediction
   - Spread-out distributions → High irreducible error → Inherently noisy problem

2. **No algorithm beats irreducible error:**
   - Complex models (neural nets, boosted trees) can't reduce it
   - More features can reduce it only if they explain the missing variance
   - Better domain knowledge → Better feature engineering → Reduced irreducible error

3. **Irreducible error is not a failure — it's reality:**
   - It tells you the fundamental difficulty of the problem
   - Use it to set expectations with stakeholders: "Even perfect models will have this error"
   - Helps evaluate whether further model improvement is worthwhile

### Estimating Irreducible Error

In practice, you can't directly compute $\sigma^2_{\text{irred}}$ without the true $E[y|x]$. But you can estimate a lower bound:

1. **Use a sufficiently flexible model** (e.g., boosted trees, neural networks)
2. **Cross-validate to get an estimate** of test error
3. **The difference between train and test error** gives you variance
4. **The floor is the irreducible error:**

```python
import numpy as np

# After fitting with cross-validation
train_mse = np.mean((y_train - predictions_train)**2)
test_mse = np.mean((y_test - predictions_test)**2)

# Residual variance (lower bound on irreducible error)
residual_variance = np.var(y_test - predictions_test)

print(f"Train MSE: {train_mse:.3f}")
print(f"Test MSE: {test_mse:.3f}")
print(f"Residual variance (lower bound on irreducible error): {residual_variance:.3f}")
```

### Connection to the Overlap

The overlap in the conditional distributions you see in the problem is **visual evidence of irreducible error**:
- Points with the same $x$ have different $y$ values
- This variation = irreducible noise
- The bigger the spread, the larger the irreducible error

---

## Question 4: Error Metric Selection in Practice

**Answer: The best metric depends on your business constraints. Here are the strategies:**

### Strategy 1: Mean Squared Error (MSE) / Root MSE (RMSE)

**Use when:** You want to minimize average squared error.

**Formula:**
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

$$\text{RMSE} = \sqrt{\text{MSE}}$$

**Pros:**
- Mathematically convenient (smooth, differentiable)
- Penalizes large errors more heavily
- Standard in most ML libraries

**Cons:**
- Sensitive to outliers (squared penalty)
- Assumes symmetric error distribution

**Example:**
- Predicting next quarter revenue
- Goal: Minimize average squared deviation from actual
- Large errors are disproportionately bad

---

### Strategy 2: Mean Absolute Error (MAE)

**Use when:** Large errors and small errors are equally penalized.

**Formula:**
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Pros:**
- Robust to outliers (linear penalty)
- Easy to interpret (error in original units)
- Makes the data-generation process clear

**Cons:**
- Not differentiable at zero (can cause optimization issues)
- Less penalizing of large errors

**Example:**
- Predicting customer delivery times
- Each hour of delay is equally costly (no exponential penalty)
- Robust to occasional very late deliveries

---

### Strategy 3: Huber Loss (Balanced Approach)

**Use when:** You want robustness to outliers but still care about large errors.

**Formula:**
$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

**Behavior:**
- Quadratic loss for small errors (like MSE)
- Linear loss for large errors (like MAE)
- Smooth transition parameter $\delta$

**Visual:**
```
Loss
  ▲
  │           ╱ Huber (quadratic then linear)
  │         ╱╱╱
  │       ╱╱╱   MSE (always quadratic)
  │      ╱ ╱╱╱╱╱
  │    ╱╱╱
  │   ╱╱  MAE (always linear)
  │ ╱╱╱
  │╱────────── Error
  └────────────────►
```

**Pros:**
- Robust to outliers but still penalizes large errors
- Smooth and differentiable everywhere

**Cons:**
- Introduces hyperparameter $\delta$ to tune
- Less interpretable than MSE or MAE

---

### Strategy 4: Quantile Loss (Asymmetric Penalty)

**Use when:** You care more about over-predicting or under-predicting.

**Formula (for quantile $q$):**
$$L_q(y, \hat{y}) = \begin{cases} q(y - \hat{y}) & \text{if } y > \hat{y} \\ (1-q)(\hat{y} - y) & \text{otherwise} \end{cases}$$

**For median ($q=0.5$):** Same as MAE

**For 90th percentile ($q=0.9$):**
- Over-predictions penalized lightly (weight 0.1)
- Under-predictions penalized heavily (weight 0.9)
- Encourages conservative (higher) predictions

**Example:**
- Inventory management: Under-predicting demand is worse than over-predicting
- Use $q=0.9$ to optimize for high-demand scenarios
- Predictions will be conservative (higher)

**Visual:**
```
Quantile Loss for Different q Values

Loss
  ▲    q=0.1 (optimistic) — Under-penalized when y>ŷ
  │  ╱╱ Heavy when y<ŷ
  │ ╱╱
  │╱╱
  ├─────────────────► Error
  │  ╱╱╱
  │ ╱ q=0.5 (median/MAE) — Symmetric
  │╱
  │╱╱╱╱╱╱╱╱
  │╱╱╱    q=0.9 (conservative) — Under-penalized when y<ŷ
```

**When to use:**
- **$q < 0.5$:** Under-predictions cost more (optimistic targets)
- **$q = 0.5$:** Symmetric cost (MAE)
- **$q > 0.5$:** Over-predictions cost more (conservative targets)

---

### Strategy 5: Custom Business Loss

**Use when:** Standard metrics don't align with business value.

**Example: Fraud Detection Costs**
- False negative (miss fraud): Lose $10,000
- False positive (flag legitimate): Lose $50 (investigation cost)

Custom loss:
$$\text{Loss} = 10000 \cdot \mathbb{1}[\text{fraud but predicted legitimate}] + 50 \cdot \mathbb{1}[\text{legitimate but predicted fraud}]$$

**Example: Medical Prediction**
- Under-predicting disease severity: Patient gets inadequate treatment (very bad)
- Over-predicting: Unnecessary precautions (bad but less)

Use quantile regression with $q=0.75$ or higher to be conservative.

---

### Practical Decision Framework

| Problem Type | Metric | Rationale |
|---|---|---|
| **General prediction** | RMSE | Standard, interpretable |
| **With outliers** | MAE | Robust to extremes |
| **Balanced robustness** | Huber | Best of both |
| **Asymmetric consequences** | Quantile loss | Penalize over/under-prediction differently |
| **Custom business value** | Custom loss | Directly optimize for business metric |
| **Multiple objectives** | Weighted combination | Combine metrics with weights |

---

## Question 5: Heteroscedasticity and Adaptive Prediction

**Answer: This is a sophisticated consideration. Here's how to handle it:**

### What is Heteroscedasticity?

**Definition:** The conditional variance $\text{Var}(y|x)$ changes with $x$.

In other words: Some regions of feature space are inherently noisier than others.

**Visual:**
```
     y
     ▲
     │          ││        ← High uncertainty (wide distribution)
     │        ││  ││
     │  ──────────────     ← Fitted mean trajectory
     │ ││││ ││
     │││ ││ ││  │││        ← Low uncertainty (tight distribution)
     │╱────────────────► x
```

**Real examples:**
- **Stock price prediction:** Volatility increases during market crashes
- **Income prediction:** Uncertainty higher for senior positions (more variation)
- **Weather forecasting:** Harder to predict in unstable conditions
- **Loan default:** Risk higher when economic uncertainty increases

### Why Standard Regression Misses This

Most regression models (linear, polynomial, trees) learn only $E[y|x]$. They assume:
$$\text{Var}(y|x) = \sigma^2 \quad \text{(constant across } x\text{)}$$

This assumption is often violated, leading to:
1. **Biased confidence intervals** (too narrow in high-variance regions)
2. **Suboptimal predictions** (same prediction uncertainty everywhere)
3. **Poor decision-making** (no signal about which predictions to trust)

### Solution 1: Quantile Regression

**Idea:** Learn multiple regression lines for different quantiles.

Learn $\hat{y}_{q}(x)$ for different $q$ (e.g., 0.1, 0.5, 0.9):
- $\hat{y}_{0.1}(x)$ = 10th percentile of $y|x$ (lower bound)
- $\hat{y}_{0.5}(x)$ = Median / 50th percentile (center)
- $\hat{y}_{0.9}(x)$ = 90th percentile of $y|x$ (upper bound)

**Advantages:**
- Captures full conditional distribution (not just mean)
- Automatically adapts to heteroscedasticity
- Prediction interval: $[\hat{y}_{0.1}(x), \hat{y}_{0.9}(x)]$

**Visualization:**
```
     y
     ▲
     │          ╱── 90th percentile (wider spread)
     │        ╱╱
     │  ────50th percentile (mean)
     │ ╱╱
     │╱── 10th percentile (narrower)
     │╱────────────────► x

Prediction interval adapts to local uncertainty
```

**Implementation:**
```python
from sklearn.linear_model import QuantileRegressor

# Learn different quantiles
quantiles = [0.1, 0.5, 0.9]
models = {}

for q in quantiles:
    model = QuantileRegressor(quantile=q)
    model.fit(X_train, y_train)
    models[q] = model

# Predict with uncertainty
y_pred_lower = models[0.1].predict(X_test)
y_pred_median = models[0.5].predict(X_test)
y_pred_upper = models[0.9].predict(X_test)

# Prediction intervals
prediction_interval = y_pred_upper - y_pred_lower
print(f"Median prediction: {y_pred_median}")
print(f"90% prediction interval: [{y_pred_lower}, {y_pred_upper}]")
print(f"Interval width: {prediction_interval}")
```

---

### Solution 2: Bayesian Regression

**Idea:** Model both parameters and predictions as distributions.

Learn $P(\theta | \text{data})$ and use it to generate $P(y|x, \text{data})$.

**Advantages:**
- Natural uncertainty quantification
- Adapts confidence based on local data density
- Can incorporate prior knowledge

**Visual:**
```
     y
     ▲
     │          ║║║║║      ← Wide posterior (high uncertainty)
     │        ║║    ║║
     │  ────────────────     ← Mean prediction
     │  ║║║║ ║║
     │ ║  ║║ ║║  ║║║        ← Narrow posterior (low uncertainty)
     │╱────────────────► x
```

**Implementation:**
```python
import pymc as pm
import arviz as az

with pm.Model() as model:
    # Prior on slope and intercept
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Linear model
    mu = alpha + beta * X_train

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)

    # Inference
    trace = pm.sample()

# Predictions with uncertainty
preds = pm.sample_posterior_predictive(trace, var_names=['y_obs'])
y_pred_mean = preds.posterior_predictive['y_obs'].mean(dim=['chain', 'draw'])
y_pred_std = preds.posterior_predictive['y_obs'].std(dim=['chain', 'draw'])

print(f"Prediction: {y_pred_mean}")
print(f"95% interval: [{y_pred_mean - 1.96*y_pred_std}, {y_pred_mean + 1.96*y_pred_std}]")
```

---

### Solution 3: Ensemble Methods with Uncertainty

**Idea:** Use multiple models and estimate variance from disagreement.

**Approach:**
1. Train $N$ different models (bootstrap samples, different architectures)
2. Make predictions on same data
3. Estimate $E[y|x]$ as mean across models
4. Estimate $\text{Var}(y|x)$ as variance across models

**Visual:**
```
     y
     ▲
     │          x x x x x   ← Disagreement = high uncertainty
     │        x     x
     │  ────────────────     ← Ensemble mean
     │  x x x x     ← Consensus = low uncertainty
     │╱────────────────► x
```

**Implementation:**
```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Ensemble approach
models = [
    BaggingRegressor(DecisionTreeRegressor(), n_estimators=100),
    RandomForestRegressor(n_estimators=100),
    GradientBoostingRegressor(n_estimators=100)
]

predictions = np.array([model.fit(X_train, y_train).predict(X_test)
                        for model in models])

# Ensemble statistics
y_pred = predictions.mean(axis=0)  # Mean prediction
y_pred_std = predictions.std(axis=0)  # Uncertainty

# Prediction intervals
y_pred_lower = y_pred - 1.96 * y_pred_std
y_pred_upper = y_pred + 1.96 * y_pred_std

print(f"Prediction: {y_pred}")
print(f"95% interval: [{y_pred_lower}, {y_pred_upper}]")
```

---

### Solution 4: Variance-Stabilizing Transformations

**Idea:** If heteroscedasticity is due to scale, transform $y$.

**Example:** If variance increases with mean (common in count/financial data):
- Use log transformation: $y' = \log(y)$
- Fit model on $y'$
- Transform predictions back: $\hat{y} = \exp(\hat{y}')$

**Benefits:**
- Simple to implement
- Stabilizes variance
- Improves prediction intervals

```python
import numpy as np

# Log transform (works if y > 0)
y_log = np.log(y_train + 1)  # +1 to avoid log(0)

# Fit model
model.fit(X_train, y_log)

# Predict and transform back
y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log) - 1
```

---

### When to Use Each Approach

| Approach | When to Use | Pros | Cons |
|---|---|---|---|
| **Quantile Regression** | Want percentile predictions | Flexible, non-parametric | Can be slower |
| **Bayesian** | Need principled uncertainty | Theoretically clean | Computationally expensive |
| **Ensemble** | Need simplicity | Easy to implement | Many models needed |
| **Transformation** | Heteroscedasticity has clear pattern | Fast and simple | Only works for certain patterns |
| **Standard regression** | Uncertainty not important | Simple, fast | Ignores variation |

---

## Putting It All Together: A Complete Example

### Scenario: Predicting House Prices

**Given:**
- Feature $x$ = Square footage
- Target $y$ = Sale price
- Data shows prices are more variable for large houses
- Goal: Predict price with uncertainty bounds

### Your Answer Flow:

1. **Initial modeling:**
   - Estimate $E[y|x]$ using linear or polynomial regression
   - Estimate $\text{Var}(y|x)$ to understand prediction uncertainty

2. **Discuss linearity:**
   - "Price likely grows roughly linearly with square footage"
   - "But there may be non-linearities (e.g., luxury premium for large homes)"
   - "Check residuals to see if linear is sufficient"

3. **Discuss irreducible error:**
   - "Even with perfect square footage, prices vary due to location, condition, etc."
   - "This irreducible noise is reflected in the spread of prices at each size"
   - "We can quantify it but not eliminate it"

4. **Choose error metric:**
   - "For house prices, I'd use RMSE (standard for regression)"
   - "RMSE is in dollars, making it interpretable to stakeholders"
   - "If we care about percentage error, could use MAPE instead"

5. **Handle heteroscedasticity:**
   - "Larger homes have more price variation — I'd use quantile regression"
   - "Learn 10th, 50th, 90th percentile prices"
   - "Prediction interval automatically widens for large homes"

6. **Evaluate and iterate:**
   - "Cross-validate to estimate irreducible error"
   - "Monitor both MSE and prediction interval widths"
   - "If intervals are too wide in some regions, engineer better features"

---

## Why This Answer Impresses Interviewers

✓ Shows probabilistic thinking (not just memorizing algorithms)
✓ Understands fundamental limits (irreducible error)
✓ Connects theory to practice (loss function and metric selection)
✓ Acknowledges heteroscedasticity (not all regions equally predictable)
✓ Proposes solutions adapted to business constraints (quantile regression, uncertainty)
✓ Avoids over-engineering (picks appropriate complexity)

---

## Appendix: Estimating Prediction Intervals

### The Problem

**Scenario:** You've fit a regression model. Now you want to provide confidence/prediction intervals.

**Question:** What's the difference between confidence intervals and prediction intervals?

- **Confidence interval for $E[y|x]$:** Where do we think the *mean* is? (narrower)
- **Prediction interval for $y|x$:** Where will *individual* observations fall? (wider)

### Mathematical Framework

For a linear regression model:
$$y = \beta_0 + \beta_1 x + \epsilon, \quad \epsilon \sim N(0, \sigma^2)$$

**Confidence interval for $E[y|x]$:**
$$E[y|x] \pm t_{\alpha/2, n-p} \cdot \text{SE}(E[y|x])$$

where $\text{SE}(E[y|x])$ is the standard error of the fitted mean.

**Prediction interval for $y|x$:**
$$\hat{y}|x \pm t_{\alpha/2, n-p} \cdot \sqrt{\text{SE}(\hat{y}|x)^2 + \hat{\sigma}^2}$$

The key difference: We add $\hat{\sigma}^2$ (residual variance) to account for individual variation.

### Implementation

```python
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate residuals and residual variance
y_train_pred = model.predict(X_train)
residuals = y_train - y_train_pred
sigma_hat = np.std(residuals)

# For confidence intervals, also need SE of fitted values
# (More complex, depends on design matrix)

# Simpler approach: Use prediction intervals directly
n = len(X_train)
p = X_train.shape[1]
dof = n - p
t_crit = stats.t.ppf(0.975, dof)  # 95% interval

# Prediction interval (simple approximation)
margin = t_crit * sigma_hat * np.sqrt(1 + 1/n)

y_lower = y_pred - margin
y_upper = y_pred + margin

print(f"Prediction: {y_pred}")
print(f"95% prediction interval: [{y_lower}, {y_upper}]")
```

### More Rigorous Approach

For more accurate intervals, use the actual standard errors:

```python
from scipy.stats import t

def prediction_interval(X_train, y_train, X_test, alpha=0.05):
    """Compute prediction intervals for linear regression"""

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Residual standard error
    y_train_pred = model.predict(X_train)
    residuals = y_train - y_train_pred
    mse = np.sum(residuals**2) / (len(y_train) - X_train.shape[1])
    sigma_hat = np.sqrt(mse)

    # Standard error of predictions
    # For simplicity, use average SE
    se_fit = sigma_hat / np.sqrt(len(X_train))
    se_pred = sigma_hat * np.sqrt(1 + 1/len(X_train))

    # Critical t-value
    dof = len(y_train) - X_train.shape[1]
    t_crit = t.ppf(1 - alpha/2, dof)

    # Intervals
    lower = y_pred - t_crit * se_pred
    upper = y_pred + t_crit * se_pred

    return y_pred, lower, upper

# Usage
y_pred, y_lower, y_upper = prediction_interval(X_train, y_train, X_test)
print(f"Prediction: {y_pred}")
print(f"95% PI: [{y_lower}, {y_upper}]")
```

### Key Insights

1. **Prediction intervals are always wider** than confidence intervals
   - Confidence interval: Uncertainty about the mean
   - Prediction interval: Uncertainty about future observations

2. **Wider intervals for new data:**
   - Points far from training data (high leverage)
   - High residual variance

3. **In heteroscedastic data, use quantile regression** for better intervals:
   ```python
   # Quantile regression gives better intervals when variance changes
   model_lower = QuantileRegressor(quantile=0.025).fit(X_train, y_train)
   model_upper = QuantileRegressor(quantile=0.975).fit(X_train, y_train)

   y_lower = model_lower.predict(X_test)
   y_upper = model_upper.predict(X_test)
   ```

---

## Key Takeaways

1. **Regression assumes a probabilistic model** of $E[y|x]$ with noise
2. **Irreducible error is fundamental** — perfect prediction is impossible
3. **Choose loss functions based on business constraints**, not just convention
4. **Heteroscedasticity matters** — adapt methods when variance changes with $x$
5. **Always quantify uncertainty** — point predictions alone are incomplete
6. **Start simple, validate empirically** — linear models often work well

