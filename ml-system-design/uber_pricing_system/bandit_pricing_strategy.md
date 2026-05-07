# Bandit & Reinforcement Learning Strategies for Dynamic Pricing

## Overview

Beyond supervised learning models that predict fixed prices, **bandit algorithms** enable iterative optimization through real-time experimentation. Instead of "predict the best price," think "continuously learn which price is best by testing variants."

**Key Question**: Should we use bandits for base price, surge price, or both?

---

## Why Bandits for Pricing?

### Problem with Pure Supervised Learning

```
Supervised Model (current approach):
  ✗ Trains on historical data
  ✗ Learns from past pricing decisions
  ✗ If past pricing was suboptimal, model inherits that bias
  ✗ Requires retraining to adapt to market changes
  ✗ No active exploration of new prices
```

### Bandit Approach

```
Bandit (exploration + exploitation):
  ✓ Continuously tests different prices
  ✓ Learns which prices maximize revenue/acceptance in real-time
  ✓ Adapts to market changes immediately
  ✓ No retraining required
  ✓ Explores price space to find optima
```

### Concrete Example

```
Supervised Learning:
  Historical data shows: demand_level=3.0 → price=$15
  Model always predicts $15
  But maybe $16 would actually generate more revenue!
  Only discovers this when you manually retrain

Bandit:
  Tries $15, $16, $17 in different requests
  Observes revenue from each
  Automatically shifts toward highest-revenue price
  Discovers optimal price in real-time
```

---

## Strategy Comparison

### 1. Epsilon-Greedy Bandit

```
Simple exploration-exploitation trade-off

Algorithm:
  With probability ε: explore (random price)
  With probability (1-ε): exploit (best price so far)

Example with ε=0.1 (10% exploration):
  90% of rides: charge best-performing price
  10% of rides: charge random price to test market

Python:

import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, price_candidates, epsilon=0.1):
        self.prices = price_candidates  # e.g., [10, 12, 14, 16, 18]
        self.epsilon = epsilon
        self.counts = np.zeros(len(price_candidates))  # trials per price
        self.revenues = np.zeros(len(price_candidates))  # total revenue per price

    def select_price(self):
        """Choose price: explore or exploit"""
        if np.random.random() < self.epsilon:
            # Explore: random price
            return self.prices[np.random.randint(len(self.prices))]
        else:
            # Exploit: best price so far
            avg_revenues = self.revenues / (self.counts + 1e-5)
            best_idx = np.argmax(avg_revenues)
            return self.prices[best_idx]

    def update(self, price, revenue):
        """Update based on observed revenue"""
        idx = self.prices.index(price)
        self.counts[idx] += 1
        self.revenues[idx] += revenue
```

**Pros:**
- ✓ Simple to implement
- ✓ Fast convergence initially
- ✓ No assumptions about price distribution

**Cons:**
- ✗ Fixed exploration rate (doesn't adapt)
- ✗ May test bad prices too often
- ✗ Doesn't use uncertainty to guide exploration

**When to use:** Rapid experimentation, simple pricing problems

---

### 2. Upper Confidence Bound (UCB)

```
Smarter exploration: test prices with high potential AND high uncertainty

Algorithm:
  For each price i:
    upper_bound_i = avg_revenue_i + C × sqrt(ln(t) / count_i)
                                     ↑
                              uncertainty term (higher if untested)

  Select price with highest upper bound

Intuition:
  - Tested prices: upper_bound ≈ avg_revenue (well-known)
  - Untested prices: upper_bound >> avg_revenue (could be great!)
  - Forces balanced exploration of all prices

Example:
  Price $10: tested 100 times, avg_revenue=$100
    upper_bound = 100 + 2×sqrt(ln(1000)/100) = 100.67

  Price $12: tested 10 times, avg_revenue=$110
    upper_bound = 110 + 2×sqrt(ln(1000)/10) = 110 + 2×0.68 = 111.36

  Price $14: tested 1 time, avg_revenue=$120
    upper_bound = 120 + 2×sqrt(ln(1000)/1) = 120 + 6.32 = 126.32  ← Choose this!

Python:

class UCBBandit:
    def __init__(self, price_candidates, c=2.0):
        self.prices = price_candidates
        self.c = c
        self.counts = np.zeros(len(price_candidates))
        self.revenues = np.zeros(len(price_candidates))
        self.t = 0  # total trials

    def select_price(self):
        """UCB formula"""
        self.t += 1
        avg_revenues = self.revenues / (self.counts + 1e-5)
        uncertainties = np.sqrt(np.log(self.t) / (self.counts + 1e-5))
        upper_bounds = avg_revenues + self.c * uncertainties

        best_idx = np.argmax(upper_bounds)
        return self.prices[best_idx]

    def update(self, price, revenue):
        idx = self.prices.index(price)
        self.counts[idx] += 1
        self.revenues[idx] += revenue
```

**Pros:**
- ✓ Principled exploration (based on uncertainty)
- ✓ Balances exploration and exploitation
- ✓ Provably optimal convergence

**Cons:**
- ✗ Assumes rewards follow specific distribution
- ✗ Doesn't account for price correlations (if $12 good, likely $11 also good)
- ✗ High variance in early exploration

**When to use:** Discrete price options, moderate exploration budget

---

### 3. Thompson Sampling

#### Understanding the Beta Distribution

The Beta distribution is the core of Thompson sampling. Here's the intuition:

**What is Beta(α, β)?**

```
β-distribution is a probability distribution parameterized by two numbers:
  α (alpha) = number of successes (e.g., rides accepted at this price)
  β (beta) = number of failures (e.g., rides rejected at this price)

Mean of distribution = α / (α + β)
  Example: Beta(80, 20) has mean = 80/100 = 0.80 (80% acceptance rate)
```

**Visual Intuition - How Beta Shapes Change:**

```
Low Data (high uncertainty):
  Beta(1, 1) → completely flat, uniform [uniform trust in all values]
  Beta(2, 1) → skewed right, "maybe this price is amazing" [1 success, 1 failure]
  Beta(1, 2) → skewed left, "maybe this price is terrible" [1 success, 1 failure]

      Beta(1,1)              Beta(2,1)              Beta(1,2)
    Wide & flat      →      Skewed right    →      Skewed left
    [Lots of uncertainty]    [Uncertain but optimistic]  [Uncertain but pessimistic]

Medium Data (moderate uncertainty):
  Beta(10, 10) → peaked at 0.5, moderate width
  Beta(15, 5) → peaked at 0.75, moderate width

        Beta(10,10)          Beta(15,5)
       ╱╲   peak at 0.5    peak at 0.75
      ╱  ╲  moderate width

High Data (low uncertainty):
  Beta(100, 20) → narrow peak at 0.83, very concentrated
  Beta(80, 80) → narrow peak at 0.50, very concentrated
  Beta(10, 100) → narrow peak at 0.09, very concentrated

      Beta(100,20)      Beta(80,80)       Beta(10,100)
      sharp peak    →   sharp peak   →    sharp peak
      near 0.83         near 0.50         near 0.09
    [High confidence]  [High confidence]  [High confidence]
```

**Why Beta Works for Pricing:**

1. **Untested prices have wide distributions**
   - Beta(1, 1) for new price: high variance
   - When you sample, you might get 0.2, or 0.8, or 0.5
   - This creates natural exploration!

2. **Well-tested prices have narrow distributions**
   - Beta(100, 20): tight peak at 0.83
   - Samples cluster around 0.83
   - Less exploration, more exploitation

3. **Automatic uncertainty quantification**
   - No need to manually set exploration rates
   - Shape automatically adapts as data grows

**Concrete Example: 3 Prices**

```
Price $10: 80 accepts, 20 rejects → Beta(81, 21)
  Mean: 81/102 = 0.794 (79.4% acceptance rate)
  Shape: narrow, concentrated
  Sample 1000 times: mostly 0.75-0.85 range

Price $12: 15 accepts, 5 rejects → Beta(16, 6)
  Mean: 16/22 = 0.727 (72.7% acceptance rate)
  Shape: wider than $10 (less data)
  Sample 1000 times: spread from 0.4 to 0.95 (high variance)

Price $14: 2 accepts, 2 rejects → Beta(3, 3)
  Mean: 3/6 = 0.5 (50% acceptance rate)
  Shape: very wide (almost no data)
  Sample 1000 times: completely scattered 0.0 to 1.0

Thompson Sampling in action:

  Iteration 1:
    Sample $10: θ₁₀ ~ Beta(81, 21) → 0.81
    Sample $12: θ₁₂ ~ Beta(16, 6) → 0.68  (happened to be lower due to variance)
    Sample $14: θ₁₄ ~ Beta(3, 3) → 0.92   (happened to be high due to high variance!)

    Choose: $14 (even though has least data, uncertainty was lucky)

  Iteration 2:
    User rejects at $14
    Update: Beta(3, 4) for $14

    Sample $10: θ₁₀ ~ Beta(81, 21) → 0.79
    Sample $12: θ₁₂ ~ Beta(16, 6) → 0.71
    Sample $14: θ₁₄ ~ Beta(3, 4) → 0.31  (now lower due to rejection)

    Choose: $10

  Iteration 3:
    After many samples, Beta(81, 21) stays as the winner
    $14 Beta(3, 4) gets more data → distribution narrows
    Eventually $14 settles to true value, stops exploring as much
```

**Key Insight: The Magic**

The Beta distribution naturally implements exploration-exploitation:
- High variance (untested) → sometimes sampled high → explore
- Low variance (well-tested) → consistently sampled at true value → exploit
- No hard-coded exploration rate needed!

Bayesian approach: maintain belief about price quality, sample from posterior

Algorithm:
  For each price i:
    Maintain Beta distribution: Beta(α, β) where:
      α = successes (rides accepted)
      β = failures (rides rejected)

    Sample from posterior: θ_i ~ Beta(α, β)
    Select price with highest sampled θ

Intuition:
  - Untested prices: wide Beta distribution → high variance → sometimes sampled high → explores
  - Tested prices: narrow Beta distribution → low variance → concentrates → exploits
  - Uncertainty automatically decreases as you gather data

```

**Python Implementation:**

```python
class ThompsonSamplingBandit:
    def __init__(self, price_candidates):
        self.prices = price_candidates
        self.successes = np.zeros(len(price_candidates))
        self.failures = np.zeros(len(price_candidates))

    def select_price(self):
        """Thompson sampling: sample from posterior, pick best"""
        samples = []
        for i in range(len(self.prices)):
            # Sample from Beta posterior
            sample = np.random.beta(
                self.successes[i] + 1,  # α (successes)
                self.failures[i] + 1    # β (failures)
            )
            samples.append(sample)

        best_idx = np.argmax(samples)
        return self.prices[best_idx]

    def update(self, price, success):
        """Update based on binary outcome"""
        idx = self.prices.index(price)
        if success:
            self.successes[idx] += 1
        else:
            self.failures[idx] += 1
```

#### Real-World Example: 3 Prices Over Time

```
Initial state (no data):
  Price $10: Beta(1, 1)  → flat, completely uncertain
  Price $12: Beta(1, 1)  → flat, completely uncertain
  Price $14: Beta(1, 1)  → flat, completely uncertain

After 100 rides:
  Price $10: 80 accepts, 20 rejects → Beta(81, 21)
    Mean: 79% acceptance
    Shape: NARROW peak ← High confidence
    Samples 1000x: mostly 73-85%

  Price $12: 60 accepts, 40 rejects → Beta(61, 41)
    Mean: 60% acceptance
    Shape: MEDIUM peak ← Medium confidence
    Samples 1000x: spread 50-70%

  Price $14: 10 accepts, 90 rejects → Beta(11, 91)
    Mean: 11% acceptance
    Shape: NARROW, LEFT-SKEWED ← High confidence it's bad
    Samples 1000x: mostly 3-18%

Thompson sampling decision (pick best sampled θ):
  Sample iteration 1:
    $10: sample = 0.78  (narrow range)
    $12: sample = 0.55  (wider range)
    $14: sample = 0.08  (narrow, near true value)
    → Choose $10 ✓

  Sample iteration 2:
    $10: sample = 0.81  (narrow)
    $12: sample = 0.68  (wider - chance to explore!)
    $14: sample = 0.05  (narrow, knows it's bad)
    → Choose $10 ✓

  Sample iteration 3:
    $10: sample = 0.76  (narrow)
    $12: sample = 0.45  (wide variance - low sample this time)
    $14: sample = 0.12  (narrow)
    → Choose $10 ✓

Key observation:
  - $10 almost always wins because of narrow, high peak
  - $12 occasionally wins when high variance produces lucky sample
  - $14 almost never wins because known to be bad
  - But all prices still get tested, proportional to uncertainty

Result: Optimal prices explored more, bad prices explored less
        Exploration happens naturally through variance
        No hand-tuned exploration rate needed!
```

**Pros:**
- ✓ Naturally models uncertainty
- ✓ Elegant Bayesian framework
- ✓ Automatically reduces exploration as confidence grows
- ✓ Handles outcomes naturally (accept/reject, revenue bins)

**Cons:**
- ✗ Requires framing outcome as success/failure
- ✗ More complex to understand
- ✗ Hyperparameter tuning (prior strength)

**When to use:** Binary outcomes (accept/reject), Bayesian teams, continuous learning

---

### 4. Contextual Bandits (Most Powerful)

```
Extend bandits to consider context (like supervised learning)

Algorithm:
  For each price i, context x:
    Maintain: model that predicts P(accept | price=i, context=x)
    Choose price with highest predicted acceptance
    (with exploration bonus)

Example:
  Context: [hour=20, requests_per_driver=3.0, weather=rainy]
  Candidate prices: [$10, $12, $14, $16]

  Model predicts acceptance rates:
    $10 → 95% accept, predicted_revenue = $9.50
    $12 → 85% accept, predicted_revenue = $10.20
    $14 → 70% accept, predicted_revenue = $9.80
    $16 → 50% accept, predicted_revenue = $8.00

  Choose $12 (highest predicted revenue)
  + explore other prices with small probability

Python (simplified):

class ContextualBandit:
    def __init__(self, price_candidates):
        self.prices = price_candidates
        self.model = self._init_model()  # LogisticRegression or similar

    def select_price(self, context, epsilon=0.1):
        """Choose price based on context"""
        if np.random.random() < epsilon:
            # Explore
            return np.random.choice(self.prices)

        # Exploit: predict acceptance for each price
        predictions = []
        for price in self.prices:
            features = np.concatenate([context, [price]])
            pred_acceptance = self.model.predict_proba(features)[0, 1]
            pred_revenue = price * pred_acceptance
            predictions.append(pred_revenue)

        best_idx = np.argmax(predictions)
        return self.prices[best_idx]

    def update(self, context, price, accepted):
        """Learn from outcome"""
        features = np.concatenate([context, [price]])
        self.model.fit(features, accepted)  # Simplified
```

**Pros:**
- ✓ Uses context (demand, weather, time) like supervised models
- ✓ Learns price-context interactions
- ✓ Optimal for dynamic pricing
- ✓ Reduces exploration variance

**Cons:**
- ✗ More complex to implement
- ✗ Requires feature engineering
- ✗ Model drift can be problematic

**When to use:** Dynamic pricing with many contextual variables (base price, surge)

---

## Base Price vs Surge Price: Which Gets Bandits?

### Base Price Analysis

```
Characteristics:
  • Prediction target: cost estimate for trip
  • Stability: relatively stable (context-dependent)
  • Data volume: high (every trip)
  • Update frequency: can be slower (weekly retraining)
  • Exploration tolerance: lower (users get surprised by big price jumps)
  • Demand elasticity: medium (affects demand moderately)

Recommendation: Contextual Bandit ⭐⭐ (moderate priority)

Reasoning:
  ✓ Enough data to support contextual learning
  ✓ Can do light exploration (1-5% of trips)
  ✓ Reduces model retraining frequency
  ✓ Learns distance/time interactions in real-time
  ✗ Users expect somewhat stable pricing
  ✗ Exploration can reduce acceptance (higher price surprises users)
```

### Surge Price Analysis

```
Characteristics:
  • Prediction target: multiplier on base price
  • Stability: highly dynamic (changes minute-to-minute)
  • Data volume: medium (aggregated by zone-time)
  • Update frequency: high (real-time needed)
  • Exploration tolerance: higher (accepted as market response)
  • Demand elasticity: high (strongly affects supply/demand balance)

Recommendation: Contextual Bandit ⭐⭐⭐⭐⭐ (high priority)

Reasoning:
  ✓ Natural fit for RL (explore prices, observe driver/user response)
  ✓ Dynamic pricing expected (users accept surge variation)
  ✓ Real-time learning needed (demand changes rapidly)
  ✓ Zone/time context critical (demand patterns shift)
  ✓ Direct feedback loop (surge → acceptance → revenue)
  ✗ Fewer zone-time samples per arm
  ✗ Non-stationary environment (demand distributions shift)
```

---

## Recommended Approach: Contextual Thompson Sampling for Surge

### Why Thompson Sampling for Surge?

```
Thompson Sampling + Contextual Bandits + Surge Pricing:

Advantages:
  ✓ Natural uncertainty handling (key for exploration)
  ✓ Bayesian framework matches pricing philosophy
  ✓ Elegant handling of rare demand scenarios (high posterior variance)
  ✓ Automatic exploration decay (as confidence increases)
  ✓ Works with limited data per arm (zone-surge combinations)
```

### Architecture

```
                    User requests ride
                           ↓
                   [1. Collect Context]
                   (demand, supply, weather, time)
                           ↓
            [2. Generate Candidate Surge Prices]
            (1.0x, 1.2x, 1.4x, 1.6x, 1.8x, 2.0x)
                           ↓
        [3. Thompson Sampling (per candidate)]
        - Maintain posterior: Beta(accepts, rejects) per surge level
        - Sample from each posterior
        - Pick surge with highest sampled acceptance prob
                           ↓
            [4. Offer Price to User]
                           ↓
        [5. Observe Outcome & Update]
        - Did user accept? (yes/no)
        - Update Beta posterior for that surge level
                           ↓
           [Track Revenue Across Surges]
        - Also monitor actual revenue achieved
        - Optional: switch objective from acceptance to revenue
```

### Implementation

```python
import numpy as np
from collections import defaultdict

class SurgeContextualThompsonSampling:
    def __init__(self, surge_candidates):
        """
        surge_candidates: e.g., [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        """
        self.surges = surge_candidates

        # Per (context_hash, surge) maintain: Beta(accepts, rejects)
        # Context hashed to reduce state space
        self.successes = defaultdict(lambda: np.zeros(len(surge_candidates)))
        self.failures = defaultdict(lambda: np.zeros(len(surge_candidates)))

    def _hash_context(self, context):
        """Convert context to discrete state"""
        zone = context['zone_id']
        hour = context['hour_of_day']
        demand_level = int(context['requests_per_driver'] * 10)  # Quantize

        return f"{zone}_{hour}_{demand_level}"

    def select_surge(self, context):
        """Thompson sampling: sample from posteriors, pick best"""
        state = self._hash_context(context)

        # Sample from Beta posterior for each surge
        samples = []
        for i in range(len(self.surges)):
            alpha = self.successes[state][i] + 1
            beta = self.failures[state][i] + 1

            # Sample probability of acceptance
            sample = np.random.beta(alpha, beta)
            samples.append(sample)

        # Pick surge with highest sampled acceptance
        best_idx = np.argmax(samples)
        return self.surges[best_idx]

    def update(self, context, surge, accepted):
        """Update posterior based on outcome"""
        state = self._hash_context(context)
        idx = self.surges.index(surge)

        if accepted:
            self.successes[state][idx] += 1
        else:
            self.failures[state][idx] += 1

    def get_stats(self, context):
        """For debugging/monitoring"""
        state = self._hash_context(context)
        stats = []

        for i, surge in enumerate(self.surges):
            success = self.successes[state][i]
            failure = self.failures[state][i]
            acceptance_rate = success / (success + failure + 1e-5)

            stats.append({
                'surge': surge,
                'accepts': success,
                'rejects': failure,
                'acceptance_rate': acceptance_rate
            })

        return stats


# Usage in production:

bandit = SurgeContextualThompsonSampling([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

# For each ride request:
context = {
    'zone_id': 'downtown_1',
    'hour_of_day': 20,
    'requests_per_driver': 3.2,
    'weather': 'rainy'
}

# Select surge via Thompson sampling
surge_multiplier = bandit.select_surge(context)

# Offer ride to user at this price...
# Later, observe if they accepted
accepted = True  # Example

# Update bandit
bandit.update(context, surge_multiplier, accepted)

# Monitor progress
print(bandit.get_stats(context))
```

### Revenue-Optimized Variant

```python
# Instead of binary acceptance, track revenue directly

class SurgeContextualBanditRevenueOptimized:
    def __init__(self, surge_candidates):
        self.surges = surge_candidates
        # Track revenue + confidence per surge
        self.revenue_sum = defaultdict(lambda: np.zeros(len(surge_candidates)))
        self.revenue_count = defaultdict(lambda: np.zeros(len(surge_candidates)))

    def select_surge(self, context):
        """UCB-style selection based on revenue"""
        state = self._hash_context(context)

        avg_revenue = self.revenue_sum[state] / (self.revenue_count[state] + 1e-5)
        uncertainties = np.sqrt(1 / (self.revenue_count[state] + 1e-5))

        # Upper confidence bound on revenue
        ucb_bounds = avg_revenue + 2.0 * uncertainties

        best_idx = np.argmax(ucb_bounds)
        return self.surges[best_idx]

    def update(self, context, surge, revenue):
        """Update based on actual revenue"""
        state = self._hash_context(context)
        idx = self.surges.index(surge)

        self.revenue_sum[state][idx] += revenue
        self.revenue_count[state][idx] += 1
```

---

## Practical Deployment Considerations

### 1. Exploration Rate

```
Too high (10% exploration):
  ✗ Too many price experiments
  ✗ Revenue loss from bad prices
  ✗ User confusion (prices changing frequently)

Too low (0.5% exploration):
  ✗ Slow to adapt to market changes
  ✗ Stuck in local optima
  ✗ Doesn't discover better prices

Recommendation:
  Start: 5% exploration
  Adjust based on:
    - Revenue impact
    - Price consistency feedback
    - Demand volatility (surge: higher exploration OK)
```

### 2. Contextual State Space

```
Problem: Too many context combinations → sparse data

Solution: Aggregate contexts smartly

Bad grouping (too fine):
  State = (zone_id, hour, minute, weather, day_of_week)
  → 50 zones × 24 hours × 60 minutes × 4 weather × 7 days = 2M states
  → Each state has ~1 sample per day (too sparse)

Good grouping (balanced):
  State = (zone_id, hour_of_day, demand_bin, is_rush_hour)
  → 50 zones × 24 hours × 4 demand bins × 2 = 9,600 states
  → Each state has ~10-100 samples per day (reasonable)

For Surge Pricing:
  requests_per_driver bucket: [0-0.5, 0.5-1.5, 1.5-3.0, 3.0+]
  hour bucket: 24 bins (hourly)
  zone: zone_id
  Total: ~400-500 states
```

### 3. Cold Start Problem

```
Challenge: New zone/hour combination has no data

Solution: Use pretrained prior
  Instead of: Beta(1, 1) [uniform]
  Use: Beta(α_prior, β_prior) from historical data

Calculation:
  Global average acceptance for price $14: 75%
  Use Beta(75, 25) as starting point

  New zone/hour observes:
    First user accepts: Beta(76, 25) - gradually overwrites prior
    After 100 observations: prior's influence negligible
```

### 4. Non-Stationarity Handling

```
Problem: Demand patterns change over time
  Morning rush: accepts at 2.0x
  But then surge subsides, acceptance drops

Solution: Decay old observations
  Give more weight to recent observations

Implementation:
  self.successes[state][i] *= decay_factor  # e.g., 0.99 per hour
  self.failures[state][i] *= decay_factor

Or: Use separate bandits for different time windows
  Morning bandit (6am-10am)
  Evening bandit (4pm-8pm)
  Off-peak bandit (everything else)
```

---

## Comparison: Bandits vs Supervised Learning

| Aspect | Supervised Model | Contextual Bandit |
|--------|------------------|-------------------|
| **Learning** | Batch retraining | Real-time updates |
| **Exploration** | None (always same prediction) | Active exploration |
| **Adaptation Speed** | Slow (weekly/daily) | Fast (immediate) |
| **Data Efficiency** | High (uses all data) | Lower (some wasted on exploration) |
| **Optimal Price Discovery** | Only from historical data | Active search |
| **Variance** | Low (averaging) | Higher (uncertainty) |
| **Best For** | Prediction when optimal is known | Finding optimal through exploration |
| **Complexity** | Medium | High |
| **Interpretability** | High (feature importance) | Lower (probabilistic) |

---

## Hybrid Approach (Recommended for Production)

```
Combine Supervised Learning + Bandits:

Day 1-7: Supervised Model
  - Train XGBoost on historical surge data
  - Make predictions
  - Use as initialization for bandit

Week 2+: Contextual Bandit
  - Initialize Thompson Sampling priors from XGBoost predictions
  - Real-time exploration + exploitation
  - Gradually optimize prices
  - Every week: retrain XGBoost on latest data as sanity check

Benefits:
  ✓ Warm start (no cold-start problem)
  ✓ Fast convergence (seeds with good estimates)
  ✓ Continuous optimization
  ✓ Safeguard (model can detect major drift)
```

### Implementation

```python
class HybridSurgeOptimizer:
    def __init__(self, xgboost_model, surge_candidates):
        self.model = xgboost_model
        self.surges = surge_candidates
        self.bandit = SurgeContextualThompsonSampling(surge_candidates)

        # Initialize bandit priors from model predictions
        self._warm_start_bandit()

    def _warm_start_bandit(self):
        """Use supervised model to initialize bandit"""
        # Generate synthetic contexts
        contexts = self._generate_synthetic_contexts()

        for context in contexts:
            state = self.bandit._hash_context(context)
            features = self._context_to_features(context)

            # Get model prediction + confidence
            pred_surge = self.model.predict([features])[0]

            for i, surge in enumerate(self.surges):
                # Peak posterior at predicted surge
                if surge == pred_surge:
                    self.bandit.successes[state][i] = 50
                    self.bandit.failures[state][i] = 10
                else:
                    # Smaller prior for other prices
                    self.bandit.successes[state][i] = 10
                    self.bandit.failures[state][i] = 30

    def select_surge(self, context):
        """Use bandit in production"""
        return self.bandit.select_surge(context)

    def update(self, context, surge, accepted):
        """Real-time learning"""
        self.bandit.update(context, surge, accepted)

    def retrain_model(self, recent_data):
        """Weekly: retrain supervised model as sanity check"""
        self.model.fit(recent_data.X, recent_data.y)
```

---

## Monitoring & Metrics

```
Key Metrics to Track:

1. Regret
   Formula: sum(optimal_revenue[t] - actual_revenue[t])
   What it means: How much revenue lost due to exploration?
   Target: Decreasing over time

2. Exploration Rate
   How often do we pick non-optimal surge?
   Track: % of rides with each surge level
   Should see: Concentration on optimal surge, rare exploration

3. Acceptance Rate by Surge
   Track separately for each surge level
   Should see: Clear correlation (higher surge → lower accept)
   Alert if: No correlation (bandit learning failing)

4. Revenue per Surge
   Track: revenue = surge × acceptance_rate × num_rides
   Should see: Convergence to highest-revenue surge
   Alert if: No convergence (market too volatile)

5. Consistency
   Track: How often does bandit change "best" surge?
   Should see: Stable decisions after warm-up period
   Alert if: Wild oscillations (indicate non-stationarity)
```

---

## Summary & Recommendations

### For Base Price
```
✓ Hybrid approach: Supervised model (primary) + Light bandit (exploration)
✓ 1-2% exploration rate
✓ Weekly model retraining
✓ Context: distance, time, demand
✓ Objective: revenue (not just acceptance)
```

### For Surge Price ⭐
```
✓ Contextual Thompson Sampling (primary)
✓ Initialize from supervised model
✓ 5-10% exploration rate (higher OK, users expect variation)
✓ Real-time updates
✓ Context: zone, hour, demand_level, weather
✓ Objective: revenue (balancing acceptance vs multiplier)
```

### Implementation Phases

```
Phase 1 (Week 1-2):
  - Build supervised base/surge models
  - Deploy to production
  - Collect baseline metrics

Phase 2 (Week 3-4):
  - Implement Contextual Thompson Sampling
  - Start with 2-3% exploration
  - Monitor regret and acceptance rate
  - A/B test vs supervised model

Phase 3 (Month 2+):
  - Increase exploration to 5-10% if safe
  - Add state decay for non-stationarity
  - Hybrid approach with weekly model retraining
  - Continuous monitoring and tuning
```
