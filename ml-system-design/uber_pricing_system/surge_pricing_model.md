# Surge Pricing Model - Uber Pricing System

## Overview

The **Surge Pricing Model** predicts the demand-based multiplier applied to base fares during peak demand periods. This model captures real-time market dynamics and generates revenue optimization signals.

**Purpose**: Calculate `surge_multiplier` which is multiplied by `base_price`:
```
final_price = base_price × surge_multiplier (1.0x - 5.0x+)
```

**Key Insight**: Surge pricing serves dual purposes:
1. **Revenue Optimization**: Maximize platform revenue during high demand
2. **Supply Incentive**: Higher prices attract more drivers to supply-constrained areas

---

## Model Objectives & Success Metrics

### Primary Objectives
1. **Revenue Maximization**: Optimize pricing to maximize total platform revenue
2. **Supply Response**: Higher surge should attract more drivers to constrained zones
3. **Market Equilibrium**: Balance demand and supply efficiently

### Secondary Objectives
1. Maintain customer satisfaction (avoid price shock)
2. Ensure fair pricing (similar demand → similar surge)
3. Smooth price transitions (avoid wild multiplier swings)

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **MAPE (Accuracy)** | <10% | Average % error in surge prediction |
| **MAE (Multiplier)** | <0.15x | Mean absolute error in multiplier (e.g., predict 1.8x when actual 1.9x) |
| **Correlation with Demand** | >0.70 | Strong positive correlation between predicted and actual surge |
| **Directional Accuracy** | >75% | Predict "up" or "down" correctly >75% of time |
| **Revenue Lift** | +5-15% | Revenue improvement vs. rule-based baseline |
| **Price Fairness** | Coef. of Var <15% | Similar demand → similar surge across zones |
| **Driver Response Correlation** | >0.6 | Higher surge predicts higher driver supply acceptance |

---

## Feature Engineering

### Features for Surge Pricing Model

#### Real-Time Demand-Supply Metrics (Zone-Level)

| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **active_requests_in_zone** | Numerical | Number of pending requests | Direct demand signal |
| **request_rate** | Numerical | Requests/minute in zone | Demand velocity (accelerating?) |
| **requests_5min_ago** | Numerical | Requests 5 min prior | Trend detection |
| **available_drivers** | Numerical | Online drivers in zone | Direct supply signal |
| **driver_utilization_rate** | Numerical | % drivers with active ride | Supply tightness |
| **requests_per_driver** | Numerical | active_requests / available_drivers | **Key metric:** supply-demand ratio |
| **avg_pickup_wait_time** | Numerical | Expected user wait (minutes) | User friction (high wait → price sensitivity) |
| **driver_acceptance_rate** | Numerical | % drivers accepting requests | Supply responsiveness |

**Most Important**: `requests_per_driver` is the strongest surge signal.

#### Temporal Features

| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **hour_of_day** | Categorical | Hour (0-23) | Predictable demand patterns (7am rush hour) |
| **day_of_week** | Categorical | Day (0-6, Mon-Sun) | Weekday vs. weekend behavior |
| **is_peak_hour** | Binary | 7-9am, 5-7pm | Rush hour indicator |
| **minutes_since_hour_start** | Numerical | 0-59 | Micro-temporal patterns within hour |
| **is_weekend** | Binary | Sat/Sun indicator | Different demand patterns |
| **is_holiday** | Binary | Holiday period | Special demand patterns (New Year's) |
| **days_until_paycheck** | Numerical | Payday effects (25-27th) | Cyclical demand variations |

#### Event & External Features

| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **nearby_events** | Categorical | Concert, game, festival, airport surge | Major demand spike indicators |
| **weather_condition** | Categorical | Clear, Rainy, Snowy, Foggy | Bad weather increases ride demand |
| **temperature** | Numerical | Degrees F/C | Extreme temps (very hot/cold) increase demand |
| **visibility** | Numerical | Miles (weather) | Poor visibility increases ride preference |
| **precipitation** | Numerical | Inches/mm (weather) | Rain increases demand significantly |

#### Historical Patterns & Baselines

| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **avg_surge_at_this_time** | Numerical | Historical avg surge for this zone-hour | Seasonal baseline |
| **surge_percentile_at_this_time** | Numerical | p90 surge historically | Baseline for volatility |
| **surge_volatility** | Numerical | Std dev of surge at this hour | How predictable is this time period? |
| **day_of_month** | Numerical | 1-31 | Payday effects, month-end patterns |
| **week_number** | Numerical | 1-52 | Seasonal cycles |

#### Contextual Metrics

| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **competitor_average_price** | Numerical | Competitors' prices | Market competitiveness affects demand |
| **zone_accessibility_score** | Numerical | How easy for drivers to reach zone | Affects supply positioning |
| **platform_incentive_active** | Binary | Running driver promotion | Artificial supply boost |
| **recent_supply_surge** | Binary | Did we recently offer bonuses? | Past actions affecting current supply |

### Derived/Interaction Features

Create these to capture complex patterns:

```python
# Core demand-supply dynamics
requests_per_driver = active_requests / (available_drivers + 1)

# Demand intensity relative to baseline
historical_avg_demand = avg_demand[zone][hour]
demand_anomaly = (active_requests - historical_avg_demand) / std_demand[zone][hour]
demand_momentum = request_rate - request_rate_5min_ago

# Supply tightness
supply_score = 1 / (1 + requests_per_driver)  # 0-1 normalized
supply_shortage_severity = clamp(requests_per_driver / 3.0, 0, 1)

# Time context combinations
is_rush_hour_weekday = (is_rush_hour and not is_weekend)
is_late_night_weekend = (hour >= 22 and is_weekend)

# Weather interaction with time
is_rainy_rush_hour = (weather == 'rain' and is_rush_hour)
is_extreme_weather = (temperature < 10 or temperature > 95) and is_night

# Expected surge based on historical patterns
baseline_surge = historical_surge_distribution[zone][hour][day_of_week]
seasonal_surge_multiplier = baseline_surge / global_avg_surge

# Demand acceleration
demand_trend = (request_rate_now - request_rate_5min_ago) > 0
demand_trend_strength = abs(request_rate_now - request_rate_5min_ago) / request_rate_5min_ago
```

---

## Training Data Generation

### Data Collection & Preprocessing

**Source Data:**
```
Real-time zone-level metrics (collected every 30 seconds):
  - active_requests_in_zone
  - available_drivers
  - completed_rides in zone
  - driver_acceptance_rate
  - avg_price_charged (at that moment)

Historical context per zone-hour:
  - Previous days' demand patterns
  - Previous days' surge multipliers
  - Weather data (from archive)
  - Event calendar

Labels:
  - Actual surge multiplier used
  - Actual price paid by users
  - Revenue generated
  - User acceptance rate (did they accept or cancel?)
```

**Data Collection Granularity:**
```
Temporal: Every 30 seconds per zone
  - Ensures fine-grained demand tracking
  - Captures demand volatility
  - Sufficient for learning patterns

Spatial: Zone-level aggregation
  - Zones: 2-5 mile radius
  - Each city has 50-500 zones
  - Enables local surge pricing

Aggregation: Never use individual user data
  - Privacy-preserving
  - More stable signals
  - Clearer demand-supply dynamics
```

**Data Quality Checks:**
```
✓ Remove periods with API/system errors
✓ Validate metrics consistency:
  - requests_per_driver >= 0
  - available_drivers > 0
  - request_rate > 0
✓ Remove anomalies:
  - Sudden spikes in requests (data glitch?)
  - Impossible metrics (drivers > population)
✓ Handle timezone conversions
✓ Align timestamps across data sources
✓ Remove duplicate records
```

### Label Generation Strategy

The **label** is the surge multiplier that should have been charged.

#### Option 1: Use Actual Historical Surge (Simple)
```
For each zone at time t:
  actual_surge = mean(prices_at_t) / baseline_price

  Input:  [active_requests, available_drivers, hour, weather, ...]
  Output: actual_surge_multiplier
```

**Considerations:**
- Actual surge reflects what the company charged
- But company pricing may not be optimal
- May be suboptimal for learning true demand-surge relationship

#### Option 2: Infer Optimal Surge from Driver Response (Better)
```
Analyze what surge level successfully:
  - Attracts enough drivers to accept requests
  - Maximizes revenue
  - Maintains acceptable cancellation rate

For each demand level:
  - What surge multiplier > 95% acceptance rate?
  - What surge multiplier maximizes revenue?
  - Use that as label
```

**Advantages:**
- Focuses on optimal pricing
- Filters out suboptimal historical decisions
- Model learns revenue-maximizing behavior

#### Option 3: Use Demand Ratio (Most Robust)
```
Pure demand-supply ratio as proxy for surge:

For each zone at time t:
  surge = f(requests_per_driver, hour, weather, ...)

Where f() is empirically derived from:
  - Historical user acceptance curves
  - Driver supply response curves
  - Revenue optimization analysis
```

**Recommendation**: Use **Option 1** for initial implementation (realistic), then **Option 2** for optimization (learning optimal pricing).

### Using Cancellations & Acceptance Rates in Training

**Key Question**: Should we only train on accepted rides? How do cancellations inform surge pricing?

#### Understanding the Selection Bias Problem

**Naive Approach (WRONG):**
```
Only use accepted rides for training:
  Input:  [active_requests, available_drivers, ...]
  Output: surge_multiplier (from accepted rides only)

Problem: Selection Bias
  - If surge is very high (3.0x), many users cancel → fewer "accepted" rides
  - Model never sees that high surge caused cancellations
  - When surge is high and user cancels, we lose that data point
  - Model becomes biased toward lower surge values
  - Underpredicts surge in high-demand scenarios
```

**Example scenario:**
```
Zone A at 8pm:
  - 200 active requests, 30 drivers
  - Company charged 2.5x surge
  - 50% users cancelled (too expensive)
  - 100 users accepted

If we only use the 100 accepted rides:
  ✗ We lose signal that 2.5x caused 50% cancellations
  ✗ Model thinks 2.5x is a good surge for this demand level
  ✗ Later when demand similar, model predicts 2.5x
  ✗ We get high cancellation rate again

Correct approach:
  ✓ Use ALL 200 requests (accepted + cancelled)
  ✓ Label all with same surge (2.5x)
  ✓ Model learns: 2.5x caused cancellations
  ✓ Can use cancellation rate as secondary signal
```

#### How to Handle Rides in Training

**1. Include ALL Rides (Accepted & Cancelled) for Label Generation:**

```python
def generate_surge_labels(zone_data_at_time_t):
    """
    Generate label using ALL rides, not just accepted ones
    """

    # zone_data_at_time_t contains:
    # - All active requests (some accepted, some cancelled)
    # - Actual price charged
    # - Base price estimate
    # - Cancellation rate

    # Label is the same for all rides in this zone-time:
    # It reflects what surge multiplier the company actually charged

    actual_prices = zone_data['price_charged']
    base_prices = zone_data['estimated_base_price']

    # Calculate surge from mean price
    surge_multiplier = mean(actual_prices) / mean(base_prices)

    # This surge applies to ALL requests (accepted and cancelled)
    # Cancellations inform us that this surge may have been too high

    return surge_multiplier
```

**Why this works:**
- Non-cancelled users → generate revenue at that surge price
- Cancelled users → indicate price sensitivity / surge too high
- Together they tell complete story: "this surge caused X% cancellation"
- Model learns optimal surge (balancing revenue vs. acceptability)

**2. Use Cancellation Rate as a Feature:**

```python
# Features for surge model include:
features = {
    'active_requests': 200,
    'available_drivers': 30,
    'requests_per_driver': 6.67,
    'hour': 20,
    'weather': 'clear',
    ...
    'recent_cancellation_rate': 0.35,  # ← Add this!
}

# If surge was too high → high cancellation_rate
# Model learns this feedback signal
```

**How to calculate cancellation_rate feature:**

```python
# For each zone-time window (e.g., last 10 minutes):
requests_made = 200
requests_cancelled = 70
requests_accepted = 130

cancellation_rate = requests_cancelled / requests_made  # 0.35 (35%)

# This becomes a feature in the next prediction cycle
# High cancellation_rate → indicate previous surge was too aggressive
```

#### Three Approaches to Using Cancellations

**Approach 1: Simple - Include cancellation_rate as Feature**

```
Pros:
  ✓ Simple to implement
  ✓ Model learns correlation: high surge → high cancellation
  ✓ Feedback loop: surge causes cancellation → cancellation informs next surge
  ✓ Works with existing label generation

Cons:
  ✗ Indirect signal (cancellation in t-10min informs surge at t)
  ✗ Lag between cause (surge) and feedback (cancellation rate)

Implementation:
  features['recent_cancellation_rate'] = moving_avg_cancellation_rate[-10min]
  # When training, include this feature
  # Model learns: high cancellation_rate → reduce next surge prediction
```

**Approach 2: Moderate - Decompose Acceptance Impact**

Key idea: Use offline revenue analysis to create optimal labels instead of actual surge.

#### Label Generation Process

**Step 1: Stratify by Demand Level**

```python
# Group historical data by demand bins
# Use requests_per_driver as the demand signal
demand_bins = [
    (0, 0.5),      # Low demand
    (0.5, 1.5),    # Medium demand
    (1.5, 3.0),    # High demand
    (3.0, 5.0),    # Very high demand
    (5.0, float('inf'))  # Extreme demand
]

for demand_min, demand_max in demand_bins:
    demand_slice = historical_data[
        (historical_data['requests_per_driver'] >= demand_min) &
        (historical_data['requests_per_driver'] < demand_max)
    ]

    # Continue to Step 2 for each demand bin
```

**Step 2: Measure Acceptance Curves**

```python
def analyze_surge_acceptance_curve(demand_slice):
    """
    For a given demand level, measure how cancellation_rate varies by surge
    """

    # Group by surge levels actually used in history
    surge_groups = demand_slice.groupby(
        pd.cut(demand_slice['surge_used'], bins=20)  # 20 surge bins
    )

    results = []

    for surge_bin, group in surge_groups:
        surge_val = surge_bin.mid

        # Calculate metrics for this surge level
        num_requests = len(group)
        cancelled = group['cancelled'].sum()
        cancellation_rate = cancelled / num_requests
        acceptance_rate = 1 - cancellation_rate

        # Revenue calculation
        revenue_per_request = surge_val * acceptance_rate
        total_revenue = revenue_per_request * num_requests

        results.append({
            'surge': surge_val,
            'cancellation_rate': cancellation_rate,
            'acceptance_rate': acceptance_rate,
            'num_samples': num_requests,
            'revenue_per_request': revenue_per_request,
            'total_revenue': total_revenue
        })

    return pd.DataFrame(results)

# Example output:
# surge  | acceptance_rate | cancellation_rate | revenue_per_request
# 1.0x   | 95%            | 5%               | 0.95
# 1.2x   | 90%            | 10%              | 1.08
# 1.5x   | 80%            | 20%              | 1.20  ← Peak!
# 2.0x   | 50%            | 50%              | 1.00
# 2.5x   | 30%            | 70%              | 0.75
```

**Step 3: Find Optimal Surge**

```python
def find_optimal_surge(acceptance_curve_df):
    """
    For this demand level, what surge maximizes revenue?
    """

    # Revenue = surge_multiplier × acceptance_rate
    acceptance_curve_df['revenue'] = (
        acceptance_curve_df['surge'] *
        acceptance_curve_df['acceptance_rate']
    )

    # Find peak revenue
    optimal_idx = acceptance_curve_df['revenue'].idxmax()
    optimal_surge = acceptance_curve_df.loc[optimal_idx, 'surge']
    peak_revenue = acceptance_curve_df.loc[optimal_idx, 'revenue']

    return optimal_surge, peak_revenue
```

**Step 4: Create Label Lookup Table**

```python
# For each demand level, store optimal surge
optimal_surge_lookup = {}

for demand_min, demand_max in demand_bins:
    demand_slice = get_demand_slice(demand_min, demand_max)
    acceptance_curve = analyze_surge_acceptance_curve(demand_slice)
    optimal_surge, peak_revenue = find_optimal_surge(acceptance_curve)

    # Store for later use
    demand_key = (demand_min, demand_max)
    optimal_surge_lookup[demand_key] = {
        'optimal_surge': optimal_surge,
        'peak_revenue': peak_revenue,
        'acceptance_curve': acceptance_curve
    }

# Example lookup:
# (0, 0.5): {'optimal_surge': 1.0x, 'peak_revenue': 1.0}
# (0.5, 1.5): {'optimal_surge': 1.2x, 'peak_revenue': 1.08}
# (1.5, 3.0): {'optimal_surge': 1.5x, 'peak_revenue': 1.20}
# (3.0, 5.0): {'optimal_surge': 1.4x, 'peak_revenue': 1.25}
# (5.0, inf): {'optimal_surge': 1.3x, 'peak_revenue': 1.28}
```

**Step 5: Apply Optimal Labels to Training Data**

```python
def generate_optimal_labels(training_data, optimal_surge_lookup):
    """
    Assign optimal surge as label for each training sample
    """

    labels = []

    for idx, row in training_data.iterrows():
        requests_per_driver = row['requests_per_driver']

        # Find which demand bin this falls into
        for demand_min, demand_max in optimal_surge_lookup.keys():
            if demand_min <= requests_per_driver < demand_max:
                optimal_surge = optimal_surge_lookup[
                    (demand_min, demand_max)
                ]['optimal_surge']
                break

        labels.append(optimal_surge)

    return np.array(labels)

# Now train model with optimal labels:
X_train, _ = training_data[feature_cols], training_data['surge_actual']
y_train = generate_optimal_labels(training_data, optimal_surge_lookup)

model.fit(X_train, y_train)  # Train on optimal surge, not actual surge!
```

#### Why This Works

```
Model learns: "For requests_per_driver=2.0, predict 1.5x (optimal)"
Instead of: "For requests_per_driver=2.0, predict 2.5x (what was actually used)"

Result:
  - Model predicts revenue-maximizing surge
  - Bypasses historical suboptimal pricing
  - Converges faster (cleaner signal)
  - Less noise from cancellation rate variations
```

#### Implementation Example

```
Training Sample:
  zone: downtown
  hour: 20
  requests_per_driver: 2.3  ← Falls in (1.5, 3.0) bin
  weather: rainy

Approach 1 (Actual):
  Label: 2.5x (whatever was charged)

Approach 2 (Optimal):
  Label: 1.5x (from optimal_surge_lookup[(1.5, 3.0)])

Model trains on cleaner signal → predicts revenue-optimal surge
```

#### Pros & Cons

```
Pros:
  ✓ Direct signal (no temporal lag like Approach 1)
  ✓ Model learns revenue-optimal behavior
  ✓ Bypasses historical suboptimal pricing
  ✓ Faster convergence
  ✓ Handles cancellations explicitly

Cons:
  ✗ Assumes historical data captures all surge-acceptance relationships
  ✗ Requires offline analysis (extra step before training)
  ✗ If acceptance curves change over time, labels become stale
  ✗ May not work well for rare demand scenarios (low sample size)
```

**Approach 3: Advanced - Weighted Loss Function**

```
During training, weight samples by outcome using sample weights (must be non-negative):

For accepted rides:
  loss_weight = 1.0 (good outcome)
  # Model should learn this surge level

For cancelled rides:
  loss_weight = 0.5 (lower weight, but still positive)
  # Reduces influence but doesn't penalize gradients

Better approach - use custom loss function instead:

def surge_weighted_loss(y_true, y_pred, accepted_mask):
    """
    Custom loss that penalizes overpredicting surge on cancellations
    """
    base_loss = (y_true - y_pred) ** 2

    # Scale loss based on outcome
    weights = np.where(accepted_mask, 1.0, 2.0)  # Penalize cancelled rides more

    return np.mean(weights * base_loss)

Implementation with custom loss:
  model.fit(X_train, y_train, loss=surge_weighted_loss,
            loss_kwargs={'accepted_mask': accepted_mask})
```

#### Recommended Approach for Production

**Hybrid: Approach 1 + 2**

```
Step 1: Train with all rides (accepted + cancelled)
  - Label: actual surge charged (Option 1)
  - Feature: include recent_cancellation_rate
  - Model learns correlation between conditions and outcomes

Step 2: Offline analysis - measure acceptance curves
  For each demand level (requests_per_driver value):
    - Find optimal surge that maximizes revenue
    - Create lookup table: demand_level → optimal_surge

Step 3: Use cancellation_rate for real-time adjustments
  If recent_cancellation_rate > threshold (e.g., 40%):
    - Reduce surge prediction by 10-20%
    - More conservative pricing to improve experience
    - Trade-off: slightly lower revenue for better retention

Step 4: Monitor feedback loop
  Every day measure:
    - Cancellation rate vs. surge prediction correlation
    - Revenue vs. user satisfaction
    - Adjust weighting if needed
```

**Python Implementation:**

```python
def prepare_surge_training_data_with_cancellations(zone_metrics):
    """
    Include cancellations in training data
    """

    # 1. Load all requests (accepted + cancelled)
    all_requests = load_all_zone_requests(zone_metrics)  # Don't filter!

    # 2. Calculate labels using ALL rides
    zone_times = all_requests.groupby(['zone_id', 'timestamp'])
    labels = {}

    for (zone, time), group in zone_times:
        # Group contains both accepted and cancelled
        actual_prices = group['price_charged']
        base_prices = group['estimated_base_price']
        cancellation_rate = group['cancelled'].mean()

        surge = mean(actual_prices) / mean(base_prices)

        labels[(zone, time)] = {
            'surge_multiplier': surge,
            'cancellation_rate': cancellation_rate,
            'num_requests': len(group),
            'num_cancelled': group['cancelled'].sum()
        }

    # 3. Engineer features including cancellation_rate
    features = engineer_features(all_requests, labels)

    # Add cancellation_rate from previous window as feature
    features['recent_cancellation_rate'] = compute_recent_cancellation_rate(
        all_requests, lookback_minutes=10
    )

    # 4. Create training labels (use actual surge, don't filter by acceptance)
    X = features[feature_cols]
    y = [labels[(z, t)]['surge_multiplier']
         for z, t in zip(features['zone_id'], features['timestamp'])]

    return X, y
```

#### Evaluating Model with Cancellation Metrics

**During model evaluation, measure cancellation impact:**

```python
def evaluate_surge_with_cancellations(model, X_test, y_test, actual_cancellations):
    """
    Evaluate surge model considering cancellation outcomes
    """

    predictions = model.predict(X_test)

    # Standard metrics
    mape = mean_absolute_percentage_error(y_test, predictions)

    # Cancellation-aware metrics
    predicted_vs_actual_cancellation = []

    for pred_surge, actual_surge, cancellation_rate in zip(
        predictions, y_test, actual_cancellations
    ):
        # Higher surge → expect higher cancellation
        # Measure: did predicted surge align with observed cancellation?

        predicted_cancellation_rate = estimate_cancellation_rate(pred_surge)
        actual_cancel = cancellation_rate

        predicted_vs_actual_cancellation.append({
            'predicted_surge': pred_surge,
            'actual_surge': actual_surge,
            'predicted_cancel_rate': predicted_cancellation_rate,
            'actual_cancel_rate': actual_cancel,
            'error': abs(predicted_cancel_rate - actual_cancel)
        })

    # Metrics
    cancellation_prediction_mae = mean(
        abs(df['error'] for df in predicted_vs_actual_cancellation)
    )

    return {
        'surge_mape': mape,
        'surge_accuracy': accuracy_of_surge_predictions,
        'cancellation_prediction_error': cancellation_prediction_mae,
        'cancellation_correlation': correlation(predictions, actual_cancellations)
    }
```

#### Key Insights

```
1. DO include cancelled rides in training data
   ✓ Use all requests (accepted + cancelled)
   ✓ Label them with the surge that caused the cancellation
   ✓ Model learns full picture of demand-surge-outcome

2. DO use cancellation_rate as a feature
   ✓ High cancellation_rate → feedback that surge was too high
   ✓ Model learns to reduce surge if seeing high cancellations
   ✓ Creates negative feedback loop for stability

3. DON'T filter to only accepted rides
   ✗ Creates selection bias
   ✗ Model misses signal that surge caused cancellations
   ✗ Overpredicts surge in high-demand scenarios

4. Monitor cancellation correlation
   ✓ Measure: does higher predicted surge correlate with higher cancellations?
   ✓ Should see moderate positive correlation
   ✓ If correlation too high: surge too aggressive
   ✓ If correlation too low: surge too conservative

5. Use cancellation data for post-processing
   ✓ If recent cancellation rate > threshold
   ✓ Reduce predicted surge by 10-20%
   ✓ Trade off: small revenue loss for better user experience
```

### Train-Test Split Strategy

**Critical**: Don't use random split! Surge has strong temporal patterns.

#### Time-Based Split

```
Train set:    Jan 1 - Aug 31, 2024 (8 months)
Validation:   Sep 1 - Sep 30, 2024 (1 month)
Test set:     Oct 1 - Oct 31, 2024 (1 month held-out)
```

**Why this works:**
- Train on diverse historical patterns
- Validate on recent, realistic data
- Test on completely held-out month
- Respects time dependency

#### Geographic Stratification

```
For each zone:
  - Allocate 80% to train
  - Allocate 10% to validation
  - Allocate 10% to test

Ensures:
  - All zones represented in all sets
  - No zone entirely in test (can't extrapolate)
  - Balanced learning across regions
```

#### Temporal Pattern Stratification

```
Ensure representation of:
  - All hours (0-23)
  - All days of week (0-6)
  - Different weather conditions
  - Peak periods (7-9am, 5-7pm)
  - Off-peak periods (3-5am)
  - Special events (if available)
```

### Data Size Requirements

| Stage | Minimum | Recommended | Ideal |
|-------|---------|-------------|-------|
| **Initial Development** | 50K zone-hours | 200K zone-hours | 500K zone-hours |
| **Production Model** | 200K zone-hours | 500K zone-hours | 1M+ zone-hours |

**Example calculation:**
```
1 city with 100 zones
Collecting data every 30 seconds
Per zone per day: 2880 data points (30s × 60 × 24)
100 zones per day: 288K data points
Per month: ~8.6M data points
Per 30 days: Very high-quality dataset

Recommendation:
  Train on 3+ months: ~25M data points
  Enables learning hourly patterns, weekly cycles, seasonal shifts
```

### Addressing Class Imbalance

**Problem:**
```
Most of the time surge ≈ 1.0 (normal pricing)
High surge (>2.0x) is rare

Distribution might be:
  1.0x surge: 60% of data
  1.1-1.5x:   25% of data
  1.5-2.0x:   10% of data
  2.0x+:      5% of data

Without handling, model learns to predict 1.0x all the time (low error overall)
But high surge predictions are poor
```

**Solution: Stratified Sampling**

```python
# Group by surge bins
low_surge_data = data[data.surge < 1.1]      # 1.0x
medium_surge = data[(data.surge >= 1.1) & (data.surge < 1.5)]  # 1.1-1.5x
high_surge = data[data.surge >= 1.5]         # 1.5x+

# Oversample rare high-surge events
train_data = pd.concat([
    low_surge_data.sample(frac=0.7),          # Use 70% of common
    medium_surge.sample(frac=0.9),            # Use 90% of medium
    high_surge.sample(frac=1.0, replace=True) # Use 100% + oversample high
])

# Result: Balanced representation of all surge levels
```

---

## Model Selection & Architecture

### Model Comparison

| Model | Latency | Accuracy | Pros | Cons | Recommendation |
|-------|---------|----------|------|------|---|
| **Linear (demand_ratio)** | <1ms | Medium | Fast, interpretable | Too simplistic | Fallback |
| **Polynomial Regression** | <1ms | Medium-High | Captures non-linearity | Limited expressiveness | Baseline |
| **Decision Tree** | <1ms | Medium | Fast | Shallow tree can't capture complexity | Baseline |
| **Random Forest** | 20-50ms | High | Robust, parallel inference | Less interpretable | Good |
| **Gradient Boosting (XGBoost)** | 20-50ms | **Very High** | Best accuracy, handles interactions | More complex | **PRIMARY** ✅ |
| **LightGBM** | 15-40ms | Very High | Faster than XGBoost | Slightly less stable | Alternative |
| **LSTM/RNN** | 100-200ms | High | Captures temporal patterns | Too slow, harder to deploy | Research only |

### Recommended: Gradient Boosting (XGBoost)

**Why XGBoost for Surge Pricing?**

✅ **Accuracy**: Captures non-linear demand-supply relationships
✅ **Speed**: 20-50ms inference (within <500ms budget)
✅ **Interactions**: Learns complex interactions (weather + hour, demand + day_of_week)
✅ **Robustness**: Handles outliers gracefully (sudden demand spikes)
✅ **Interpretability**: SHAP values explain predictions
✅ **Production Ready**: Proven in high-frequency trading, pricing systems
✅ **Feature Importance**: Identifies key surge drivers (requests_per_driver > hour > weather)

**Architecture Configuration:**

```
Model: XGBoost Regressor

Hyperparameters:
  - num_rounds: 150-250 trees (more than base model)
  - max_depth: 6-8 (captures surge complexity)
  - learning_rate: 0.05-0.08 (balanced learning)
  - subsample: 0.8 (sample rows)
  - colsample_bytree: 0.8 (sample features)
  - objective: 'reg:squarederror' (regression)
  - eval_metric: 'rmse'

Performance:
  - Training time: 30 min - 2 hours on 500K zone-hours
  - Inference latency: ~25-40ms
  - Model size: ~100-150MB
  - Memory: ~3GB for serving
```

### Fallback: Simple Linear Model

**For rapid deployment or fallback:**

```
surge = 1.0 +
        w1 × log(requests_per_driver) +
        w2 × hour_spike_factor +
        w3 × is_peak_hour +
        w4 × weather_factor +
        w5 × day_of_week_factor

Example formula:
surge = 1.0 + 0.5×log(requests_per_driver) + 0.3×is_peak_hour + ...

Inference: <1ms
Accuracy: ~12-15% MAPE (acceptable as fallback)
```

This captures the core logic without ML complexity.

---

## Model Training Pipeline

### Training Data Processing

```python
def prepare_surge_training_data(zone_metrics):
    """
    Complete pipeline for surge model training data
    """

    # 1. Load zone-level metrics (30-second granularity)
    metrics = load_zone_metrics(zone_metrics)

    # 2. Validate data quality
    metrics = validate_surge_metrics(metrics)

    # 3. Feature engineering
    metrics['requests_per_driver'] = (
        metrics['active_requests'] / (metrics['available_drivers'] + 1)
    )
    metrics['supply_score'] = 1 / (1 + metrics['requests_per_driver'])
    metrics['demand_momentum'] = (
        metrics['request_rate'] -
        metrics['request_rate_5min_ago']
    )

    # 4. Historical baseline features
    metrics['avg_surge_this_time'] = metrics.apply(
        lambda row: get_historical_surge(
            zone=row['zone_id'],
            hour=row['hour'],
            day_of_week=row['day_of_week']
        ),
        axis=1
    )

    # 5. Label creation (actual surge used)
    metrics['surge_multiplier'] = (
        metrics['actual_price'] /
        metrics['base_price']  # Use base price model predictions
    )

    # 6. Temporal split
    train = metrics[metrics['date'] <= '2024-08-31']
    val = metrics[(metrics['date'] >= '2024-09-01') & (metrics['date'] <= '2024-09-30')]
    test = metrics[metrics['date'] >= '2024-10-01']

    # 7. Feature selection
    feature_cols = [
        'active_requests', 'available_drivers', 'requests_per_driver',
        'request_rate', 'driver_utilization_rate', 'avg_pickup_wait_time',
        'hour_of_day', 'day_of_week', 'is_peak_hour',
        'weather_condition', 'temperature', 'is_holiday',
        'avg_surge_this_time', 'surge_volatility'
    ]

    # 8. Address class imbalance (oversample high surge)
    train = balance_surge_dataset(train, label='surge_multiplier')

    return {
        'train': (train[feature_cols], train['surge_multiplier']),
        'val': (val[feature_cols], val['surge_multiplier']),
        'test': (test[feature_cols], test['surge_multiplier']),
        'feature_names': feature_cols
    }
```

### Hyperparameter Tuning

```python
from optuna import create_study
import xgboost as xgb

def objective(trial):
    """Objective function for surge model tuning"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
    }

    model = xgb.XGBRegressor(**params, n_estimators=200)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20
    )

    predictions = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, predictions)

    return mape

# Optimize for MAPE
study = create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

### Training Frequency & Schedule

**Surge Model Retraining (More Frequent):**

```
Frequency: Daily (every day at 2:00 AM UTC)
Data window: Last 30 days of metrics
Duration: 30 min - 2 hours
Why daily:
  - Demand patterns change rapidly
  - Market dynamics shift (competitors)
  - Short-term seasonal effects
  - Regular retraining keeps model fresh

Deployment:
  1. Train new model in shadow mode
  2. Validate MAPE <10% on validation set
  3. Compare to previous model on recent data
  4. If improvement: deploy to 1% traffic
  5. Monitor for 1-2 hours
  6. Expand to 100% if no regressions
  7. Keep previous version for rollback
```

---

## Feature Preprocessing & Scaling

### Categorical Features

**Target Encoding by Demand:**

```python
def encode_categorical(feature, target_metric):
    """
    Encode categorical features using surge as target
    """

    # For hour_of_day: compute avg surge per hour
    hour_surge_mean = data.groupby('hour')['surge_multiplier'].mean()
    # hour_0: 1.05x (night, low surge)
    # hour_8: 1.85x (rush hour, high surge)
    # hour_14: 1.20x (afternoon, moderate)

    hour_encoding = hour_surge_mean.to_dict()

    # For day_of_week
    dow_surge_mean = data.groupby('day_of_week')['surge_multiplier'].mean()
    # Monday: 1.45x (commuting)
    # Friday: 1.52x (evening events)
    # Sunday: 1.35x (lower)

    return hour_encoding, dow_encoding
```

**Why this works:**
- Captures relationship between category and surge
- Reduces dimensionality (1 column instead of 24 or 7)
- Interpretable (hour_8 = 1.85x means morning rush historically 1.85x)

### Numerical Features

**Normalization for XGBoost:**
```
✗ NOT required for tree models
  - XGBoost splits on actual values
  - Invariant to scaling
  - 100 requests same as 100,000 requests (just uses threshold)

✓ Scale for interpretability/comparison
  - Range normalize: (value - min) / (max - min)
  - Standard scale: (value - mean) / std
```

### Missing Value Handling

**Strategy:**

```
active_requests:         Use 0 (no requests)
available_drivers:       Use zone median (fallback)
request_rate:            Use previous 30min avg
weather_condition:       Use previous non-missing value
temperature:             Use weather service API
nearby_events:           Use event database lookup

NEVER forward-fill (using future data)
NEVER drop rows (lose valuable training data)
```

---

## Model Inference & Serving

### Inference Pipeline (per ride request)

```
User requests ride
    ↓
[1. Collect Current Metrics] (~5ms)
    ├── Get active_requests in zone
    ├── Get available_drivers in zone
    ├── Calculate requests_per_driver
    ├── Get traffic_score
    ├── Get weather data
    └── Encode categorical features

    ↓
[2. Model Inference] (~30-40ms)
    ├── Load surge XGBoost model (pre-loaded)
    ├── Feed features
    └── Output: surge_multiplier (1.0 - 5.0+)

    ↓
[3. Post-processing] (~5ms)
    ├── Round to nearest 0.1x (1.5x, 1.6x, 1.7x)
    ├── Apply bounds:
    │   ├── Min: 1.0x (no negative surge)
    │   └── Max: 5.0x (cap extreme surge)
    ├── Smooth transition:
    │   └── Don't change by >0.5x from previous price
    └── Create response

    ↓
[Combined with Base Price] (~10ms)
    ├── final_price = base_price × surge_multiplier
    └── Return to user

[Total Latency: 350-450ms]
```

### Real-Time Metrics Collection

**Critical for surge model**: Need real-time zone-level signals.

```
Architecture:

    Driver Location Stream
         ↓
    [Stream Processor]
         ├── Aggregate drivers per zone
         ├── Calculate utilization rate
         └── Update every 30 seconds
         ↓
    Redis Cache
    ├── zone_1_active_drivers: 45
    ├── zone_1_active_requests: 150
    ├── zone_1_request_rate: 2.5 req/min
    └── zone_1_surge_prediction: 1.8x

When ride request arrives:
    ├── Query Redis for metrics (< 5ms)
    ├── Pass to surge model
    └── Get prediction (< 40ms)
```

### Serving Architecture

```
                    API Gateway
                         ↓
                   Load Balancer
                    /      |      \
                   /       |       \
             Server 1   Server 2   Server 3
          [surge_v3] [surge_v3] [surge_v3]

Per server:
  - Load model at startup
  - Keep in memory (~3GB)
  - Max 3000 req/sec per server
  - Use ONNX Runtime for optimization
```

**Model Versioning:**

```
Version History:
  surge_v1: Initial model
  surge_v2: Added weather features
  surge_v3: Current production (1M data points)
  surge_v4: Candidate (new data, improved)

Deployment Process:
  1. Train surge_v4 with latest 30 days
  2. Validate MAPE < 10%
  3. A/B test 1% traffic
  4. Expand to 100% if successful
  5. Keep surge_v3 for quick rollback
```

### Caching & Fallback

```
Cache historical surge patterns:
  zone_1_hour_8_dow_1: 1.85x (Monday 8am typical)
  zone_1_hour_18_dow_5: 2.20x (Friday 6pm typical)

Use when:
  - Model inference fails
  - Metrics unavailable
  - Under extreme traffic load
  - Provides quick approximate surge

Fallback to simple formula:
  surge = 1.0 + 0.4 × log(requests_per_driver)

Ensure pricing always available (never fail open)
```

---

## Model Evaluation & Monitoring

### Offline Evaluation Metrics

**Primary Metrics:**
```
MAPE (Mean Absolute Percentage Error):
  Target: <10%
  (Surge is harder to predict than base price)

MAE (Mean Absolute Error in Multiplier):
  Formula: mean(|actual_surge - pred_surge|)
  Target: <0.15x
  (e.g., predict 1.8x when actual 1.9x)

Directional Accuracy:
  % of time model predicts correct direction (up/down)
  Target: >75%

Correlation:
  Pearson correlation between predicted and actual
  Target: >0.70
```

**Fairness Metrics:**
```
Zone Fairness:
  - MAPE per zone
  - Ensure no zone systematically over/underpriced
  - Target: <12% MAPE in each zone

Time Fairness:
  - MAPE by hour
  - Avoid systematic bias toward peak/off-peak
  - Target: consistent across all hours

Supply Correlation:
  - Does model's surge prediction correlate with driver supply response?
  - Higher predicted surge → more drivers should accept
  - Target: >0.60 correlation
```

### Online Monitoring Dashboard

**Real-Time Metrics:**

```
1. Prediction Accuracy
   - Actual vs Predicted surge (scatter plot)
   - MAPE trending (line chart)
   - Alert if MAPE > 12% for 30 min

2. Revenue Impact
   - Estimated revenue with model vs baseline
   - Revenue lift vs rule-based pricing
   - Target: +5-15% improvement

3. Supply Response
   - Surge prediction vs driver supply changes
   - Do higher predictions attract drivers?
   - Correlation metric (target: >0.60)

4. Price Fairness
   - Coefficient of variation across zones
   - Are zones pricing consistently?
   - Target: CoV < 15%

5. Latency & Performance
   - Model inference latency (p50, p95, p99)
   - Feature collection latency
   - Alert if p99 > 100ms

6. Feature Distribution
   - Active requests distribution
   - Available drivers distribution
   - Traffic/weather patterns
   - Alert on significant shifts
```

### Drift Detection

**Monitor for Changes:**

```
Data Drift:
  - requests_per_driver distribution shifted
  - available_drivers distribution changed
  - weather/traffic patterns different
  → Trigger retrain if KL divergence > 0.3

Label Drift:
  - Actual surge patterns changing
  - Competitor pricing pressure
  - Market dynamics shift
  → Trigger retrain if actual_surge distribution shifted >20%

Concept Drift:
  - Relationship between features and surge changing
  - MAPE increasing despite no data drift
  - Model losing predictive power
  → Trigger retrain if MAPE > 12% for 3+ hours
```

**Automatic Retrain Triggers:**
```
✓ MAPE > 12% for 2+ hours consecutive
✓ Feature distribution KL divergence > 0.3
✓ Supply response correlation < 0.50
✓ Revenue lift dropped below +3%
✓ > 20% increase in price anomalies
```

---

## Model Interpretability & Debugging

### Feature Importance (SHAP)

```python
import shap
import xgboost as xgb

explainer = shap.TreeExplainer(surge_model)
shap_values = explainer.shap_values(X_test)

# Global importance
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Per-prediction explanation
shap.force_plot(explainer.expected_value, shap_values[i], X_test[i])
```

**Expected Feature Importance:**
```
Rank 1: requests_per_driver          (+0.90) ← Dominant surge signal
Rank 2: hour_of_day                   (+0.50) ← Time patterns
Rank 3: is_peak_hour                  (+0.35)
Rank 4: day_of_week                   (+0.25)
Rank 5: weather_condition             (+0.20)
Rank 6: temperature                   (+0.15)
Rank 7: avg_surge_this_time          (+0.15) ← Historical baseline
Rank 8: is_holiday                    (+0.10)

requests_per_driver > all others combined
```

### Debugging Surge Anomalies

**When surge prediction seems wrong:**

```
Step 1: Check input metrics
  - requests_per_driver correct?
  - available_drivers reasonable?
  - Metrics up-to-date (<30s old)?

Step 2: Check SHAP values
  Example 1: Predicted 2.5x surge
    ├── requests_per_driver = 3.0 (+1.2x from SHAP)
    ├── is_peak_hour = true (+0.4x)
    ├── avg_surge_this_time = 1.1x (+0.1x)
    └── Weather = clear (-0.1x)
    = 1.0 + 1.2 + 0.4 + 0.1 - 0.1 = 2.6x ✓ Makes sense

Step 3: Analyze similar historical periods
  - Find times with similar requests_per_driver
  - What was actual surge?
  - Is prediction in reasonable range?

Step 4: Check for infrastructure issues
  - Feature latency: are metrics fresh?
  - Model staleness: last retrain when?
  - Downstream service delays?
```

---

## A/B Testing & Deployment

### Revenue Impact A/B Test

```
Test new surge model:

Control (50%): Old model (current production)
Treatment (50%): New surge model

Duration: 1-2 weeks (need statistical significance)

Primary Metrics:
  - Revenue per ride
  - Revenue per hour
  - Total platform revenue

Secondary Metrics:
  - Driver supply (more accept?)
  - User acceptance rate (accept or cancel?)
  - Price fairness (variation across zones)
  - User satisfaction scores

Success Criteria:
  ✓ Revenue lift: +5% or higher
  ✓ No reduction in driver supply
  ✓ No increase in cancellations
  ✓ Fair pricing maintained
```

### Deployment Process

```
Day 1 - Shadow Deployment:
  ├── Deploy surge_v4 in shadow mode
  ├── Run predictions but don't use them
  ├── Validate MAPE < 10% on real data
  └── Proceed if validation passes

Day 2 - Canary 1%:
  ├── Deploy to 1% of ride requests
  ├── Monitor errors, latency, metrics
  ├── Verify no crashes or timeouts
  └── Proceed if stable

Day 3 - Canary 5%:
  ├── Expand to 5% traffic
  ├── Run statistical A/B test
  ├── Measure revenue impact
  └── If good metrics: proceed

Day 4+ - Rollout:
  ├── If revenue lift confirmed: 100%
  ├── If unclear: extend A/B test
  ├── If regression: rollback immediately
  └── Monitor closely for 1 week
```

---

## Summary & Deployment Checklist

### Model Specifications
- **Algorithm**: XGBoost Regressor
- **Features**: 15-20 demand-supply metrics
- **Model Size**: ~100-150MB
- **Training Time**: 30 min - 2 hours
- **Inference Latency**: 25-40ms
- **Accuracy Target**: MAPE <10%
- **Retraining**: Daily (vs. weekly for base price)

### Implementation Steps

- [ ] Collect zone-level metrics (500K+ zone-hours)
- [ ] Engineer features (demand-supply, temporal, weather)
- [ ] Generate labels (actual surge multipliers used)
- [ ] Temporal split (train/val/test by date)
- [ ] Handle class imbalance (oversample high surge)
- [ ] Hyperparameter tuning
- [ ] Train XGBoost model
- [ ] Evaluate (target: <10% MAPE)
- [ ] Build real-time feature pipeline
- [ ] Deploy inference service
- [ ] Set up monitoring dashboard
- [ ] A/B test with 5% traffic for revenue impact
- [ ] Rollout to 100%

### Monitoring & Maintenance

- Monitor MAPE hourly (alert if >12%)
- Retrain daily at 2am UTC
- Track revenue impact vs baseline
- Monitor supply response correlation
- Track feature distribution drift
- Maintain rule-based fallback pricing
- Regular SHAP analysis

### Key Differences from Base Price Model

| Aspect | Base Price | Surge |
|--------|-----------|-------|
| **Retraining** | Weekly | Daily |
| **Features** | Static (trip details) | Dynamic (real-time metrics) |
| **Prediction Horizon** | Single trip | Zone-level aggregate |
| **Accuracy Target** | <5% MAPE | <10% MAPE |
| **Time Sensitivity** | Low | High |
| **Data Freshness** | Hours OK | Seconds critical |
| **Model Latency** | 15-20ms | 25-40ms |
| **Business Impact** | Cost accuracy | Revenue optimization |
