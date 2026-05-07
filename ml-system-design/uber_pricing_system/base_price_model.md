# Base Price Model - Uber Pricing System

## Overview

The **Base Price Model** predicts the fundamental fare for a trip before any surge/demand-based adjustments. This model estimates the cost of a trip based on trip characteristics, location, time, and conditions.

**Purpose**: Calculate `base_price` which is then multiplied by `surge_multiplier` to get the final price:
```
final_price = base_price × surge_multiplier
```

---

## Model Objectives & Success Metrics

### Primary Objectives
1. **Accuracy**: Predict trip cost within ±5% for 90% of trips
2. **Fairness**: Similar trips (same route, time, conditions) → similar prices
3. **Smoothness**: Avoid discontinuities and unexpected price jumps

### Secondary Objectives
1. Capture location-based pricing variations (airport, city center vs. suburbs)
2. Account for time-of-day effects (rush hour, night surcharge)
3. Include traffic/weather impacts on trip cost

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **MAPE (Mean Absolute % Error)** | <5% | Average percentage error across all trips |
| **RMSE (Root Mean Squared Error)** | <$2.50 | Penalizes large outliers |
| **p90 Error** | <10% | 90th percentile error |
| **Zone-level MAPE** | <8% per zone | Fairness: no zone systematically over/underpriced |
| **Ride-type MAPE** | <7% per type | Consistent across UberX, XL, Comfort, etc. |
| **Percentile Errors** | p50, p75, p95 | Distribution of errors (not just mean) |

---

## Feature Engineering

### Features for Base Price Model

#### Trip Characteristics
| Feature | Type | Range/Values | Why Important |
|---------|------|--------------|-----------------|
| **distance** | Numerical | 0.5 - 50+ miles | Direct cost driver - main pricing component |
| **duration** | Numerical | 5 - 120+ minutes | Driver time cost - per-minute charges |
| **pickup_hour** | Categorical | 0-23 | Time-of-day pricing (night surcharge, rush hour) |
| **day_of_week** | Categorical | 0-6 (Mon-Sun) | Weekday vs. weekend behavior |
| **is_holiday** | Binary | True/False | Holidays have different demand/pricing |
| **ride_type** | Categorical | UberX, XL, Comfort | Different pricing tiers and vehicle costs |

#### Location Features
| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **pickup_zone_tier** | Categorical | City center, suburbs, rural | Zone-based pricing tiers |
| **dropoff_zone_tier** | Categorical | Destination tier | Affects distance/difficulty |
| **zone_based_multiplier** | Numerical | 0.8 - 1.5x | Pre-computed zone pricing multiplier |
| **airport_trip** | Binary | Trip to/from airport | Premium pricing (drop-off fees, regulations) |
| **is_high_traffic_area** | Binary | Known congestion zones | Affects actual trip duration |
| **cross_zone_trip** | Binary | Pickup and dropoff in different zones | Long-distance trips |

#### Traffic & Environmental
| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **traffic_score** | Numerical | 0-10 scale | Increases estimated duration → higher cost |
| **weather_condition** | Categorical | Clear, Rainy, Snowy, Foggy | Bad weather increases costs (more time) |
| **is_weekend** | Binary | Sat/Sun | Weekend pricing patterns |
| **is_peak_hour** | Binary | 7-9am, 5-7pm | Rush hour time adjustments |
| **temperature** | Numerical | F or C degrees | Extreme temps may affect driver costs |

#### User Features
| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **user_rating** | Numerical | 4.0 - 5.0 | May affect driver acceptance (if low rating) |
| **account_age_days** | Numerical | 0 - 3000+ | Proxy for user reliability |
| **is_new_user** | Binary | <30 days | New users may see promotional pricing |
| **device_type** | Categorical | iOS, Android, Web | Proxy for user segment |

#### Historical Context
| Feature | Type | Description | Importance |
|---------|------|-------------|-----------|
| **user_trip_count** | Numerical | 1, 5, 100, 1000+ | User familiarity with pricing |
| **avg_user_trip_price** | Numerical | Historical average | Baseline expectation |
| **user_cancellation_rate** | Numerical | 0 - 100% | Risk of trip not completing |

### Derived/Interaction Features

Create these features to capture relationships:

```python
# Time interactions
is_early_morning = (hour < 6)
is_night = (hour >= 22)
is_rush_hour = ((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19))

# Distance-time relationship
avg_speed = distance / duration  # Can identify traffic impacts

# Location interactions
is_CBD_to_residential = (pickup_zone == 'CBD' and dropoff_zone == 'residential')
is_airport_drop = (dropoff_zone == 'airport')

# Time of week patterns
is_weekend_night = (is_weekend and hour >= 22)
is_rush_hour_weekday = (is_rush_hour and not is_weekend)
```

---

## Training Data Generation

### Data Collection & Preprocessing

**Source Data:**
```
Historical ride transactions containing:
- Trip details: distance, duration, pickup/dropoff locations
- Pricing: actual_price_paid, base_price (if available)
- Surge info: surge_multiplier at time of ride
- User info: user_id, user_rating, account_age
- Context: weather, traffic, time, date
- Promotions: discount_amount, promo_code
```

**Data Quality Checks:**
```
✓ Remove rides with missing core fields (distance, duration, price)
✓ Filter outliers:
  - Distance: 0.1 - 300 miles (flag >500 miles)
  - Duration: 1 - 600 minutes (flag >720 min)
  - Price: <$2 or >$200 may be invalid
  - Speed: <2 mph or >100 mph suspicious
✓ Remove duplicate records (same user, time, location)
✓ Validate coordinates (lat: -90 to 90, lng: -180 to 180)
✓ Handle timezone conversions properly
✓ Remove fraud/test transactions
```

### Label Generation Strategy

The **label** is the base fare that should be charged (independent of surge).

#### Option 1: Direct Usage (Simple, but Biased)
```
Input:  [distance, duration, location, time, ...]
Output: actual_price_paid
```

**Problems:**
- Biased by surge pricing (surge affects the label)
- Affected by promotions/discounts that shouldn't affect base price
- Result: model learns to predict "actual price" not "base price"

#### Option 2: Decompose from Actual Price (Better)
```
Raw data: actual_price = base_price × surge_multiplier + discounts

Reverse engineer:
estimated_base_price = (actual_price - discounts) / surge_multiplier

Input:  [distance, duration, location, time, ...]
Output: estimated_base_price
```

**Advantages:**
- Removes surge effect from label
- Removes promotional discounts
- Focuses model on intrinsic trip cost

**Implementation:**
```python
# Get surge multiplier from demand-supply metrics at time of ride
surge_multiplier = get_surge_multiplier(ride_time, zone)

# Remove known discounts/promotions
discounts = ride.promo_discount + ride.platform_credit

# Decompose
base_price = (ride.actual_price - discounts) / surge_multiplier
```

#### Option 3: Historical Reference Prices (Most Robust)
```
For each origin-destination pair:
  Use the median price when surge ≈ 1.0 (off-peak)

Input:  [distance, duration, location, time, ...]
Output: reference_base_price (p50 from low-surge periods)
```

**Advantages:**
- Completely removes surge effects
- Stable, representative baseline
- Less affected by promotional campaigns

**Implementation:**
```python
# For each origin-destination-ride_type combo:
# Find trips with surge_multiplier between 0.95 - 1.05 (normal pricing)
normal_surge_trips = data[
  (data.origin == origin) and
  (data.destination == destination) and
  (data.ride_type == ride_type) and
  (data.surge_multiplier.between(0.95, 1.05))
]

# Use median price as reference
reference_price = normal_surge_trips.actual_price.median()
```

**Recommendation**: Use **Option 3** for clearest base price signal.

### Train-Test Split Strategy

**Critical**: Don't use random split! Price patterns are time-dependent.

#### Temporal Split (Recommended)

```
Train set:    Jan 1, 2024 - Aug 31, 2024 (8 months)
Validation:   Sep 1, 2024 - Sep 30, 2024 (1 month)
Test set:     Oct 1, 2024 - Oct 31, 2024 (1 month, held-out)
```

**Reasoning:**
- Train on older data, validate on recent
- Test on completely held-out month
- Respects time dependency of prices
- Avoids data leakage from future to past

#### Seasonal Adjustments

If data spans multiple years:

```
Year 1 (2023):
  Train: Jan - Dec 2023
  Test: Hold out Oct-Dec 2023

Year 2 (2024):
  Train: Jan - Sep 2024
  Test: Oct - Dec 2024 (compare with same months in 2023)
```

This captures seasonal patterns (summer vs. winter pricing).

### Data Stratification

Ensure train and test sets are representative:

```python
# Stratify by key dimensions
for zone in zones:
  for time_period in ['0-6am', '7-12pm', '1-6pm', '7-11pm']:
    for weather in ['clear', 'rain', 'snow', 'fog']:
      for ride_type in ['UberX', 'XL', 'Comfort']:

        subset = data[
          (data.pickup_zone == zone) and
          (data.hour in time_period) and
          (data.weather == weather) and
          (data.ride_type == ride_type)
        ]

        # Allocate 80% to train, 20% to test
        if len(subset) > 0:
          train_samples = subset.sample(frac=0.8, random_state=42)
          test_samples = subset.drop(train_samples.index)
```

This ensures:
- All geographic zones represented
- All time-of-day patterns captured
- Weather conditions balanced
- Ride types distributed

### Data Size Requirements

| Stage | Minimum | Recommended | Ideal |
|-------|---------|-------------|-------|
| **Initial Development** | 100K rides | 500K rides | 1M rides |
| **Production Model** | 1M rides | 5M rides | 10M+ rides |

**Why larger is better:**
- Captures edge cases (rare weather, special events)
- Allows geographic segmentation (model per city)
- Enables ride-type specialization
- Improves generalization to new scenarios

---

## Model Selection & Architecture

### Model Comparison

| Model | Latency | Accuracy | Interpretability | Training Time | Recommendation |
|-------|---------|----------|------------------|---------------|---|
| **Linear Regression** | <1ms | Low | Very High | 10s | Fallback only |
| **Decision Tree** | <1ms | Medium | Very High | 1min | Baseline |
| **Random Forest** | 10-50ms | High | Medium | 30min | Good alternative |
| **Gradient Boosting (XGBoost)** | 10-50ms | **Very High** | Medium | 1-2hrs | **PRIMARY** ✅ |
| **LightGBM** | 10-50ms | Very High | Medium | 30min | Alternative to XGBoost |
| **Neural Network (2-3 layer)** | 20-50ms | High | Low | 1-2hrs | For scale |
| **Deep NN** | 50-200ms | Very High | Very Low | 5-10hrs | Too slow |

### Recommended: Gradient Boosting (XGBoost or LightGBM)

**Why XGBoost for Base Price?**

✅ **Accuracy**: Captures non-linear relationships (distance and time aren't perfectly linear)
✅ **Speed**: <50ms inference (well within <500ms budget)
✅ **Categorical Handling**: Natively handles zones, ride types, weather
✅ **Robustness**: Handles outliers gracefully
✅ **Interpretability**: SHAP values explain predictions
✅ **Production Ready**: Battle-tested in industry
✅ **Feature Importance**: Identifies key drivers (distance, location, time)

**Architecture Configuration:**

```
Model: XGBoost Regressor

Hyperparameters:
  - num_rounds: 100-200 trees
  - max_depth: 5-7 (avoid overfitting)
  - learning_rate: 0.05-0.1 (balance learning speed)
  - subsample: 0.8 (sample rows for each tree)
  - colsample_bytree: 0.8 (sample columns for each tree)
  - objective: 'reg:squarederror' (regression task)
  - eval_metric: 'rmse'

Performance:
  - Training time: 1-2 hours on 5M rides
  - Inference latency: ~15-20ms per prediction
  - Model size: ~50-100MB
  - Memory requirement: ~2GB for serving
```

### Fallback: Simple Linear Model

**For emergency or high-availability scenarios:**

```
base_price = w0 +
             w1 × distance +
             w2 × duration +
             w3 × traffic_score +
             w4 × zone_multiplier +
             w5 × time_of_day_multiplier +
             w6 × is_airport +
             interaction_terms

Inference: <1ms
Accuracy: ~8-10% MAPE (acceptable as fallback)
```

This can be deployed without ML infrastructure if primary model fails.

---

## Model Training Pipeline

### Training Data Processing

```python
def prepare_base_price_training_data(raw_rides):
    """
    Complete pipeline for training data preparation
    """

    # 1. Load and validate
    rides = load_rides(raw_rides)
    rides = validate_data(rides)  # Remove bad rows

    # 2. Feature engineering
    rides['is_early_morning'] = rides['hour'] < 6
    rides['is_rush_hour'] = (
        ((rides['hour'] >= 7) & (rides['hour'] <= 9)) |
        ((rides['hour'] >= 17) & (rides['hour'] <= 19))
    )
    rides['is_airport'] = rides['dropoff_zone'].isin(['JFK', 'LAX', 'ORD'])
    rides['avg_speed'] = rides['distance'] / (rides['duration'] + 0.1)

    # 3. Label creation (base price decomposition)
    rides['surge_multiplier'] = get_surge_at_time(rides['zone'], rides['timestamp'])
    rides['discounts'] = rides['promo_discount'] + rides['platform_credit']
    rides['base_price'] = (
        (rides['actual_price'] - rides['discounts']) /
        (rides['surge_multiplier'] + 0.01)
    )

    # 4. Temporal split (crucial!)
    train = rides[rides['date'] <= '2024-08-31']
    val = rides[(rides['date'] >= '2024-09-01') & (rides['date'] <= '2024-09-30')]
    test = rides[rides['date'] >= '2024-10-01']

    # 5. Stratification
    train = stratify_by_zone_time_weather(train)
    val = stratify_by_zone_time_weather(val)
    test = stratify_by_zone_time_weather(test)

    # 6. Feature selection
    feature_cols = [
        'distance', 'duration', 'pickup_hour', 'day_of_week',
        'pickup_zone_tier', 'airport_trip', 'traffic_score',
        'weather_condition', 'is_peak_hour', 'ride_type',
        'user_rating', 'is_new_user'
    ]

    return {
        'train': (train[feature_cols], train['base_price']),
        'val': (val[feature_cols], val['base_price']),
        'test': (test[feature_cols], test['base_price']),
        'feature_names': feature_cols
    }
```

### Hyperparameter Tuning

Use **Bayesian Optimization** to find best hyperparameters:

```python
from optuna import create_study
import xgboost as xgb

def objective(trial):
    """Objective function for hyperparameter tuning"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

    model = xgb.XGBRegressor(**params, n_estimators=150)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    predictions = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, predictions)

    return mape

# Run optimization
study = create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

### Training Frequency & Schedule

**Base Price Model Retraining:**

```
Frequency: Weekly (every Sunday at 2am UTC)
Data window: Last 6 months of historical rides
Duration: ~1-2 hours

Why weekly:
  - Captures seasonal patterns (weekly cycle)
  - Adapts to gradual price changes
  - Not too frequent (stability)
  - Not too infrequent (staleness)

Deployment:
  1. Train new model in shadow mode
  2. Validate MAPE <5% on test set
  3. A/B test with 1% traffic for 1 day
  4. If metrics improve: deploy to 100%
  5. If regression: rollback to previous version
```

---

## Feature Preprocessing & Scaling

### Categorical Features

**Method: Target Encoding with Smoothing**

```python
def target_encode(feature, target, smoothing_factor=1.0):
    """
    Encode categorical feature using target variable
    """

    # Compute mean target per category
    category_means = df.groupby(feature)[target].agg(['mean', 'count'])

    # Smoothing: avoid extreme values for rare categories
    global_mean = target.mean()
    smoothed_encoding = (
        (category_means['count'] * category_means['mean'] +
         smoothing_factor * global_mean) /
        (category_means['count'] + smoothing_factor)
    )

    return smoothed_encoding.to_dict()

# Example for zone_id
zone_encoding = target_encode('zone_id', df['base_price'], smoothing_factor=10)
# zone_1 -> $12.50 (mean price in zone 1, smoothed)
# zone_2 -> $18.75 (mean price in zone 2, smoothed)
```

**Why target encoding:**
- Better for tree models than one-hot encoding
- Reduces dimensionality (1 column instead of N)
- Captures relationship between category and price
- Smoothing prevents overfitting on rare categories

### Numerical Features

**Scaling for XGBoost:**
```
✗ NOT needed for tree models (XGBoost)
  - Trees are invariant to monotonic transformations
  - Distance of 10 miles vs 100 miles: tree splits on actual values
✓ NEEDED for linear models or neural networks
  - Use StandardScaler or MinMaxScaler
```

### Missing Value Handling

**Strategy:**
```
distance:       Impute with routing API (maps distance)
duration:       Impute with routing API (maps time)
traffic_score:  Impute with historical avg for zone-hour
weather:        Impute with nearest past non-missing value
user_rating:    Impute with platform median (4.8)
zone_tier:      Lookup from zone database

NEVER just drop rows - you lose valuable training data
```

---

## Model Inference & Serving

### Inference Pipeline (per ride request)

```
User requests ride (pickup: lat1,lng1, dropoff: lat2,lng2)
    ↓
[1. Feature Extraction] (~10ms)
    ├── Call routing API → distance, duration
    ├── Lookup zone_tiers for pickup/dropoff
    ├── Get traffic_score from traffic service
    ├── Get weather from weather API
    ├── Encode categorical features (zone→numeric)
    └── Create feature vector [40 values]
    ↓
[2. Model Inference] (~15-20ms)
    ├── Load XGBoost model (pre-loaded in memory)
    ├── Feed features to model
    └── Output: base_price prediction
    ↓
[3. Post-processing] (~5-10ms)
    ├── Round to nearest $0.50
    ├── Apply business rules:
    │   ├── Min price: $2.50
    │   ├── Max price: $200
    │   └── Airport trip: add $5 fee
    └── Create response
    ↓
[Return to App] (Total: 300-400ms)
    └── User sees estimated price
```

### Model Serving Architecture

**Recommended setup for high availability:**

```
                    API Gateway
                         ↓
                   Load Balancer (round-robin)
                    /      |      \
                   /       |       \
             Server 1   Server 2   Server 3
           [model_v5] [model_v5] [model_v5]

Per server:
  - Load model once at startup (~1s)
  - Keep in memory for ~1μs latency
  - Use ONNX Runtime for optimization
  - Max 2000 req/sec per server
```

**Model Versioning & Deployment:**

```
Version tracking:
  model_v1: initial production model
  model_v2: improved training data
  model_v3: new features (airport, weather)
  model_v4: current production
  model_v5: candidate for next release

Blue-Green Deployment:
  Blue (current):  100% traffic → model_v4
  Green (canary):   1% traffic → model_v5 (candidate)
  If metrics good:  Green → Blue (100% traffic)
  If regression:    Rollback to model_v4
```

### Caching Strategy

```
Cache base fare templates (update daily):

  Cache key: zone_{zone_id}_hour_{hour}_distance_{dist_bin}
  Cache value: estimated_base_price
  TTL: 24 hours

  Examples:
    zone_1_hour_9_distance_5:  $18.50
    zone_1_hour_15_distance_5: $16.25
    zone_2_hour_9_distance_5:  $22.00

Benefits:
  - Fallback if model inference fails
  - Quick approximate prices
  - Reduces model load
  - Improves p99 latency
```

---

## Model Evaluation & Monitoring

### Offline Evaluation Metrics

**Primary Metrics:**
```
MAPE (Mean Absolute Percentage Error):
  Formula: (1/n) × Σ|y_actual - y_pred| / y_actual
  Target: <5%
  Interpretation: Average % error in price prediction

RMSE (Root Mean Squared Error):
  Formula: √[(1/n) × Σ(y_actual - y_pred)²]
  Target: <$2.50
  Interpretation: Penalizes large errors heavily

Percentile Errors:
  p50 error: 50th percentile (median)
  p90 error: 90th percentile error
  p95 error: 95th percentile error
  Interpretation: Distribution of errors
```

**Fairness Metrics:**
```
Zone fairness: MAPE per zone
  - Ensure no zone systematically over/underpriced
  - Target: <8% MAPE in each zone

Ride-type fairness: MAPE per type
  - Target: <7% MAPE for UberX, XL, Comfort

Time fairness: MAPE by hour
  - Target: consistent across all hours
```

### Online Monitoring Dashboard

**Key Metrics (real-time):**

```
1. Prediction Accuracy
   - Actual vs Predicted price (scatter plot)
   - Error distribution (histogram)
   - Trend over time (line chart)

2. Latency Monitoring
   - p50, p95, p99 latency
   - Alert if p99 > 200ms (half of budget)

3. Feature Distribution Drift
   - Compare training vs production distributions
   - Flag if distance distribution shifts >15%
   - Alert on new values (e.g., new zone)

4. Model Staleness
   - Days since last retrain
   - Alert if >8 days without retraining

5. Performance by Segment
   - Accuracy by zone
   - Accuracy by ride type
   - Accuracy by time of day
```

### Drift Detection & Retraining Triggers

**Data Drift Detection:**
```
Monitor feature distributions in production:

If any feature distribution differs significantly:
  KL_divergence = Σ p(x)_train × log(p(x)_train / p(x)_prod)
  Trigger retrain if KL_divergence > threshold

Example: If average distance shifts from 5 miles to 8 miles
  → indicates market change
  → retrain to capture new patterns
```

**Label Drift Detection:**
```
Monitor actual price distribution:

If prices shift (e.g., competitor raises prices):
  Actual_price_dist shifts from historical
  → May indicate market change
  → May need pricing strategy review
```

**Concept Drift Detection:**
```
Monitor model error trends:

If MAPE increases from 4% to 6%:
  → Model performance degrading
  → Data distribution has shifted
  → Retrain needed

Trigger retraining if:
  - MAPE on recent data >6%
  - Feature distribution KL divergence >0.3
  - Consistent upward trend in errors over 3 days
```

---

## Model Interpretability & Debugging

### Feature Importance Analysis (SHAP)

```python
import shap
import xgboost as xgb

# Train model
model = xgb.XGBRegressor(...)
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)
# Output: Shows which features most impact price across all predictions

# Feature importance for single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
# Output: Shows why model predicted this specific price
```

**Expected Top Features (by importance):**
```
Rank 1: distance                (+0.8)  ← Primary driver
Rank 2: estimated_duration      (+0.6)  ← Time cost
Rank 3: is_airport              (+0.45) ← Premium pricing
Rank 4: zone_multiplier         (+0.4)  ← Location pricing
Rank 5: traffic_score           (+0.25) ← Congestion
Rank 6: pickup_hour             (+0.15) ← Time variation
Rank 7: is_peak_hour            (+0.12)
Rank 8: ride_type               (+0.10)
...
```

### Debugging Price Anomalies

**When predicted price seems wrong:**

```
Step 1: Extract features for that ride
  - Check distance, duration are correct
  - Verify zone assignments
  - Confirm weather/traffic data

Step 2: Check SHAP values
  - Which features pushed price up/down?
  - Are they reasonable?

  Example:
    Distance +$8 (reasonable)
    Zone +$2 (reasonable)
    Traffic +$3 (unexpected during night?)
    → May indicate bug in traffic data source

Step 3: Analyze similar rides
  - Find 10 similar rides in history
  - What was their typical price?
  - Is prediction in range?

Step 4: Compare to rule-based baseline
  - Calculate price using simple formula
  - If model ≫ formula, investigate model
  - If formula ≫ model, model may be underpredicting

Step 5: Check for data issues
  - Stale demand/supply metrics?
  - Bad feature values (null, extreme)?
  - API failures (routing, weather)?
```

---

## A/B Testing & Deployment Strategy

### Canary Deployment Process

```
Day 1 - Canary 1%:
  ├── Deploy model_v5 to 1% of traffic
  ├── Monitor for errors, crashes, latency spikes
  ├── Check MAPE on 1% sample
  └── If OK, proceed to next step

Day 2 - Canary 5%:
  ├── Deploy to 5% traffic (enough for stats)
  ├── Run A/B test (5% treatment vs 5% control)
  ├── Measure: MAPE, latency, price fairness
  ├── Statistical significance test (p-value < 0.05)
  └── If metrics improve, proceed

Day 3+ - Full Rollout or Rollback:
  ├── If all metrics good: deploy to 100%
  ├── If any regression: rollback to model_v4
  ├── If unclear: extend A/B test another day
  └── Once rolled out: monitor daily
```

### Success Criteria for New Model

✅ MAPE improved or ≤5% (not worse)
✅ Latency p99 < 200ms (not slower)
✅ Price fairness: std dev of errors < 8%
✅ No systematic bias by zone or ride type
✅ User acceptance rate maintained or improved

---

## Summary & Deployment Checklist

### Model Specifications
- **Algorithm**: XGBoost Regressor
- **Feature Count**: 25-30 engineered features
- **Model Size**: ~50-100MB
- **Training Time**: 1-2 hours
- **Inference Latency**: 15-20ms
- **Accuracy Target**: MAPE <5%

### Key Implementation Steps

- [ ] Collect and validate historical ride data (1M+ rides)
- [ ] Engineer features (distance, zone, time, weather, user)
- [ ] Generate labels (base price decomposition)
- [ ] Temporal split (train/val/test by date)
- [ ] Hyperparameter tuning (Bayesian optimization)
- [ ] Train XGBoost model
- [ ] Evaluate on test set (target: <5% MAPE)
- [ ] Build inference service (feature pipeline + model)
- [ ] Set up monitoring dashboard
- [ ] Implement canary deployment
- [ ] A/B test with 1% traffic
- [ ] Rollout to 100% if successful

### Monitoring & Maintenance

- Monitor MAPE daily (alert if >6%)
- Retrain weekly with new data
- Track latency (alert if p99 >200ms)
- Monitor feature drift
- Maintain fallback rule-based pricing
- Regular SHAP analysis (top features, anomalies)
