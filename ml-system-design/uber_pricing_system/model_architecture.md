# Uber Pricing System - ML Model Architecture

## Overview

The pricing system uses two complementary ML models:
1. **Surge Pricing Model** - Predicts demand-based surge multiplier (1.0x - 5.0x+)
2. **Base Price Model** - Predicts base fare before surge adjustment

These models work together to calculate final price: `final_price = base_price × surge_multiplier`

---

## 1. Model Objectives & Success Metrics

### Surge Pricing Model Objectives
- **Primary**: Maximize platform revenue by setting optimal surge multiplier
- **Secondary**: Maintain driver supply (high surge attracts drivers)
- **Constraint**: Keep customer satisfaction high (avoid price shock)

### Base Price Model Objectives
- **Primary**: Accurately predict trip cost (distance, duration, location)
- **Secondary**: Smooth predictions (avoid discontinuities)
- **Constraint**: Fair pricing for similar trips

### Evaluation Metrics

**For Surge Model:**
- **Revenue-weighted accuracy**: Weighted by trip volume and price
- **Supply response correlation**: Does surge price attract more drivers?
- **Price stability**: Temporal smoothness (penalize large jumps)
- **Fairness**: Similar demand scores → similar surge multipliers

**For Base Price Model:**
- **Mean Absolute Percentage Error (MAPE)**: $\text{MAPE} = \frac{1}{n}\sum \frac{|y_i - \hat{y}_i|}{y_i}$
- **Root Mean Squared Error (RMSE)**: Penalize large outliers
- **Percentile errors**: p50, p90, p95 (not just mean)
- **Location fairness**: Similar routes have similar prices

---

## 2. Feature Engineering

### 2.1 Base Price Model Features

#### Trip Characteristics
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **distance** | Numerical | Estimated distance (miles) | Direct cost driver |
| **duration** | Numerical | Estimated trip duration (minutes) | Driver time cost |
| **pickup_hour** | Categorical | Hour of day (0-23) | Time-of-day pricing |
| **day_of_week** | Categorical | Day (0-6, Mon-Sun) | Peak vs. off-peak patterns |
| **is_holiday** | Binary | Whether trip during holiday | Demand patterns |
| **ride_type** | Categorical | UberX, XL, Comfort, etc. | Different pricing tiers |

#### Location Features
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **pickup_zone_tier** | Categorical | City center, suburbs, rural | Zone-based pricing |
| **dropoff_zone_tier** | Categorical | Destination tier | Affects demand |
| **zone_based_multiplier** | Numerical | Pre-computed zone multiplier | Location pricing |
| **airport_trip** | Binary | Trip to/from airport | Premium pricing |
| **is_high_traffic_area** | Binary | Known congestion area | Affects duration |

#### Traffic & Environmental
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **traffic_score** | Numerical | Current traffic level (0-10) | Increases duration/cost |
| **weather_condition** | Categorical | Clear, rainy, snowy, foggy | Driver incentive needed |
| **is_weekend** | Binary | Saturday/Sunday indicator | Demand pattern |
| **is_peak_hour** | Binary | Rush hour indicator (7-9am, 5-7pm) | High demand |

#### User Features
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **user_rating** | Numerical | User's average rating (4.0-5.0) | May affect driver acceptance |
| **account_age_days** | Numerical | Days since account created | Proxy for user reliability |
| **is_new_user** | Binary | First 30 days of account | Cold start users may see promo pricing |
| **device_type** | Categorical | iOS, Android, Web | Proxy for user segment |

#### Historical Context
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **user_trip_count** | Numerical | Number of previous trips | User familiarity with pricing |
| **avg_user_trip_price** | Numerical | User's historical avg price | Baseline expectation |
| **cancellation_rate** | Numerical | User's historical cancel rate | Risk of not completing trip |

### 2.2 Surge Pricing Model Features

#### Demand-Supply Features (Real-time, zone-level)
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **active_requests_in_zone** | Numerical | Number of pending requests | Direct demand signal |
| **request_rate** | Numerical | Requests/minute in zone | Demand velocity |
| **available_drivers** | Numerical | Drivers online in zone | Direct supply signal |
| **driver_utilization_rate** | Numerical | % drivers with active ride | Supply tightness |
| **avg_pickup_wait_time** | Numerical | Expected wait (minutes) | User friction metric |
| **requests_per_driver** | Numerical | active_requests/available_drivers | Supply-demand ratio |

#### Temporal Features
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **hour_of_day** | Categorical | Hour (0-23) | Predictable demand patterns |
| **day_of_week** | Categorical | Day (0-6) | Weekday vs. weekend behavior |
| **is_peak_hour** | Binary | Rush hour window | Demand concentration |
| **minutes_since_hour_start** | Numerical | Position in hour (0-59) | Micro-patterns |

#### Event & External Features
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **nearby_events** | Categorical | Concert, game, festival | Demand spike indicators |
| **weather_condition** | Categorical | Clear, rain, snow | Affects ride demand |
| **temperature** | Numerical | Degrees (F/C) | Extreme temps increase demand |
| **is_holiday** | Binary | Holiday period | Predictable demand shift |

#### Historical Patterns
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **avg_surge_at_this_time** | Numerical | Historical average surge | Seasonal baseline |
| **surge_volatility** | Numerical | Std dev of surge at this time | Prediction uncertainty |
| **day_of_month** | Numerical | 1-31 | Payday effects (26-28th) |

#### Contextual Metrics
| Feature | Type | Description | Why Important |
|---------|------|-------------|-----------------|
| **competitor_average_price** | Numerical | Competitors' prices (if available) | Market competitiveness |
| **zone_accessibility_score** | Numerical | Ease of drivers reaching zone | Supply positioning |
| **platform_incentive_active** | Binary | Running driver promo | Artificial supply boost |

### 2.3 Feature Interactions & Engineering

**Derived Features to Create:**
```python
# Demand-supply imbalance
requests_per_driver = active_requests / (available_drivers + 1)
supply_score = 1 / (1 + requests_per_driver)  # 0-1 normalized

# Demand intensity
recent_demand_trend = (request_rate - avg_request_rate) / std_request_rate
demand_momentum = request_rate_diff_1min / request_rate_diff_5min

# Time context
is_early_morning = hour < 6
is_night = hour >= 22
is_rush_hour = ((hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19))

# Location-time interaction
zone_hour_interaction = zone_id + hour * 1000  # for embeddings
location_demand_bias = historical_avg_demand[zone][hour]
```

---

## 3. Training Data Generation from Historical Data

### 3.1 Data Collection & Preprocessing

**Source Data:**
- Completed ride transactions
- Real-time demand-supply snapshots (zone-level, every 30s)
- Driver location history
- Weather data, events, holidays
- User interaction logs

**Data Quality Checks:**
```
- Remove rides with missing core fields
- Filter outliers (price >10x median, distance >500 miles)
- Handle timezone conversions properly
- Remove duplicate records
- Validate coordinate bounds (valid lat/lng)
```

### 3.2 Labeling Strategy

#### For Base Price Model

**Option 1: Use Actual Trip Prices (Simple)**
```
Input:  [distance, duration, location, time, ...]
Output: actual_price_paid
```

**Issues:**
- Biased by surge pricing (don't want surge to affect base price learning)
- Affected by promotions/discounts

**Option 2: Decompose from Actual Price (Better)**
```
actual_price = base_price × surge_multiplier + promotions

Reverse engineer:
base_price = (actual_price - promotions) / surge_multiplier

Input:  [distance, duration, location, time, ...]
Output: estimated_base_price
```

**Option 3: Use Historical Reference Prices (Robust)**
```
For each trip route, use median/p50 price from low-surge periods (surge_multiplier ≈ 1.0)

Input:  [distance, duration, location, time, ...]
Output: reference_base_price
```

#### For Surge Pricing Model

**Label Generation:**
```
For each zone at time t:
  actual_surge = mean(prices_at_t) / baseline_price

  Input:  [active_requests, available_drivers, hour, weather, ...]
  Output: actual_surge_multiplier
```

**Time Alignment:**
- Requests at time t → Use demand metrics from t-1 min to t (leading indicator)
- Don't use metrics from t onwards (data leakage)

### 3.3 Train-Test Split Strategy

**Don't use random split!** Price patterns are time-dependent.

**Recommended Approach: Temporal Split**
```
Train set:    Jan 1 - Aug 31 (8 months historical data)
Validation:   Sep 1 - Sep 30 (recent month)
Test set:     Oct 1 - Oct 31 (held-out month)
```

**Seasonal Adjustment:**
```
If using data across seasons (summer vs. winter):
  Train set:  Full year or 2+ years of history
  Test set:   Recent similar season year-over-year

Example:
  Train: Jan 2023 - Dec 2023
  Test:  Jan 2024 - Mar 2024 (compare to same months in 2023)
```

### 3.4 Data Stratification

**Stratify by:**
- Geographic zones (ensure all zones represented)
- Time periods (peak, off-peak, night)
- Ride types (balance XL, Comfort, UberX)
- Weather conditions (don't overrepresent sunny days)

```python
# Example stratification for base price model
for zone in zones:
  for hour in [0-6, 7-12, 13-18, 19-23]:
    for weather in [clear, rain, snow]:
      subset = data[
        (data.zone == zone) &
        (data.hour in hour_range) &
        (data.weather == weather)
      ]
      # Allocate 80% to train, 20% to test
      train_subset = subset.sample(frac=0.8)
      test_subset = subset.drop(train_subset.index)
```

### 3.5 Data Size Requirements

**Minimum recommended:**
- **Base Price Model**: 1M+ rides (covers seasonal patterns, edge cases)
- **Surge Model**: 100K+ zone-hour combinations (allows learning temporal patterns)

**Ideal:**
- **Base Price**: 10M+ rides (covers all zones, ride types, seasons)
- **Surge Model**: 1M+ zone-hour samples (deep historical patterns)

---

## 4. Model Selection & Architecture

### 4.1 Candidate Models

#### For Base Price Model

| Model | Latency | Accuracy | Interpretability | Training Time |
|-------|---------|----------|------------------|---------------|
| **Linear Regression** | <1ms | Low | High | 10s |
| **Decision Tree** | <1ms | Medium | High | 1min |
| **Random Forest** | 10-50ms | High | Medium | 30min |
| **Gradient Boosting (XGBoost)** | 10-50ms | Very High | Medium | 1-2hrs |
| **Neural Network (1-2 layers)** | 5-10ms | High | Low | 1-2hrs |
| **Deep Neural Network** | 20-100ms | Very High | Very Low | 5-10hrs |

#### For Surge Pricing Model

| Model | Latency | Accuracy | Pros | Cons |
|-------|---------|----------|------|------|
| **Linear (demand_ratio)** | <1ms | Medium | Fast, interpretable | Too simplistic |
| **Polynomial Regression** | <1ms | Medium-High | Fast, captures non-linearity | Limited |
| **Tree-based (XGBoost)** | 10-50ms | Very High | Handles interactions, robust | More complex |
| **LSTM/RNN** | 50-200ms | High | Captures temporal patterns | Slower, harder to deploy |
| **Ensemble (Gradient Boosting + Linear)** | 10-50ms | Very High | Best of both | Complexity |

### 4.2 Recommended Models

#### Primary: Gradient Boosting (XGBoost or LightGBM)

**Why:**
- ✅ Excellent accuracy (captures non-linear relationships)
- ✅ Fast inference (<50ms, meets <500ms requirement)
- ✅ Handles categorical features natively
- ✅ Robust to outliers
- ✅ Interpretable via SHAP values
- ✅ Proven in production systems

**Architecture:**
```
Base Price Model:
  - 100-200 trees
  - Max depth: 5-7
  - Learning rate: 0.05-0.1
  - Feature importance: distance > time > zone > hour
  - Inference: ~10-20ms

Surge Model:
  - 150-250 trees
  - Max depth: 6-8
  - Learning rate: 0.05
  - Feature importance: requests_per_driver > hour > weather
  - Inference: ~20-30ms
```

#### Secondary: Linear Model with Feature Engineering

**Use Case:** As fallback when serving infrastructure fails

```
base_price = w0 + w1×distance + w2×duration + w3×traffic_score +
             w4×zone_multiplier + w5×time_multiplier + interaction_terms

surge = 1.0 + w1×log(requests_per_driver) + w2×hour_spike + w3×weather_factor
```

**Advantages:**
- <1ms inference latency
- Highly interpretable
- Easy to debug and update
- Can deploy without ML infrastructure

#### Optional: Neural Network for Scaling

**When to use:** If serving millions of requests/second becomes bottleneck

```
Architecture (base price):
  Input layer: 40 features (categorical embedded)
  Hidden 1: 128 units, ReLU, Dropout(0.2)
  Hidden 2: 64 units, ReLU, Dropout(0.2)
  Output: 1 (price, ReLU activation)

Latency: 5-10ms (GPU), 20-50ms (CPU)
```

---

## 5. Model Training Pipeline

### 5.1 Training Data Processing

```python
# Pseudocode for training data pipeline

def prepare_training_data(historical_rides, zone_demand_metrics):
    """
    Prepare features and labels from historical data
    """

    # 1. Load and validate data
    rides = load_and_validate(historical_rides)
    metrics = load_and_validate(zone_demand_metrics)

    # 2. Feature engineering
    features = engineer_features(rides, metrics)

    # 3. Label creation
    labels_base_price = create_base_price_labels(rides)
    labels_surge = create_surge_labels(metrics)

    # 4. Temporal split
    train, val, test = temporal_split(
        features, labels,
        train_end='2024-08-31',
        val_end='2024-09-30'
    )

    # 5. Normalization/scaling
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train.features)
    val_features = scaler.transform(val.features)
    test_features = scaler.transform(test.features)

    # 6. Address class imbalance (for surge model)
    # If most trips have 1.0x surge, oversample high-surge samples
    train = balance_dataset(train, by='surge_bin')

    return train, val, test, scaler, feature_names
```

### 5.2 Hyperparameter Tuning

```python
# Bayesian optimization for XGBoost hyperparameters

param_space = {
    'max_depth': (4, 10),
    'learning_rate': (0.01, 0.2),
    'num_rounds': (100, 500),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
}

# Optimize for MAPE on validation set
best_params = bayesian_optimization(
    objective=train_xgboost,
    param_space=param_space,
    n_trials=50,
    metric='validation_mape'
)
```

### 5.3 Training Frequency

**Offline Model Retraining Schedule:**
- **Base Price Model**: Weekly (capture seasonal patterns)
  - Retrain every Sunday with last 6 months of data
  - A/B test with canary deployment (1% traffic first)

- **Surge Model**: Daily (demand patterns change rapidly)
  - Retrain every day at 2am UTC with last 30 days of data
  - Rapid deployment (within 30 min if metrics improve)

---

## 6. Feature Preprocessing & Scaling

### 6.1 Categorical Features

**Method: Target Encoding + Smoothing**
```
For categorical feature (e.g., zone_id):
  1. Compute mean price per zone from training data
  2. Add smoothing: encoded_value = (count × mean + global_mean) / (count + smoothing_factor)
  3. Use encoded values as input to model
```

**Why target encoding:**
- Better for tree models than one-hot encoding
- Reduces dimensionality
- Captures relationship between category and target

### 6.2 Numerical Features

**Scaling:**
- **For tree models (XGBoost)**: No scaling needed
- **For neural networks**: StandardScaler or MinMaxScaler
- **For linear models**: StandardScaler (critical for interpretation)

```python
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

### 6.3 Missing Value Handling

**Strategy:**
- Distance/duration: Impute with routing API calls
- Weather: Impute with nearest non-missing time window
- Traffic: Impute with historical average for that zone-time
- Never just drop rows - you'll lose valuable data

---

## 7. Model Inference & Serving

### 7.1 Inference Pipeline (per ride request)

```
Ride Request
    ↓
[Feature Extraction] (10ms)
  - Calculate distance/duration (routing API)
  - Lookup zone tier
  - Get current demand/supply metrics
  - Encode categorical features
    ↓
[Model Inference] (20-40ms)
  - Load XGBoost model from memory
  - Run prediction on features
    ├── Base price prediction
    └── Surge multiplier prediction
    ↓
[Post-processing] (5-10ms)
  - Apply business rules
  - Round to nearest $0.50
  - Apply minimum/maximum bounds
  - Format response
    ↓
[Return Price] (Total: 300-400ms)
```

### 7.2 Model Serving Architecture

**Recommended Setup:**
```
API Gateway
    ↓
Load Balancer (round-robin)
    ├── Server 1 [Model in memory]
    ├── Server 2 [Model in memory]
    └── Server 3 [Model in memory]

Per server:
  - Load model once at startup (~1s)
  - Keep in memory for <1μs access time
  - Use ONNX Runtime for faster inference
```

**Deployment:**
- Use model versioning (model_v1, model_v2)
- Blue-green deployment for zero-downtime updates
- Fallback to rule-based pricing if model fails

### 7.3 Caching Strategy

```
Cache demand-supply metrics (update every 30s):
  zone_1_demand: 150 requests
  zone_1_supply: 45 drivers
  zone_1_surge: 1.8x

  [Reduces need to recompute metrics per request]

Cache base fare templates (update daily):
  zone_1_hour_7_distance_10: $18.50 base
  zone_1_hour_8_distance_10: $20.00 base

  [Approximate price if exact features not available]
```

---

## 8. Model Evaluation & Monitoring

### 8.1 Offline Evaluation

**Metrics:**
```
For Base Price Model:
  - MAPE: Mean absolute % error (target: <5%)
  - RMSE: Root mean squared error
  - p90_error: Error at 90th percentile
  - By zone: Ensure <8% MAPE in each zone
  - By ride_type: Ensure <7% MAPE per type

For Surge Model:
  - MAPE: (target: <10%)
  - MAE: Mean absolute error in multiplier (target: <0.15x)
  - Correlation with demand ratio: >0.7
  - Directional accuracy: Predict up/down correctly >75% of time
```

### 8.2 Online Monitoring

**Real-time Metrics (dashboard):**
```
Base Price Model:
  - Prediction error (actual vs predicted)
  - Latency (p50, p95, p99)
  - Feature distribution drift
  - Model staleness (time since retrain)

Surge Model:
  - Revenue impact (vs rule-based baseline)
  - Driver supply response (does higher surge attract drivers?)
  - User acceptance rate (do users accept predicted prices?)
  - Price fairness: Coefficient of variation <15%
```

### 8.3 Drift Detection

**Monitor for:**
- **Data drift**: Feature distributions changing
- **Label drift**: Price patterns changing (competitive pressure)
- **Concept drift**: Relationships between features and price changing

**Trigger retraining if:**
- MAPE on recent data >6% (base price)
- MAPE on recent data >12% (surge)
- KL divergence of features >threshold
- Demand-supply ratio distribution shifted >20%

---

## 9. Model Interpretability & Debugging

### 9.1 Feature Importance (SHAP)

```python
import shap

# Get SHAP values for test set
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)  # Overall importance
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])  # Per-prediction
```

**Insights from SHAP:**
```
Base Price Model Top Features:
  1. distance: +0.8 correlation with price
  2. estimated_duration: +0.6
  3. is_airport: +0.45 (premium)
  4. zone_multiplier: +0.4
  5. traffic_score: +0.25

Surge Model Top Features:
  1. requests_per_driver ratio: +0.9
  2. hour_of_day: +0.5
  3. day_of_week: +0.3
  4. weather: +0.2
```

### 9.2 Debugging Price Anomalies

**When price seems wrong:**
```
1. Check features (are they correct?)
2. Check SHAP values (is model behaving as expected?)
3. Analyze similar historical rides (what was typical price?)
4. Compare to rule-based baseline
5. Check demand/supply metrics (are they stale/wrong?)
```

---

## 10. A/B Testing & Deployment

### 10.1 A/B Test Design

**Test new surge model:**
```
Control (10%): Old model (current production)
Treatment (10%): New model
No experiment (80%): Keep current model

Duration: 1-2 weeks
Primary metric: Revenue per ride
Secondary metrics: Driver supply, user acceptance rate, user satisfaction
```

### 10.2 Canary Deployment

```
Day 1: Deploy to 1% of traffic
  - Monitor for errors, latency spikes, anomalies

Day 2: Deploy to 5% of traffic
  - A/B test statistical significance

Day 3+: Deploy to 100% if successful
  - Or rollback if metrics regress
```

---

## Summary

**Recommended approach:**
- **Model**: XGBoost for both base price and surge
- **Features**: 40-50 most important features (distance, demand ratio, time, location, weather)
- **Training**: Weekly base price, daily surge model retraining
- **Latency**: <50ms per model (total request <500ms with features)
- **Accuracy**: <5% MAPE for base price, <10% MAPE for surge
- **Fallback**: Rule-based pricing if models unavailable
- **Monitoring**: Real-time dashboard for drift and performance
