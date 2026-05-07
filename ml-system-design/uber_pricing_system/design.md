# Uber Pricing System Design

## Problem Statement

Design a dynamic pricing system for a ride-sharing platform (Uber) that determines optimal prices in real-time. The system must balance multiple objectives: maximizing platform revenue, ensuring driver availability and earnings, maintaining driver-rider fairness, and responding to market demand fluctuations.

The pricing system should:
- Calculate fares in real-time when a user requests a ride
- Adjust prices based on demand, supply, and market conditions
- Be transparent to users while remaining competitive
- Incentivize drivers to accept rides and maintain availability
- Optimize for both short-term revenue and long-term platform health

## System Requirements

### Functional Requirements

1. **Real-time Price Calculation**
   - Calculate fare within <500ms for user requests
   - Support multiple ride types (UberX, UberXL, Uber Eats, etc.)
   - Provide upfront pricing for customers

2. **Dynamic Adjustment**
   - Adjust prices based on real-time demand and supply
   - Implement surge pricing during peak hours
   - Support location-based pricing variations
   - Handle time-of-day pricing patterns

3. **Driver Incentive System**
   - Offer surge pricing to attract drivers to high-demand areas
   - Balance driver earnings with customer affordability
   - Support peak hour bonuses and promotions

4. **Pricing Components**
   - Base fare
   - Per-mile charge
   - Per-minute charge
   - Service fees and platform fees
   - Surge/multiplier pricing
   - Promotions and discounts

5. **Transparency & Fairness**
   - Show price estimates before trip confirmation
   - Provide price breakdowns
   - Consistent pricing across similar rides
   - Prevent price discrimination

### Non-Functional Requirements

1. **Scale**
   - Support millions of concurrent ride requests
   - Handle 100K+ pricing requests per second
   - Process across multiple cities/regions

2. **Performance**
   - Price calculation latency: <500ms (p99)
   - 99.99% availability
   - Handle 10x traffic spikes

3. **Consistency & Fairness**
   - Two users on similar routes at same time see similar prices (within tolerance)
   - Smooth price transitions (avoid sudden jumps)
   - Prevent arbitrage opportunities

4. **Elasticity & Optimization**
   - Maximize total platform revenue
   - Maintain optimal supply-demand balance
   - Maximize driver utilization
   - Minimize customer churn

## High-Level Approach

### 1. Pricing Model Architecture

```
Ride Request
    ↓
[Load Balancer]
    ↓
[Pricing Service]
    ├── Real-time Data Layer
    │   ├── Current demand (active requests)
    │   ├── Available supply (driver locations)
    │   └── Traffic conditions
    ├── Feature Engineering
    │   ├── Demand score
    │   ├── Supply score
    │   ├── Temporal features
    │   └── Location features
    ├── ML Model Layer
    │   ├── Surge pricing model
    │   ├── Base price predictor
    │   └── Demand forecaster
    ├── Business Rules Engine
    │   ├── Fairness constraints
    │   ├── Promotion rules
    │   └── Price bounds
    └── Cache Layer (Redis)
        ├── Surge multipliers by zone
        └── Base fares
    ↓
[Price Response]
```

### 2. Key Components

**Demand-Supply Metrics**
- Request rate per zone
- Available driver count per zone
- Wait time for users
- Estimated driver earning opportunity

**Surge Pricing Model**
- Calculate surge multiplier (1.0x - 5.0x+)
- Account for: demand intensity, supply scarcity, temporal factors
- Apply smoothing to prevent wild price swings

**Base Price Predictor**
- Estimate trip duration and distance
- Calculate per-mile and per-minute rates
- Include service fees and platform costs

**Demand Forecaster**
- Predict demand peaks (events, weather, time)
- Pre-position drivers and adjust pricing proactively
- Handle unexpected demand spikes

### 3. Key Data Flows

**Online (Real-time Pricing)**
- User submits ride request (pickup, dropoff, ride type)
- System calculates trip characteristics (estimated distance, duration)
- Fetch current demand-supply metrics
- Generate features for ML model
- Model predicts base price and surge multiplier
- Apply business rules and constraints
- Return final price to user

**Offline (Model Training & Optimization)**
- Collect historical ride data (prices, demand, supply, outcomes)
- Analyze price elasticity and market dynamics
- Train demand forecasting models
- Optimize pricing parameters
- A/B test new pricing strategies
- Deploy updated models

## Detailed Design

### Pricing Components

**1. Trip Estimation Module**
- Input: pickup location, dropoff location, ride type
- Output: estimated distance, duration, base fare
- Uses: maps API, historical routing data, traffic patterns

**2. Surge Pricing Engine**
- Calculate demand-supply ratio per zone
- Map ratio to surge multiplier using ML model
- Apply temporal adjustments (peak hours, events)
- Formula: `surge_multiplier = f(demand_ratio, time, location, day)`

**3. Dynamic Base Fare**
- Adjust base fare based on:
  - Trip type and distance
  - Location tier (city center, suburbs, rural)
  - Time of day (peak vs. off-peak)
  - Capacity constraints

**4. Revenue Optimization**
- Objective: maximize total platform revenue = rides × fare
- Constraint: fair pricing, driver earnings, customer satisfaction
- Approach: price elasticity modeling
  - Estimate demand curve: how price affects acceptance rate
  - Find price that maximizes revenue × driver supply

**5. Fairness & Constraints**
- Similar trips → similar prices (within 5-10%)
- Price bounds: [min_price, max_price]
- Temporal smoothing: avoid sudden price jumps >20%
- Prevent cross-subsidization: don't overprice specific groups

### Demand Metrics & Scoring

**Supply Score**
- Available driver count in zone
- Average driver wait time
- Driver utilization rate
- Recent acceptance rate

**Demand Score**
- Active ride requests
- Request rate (requests/minute)
- Estimated pickup wait time
- Cancellation rate

**Surge Formula**
```
demand_ratio = demand_score / supply_score
surge = 1.0 + k * log(demand_ratio)  // smooth scaling
surge = clamp(surge, 1.0, max_surge)  // bounds
```

### Pricing Strategy by Scenario

**Off-Peak (Supply > Demand)**
- Base price only (1.0x)
- May offer discounts to stimulate demand
- Focus on utilization

**Peak (Demand ≈ Supply)**
- Small surge (1.1x - 1.5x)
- Balance customer acquisition with driver incentives
- Smooth price transitions

**High-Demand (Supply << Demand)**
- Significant surge (2.0x - 3.0x+)
- Maximize driver earnings to attract more supply
- May reach maximum surge limits

**Emergency (Extreme shortage)**
- Capped surge (e.g., 5.0x max)
- Proactive driver recruitment
- Pre-booking incentives

## Data Models

### Pricing Request
```
ride_id
user_id
pickup_location (lat, lng, zone_id)
dropoff_location (lat, lng, zone_id)
ride_type (UberX, UberXL, Uber Eats, etc.)
requested_time
device_type
user_rating
user_history (payment method, churn risk)
```

### Pricing Response
```
ride_id
estimated_distance
estimated_duration
base_fare
per_mile_charge
per_minute_charge
surge_multiplier
service_fee
platform_fee
estimated_total_fare
price_breakdown
currency
```

### Demand-Supply Metrics (per zone, per 10-30 seconds)
```
zone_id
timestamp
active_requests
available_drivers
request_rate
acceptance_rate
avg_pickup_wait_time
avg_driver_earnings_rate
cancellation_rate
```

### Historical Ride Data
```
ride_id
pickup_location
dropoff_location
actual_distance
actual_duration
quoted_price
actual_price (after surge)
surge_multiplier
ride_type
user_id
driver_id
acceptance_time
completion_time
user_rating
driver_rating
demand_score_at_request
supply_score_at_request
timestamp
```

## Key Design Challenges

### 1. Balancing Multiple Objectives
- **Revenue maximization** vs. **customer fairness**
- **Driver supply incentives** vs. **customer affordability**
- **Short-term revenue** vs. **long-term platform health**

### 2. Fairness & Discrimination
- Similar users on similar routes should see similar prices
- Avoid systematic overcharging of groups
- Transparent pricing methodology

### 3. Price Elasticity & Dynamics
- User acceptance drops as price increases
- Need to model demand curve accurately
- Adaptive pricing based on acceptance rates

### 4. Driver Supply Response
- Drivers respond to earning opportunities
- Need to forecast driver supply changes
- Incentivize strategic repositioning

### 5. Market Dynamics & Competition
- Competitor pricing affects demand
- User switching behavior
- Network effects and liquidity

### 6. Regulatory Constraints
- Price gouging regulations
- Transparency requirements
- Labor laws and driver classification

## Scalability Considerations

1. **Caching Strategy**
   - Cache zone-level surge multipliers (update every 30s)
   - Cache base fares by route types
   - Reduce real-time computation load

2. **Distributed Processing**
   - Partition pricing by geographic zones
   - Parallel model inference
   - Real-time streaming for metrics updates

3. **Data Pipeline**
   - Stream processing for demand-supply metrics
   - Batch processing for historical analysis
   - Model training pipeline with scheduled retraining

4. **Database Design**
   - Time-series DB for demand-supply metrics
   - Cache layer (Redis) for hot data
   - Data warehouse for analytics and training

5. **Model Serving**
   - Multiple model replicas for high throughput
   - Model A/B testing framework
   - Fallback to rule-based pricing if models fail
