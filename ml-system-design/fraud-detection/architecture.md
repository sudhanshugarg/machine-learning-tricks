# Fraud Detection System - Architecture

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT/MERCHANT                              │
│                    (Payment/Transaction Request)                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   API Gateway / LB      │
                    │  (Rate Limiting)        │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼─────────┐    ┌─────────▼──────────┐   ┌────────▼────────┐
│ Feature Store   │    │ Decision Service   │   │ Model Service   │
│ (Online)        │    │ (Orchestration)    │   │ (Model Scoring) │
│                 │    │                    │   │                 │
│ - Redis/DynamoDB   │    │ - Fraud Rules      │   │ - XGBoost Model │
│ - Real-time lag    │    │ - Risk Scoring     │   │ - v1, v2, v3... │
│   features          │    │ - Thresholding     │   │ - < 20ms latency│
└─────────────────┘    │ - Logging/Audit    │   │                 │
                       └────────┬───────────┘   └────────┬────────┘
                                │                        │
                                │ (async telemetry)      │
                    ┌───────────┴────────────┐           │
                    │  Telemetry Pipeline    │◄──────────┘
                    │  (Kafka/Pub-Sub)       │
                    └───────────┬────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼──────────┐  ┌─────────▼─────────┐  ┌─────────▼────────┐
│  Data Lake       │  │  Metrics Store    │  │  Monitoring &    │
│  (Event Log)     │  │  (Time Series)    │  │  Alerting        │
│                  │  │                   │  │                  │
│ - S3 / GCS       │  │ - Prometheus      │  │ - DataDog /      │
│ - Transactions   │  │ - Model metrics   │  │   Grafana        │
│ - Events         │  │ - System metrics  │  │ - Alerting rules │
└──────┬───────────┘  └─────┬─────────────┘  └──────────────────┘
       │                    │
       │ (historical data)  │
       │                    │
┌──────▼────────────────────▼────────┐
│   Offline ML Pipeline              │
│   (Batch Processing)               │
│                                    │
│ 1. Feature Generation (daily)      │
│    - Compute aggregated features   │
│    - Update feature store          │
│                                    │
│ 2. Label Aggregation (7+ days)     │
│    - Wait for delayed labels       │
│    - Aggregate fraud signals       │
│                                    │
│ 3. Model Training (daily/weekly)   │
│    - XGBoost training              │
│    - Hyperparameter tuning         │
│    - Validation on test set        │
│                                    │
│ 4. Model Evaluation                │
│    - Offline metrics               │
│    - Canary & A/B comparison       │
│                                    │
│ 5. Model Promotion                 │
│    - Shadow traffic testing        │
│    - Canary rollout (5% → 100%)    │
│                                    │
└────────────────────────────────────┘
```

---

## Component Details

### 1. API Gateway & Request Routing

**Purpose**: Handle incoming transaction requests, route to decision service.

**Responsibilities**:
- Rate limiting per customer/merchant
- Request validation and sanitization
- Load balancing across decision service instances
- Timeout handling (fail-open or fail-closed based on config)

**Implementation Considerations**:
- Use proven framework (Kong, AWS API Gateway, Envoy)
- Implement circuit breaker pattern
- Log all requests for audit trail

---

### 2. Feature Store (Online)

**Purpose**: Serve real-time and precomputed features with low latency.

**Components**:
- **Online Store** (Redis/DynamoDB): Sub-millisecond feature lookups
- **Batch Store** (BigQuery/Snowflake): Historical features, aggregations
- **Feature Registry**: Feature definitions, computation logic, versions

**Features Served**:
- **Real-time computed**: 
  - Velocity features (transactions in last 1h, 24h)
  - Device consistency (is this device new?)
  - Location consistency (distance from last txn)
- **Precomputed daily**:
  - Account-level statistics (avg amount, std dev)
  - Device history (devices used, switching frequency)
  - Merchant affinity (favorite merchants)

**Latency Requirement**: < 20ms for full feature vector lookup

**Tools**: Feast, Tecton, Hopsworks, or custom solution

---

### 3. Decision Service (Orchestration)

**Purpose**: Orchestrate the fraud decision pipeline.

**Flow**:
```
Request → Validate → Lookup Features → Score Model → Apply Rules → Decide
         ↓ (if fails) → Fallback Decision
```

**Components**:
- **Input Validation**: Check required fields, data types
- **Feature Lookup**: Call feature store for feature vector
- **Model Scoring**: Call model service with features
- **Post-Processing Rules**:
  - Manual review list (known high-risk merchants)
  - Whitelist trusted users/devices
  - Geographic restrictions
- **Decision Logic**:
  - Score > 0.8 → BLOCK
  - Score 0.5-0.8 → CHALLENGE (require 2FA, additional verification)
  - Score < 0.5 → ALLOW
  - Adjust thresholds based on business rules
- **Fallback**: If model service unavailable, use rule-based decision

**Output**: Decision (ALLOW, CHALLENGE, BLOCK, REVIEW)

**Latency Budget**: < 100ms end-to-end

---

### 4. Model Service

**Purpose**: Load and serve the fraud detection model.

**Architecture**:
```
┌─────────────────────────────────────────┐
│         Model Service (Singleton)       │
├─────────────────────────────────────────┤
│ Model v1 (80% traffic)                  │
│ Model v2 (15% traffic - canary)         │
│ Model v3 (5% traffic - shadow)          │
└─────────────────────────────────────────┘
       ↓ (feature vector)
   Inference
       ↓
   Risk Score [0, 1]
```

**Key Responsibilities**:
- Load trained XGBoost/LightGBM model
- Serve predictions with < 20ms latency
- Handle model versioning and canary rollouts
- Track prediction metrics per model version

**Implementation**:
- Language: Python/Java/C++ for speed
- Framework: Seldon, KServe, BentoML, or custom
- Containerization: Docker with resource limits
- Autoscaling: Scale replicas based on latency/throughput

**Model Serving Strategy**:
- **Canary Deployment**: Route small % of traffic to new model
- **Shadow Mode**: Log predictions from new model without affecting decisions
- **A/B Testing**: Compare models on statistically significant sample
- **Rollback**: Instant switch back if metrics degrade

---

### 5. Telemetry Pipeline (Kafka / Pub-Sub)

**Purpose**: Collect events for audit, analytics, and model retraining.

**Events Captured**:
- Transaction request (features, model score, decision)
- Decision outcome (ALLOW, BLOCK, CHALLENGE)
- Delayed feedback (fraud label when it arrives hours later)
- Model metrics (predictions, probabilities, latencies)

**Data Flow**:
```
Decision Service → Kafka Topic → {Data Lake, Metrics Store, Real-time Processors}
```

**Topics**:
- `transactions-v1`: All transaction requests
- `decisions-v1`: Fraud decisions with scores
- `feedback-v1`: Delayed fraud labels (fraud, chargeback, dispute signals)
- `metrics-v1`: System metrics and performance

---

### 6. Data Lake (S3 / GCS)

**Purpose**: Long-term storage of all events for analytics, auditing, and retraining.

**Structure**:
```
s3://fraud-detection/
├── transactions/
│   └── YYYY/MM/DD/HH/
│       └── transactions-*.parquet
├── feedback/
│   └── YYYY/MM/DD/
│       └── feedback-*.parquet
├── decisions/
│   └── YYYY/MM/DD/
│       └── decisions-*.parquet
└── models/
    └── {model_name}/{version}/
        └── model.pkl, metadata.json
```

**Retention Policy**: 
- Raw events: 2-3 years (compliance, auditing)
- Aggregated features: indefinite
- Models: All versions kept for rollback

---

### 7. Metrics Store (Prometheus / Time-Series DB)

**Purpose**: Store metrics for monitoring and alerting.

**Key Metrics**:

#### Model Metrics
- `fraud_detection_model_precision`: Precision per model version
- `fraud_detection_model_recall`: Recall per model version
- `fraud_detection_model_auc`: ROC-AUC score
- `fraud_detection_predictions_per_minute`: Throughput

#### System Metrics
- `decision_service_latency_p50/p99`: Request latency percentiles
- `feature_store_latency_p50/p99`: Feature lookup latency
- `model_service_latency_p50/p99`: Model scoring latency
- `request_throughput`: Requests per second

#### Business Metrics
- `fraud_catch_rate`: % of actual fraud caught
- `false_positive_rate`: % of legitimate txns blocked
- `fraud_prevented_revenue`: $$ saved by blocking fraud
- `customer_friction`: Support tickets related to fraud blocks

---

### 8. Monitoring & Alerting (Grafana, DataDog)

**Purpose**: Real-time visibility into system health and model performance.

**Dashboards**:
1. **Real-time Dashboard**: Latency, throughput, error rates
2. **Model Performance**: Predictions, scores, decisions per hour
3. **Fraud Metrics**: Fraud rate, catch rate, false positive rate
4. **Data Quality**: Feature freshness, missing features, distribution drift

**Alerts**:
- Latency > 100ms (P99): Check model service or feature store
- Model prediction drift: Alert if fraud prediction rate > 2x baseline
- Feature distribution drift: Alert if features shift > 2 std dev
- Fraud catch rate degradation: If < 95% for 1 hour, page on-call

---

### 9. Offline ML Pipeline (Batch Processing)

**Purpose**: Daily/weekly training and evaluation of new models.

#### 9a. Feature Generation (Daily)
```
┌──────────────────────────────────────────┐
│ Feature Generation Pipeline (daily)      │
├──────────────────────────────────────────┤
│ 1. Read raw transaction events from lake │
│ 2. Compute aggregated features:          │
│    - 1-day, 7-day, 30-day windows        │
│ 3. Join with static features (account)   │
│ 4. Output feature table → Feature Store  │
│                                           │
│ Output: Feature table (users, features)  │
└──────────────────────────────────────────┘
```

**Features Generated**:
- Velocity: txn count and amount in time windows
- Behavioral: avg amount, std dev, percentiles
- Device: number of devices, switching frequency
- Merchant: favorite merchants, category distribution

#### 9b. Label Aggregation (7+ Days Delay)
```
┌───────────────────────────────────────────┐
│ Label Aggregation (run 7+ days after txn) │
├───────────────────────────────────────────┤
│ 1. Read feedback events (fraud labels)    │
│ 2. Join with transactions                 │
│ 3. Create target variable (fraud=1 or 0)  │
│    - Fraud: chargeback, customer dispute  │
│    - Legitimate: no fraud signal in 7 days│
│ 4. Output labeled dataset                 │
└───────────────────────────────────────────┘
```

**Label Sources**:
- Chargebacks (customer disputes transaction)
- Fraud reports (customer reports unauthorized txn)
- Manual investigation (fraud analyst confirms)
- After 7+ days: Assume legitimate if no signals

#### 9c. Model Training (Daily/Weekly)
```
┌──────────────────────────────────────────┐
│ Model Training Pipeline                  │
├──────────────────────────────────────────┤
│ 1. Read labeled training dataset          │
│ 2. Stratified split (train/val/test)     │
│    - Train: Last 30 days                 │
│    - Val: Last 7 days                    │
│    - Test: Last 2 days (freshest data)   │
│ 3. Handle class imbalance:               │
│    - Undersample legitimate (0 class)    │
│    - Oversample fraud (1 class)          │
│    - Class weight = len(neg) / len(pos)  │
│ 4. Train XGBoost model                   │
│    - Hyperparameters: learning_rate=0.1  │
│    - max_depth=6, num_rounds=500         │
│    - Sample weights = class weights      │
│ 5. Evaluate on validation set            │
│ 6. If better than baseline: promote      │
└──────────────────────────────────────────┘
```

**Hyperparameter Tuning**:
- Grid search or Bayesian optimization
- Optimize for PR-AUC (better for imbalanced data)
- Constraint: Model latency < 20ms on target hardware

#### 9d. Model Evaluation & Comparison
```
┌──────────────────────────────────────────┐
│ Model Evaluation & Comparison            │
├──────────────────────────────────────────┤
│ New Model vs Current Production:         │
│ - Precision, Recall, F1-Score, ROC-AUC   │
│ - Business metrics (fraud caught, FP)    │
│                                           │
│ If new model wins:                       │
│  → Promote to canary (5% traffic)        │
│  → Monitor for 24h                       │
│  → If good: Canary → Prod (100% traffic) │
│  → Keep old model for quick rollback     │
└──────────────────────────────────────────┘
```

#### 9e. Canary Deployment
```
┌──────────────────────────────────────────┐
│ Canary Deployment Strategy               │
├──────────────────────────────────────────┤
│ Stage 1: Shadow (0% decision impact)     │
│  - Run new model, log predictions        │
│  - Compare with production model         │
│  - 24 hours                              │
│                                           │
│ Stage 2: Canary (5% traffic)             │
│  - New model makes actual decisions      │
│  - Monitor metrics closely               │
│  - 24 hours                              │
│                                           │
│ Stage 3: Gradual Rollout                 │
│  - 5% → 25% → 50% → 100%                │
│  - 24 hours per stage                    │
│  - Alert if recall/precision degrades    │
│                                           │
│ Rollback: Instant switch if issues       │
└──────────────────────────────────────────┘
```

---

## Data Flow Diagram

### Online Request Path (Low Latency)
```
1. API Gateway receives transaction request (< 10ms)
   ↓
2. Decision Service validates request (< 5ms)
   ↓
3. Feature Store lookup (real-time + precomputed) (< 20ms)
   ↓
4. Model Service scores transaction (< 20ms)
   ↓
5. Decision Service applies business rules (< 5ms)
   ↓
6. Response sent to client (ALLOW/BLOCK/CHALLENGE)
   ↓
7. Async: Event logged to Kafka (doesn't block response)

Total: < 100ms P99
```

### Offline Feedback Loop (Hours to Days)
```
1. Fraud feedback event (chargeback, dispute) arrives
   ↓
2. Event logged to Kafka `feedback-v1` topic
   ↓
3. Batch job (runs daily): Join feedback with historical transactions
   ↓
4. Generate labeled dataset
   ↓
5. Train new model
   ↓
6. Evaluate and compare against production
   ↓
7. If better: Deploy canary → prod
```

---

## Scalability Considerations

### Handling High Throughput (10k+ RPS)

**Decision Service**:
- Stateless design (scale horizontally)
- Load balance across replicas
- Cache frequently accessed features
- Async logging (non-blocking)

**Feature Store**:
- Redis cluster for online serving
- Distributed cache with replication
- Warm cache with popular features
- Fallback to default values for missing features

**Model Service**:
- Model loaded in memory (fast inference)
- Batch inference where possible
- GPU acceleration for large models
- Replicas scale based on latency/throughput

### Handling Failures

**Feature Store Unavailable**:
- Use cached features (stale but better than nothing)
- If cache miss: Use default values (e.g., 0 for velocity features)
- Return CHALLENGE decision (require additional verification)

**Model Service Unavailable**:
- Fallback to rule-based scoring
- Route to manual review queue
- Page on-call to restore service

**Kafka Unavailable**:
- Buffer events in local queue
- Flush to Kafka when recovered
- Don't block decision (async logging)

---

## Monitoring & Alerting Rules

### Critical Alerts
1. **Model latency > 100ms P99**: Page on-call (user experience impact)
2. **Model unavailable**: Page on-call (decisions blocked)
3. **Feature store unavailable**: Page on-call (features unavailable)
4. **Fraud catch rate < 95%**: Page on-call (potential model degradation)

### Warning Alerts
1. **Feature distribution drift > 2 std dev**: Investigate, may need retraining
2. **Prediction distribution change**: New fraud patterns emerging?
3. **False positive rate > 2%**: Customer friction increasing

---

## Next Steps

See **tradeoffs.md** for design tradeoff discussions and **template.py** for implementation starter code.
