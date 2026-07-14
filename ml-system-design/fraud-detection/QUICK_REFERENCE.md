# Fraud Detection System - Quick Reference

## System at a Glance

```
Request → API Gateway → Feature Store → Model → Decision Service → Response
                              ↓
                       Kafka Event Log
                              ↓
                    Data Lake + Metrics Store
                              ↓
                    Daily Batch Training Pipeline
```

---

## Key Numbers

| Metric | Target | Notes |
|--------|--------|-------|
| Latency | < 100ms P99 | End-to-end request time |
| Throughput | 10k+ RPS | Requests per second |
| Model AUC | > 95% | Offline evaluation metric |
| Fraud Catch Rate | > 95% | % of actual fraud blocked |
| False Positive Rate | < 1% | % of legit txns blocked |
| Model Training Time | < 1 hour | Daily retraining |
| Label Delay | 7+ days | Training data freshness |
| Feature Lookup | < 20ms | Feature store latency |
| Model Scoring | < 20ms | XGBoost inference |

---

## Components Checklist

### Online (Request Path)
- [ ] API Gateway (rate limiting, load balancing)
- [ ] Feature Store (Redis/DynamoDB for fast lookup)
- [ ] Decision Service (orchestration)
- [ ] Model Service (XGBoost scoring)
- [ ] Fallback strategy (use rules if model unavailable)

### Offline (Batch Pipeline)
- [ ] Data Lake (S3/GCS for event storage)
- [ ] Feature Generation (daily batch)
- [ ] Label Aggregation (wait 7+ days)
- [ ] Model Training (XGBoost)
- [ ] Model Evaluation (compare to baseline)
- [ ] Canary Deployment (5% traffic first)

### Observability
- [ ] Metrics Store (Prometheus/time-series DB)
- [ ] Logging (Kafka for event streaming)
- [ ] Monitoring (latency, throughput, errors)
- [ ] Alerting (model degradation, feature drift)
- [ ] Dashboards (real-time + historical analysis)

---

## Decision Tiers

| Score | Decision | Action |
|-------|----------|--------|
| < 0.3 | ALLOW | Process normally |
| 0.3-0.7 | CHALLENGE | Require 2FA or additional verification |
| 0.7-0.9 | BLOCK | Decline transaction, allow manual appeal |
| > 0.9 | REVIEW | Queue for manual investigation |

---

## Design Decisions

### Model: XGBoost (not neural networks)
- ✅ 20ms latency (beats P99 budget)
- ✅ 95%+ AUC (sufficient accuracy)
- ✅ Interpretable (can explain decisions)
- ❌ Not as powerful as deep networks

### Features: Hybrid (real-time + batch)
- ✅ Real-time: Velocity, device consistency (can't precompute)
- ✅ Batch: Historical aggregates (computed daily)
- ✅ Combined: Fresh signals + low latency

### Training: Daily batch retraining
- ✅ Clean, stable models
- ✅ Easy to debug and rollback
- ✅ Handles delayed labels
- ❌ Slower to adapt to new patterns

### Deployment: Canary rollout
1. Shadow (0% impact): 24h
2. Canary (5% traffic): 24h
3. Gradual (5% → 25% → 50% → 100%): 24h per stage
4. Full (100% traffic)

---

## Key Interview Points

### 1. Explain Latency Budget
```
API Gateway      (5-10ms)
Feature Lookup   (10-20ms)
Model Scoring    (10-20ms)
Decision Logic   (5-10ms)
Response         (10-20ms)
─────────────────────────
Total P99        < 100ms ✓
```

### 2. Handle Class Imbalance
- Fraud: < 0.1% of transactions
- **Solution**: Oversample fraud during training
- Use stratified train/val/test to preserve ratio
- Adjust decision threshold (optimize precision-recall)

### 3. Handle Delayed Labels
- Labels arrive: 1-7 days later
- **Solution**: Train on labels >= 7 days old
- Use weak labels (chargebacks) for faster feedback
- Retrain daily as new labels arrive

### 4. Avoid Training-Serving Skew
- Feature store: single source of truth
- Same computation logic for batch + streaming
- Automated testing: verify consistency
- Monitor feature distributions

### 5. Handle Model Failure
- **Fallback**: Rule-based scoring
- **Example**: High-risk merchants → BLOCK
- Don't let fraud detector become a bottleneck

---

## Feature Engineering

### Real-Time Features (Computed at Request)
```python
velocity_1h = count(transactions in last 1 hour)
velocity_24h = count(transactions in last 24 hours)
device_is_new = is_device_new_for_account(device_id)
location_distance = distance(current_ip, last_ip)
```

### Batch Features (Computed Daily)
```python
avg_transaction_amount = mean(past_30_days_amounts)
std_transaction_amount = std(past_30_days_amounts)
merchant_affinity = fav_merchants / total_merchants
account_age_days = days_since_account_creation()
num_devices = count(unique_devices_used)
txn_per_day = transactions / account_age_days
```

---

## Model Training Pipeline

```python
# 1. Load labeled data (transactions from 7+ days ago)
X_train, y_train = load_training_data(older_than_7_days)

# 2. Handle class imbalance (fraud is rare)
X_train, y_train = oversample_fraud(X_train, y_train)

# 3. Train XGBoost model
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(neg_samples) / len(pos_samples),
)
model.fit(X_train, y_train)

# 4. Evaluate on recent data
metrics = evaluate(model, X_test, y_test)
# precision, recall, f1, auc

# 5. If better than baseline, promote
if metrics['auc'] > baseline_auc:
    deploy_canary(model)  # 5% traffic first
```

---

## Monitoring Alerts

### Critical (Page On-Call)
- Model latency > 100ms P99
- Model unavailable
- Feature store unavailable
- Fraud catch rate < 95%

### Warning (Investigate)
- Feature distribution drift > 2σ
- Prediction distribution changed
- False positive rate > 2%

### Info (Track)
- Model training failures
- Feature generation delays
- Canary metrics

---

## Questions to Answer in Interview

1. **Clarify Requirements**
   - Latency budget? (< 100ms)
   - Throughput? (10k+ RPS)
   - Fraud rate? (< 0.1%)
   - Acceptable fraud loss? (vs. false positives)

2. **Architecture Design**
   - Draw online request path
   - Draw offline pipeline
   - Identify latency bottlenecks

3. **Model Selection**
   - Why XGBoost? (latency + accuracy + interpretability)
   - How handle imbalance? (oversample fraud)
   - How handle delayed labels? (wait 7+ days)
   - How avoid skew? (feature store)

4. **Operational**
   - How deploy safely? (canary, shadow, gradual)
   - How monitor? (metrics store, alerts)
   - How rollback? (instant switch)

5. **Scale & Reliability**
   - What fails first? (feature store or model)
   - Fallback strategy? (rules)
   - Graceful degradation? (CHALLENGE if unsure)

---

## Related Problems

- **Recommendation System**: Similar serving architecture, different ML problem
- **Search Ranking**: Feature store, model serving, A/B testing
- **Click Prediction**: Real-time scoring, concept drift
- **Churn Prediction**: Batch predictions, delayed feedback

---

## Common Mistakes

- ❌ Using only online features (miss historical patterns)
- ❌ Treating ML as purely model problem (ignoring system design)
- ❌ Not handling class imbalance (poor fraud detection)
- ❌ Not accounting for label delay (stale training data)
- ❌ Ignoring latency requirements (model too slow)
- ❌ No fallback strategy (SPOF: single point of failure)
- ❌ Not versioning models (can't rollback)
- ❌ Insufficient monitoring (blind to problems)

---

## Time Breakdown (45-60 min interview)

| Phase | Time | Focus |
|-------|------|-------|
| 1. Clarify Requirements | 5 min | Latency, throughput, fraud rate |
| 2. High-Level Architecture | 10 min | Draw systems and data flow |
| 3. Component Deep-Dives | 20 min | Feature store, model, decision logic |
| 4. Tradeoffs & Decisions | 15 min | Why XGBoost, class imbalance, deployment |
| 5. System Reliability | 5 min | Failures, fallbacks, monitoring |

---

## One-Minute Pitch

*"I would design a two-path system: an online request path for low-latency decisions, and an offline batch pipeline for model training. For the online path, use an XGBoost model for the balance of accuracy and latency, with real-time features (velocity, device consistency) looked up from a feature store. The offline path trains daily on labels that are 7+ days old to ensure accuracy, handles class imbalance by oversampling fraud, and safely deploys through canary testing. Key to this design is avoiding training-serving skew through a centralized feature store, graceful fallback to rules if the model fails, and comprehensive monitoring for drift and degradation."*

---

## Resources

- Read: `design.md` for problem requirements
- Deep dive: `architecture.md` for component details
- Understand: `tradeoffs.md` for design rationale
- Code: `template.py` for implementation starter
- Prepare: `README.md` for interview guide
