# Fraud Detection System - Design

## Problem Statement

Design a machine learning system to support **online fraud detection** for a large-scale consumer platform (e.g., payment processor, e-commerce, lending platform). The system should:

- Make **low-latency fraud decisions** (< 100ms) at high throughput (10k+ RPS)
- Support **offline analytics** for investigation, reporting, and model retraining
- Serve **production ML models** with version control and safe rollouts
- Collect **comprehensive metrics** for model quality, business impact, and system health
- Handle **delayed feedback** (fraud labels arrive hours to days later)
- Detect **adversarial behavior** as fraud patterns evolve
- Balance **recall vs precision** based on business costs (false positives vs false negatives)

---

## Key Design Considerations

### 1. Data Processing

#### What Data to Collect?
- **Transaction features**: amount, merchant, category, timestamp, currency
- **Account features**: age, location, historical transaction patterns, device history
- **Device features**: IP address, device fingerprint, browser, OS, device type
- **Session features**: login method, multiple login attempts, geographic changes
- **Behavioral features**: spending velocity, typical purchase patterns, weekend vs weekday behavior

#### Data Quality Challenges
- **Missing values**: Handle NaN in older accounts or new merchants
- **Noisy labels**: Fraud labels arrive with delays (some arrive 30+ days later)
- **Data imbalance**: Fraud typically represents < 0.1% of transactions
- **Privacy**: Compliance with regulations (GDPR, PCI-DSS) for sensitive data

#### Preprocessing Strategy
- Standardize and normalize numerical features
- One-hot encode categorical features (with cardinality considerations)
- Handle outliers in transaction amounts (use percentile-based clipping)
- Aggregate historical features with different time windows (1d, 7d, 30d)

---

### 2. Feature Engineering

#### Real-Time Features (Required for Online Scoring)
Features computed at request time from current transaction and recent history:
- **Velocity features**: transactions in last 1h, 24h; spending amount in last 24h
- **Device consistency**: is this device new for the account?
- **Location consistency**: geographic distance from last transaction
- **Merchant patterns**: is this merchant category typical for this user?
- **Time patterns**: typical transaction time of day?

#### Offline Features (For Model Training)
Features computed during batch feature generation:
- **Historical statistics**: average transaction amount, std dev, quantiles
- **Behavioral patterns**: purchases per day (weekly/monthly)
- **Device history**: number of devices used, device switching frequency
- **Merchant affinity**: favorite merchants, merchant category distribution
- **Network features**: graph-based features (user-to-user connections)

#### Avoiding Training-Serving Skew
**Challenge**: Features computed in training pipeline must match features in production.

**Solutions**:
1. **Feature Store**: Centralized repository (e.g., Feast, Tecton) for feature definitions
   - Single source of truth for feature computation logic
   - Automated feature backfill and online serving
2. **Consistent Data Pipeline**: Use same code paths for batch and streaming
3. **Feature Versioning**: Track feature schema and computation logic
4. **Monitoring**: Alert if production feature distributions drift from training

---

### 3. Model Design

#### Suitable Models for Online Fraud Detection
- **Logistic Regression**: Fast, interpretable, production-friendly baseline
- **XGBoost/LightGBM**: Better performance, still fast enough for online (< 10ms)
- **Neural Networks**: Larger capacity but higher latency (use if features are image/text)
- **Ensemble Methods**: Combine multiple models for robustness

**Recommendation**: Start with XGBoost/LightGBM for good performance-latency tradeoff.

#### Handling Class Imbalance
- **Upsampling/Downsampling**: Oversample fraud cases, undersample legitimate cases
- **Class Weights**: Assign higher weight to minority (fraud) class during training
- **Sampling Strategy**: SMOTE or stratified sampling for balanced batches
- **Threshold Tuning**: Adjust decision threshold based on business costs (not default 0.5)

#### Handling Delayed Fraud Labels
**Challenge**: True labels (fraud vs legitimate) may arrive days later.

**Solutions**:
1. **Delayed Feedback Window**: Train on labels that arrived >= 7 days ago
   - Sacrifices freshness but ensures label accuracy
2. **Weak Labels**: Use chargeback, dispute signals as proxy labels
3. **Retraining Schedule**: Daily/weekly retraining as new labels arrive
4. **Offline Feedback Loop**: Periodically relabel historical predictions with true labels

---

### 4. Evaluation Metrics

#### Offline Metrics (Model Quality)
- **Precision**: TP / (TP + FP) - of predicted fraud cases, how many are true fraud?
- **Recall**: TP / (TP + FN) - of actual fraud cases, how many did we catch?
- **ROC-AUC**: Performance across all thresholds
- **PR-AUC**: Precision-recall tradeoff (more informative for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall

**Selection Strategy**: Choose based on business costs
- If false positives are very costly (e.g., blocking legitimate users), optimize for precision
- If false negatives are costly (e.g., fraud losses), optimize for recall
- Often use **custom threshold** that maximizes business-relevant metric

#### Online Metrics (Business Impact)
- **Fraud Caught Rate**: % of actual fraud we block (requires delayed labels)
- **False Positive Rate**: % of legitimate transactions incorrectly flagged
- **Customer Impact**: account lockouts, friction added to checkout flow
- **Financial Impact**: fraud losses prevented vs. customer complaints
- **Latency**: P50, P99 response time for model scoring
- **Coverage**: % of transactions scored by model (vs. fallback decisions)

#### Monitoring Metrics
- **Feature Distribution Drift**: Alert if feature distributions shift significantly from training
- **Prediction Distribution Drift**: Alert if model predictions shift (fewer/more fraud predictions)
- **Model Performance Degradation**: Daily/weekly monitoring of recall/precision on delayed labels

---

### 5. Deployment & Serving

#### Model Versioning & Rollout
1. **Canary Deployment**: Route small % of traffic to new model before full rollout
2. **A/B Testing**: Compare new model to baseline in production
3. **Gradual Rollout**: Increase traffic % to new model if metrics are good
4. **Quick Rollback**: Ability to instantly switch back to previous model

#### Latency Requirements
- **Target Latency**: < 100ms per request (includes feature lookup + model scoring)
- **Budget Allocation**:
  - Feature store lookup: 10-20ms
  - Feature computation: 20-30ms
  - Model scoring: 10-20ms
  - Decision service: 20-30ms

#### Handling Failures
- **Fallback Strategy**: If model service fails, use rule-based fallback (e.g., manual review)
- **Rate Limiting**: Graceful degradation under high load
- **Circuit Breaker**: Stop calling failing service, return fallback decision

---

### 6. Post-Deployment Monitoring

#### Drift Detection
1. **Data Drift**: Monitor feature distributions
2. **Prediction Drift**: Monitor model prediction distribution
3. **Label Drift**: Monitor actual fraud rate in delayed labels

#### Adversarial Behavior
- Fraudsters adapt to detection patterns
- Monitor emerging fraud patterns (new merchant categories, device types, etc.)
- Retrain model regularly (daily/weekly) with new fraud examples
- Implement **online learning** mechanisms for rapid adaptation

#### False Positives & Customer Impact
- Track accounts blocked incorrectly
- Monitor customer friction (support tickets, account lockouts)
- Trade off security with customer experience

#### Model Performance Degradation
- Daily comparison of model predictions vs delayed labels
- Alert if recall/precision drops below thresholds
- Trigger automatic retraining if needed

---

## System Requirements Summary

| Requirement | Target |
|------------|--------|
| **Latency** | < 100ms (P99) |
| **Throughput** | 10k+ RPS |
| **Accuracy** | ROC-AUC > 0.95 |
| **Fraud Recall** | > 95% (catch most fraud) |
| **False Positive Rate** | < 1% (minimize customer friction) |
| **Availability** | 99.99% uptime |
| **Model Training Time** | < 1 hour for daily retraining |

---

## Next Steps

See **architecture.md** for detailed system architecture, data flow, and component interactions.
