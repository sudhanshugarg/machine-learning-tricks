# Fraud Detection System - Design Tradeoffs

## Core Tradeoffs

### 1. Latency vs Accuracy

**The Tradeoff**:
- More complex models (deep networks, large ensembles) → Better accuracy but higher latency
- Simple models (logistic regression) → Lower latency but lower accuracy

**Requirements Conflict**:
- Product needs: < 100ms P99 latency
- ML needs: More features and complexity for better accuracy

**Solutions**:
| Approach | Pros | Cons |
|----------|------|------|
| **XGBoost/LightGBM** | ~20ms latency, good accuracy (AUC > 0.95) | Less powerful than deep networks |
| **Neural Networks** | Better accuracy potential | 50-100ms+ latency, harder to debug |
| **Feature Selection** | Reduce feature count → faster inference | May lose signal for harder cases |
| **Model Quantization** | Faster inference without retraining | Small accuracy loss (usually < 1%) |
| **GPU Acceleration** | 5-10x faster inference | High cost, added complexity |

**Recommendation**: Use XGBoost/LightGBM as baseline. Only move to neural networks if AUC plateaus and latency budget allows.

---

### 2. Recall vs False Positive Rate

**The Tradeoff**:
- High recall (catch all fraud) → Block many legitimate transactions (high FP rate)
- Low false positive rate (minimal customer friction) → Miss some fraud

**Business Cost Analysis**:
- False Negative (miss fraud): Lost money + reputational cost
- False Positive (block legitimate): Customer friction, support tickets, loss of trust

**Solutions**:

| Strategy | Recall | FP Rate | Impact |
|----------|--------|---------|--------|
| **Low threshold (0.3)** | 98% | 5% | Block many legitimate users |
| **Medium threshold (0.5)** | 95% | 1% | Balanced approach |
| **High threshold (0.8)** | 80% | 0.1% | Miss some fraud but minimal friction |
| **Tiered Decision** | 95% | 1% | BLOCK (high score), CHALLENGE (medium), ALLOW (low) |

**Recommendation**: Use **tiered decisions**:
- BLOCK: High-risk transactions (require manual investigation)
- CHALLENGE: Medium-risk (2FA, additional verification)
- ALLOW: Low-risk

This balances recall with customer experience.

---

### 3. Real-Time vs Batch Features

**The Tradeoff**:
- Real-time computed features (velocity, location consistency) → Fresh, catches new patterns but requires low-latency computation
- Batch precomputed features (historical aggregates) → More accurate but may be stale (updated daily)

**Feature Type | Computation | Freshness | Latency Impact |**
|---|---|---|---|
| Real-time velocity | Computed at request time | Minutes to hours lag | 5-10ms |
| Batch aggregates | Computed daily | 1 day staleness | 1-2ms (lookup only) |
| Static features | Rarely change | No issue | 1ms |

**Solutions**:
1. **Hybrid Approach** (Recommended): Mix real-time + batch features
   - Real-time: Velocity, device consistency (can't precompute)
   - Batch: Historical aggregates, device history (change slowly)
2. **Near-Real-Time Updates**: Update batch features every hour instead of daily
3. **Streaming Aggregation**: Use stream processing (Spark Streaming, Kafka Streams) for faster updates

**Recommendation**: Hybrid approach with hourly batch updates for better freshness.

---

### 4. Training Data Freshness vs Label Accuracy

**The Tradeoff**:
- Train immediately (stale labels) → Respond quickly to new patterns but with noisy labels
- Wait for labels to arrive (7+ days) → Clean labels but slower feedback loop

**Timeline**:
- Day 1: Transaction occurs
- Day 1-3: Customer may dispute (some fraud detected quickly)
- Day 3-7: Additional disputes arrive (chargeback processing)
- Day 7+: Reliable fraud label available

**Solutions**:

| Approach | Training Latency | Label Accuracy | Impact |
|----------|------------------|-----------------|--------|
| **Immediate training** | 1 day | 70% labeled | Noisy labels, unstable models |
| **Wait 7 days** | 7 days | 95%+ labeled | Clean labels but slow feedback |
| **Hybrid** | 3-4 days | 85% labeled | Reasonable tradeoff |
| **Weak Labels** | 1 day | 60% labeled | Use proxies (dispute signals) |

**Weak Label Sources**:
- Chargebacks (1-3 days)
- Fraud reports (1-2 days)
- Disputed transactions (1-3 days)
- Manual investigation (longer but highest confidence)

**Recommendation**: 
- Train on labels available >= 7 days old (high confidence)
- Separately monitor real-time metrics on current predictions vs weak labels
- Retrain daily/weekly as new labels arrive

---

### 5. Feature Store Consistency vs Computation Cost

**The Tradeoff**:
- Compute features on-demand → Always fresh, consistent with training, but slower (requires real-time computation)
- Precompute features → Faster serving but risks training-serving skew if computation logic differs

**Implementation Options**:

| Approach | Speed | Consistency | Cost |
|----------|-------|-------------|------|
| **On-demand computation** | 20-50ms per request | Perfect | High CPU, latency risk |
| **Precomputed (daily)** | 1-2ms lookup | Risk of skew if not careful | Low compute, low serving cost |
| **Hybrid Feature Store** | 5-10ms | Good if well managed | Medium |

**Ensuring Consistency**:
1. **Single Source of Truth**: Feature definitions in version-controlled code
2. **Same Code Path**: Use same logic for batch and streaming
3. **Automated Testing**: Validate that batch and streaming produce same results
4. **Monitoring**: Alert if feature values diverge

**Recommendation**: Use managed feature store (Feast, Tecton) that handles consistency automatically.

---

### 6. Model Complexity vs Interpretability

**The Tradeoff**:
- Complex models (deep networks, large ensembles) → Better accuracy but black-box
- Simple models (logistic regression, decision trees) → Lower accuracy but explainable

**Regulatory & Operational Implications**:
- Need to explain why transaction was blocked (fair lending laws)
- Need to debug when model makes mistakes
- Need to handle model appeals

**Solutions**:

| Model Type | Accuracy | Interpretability | Use Case |
|-----------|----------|------------------|----------|
| **Logistic Regression** | 85% AUC | Excellent (feature weights) | Baseline, regulatory compliance |
| **XGBoost** | 95% AUC | Good (feature importance) | Production workhorse |
| **Deep Network** | 97% AUC | Poor (black box) | Use if absolutely needed |
| **Ensemble + LR** | 96% AUC | Medium (aggregate importance) | Interpretable accuracy |

**Recommendation**: XGBoost provides good balance of accuracy and interpretability. Can use SHAP/LIME to explain predictions when needed.

---

### 7. Online Decisions vs Manual Review

**The Tradeoff**:
- Automated decisions (ML model) → Fast, consistent, scalable but may have errors
- Manual review → Accurate, can handle edge cases but slow and costly

**Decision Tier Architecture**:
```
ALLOW (score < 0.5)
  ↓ (100% auto-approved)

CHALLENGE (score 0.5-0.7)
  ↓ (2FA, additional verification)
  
REVIEW (score > 0.9 or high-risk signals)
  ↓ (Queue for manual investigation)
```

**Cost Analysis**:
- Auto-decision: $0.001 per transaction
- Manual review: $0.50-$1.00 per transaction
- Fraud loss if missed: $50-$500 per transaction

**Recommendation**: 
- Auto-approve low-risk (> 95% confidence)
- Challenge medium-risk (2FA, additional steps)
- Manual review very high-risk or edge cases

---

### 8. Batch Retraining vs Online Learning

**The Tradeoff**:
- Batch retraining (daily/weekly) → Clean, stable models but slow to adapt to new fraud patterns
- Online learning → Adapts quickly to new patterns but risks instability and degradation

**Timeline to Adapt to New Fraud Pattern**:
| Approach | Latency | Complexity | Stability |
|----------|---------|-----------|-----------|
| **Daily batch** | 1 day | Low | High (stable) |
| **Hourly batch** | 1 hour | Medium | Medium |
| **Online learning** | 5 minutes | High | Risk of degradation |

**Online Learning Risks**:
- Can overfit to recent anomalies
- Catastrophic forgetting (forget old patterns)
- Harder to debug and rollback
- May introduce subtle biases

**Recommendation**: Start with daily/hourly batch retraining. Only consider online learning if:
- Fraud patterns change very rapidly
- You have strong monitoring and rollback procedures
- You have safeguards (confidence thresholds, drift detection)

---

### 9. Single Model vs Ensemble

**The Tradeoff**:
- Single model → Simpler, faster inference, easier to maintain
- Ensemble → Better accuracy, more robust to distribution shift, but higher latency and complexity

**Performance Comparison**:
| Approach | AUC | Latency | Complexity | Cost |
|----------|-----|---------|-----------|------|
| **Single XGBoost** | 95% | 20ms | Low | $$ |
| **2-Model Ensemble** | 96% | 35ms | Medium | $$$ |
| **3+ Model Ensemble** | 96.5% | 50ms | High | $$$$ |

**Ensemble Benefits**:
- Better performance on edge cases
- More robust to concept drift
- Can combine different model types (gradient boosting + neural net)

**Recommendation**: Start with single XGBoost. Move to ensemble if:
- Single model AUC plateaus
- Latency budget allows (< 50ms)
- Accuracy gains justify added complexity

---

### 10. Real-Time Monitoring vs Batch Analysis

**The Tradeoff**:
- Real-time monitoring → Immediate alerts but more false alarms (noisy)
- Batch analysis → Reliable insights but delayed detection

**Alert Examples**:
| Alert Type | Latency | False Alarms | Utility |
|-----------|---------|--------------|---------|
| **Real-time spike** | 1 min | High | Useful for traffic anomalies |
| **Hourly drift detection** | 1 hour | Medium | Useful for feature drift |
| **Daily metrics** | 24 hours | Low | Reliable insights |

**Recommendation**: Hybrid approach:
- Real-time: Monitor system health (latency, throughput, errors)
- Hourly: Monitor model metrics (drift, distribution changes)
- Daily: Deep analysis and investigation

---

## System-Level Tradeoffs

### A. Consistency vs Availability

**CAP Theorem Applied to Fraud Detection**:
- **Consistency**: All replicas have same latest fraud rules/model
- **Availability**: System never goes down, always makes a decision
- **Partition tolerance**: System survives network failures

**Choice**: Favor **Availability** (prefer to let through a few frauds than block legitimate users)
- Fraud has financial impact but system downtime damages customer trust
- Use eventual consistency for feature store and models

**Implementation**:
- Cache features aggressively (stale but fast)
- Fallback to rule-based scoring if model unavailable
- Async updates to models and rules (don't block on consistency)

---

### B. Latency vs Throughput

**The Tradeoff**:
- Optimize for latency (low P99) → Use batch inference, GPUs, caching
- Optimize for throughput (high RPS) → Use simpler models, CPU-efficient inference

**Decision Service Bottleneck**:
- Feature store lookup: 20ms (parallelizable)
- Model scoring: 20ms (CPU-bound for XGBoost)
- Total: 40ms + network overhead

**Scaling Solutions**:
1. **Vertical**: Use faster hardware (high-end CPUs, GPUs)
2. **Horizontal**: Replicate decision service, load balance
3. **Model optimization**: Quantize, prune, or use simpler model

**Recommendation**: Horizontal scaling (easiest and most cost-effective for this workload).

---

### C. Cost vs Performance

**Infrastructure Costs** (rough estimates per 1M transactions/day):
| Component | Simple Setup | Optimized | High-Performance |
|-----------|-------------|----------|-----------------|
| **Feature Store** | $500/month | $2k/month | $5k/month |
| **Model Serving** | $1k/month | $3k/month | $10k/month |
| **Data Lake** | $2k/month | $5k/month | $10k/month |
| **ML Pipeline** | $1k/month | $3k/month | $8k/month |
| **Total** | ~$4.5k | ~$13k | ~$33k |

**Cost vs Accuracy Tradeoff**:
- Baseline (single model, daily retraining): ~$13k, AUC 95%
- Enhanced (2-model ensemble, hourly retraining): ~$20k, AUC 96%
- Premium (3-model ensemble, online learning, GPU inference): ~$35k, AUC 96.5%

**Recommendation**: Start with baseline, invest in optimization only if:
- Model accuracy is insufficient
- Fraud losses exceed added infrastructure cost
- Latency requirements become critical

---

## Summary: Recommended Design Decisions

| Tradeoff | Decision | Rationale |
|----------|----------|-----------|
| Latency vs Accuracy | XGBoost (not neural nets) | 20ms latency, AUC 95%+ |
| Recall vs FP Rate | Tiered decisions (Allow/Challenge/Block) | Balances fraud catch with UX |
| Real-time vs Batch Features | Hybrid (real-time velocity + batch aggregates) | Fresh signals + low latency |
| Label Freshness vs Accuracy | Wait 7+ days for labels | High-quality training data |
| Feature Consistency | Use managed feature store | Automates consistency checks |
| Model Complexity | XGBoost (simple enough to debug) | Accuracy + interpretability |
| Online Decisions | Auto + Challenge + Manual review tiers | Optimizes cost and accuracy |
| Retraining Strategy | Daily batch (not online learning) | Stable, debuggable, safe |
| Model Ensemble | Start with single, move to 2-model if needed | Simplicity first |
| Availability vs Consistency | Favor availability (eventual consistency) | Better customer experience |

---

## Next Steps

See **template.py** for starter implementation code and **design.md** for problem requirements.
