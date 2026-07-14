# Fraud Detection System - ML System Design Problem

## Overview

This is a comprehensive ML system design problem for building a **real-time fraud detection system** at scale. It covers the full spectrum of challenges in production ML systems: from feature engineering and model serving to monitoring and operational considerations.

**Difficulty**: Hard  
**Time to Complete**: 45-60 minutes in an interview setting  
**Best For**: Senior ML engineers, ML system design roles

---

## Problem Statement

Design a machine learning system to support **online fraud detection** for a large-scale consumer platform (e.g., payment processor, e-commerce, lending). The system should:

- Make **low-latency fraud decisions** (< 100ms at 10k+ RPS)
- Support **offline analytics** for investigation and model retraining
- Serve **production ML models** with safe version control and rollouts
- Collect **metrics** for model quality, business impact, and system health
- Handle **delayed fraud labels** (feedback arrives hours to days later)
- Detect **adversarial behavior** as fraud patterns evolve
- Balance **recall vs precision** based on business costs

---

## Key Interview Topics Covered

### ML System Design
1. **Data Processing**: Collection, cleaning, preprocessing of transaction/account/device/behavioral data
2. **Feature Engineering**: Real-time vs offline features, avoiding training-serving skew
3. **Model Design**: Model selection (XGBoost vs neural nets), handling class imbalance, dealing with delayed labels
4. **Evaluation Metrics**: Offline metrics (precision, recall, AUC) vs online metrics (fraud catch rate, false positives)
5. **Deployment**: Model versioning, canary deployments, safe rollouts
6. **Post-Deployment**: Monitoring drift, detecting adversarial behavior, responding to degradation

### System Architecture
1. **Online Decisioning**: Low-latency feature fetching, model scoring, decision logic
2. **Offline Analytics**: Event storage, label aggregation, investigation capabilities
3. **Model Serving**: Version management, canary traffic routing, quick rollback
4. **Feature Store**: Real-time feature serving, batch feature generation, consistency
5. **Metrics Collection**: What metrics to track for model, data, product, and infrastructure
6. **Scalability**: Handling high throughput, managing failures, graceful degradation

---

## Files in This Problem

### 1. `design.md` - Problem Statement & High-Level Design
Covers the core ML system design questions:
- Data processing and feature engineering strategies
- Model design considerations (class imbalance, delayed feedback)
- Evaluation metrics (offline and online)
- Deployment and post-deployment monitoring

**Read this first** to understand the problem requirements.

### 2. `architecture.md` - Detailed System Architecture
Deep dive into the complete system architecture including:
- High-level system diagram with all components
- Detailed explanation of each component (API Gateway, Feature Store, Decision Service, Model Service, etc.)
- Data flow diagrams (online request path and offline feedback loop)
- Scalability and failure handling strategies
- Monitoring and alerting rules

**Read this second** to understand how components interact.

### 3. `tradeoffs.md` - Design Tradeoffs & Decision Rationale
Explores key design tradeoffs:
- Latency vs Accuracy (XGBoost vs neural nets)
- Recall vs False Positive Rate (tiered decisions)
- Real-time vs Batch Features (hybrid approach)
- Training data freshness vs label accuracy (delayed feedback handling)
- Feature consistency vs computation cost
- Model complexity vs interpretability
- Online decisions vs manual review
- Batch retraining vs online learning
- Single model vs ensemble
- Real-time monitoring vs batch analysis

**Read this for interview discussions** to show tradeoff thinking.

### 4. `template.py` - Implementation Starter Code
A working Python implementation of key components:
- **FeatureStore**: Feature lookup and serving
- **FraudScoringModel**: Model scoring
- **DecisionService**: Orchestration and decision logic
- **TrainingPipeline**: Offline training and evaluation
- **Example usage**: End-to-end demonstration

**Use this as a starting point** for implementing your solution.

---

## Interview Discussion Guide

### What to Prepare

1. **System Components**: Be able to draw and explain:
   - Request path (API → features → model → decision)
   - Offline pipeline (data lake → feature generation → training → evaluation → deployment)
   - Monitoring and alerting

2. **Key Decisions to Justify**:
   - Why XGBoost/LightGBM instead of neural networks?
   - Why use tiered decisions (ALLOW/CHALLENGE/BLOCK/REVIEW)?
   - How to handle class imbalance (rare fraud)?
   - How to handle delayed fraud labels (wait 7+ days)?
   - How to ensure no training-serving skew?

3. **Real-World Constraints**:
   - Latency budget: 100ms P99
   - Throughput: 10k+ RPS
   - Fraud rate: < 0.1% (highly imbalanced)
   - Label delay: 1-7 days
   - False positive cost: customer friction
   - False negative cost: fraud losses

### Common Interview Questions

**On Model Selection**:
> "Why not use a deep neural network for better accuracy?"

**Answer Strategy**:
- Neural networks have higher latency (> 50ms vs 20ms for XGBoost)
- Harder to interpret (important for fraud blocking decisions)
- XGBoost achieves 95%+ AUC, which is sufficient
- Only use neural networks if XGBoost AUC plateaus and latency budget allows

**On Class Imbalance**:
> "How do you handle fraud being < 0.1% of transactions?"

**Answer Strategy**:
- Use stratified train/val/test splits to preserve fraud ratio
- Oversample fraud cases during training (SMOTE or duplication)
- Use class weights (higher weight on fraud)
- Adjust decision threshold (optimize for PR-AUC, not ROC-AUC)
- Use appropriate metrics (precision-recall, not accuracy)

**On Delayed Labels**:
> "Fraud labels arrive 1-7 days later. How do you build the training set?"

**Answer Strategy**:
- Train on labels that are >= 7 days old (high confidence)
- Separate older labels from recent data
- Maintain separate evaluation on more recent data
- Use weak labels (chargebacks, disputes) for faster feedback
- Retrain daily/weekly as new labels arrive

**On Training-Serving Skew**:
> "How do you ensure features in production match training?"

**Answer Strategy**:
- Use a centralized feature store (Feast, Tecton) as single source of truth
- Define features once, use for both batch and streaming
- Version feature schemas and computation logic
- Automated testing to verify batch and streaming compute same values
- Monitor feature distributions in production

**On Deployment Safety**:
> "How do you safely deploy a new model?"

**Answer Strategy**:
- Shadow mode: Run new model, log predictions, don't affect decisions (24h)
- Canary: Route 5% of traffic to new model, monitor metrics (24h)
- Gradual rollout: 5% → 25% → 50% → 100%, 24h per stage
- Rollback: Instant switch if metrics degrade
- A/B testing: Compare against baseline on statistically significant sample

---

## Key Metrics to Know

### Model Metrics
- **Precision**: Of predicted fraud, how many are true fraud?
- **Recall**: Of actual fraud, how many did we catch?
- **ROC-AUC**: Performance across all thresholds
- **PR-AUC**: Better for imbalanced data than ROC-AUC
- **Threshold Tuning**: Optimize for business objective (usually recall or F1)

### Business Metrics
- **Fraud Catch Rate**: % of actual fraud blocked (requires delayed labels)
- **False Positive Rate**: % of legitimate transactions incorrectly flagged
- **Fraud Prevented**: Revenue saved by blocking fraud
- **Customer Friction**: Support tickets, account lockouts

### System Metrics
- **Latency**: P50, P99 request latency (target: < 100ms)
- **Throughput**: Requests per second (target: 10k+)
- **Model Serving Latency**: Time for model scoring (target: < 20ms)
- **Feature Lookup Latency**: Time for feature store query (target: < 20ms)

### Drift Metrics
- **Feature Distribution Drift**: Alert if features shift significantly
- **Prediction Distribution Drift**: Alert if model predictions shift
- **Label Drift**: Alert if fraud rate changes unexpectedly

---

## Latency Budget Breakdown

Total Budget: **< 100ms P99**

| Component | Target | Notes |
|-----------|--------|-------|
| API Gateway + Load Balancing | < 10ms | Network + request routing |
| Feature Store Lookup | < 20ms | Real-time + batch feature serve |
| Feature Computation | < 20ms | Velocity, device consistency, etc |
| Model Scoring | < 20ms | XGBoost inference |
| Decision Service Logic | < 10ms | Thresholding, rules application |
| Response + Logging | < 20ms | Network + async event logging |
| **Total** | **< 100ms** | **P99 latency** |

---

## Architecture Decisions at a Glance

| Decision | Choice | Why |
|----------|--------|-----|
| Model | XGBoost/LightGBM | Balance of accuracy (AUC 95%+) and latency (20ms) |
| Decision Tier | ALLOW/CHALLENGE/BLOCK/REVIEW | Balances fraud catch with customer experience |
| Feature Mix | Real-time velocity + daily batch aggregates | Fresh signals + low latency |
| Label Freshness | Wait 7+ days | High-confidence training data |
| Feature Consistency | Managed feature store | Automated skew prevention |
| Retraining | Daily batch (not online learning) | Stable, debuggable, safe |
| Availability | Prefer availability over consistency | Better customer experience |
| Monitoring | Real-time systems + hourly drift checks + daily analysis | Catches issues at different timescales |

---

## How to Use This Problem

### For Interview Preparation
1. Read `design.md` to understand requirements
2. Read `architecture.md` to understand component design
3. Study `tradeoffs.md` to understand decision rationale
4. Review `template.py` to understand implementation
5. Practice drawing the full architecture on whiteboard
6. Practice explaining key decisions (latency vs accuracy, recall vs FP, etc.)

### For System Design Interview
1. **Clarify Requirements** (5 min):
   - Latency: < 100ms P99
   - Throughput: 10k+ RPS
   - Accuracy: AUC > 0.95
   - Balance: prevent fraud vs minimize false positives

2. **Propose High-Level Architecture** (10 min):
   - Draw request path: API → features → model → decision
   - Draw offline pipeline: data lake → features → training → deployment
   - Identify bottlenecks (feature lookup, model serving)

3. **Deep Dive on Tradeoffs** (20 min):
   - Model selection: Why XGBoost not neural networks?
   - Class imbalance: How handle rare fraud?
   - Label delay: How get training data with delayed labels?
   - Training-serving skew: How ensure consistency?

4. **System Reliability** (10 min):
   - Failure modes and fallbacks
   - Monitoring and alerting
   - Rollback procedures

5. **Scale & Optimization** (5 min):
   - How to handle 10k+ RPS?
   - Cost optimization

### For Learning & Implementation
- Start with `template.py` to understand the basic components
- Extend the template with real feature engineering
- Implement a simple model training pipeline
- Add monitoring and metrics collection
- Test with simulated transaction data

---

## Additional Resources

### Related Topics
- Feature engineering for fraud detection
- Handling imbalanced datasets in ML
- Online learning and concept drift
- A/B testing and experimentation
- Model interpretability and explainability
- Real-time feature stores (Feast, Tecton)
- Model serving (Seldon, KServe, BentoML)

### Recommended Readings
- "Machine Learning Systems Design" by Chip Huyen - covers many of these topics
- Feature Store research papers (DoorDash, Uber, AirBnB)
- "Rules of Machine Learning" by Google
- Papers on online learning and drift detection

---

## Questions for Your Interviewer

Good questions to ask after presenting your design:

1. "What's the most critical metric we should optimize for — fraud catch rate or false positive rate?"
2. "How would you handle a sudden shift in fraud patterns (e.g., new fraud type)?"
3. "What's the cost per false positive vs false negative in your business?"
4. "How do you currently handle fraud before ML? What's the manual process?"
5. "What's your tolerance for false positives (customer friction)?"
6. "Do you have a feature store already, or would we build from scratch?"
7. "How would model governance work (approval, compliance, monitoring)?"

---

## Summary

This fraud detection system design problem covers **all major aspects of production ML systems**:
- ✅ Data pipelines and feature engineering
- ✅ Model selection and training
- ✅ Production serving and latency optimization
- ✅ Offline analytics and feedback loops
- ✅ Monitoring, drift detection, and alerting
- ✅ Safety, reliability, and failure handling
- ✅ Tradeoff analysis and decision-making

Master this problem and you'll be well-prepared for any ML system design interview!
