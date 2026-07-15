# Solution: Evaluation System with Human + LLM Evaluators

## Clarifying Questions

Before diving into the architecture, a strong candidate would ask:

1. **Evaluation scope**: Are all items the same type (essays, code, images, mixed)? Does rubric complexity vary?
2. **Batch semantics**: Is a "batch" a full day's requests, or smaller windows? Can we adjust routing mid-batch?
3. **SLA trade-offs**: Is latency (2s p99) hard, or is cost the primary constraint?
4. **Ground truth**: Do we have a gold-standard validation set to measure accuracy against?
5. **Human behavior**: Can annotators reject ambiguous items, or must we always return a score?
6. **LLM drift**: Should we monitor for degradation in LLM judge performance over time?

**Assumptions we'll proceed with:**
- All items evaluated on the same 5-point Likert scale
- Batches are daily (86.4M items) with dynamic re-routing as humans become available/exhausted
- Cost is primary constraint; latency is soft (aim for p99 < 2s)
- We have a ~5% holdout validation set for calibration
- Annotators can flag items as unevaluable; system retries or escalates
- LLM models may drift; we monitor quarterly

---

## Goals & Constraints

| Goal | Constraint |
|------|-----------|
| **Quality** | Spearman's ρ ≥ 0.85 vs. gold-standard human evaluation |
| **Cost** | ≤ $0.10/item average |
| **Latency** | p99 ≤ 2s (initial score); full audit ≤ 5 min |
| **Coverage** | Score 100% of items (don't drop items due to budget) |
| **Observability** | Track per-rater accuracy, LLM confidence, inter-rater κ |

---

## System Architecture

### High-Level Data Flow

```
Item Intake
    ↓
┌─────────────────────────────────────────┐
│  Router & Triage                        │
│  - LLM confidence prediction            │
│  - Budget remaining?                    │
│  - Human availability?                  │
│  → Decide: LLM-only / Human-only / Both │
└─────────────────────────────────────────┘
    ├─→ LLM Judge Evaluator
    │       ├─ Ensemble (3 models)
    │       └─ Structured output + confidence
    │
    └─→ Human Pool Coordinator
            ├─ Queue by skill tier
            ├─ Assign via load balancer
            └─ Track per-rater accuracy
                        ↓
            ┌────────────────────────────┐
            │  Score Aggregation Layer   │
            │  - Merge LLM + human       │
            │  - Weighted averaging      │
            │  - Consensus voting        │
            └────────────────────────────┘
                        ↓
            ┌────────────────────────────┐
            │  Calibration & Confidence  │
            │  - Platt scaling           │
            │  - Recalibrate per domain  │
            │  - Assign final confidence │
            └────────────────────────────┘
                        ↓
            ┌────────────────────────────┐
            │  Output & Audit Trail      │
            │  - Final score (1-5)       │
            │  - Confidence interval     │
            │  - Evaluator trace         │
            │  - Reasoning (if avail)    │
            └────────────────────────────┘
```

---

## Component Deep Dives

### 1. Router & Triage

**Goal**: Decide whether each item goes to LLM-only, human-only, or both—minimizing cost while staying within quality bounds.

**Strategy**:

1. **Pre-compute item difficulty** (once per batch):
   - Run LLM judge on a sample (10K items)
   - Cluster by confidence distribution
   - Map to estimated accuracy (via calibration set)

2. **Route by confidence tier**:
   ```
   LLM confidence ≥ 0.90   → LLM-only (estimated accuracy ~90%)
   LLM confidence ∈ [0.70, 0.90) → Send to human (highest ROI for quality)
   LLM confidence < 0.70   → Send to human + secondary LLM ensemble
   
   + Always reserve 5% of daily budget for calibration
       (random sample, evaluate with both LLM & human)
   ```

3. **Dynamic budget reallocation**:
   ```
   budget_per_item = 0.10  // avg cost target
   n_humans = available annotators at this hour
   daily_human_capacity = n_humans * 1000 items/day
   
   if (human_capacity_used / daily_human_capacity) > 0.8:
       # Humans are running out; shift remaining items to LLM-only
       override_confidence_threshold = 0.60
   elif (daily_cost_spent / daily_budget) > 0.9:
       # Budget is tight; pull back human assignments
       override_confidence_threshold = 0.75
   ```

**Data Structures**:
```python
class RoutingDecision:
    item_id: str
    route: Literal["llm_only", "human_only", "both"]
    llm_confidence: float
    estimated_accuracy: float
    assigned_tier: Optional[str]  # e.g., "tier_1" for experts
    budget_allocated: float
```

**Tradeoff**:
- **Decision**: Route uncertain items (0.70–0.90) to humans first; extreme confidence items to LLM-only.
- **Why**: Humans are expensive but high-quality. LLM is cheap but unreliable in the middle range.
- **Alternative**: Always use LLM + spot-check with human. (Cheaper, but quality suffers; ρ might drop to 0.75.)

---

### 2. LLM Judge Evaluator

**Goal**: Get fast, reliable scores from LLM models; generate confidence estimates for routing.

**Architecture**:

#### 2.1 Ensemble of Models
Use **3 models with different strengths**:
- **Claude 3.5 Sonnet** (strong reasoning, context-aware)
- **GPT-4o** (fast, robust)
- **Gemini 2.0 Pro** (multimodal if needed)

Why 3? Majority voting reduces single-model hallucinations. Each model has different biases; ensemble smooths them.

#### 2.2 Structured Output

**System Prompt** (anonymized, rubric-agnostic):
```
You are an expert evaluator. You will assess a submission against the provided rubric.
Output a JSON with:
- score: integer 1-5
- confidence: float 0.0-1.0 (how sure are you?)
- reasoning: str (key evidence for the score)
- flag: bool (is this ambiguous or potentially unevaluable?)

Anonymize the author/context. Focus on the work quality, not the source.
Do not explain what you're evaluating—just score it.
```

**Input Format**:
```json
{
  "submission": "...",
  "rubric": "...",
  "anonymized_metadata": {
    "word_count": 500,
    "submission_type": "essay"
  }
}
```

**Output** (structured):
```json
{
  "score": 4,
  "confidence": 0.85,
  "reasoning": "Well-argued thesis with supporting evidence. Minor logical gap in section 2.",
  "flag": false
}
```

#### 2.3 Confidence Calibration via Platt Scaling

Train a logistic regression on calibration set to map LLM confidence → accuracy:

```python
# After collecting (LLM_confidence, human_accuracy) pairs from calibration set:
from sklearn.linear_model import LogisticRegression

calibration_data = [
    (0.95, 0.92),  # LLM said 95% confident; turned out 92% accurate
    (0.70, 0.68),
    ...
]

platt_scaler = LogisticRegression()
platt_scaler.fit(
    X=np.array([c[0] for c in calibration_data]).reshape(-1, 1),
    y=np.array([c[1] > 0.8 for c in calibration_data])  # binary: correct/incorrect
)

# Then, for any new LLM prediction:
recalibrated_confidence = platt_scaler.predict_proba([[raw_confidence]])[0][1]
```

#### 2.4 Ensemble Aggregation

For items routed to "both" or marked as "flag=true":

```python
def aggregate_llm_scores(scores: List[int], confidences: List[float]) -> Tuple[int, float]:
    # Weighted median by confidence
    weights = np.array(confidences) / sum(confidences)
    weighted_scores = scores * weights
    final_score = np.round(np.average(scores, weights=weights))
    
    # Consensus confidence: if all 3 agree, confidence is high
    agreement = 1 if len(set(scores)) == 1 else max(confidences) * 0.8
    
    return int(final_score), agreement
```

**Tradeoff**:
- **Decision**: Use 3-model ensemble with Platt-calibrated confidence.
- **Why**: Ensemble reduces hallucination; Platt scaling maps "confidence" to actual accuracy, enabling routing.
- **Alternative**: Single model (faster, cheaper). Cost: ρ drops to ~0.78; routing becomes unreliable.

---

### 3. Human Pool Management

**Goal**: Assign items to raters efficiently; track accuracy per rater; balance workload by skill tier.

#### 3.1 Skill Tiers

```
Tier 1 (Expert):
  - Accuracy on calibration set: ≥ 92%
  - Cost: $2.00/item
  - Use for: High-value items, difficult edge cases, calibration

Tier 2 (Experienced):
  - Accuracy: 80–92%
  - Cost: $1.00/item
  - Use for: Medium-difficulty items, consensus votes

Tier 3 (General):
  - Accuracy: 65–80%
  - Cost: $0.50/item
  - Use for: High-confidence LLM items that need human validation
```

#### 3.2 Rater Assignment Algorithm

For each item routed to humans:

```python
def assign_rater(item, confidence, budget_remaining):
    """Route to appropriate tier based on difficulty and budget."""
    
    # Difficulty metric: inverse of LLM confidence (or explicit rubric complexity)
    difficulty = 1 - confidence
    
    if difficulty > 0.7 or budget_remaining < $0.50:
        # Hard item or tight budget: try Tier 3 first, escalate if needed
        tier = assign_from_pool([Tier3, Tier2, Tier1], max_wait=1000ms)
    elif difficulty > 0.4:
        tier = assign_from_pool([Tier2, Tier1, Tier3], max_wait=500ms)
    else:
        tier = assign_from_pool([Tier3, Tier2], max_wait=200ms)
    
    return tier
```

#### 3.3 Per-Rater Accuracy Tracking

Maintain a confusion matrix per rater against a gold standard (sampled):

```python
class RaterProfile:
    rater_id: str
    tier: str
    accuracy: float  # P(rater's score == gold)
    confusion_matrix: np.ndarray  # 5x5 for Likert
    bias: float  # mean(rater_score - gold_score); detect systematic under/overrating
    last_recalibrated: datetime
    n_items_evaluated: int
```

**Monthly recalibration**: If `accuracy < tier_threshold - 5%`, demote or retrain.

#### 3.4 Consensus Voting

For important items (or calibration batch), assign to 2–3 raters:

```python
def consensus_score(rater_scores: List[int]) -> Tuple[int, float]:
    """Majority vote with confidence based on agreement."""
    mode = statistics.mode(rater_scores)
    agreement_ratio = rater_scores.count(mode) / len(rater_scores)
    confidence = agreement_ratio  # high agreement → high confidence
    return mode, confidence
```

**Tradeoff**:
- **Decision**: Skill tiers + per-rater accuracy tracking + monthly recalibration.
- **Why**: Humans are expensive; tracking accuracy ensures we use each rater effectively.
- **Alternative**: Uniform payment per item (no tiers). Cost: Either overpay good raters or get consistently bad scores.

---

### 4. Score Aggregation & Consensus

**Goal**: Merge LLM and human signals into a single final score with confidence.

#### 4.1 Aggregation Rules

```python
def aggregate_scores(
    llm_score: Optional[int],
    llm_confidence: float,
    human_scores: List[int],
    human_confidences: List[float]
) -> Tuple[int, float, str]:
    """
    Merge LLM and human evaluations into final score.
    
    Returns:
      (final_score, confidence, source)
    """
    
    # Case 1: Only LLM
    if not human_scores:
        return llm_score, llm_confidence, "llm_only"
    
    # Case 2: Only humans
    if llm_score is None:
        if len(human_scores) == 1:
            return human_scores[0], 0.8, "human_single"
        else:
            # Majority vote
            mode = statistics.mode(human_scores)
            agreement = human_scores.count(mode) / len(human_scores)
            return mode, agreement, "human_consensus"
    
    # Case 3: Both LLM and human
    # Weight by accuracy/confidence
    llm_weight = llm_confidence  # e.g., 0.85
    human_weight = np.mean(human_confidences)  # e.g., 0.90
    
    total_weight = llm_weight + human_weight
    
    # If human and LLM strongly disagree (|diff| > 1.5), flag for review
    human_avg = np.mean(human_scores)
    if abs(llm_score - human_avg) > 1.5:
        return int(human_avg), 0.6, "disagreement_flagged"
    
    # Weighted average
    final = (llm_score * llm_weight + human_avg * human_weight) / total_weight
    confidence = min(llm_weight, human_weight)  # confidence = min of the two
    
    return int(np.round(final)), confidence, "llm_human_merged"
```

#### 4.2 Tie-Breaking & Edge Cases

- **Odd number of human raters**: Use majority vote
- **Even number**: Take median or escalate to a tie-breaker (expert rater)
- **Unanimous disagreement**: Flag for manual review; use human average as default

**Tradeoff**:
- **Decision**: Weight by confidence; use human average as primary if both available.
- **Why**: Humans are more reliable (~0.90 accuracy) than LLM (~0.80). Confidence-weighted averaging respects this.
- **Alternative**: Always trust human if available. Cost: Loses benefit of LLM speed; budget blows up.

---

### 5. Cost Optimization

**Goal**: Stay within $0.10/item budget while maintaining quality.

#### 5.1 Daily Budget Allocation

```python
def allocate_budget(
    daily_volume: int,  # 86.4M items
    target_cost_per_item: float,  # $0.10
    human_availability: Dict[str, int],  # tier → count of available raters
) -> Dict[str, float]:
    """
    Allocate daily budget across LLM and human evaluators.
    """
    
    total_budget = daily_volume * target_cost_per_item  # $8.64M
    
    # Reserve 5% for calibration (both LLM and human)
    calibration_budget = total_budget * 0.05
    remaining_budget = total_budget - calibration_budget
    
    # Estimate human capacity
    tier_1_capacity = human_availability["tier_1"] * 1000 * 0.3  # only 30% for non-calibration
    tier_2_capacity = human_availability["tier_2"] * 1000 * 0.4
    tier_3_capacity = human_availability["tier_3"] * 1000 * 0.6
    
    max_human_items = tier_1_capacity + tier_2_capacity + tier_3_capacity
    max_human_spend = (
        tier_1_capacity * 2.00 +
        tier_2_capacity * 1.00 +
        tier_3_capacity * 0.50
    )
    
    # Allocate greedily: send high-value uncertain items to humans up to budget
    if max_human_spend < remaining_budget:
        human_budget = max_human_spend
    else:
        human_budget = remaining_budget * 0.40  # cap human at 40% of remaining
    
    llm_budget = remaining_budget - human_budget
    llm_items = llm_budget / 0.005  # assume $0.005/item for LLM
    
    return {
        "human_budget": human_budget,
        "llm_budget": llm_budget,
        "human_items": max_human_items,
        "llm_items": llm_items,
        "calibration_budget": calibration_budget,
    }
```

#### 5.2 Greedy Routing by Confidence

Sort all items by LLM confidence (descending). Assign greedily:

```
Sort items by confidence DESC

for item in items:
    if human_budget_remaining > 0 and item.confidence < 0.75:
        route to human
        human_budget_remaining -= cost_for_tier
    else:
        route to llm_only
        llm_budget_remaining -= 0.005
    
    if llm_budget_remaining < 0:
        # Overflow: reject item or queue for next batch
        queue_for_next_batch(item)
```

#### 5.3 Cost Monitoring & Real-Time Adjustment

```python
def realtime_budget_check(items_evaluated, budget_spent, hours_remaining):
    """Warn if we're going to overspend."""
    daily_budget = 8.64e6  # $8.64M
    pace = budget_spent / (24 - hours_remaining)  # $/hour
    projected_spend = pace * 24
    
    if projected_spend > daily_budget * 1.05:
        # Flip a switch: be more aggressive with LLM-only routing
        log.warning(f"Budget overrun projected: {projected_spend}. Tightening human assignments.")
        set_confidence_threshold(0.80)  # only send very uncertain items to humans
    elif projected_spend < daily_budget * 0.80:
        # Underutilized budget; can afford more human evaluations
        set_confidence_threshold(0.65)
```

**Tradeoff**:
- **Decision**: Reserve 5% calibration, allocate 40–60% to humans (greedy by confidence), rest to LLM.
- **Why**: Captures ROI: high-confidence items (LLM-only) are cheap; uncertain items get human review, which improves quality the most.
- **Alternative**: Fixed 50% human / 50% LLM split. Cost: Misallocates; wastes humans on easy items and LLM on hard ones.

---

### 6. Evaluation & Feedback Loop

**Goal**: Measure quality, detect drift, and continuously improve the system.

#### 6.1 Quality Metrics

```python
class EvaluationMetrics:
    # Primary metric
    spearman_rho: float  # correlation with gold-standard human evaluation
    target = 0.85
    
    # Sub-metrics
    mae_vs_gold: float  # mean absolute error (vs gold)
    per_score_accuracy: Dict[int, float]  # recall per Likert score (1-5)
    
    # LLM-specific
    llm_accuracy: float  # P(llm_score == human_consensus_score)
    llm_drift: float  # ρ_this_month vs ρ_last_month
    
    # Human-specific
    inter_rater_reliability_kappa: float  # Cohen's κ between raters (target ≥ 0.70)
    per_rater_accuracy: Dict[str, float]
    
    # System
    cost_per_item: float
    p99_latency: float
```

#### 6.2 Validation Set & Quarterly Recalibration

Maintain a **5% holdout validation set** (4.32M items/day):

```python
def quarterly_recalibration():
    """Re-evaluate calibration every 3 months."""
    
    # Sample 10K items from validation set
    # Have both LLM and humans re-evaluate them
    
    for item, llm_score, human_scores in validation_set:
        true_score = np.median(human_scores)
        
        # Measure LLM accuracy
        llm_correct = (llm_score == true_score)
        
        # Measure human rater agreement
        kappa = cohens_kappa(human_scores, [true_score] * len(human_scores))
    
    # Recompute Spearman's ρ
    all_predictions = [llm_score for _, llm_score, _ in validation_set]
    all_gold = [np.median(human_scores) for _, _, human_scores in validation_set]
    new_rho = spearmanr(all_predictions, all_gold)[0]
    
    if new_rho < 0.85:
        alert("Spearman ρ dropped below target. Investigating LLM drift...")
        # Possible actions:
        # 1. Retrain Platt scaler
        # 2. Update LLM system prompt
        # 3. Switch to different LLM models
        # 4. Increase human evaluation %, lower LLM routing threshold
```

#### 6.3 Anomaly Detection: Rater Bias

```python
def detect_rater_bias(rater_id, bias_threshold=0.3):
    """Flag raters with systematic bias."""
    
    profile = rater_profiles[rater_id]
    bias = np.mean(profile.confusion_matrix.diagonal()) - 0.5  # simplistic
    
    if abs(bias) > bias_threshold:
        log.warning(f"Rater {rater_id} has systematic bias: {bias}")
        # Actions:
        # 1. Demote to lower tier
        # 2. Schedule retraining
        # 3. Reduce future assignments
```

#### 6.4 Continuous Improvement Loop

```
Daily:
  - Monitor cost/item, p99 latency, items_evaluated
  - Alert if ρ drops > 0.05 from baseline
  
Weekly:
  - Analyze human rater accuracy; flag outliers
  - Check for LLM model drift (confidence calibration)
  
Monthly:
  - Retrain Platt scaler on new calibration data
  - Update LLM system prompt if needed (based on flagged items)
  - Rotate training set for rater accuracy tracking
  
Quarterly:
  - Full recalibration on validation set
  - Reassess skill tier assignments
  - Decide on new LLM models to evaluate
```

**Tradeoff**:
- **Decision**: Quarterly recalibration with continuous weekly monitoring.
- **Why**: Detects drift early (weekly) without over-engineering (daily retraining).
- **Alternative**: No monitoring (save cost). Cost: ρ drifts silently; one day you realize accuracy collapsed.

---

## Deployment & Operational Considerations

### 6.5 Database Schema

```sql
-- Items to be evaluated
CREATE TABLE items (
    id STRING PRIMARY KEY,
    submission BLOB,
    rubric STRING,
    batch_id STRING,
    created_at TIMESTAMP,
    routed_to STRING,  -- "llm_only", "human_only", "both"
    status STRING,  -- "pending", "evaluated", "flagged"
);

-- LLM evaluations
CREATE TABLE llm_evaluations (
    id STRING PRIMARY KEY,
    item_id STRING FOREIGN KEY,
    model STRING,  -- "sonnet", "gpt4o", "gemini"
    score INT,
    confidence FLOAT,
    reasoning STRING,
    flag BOOL,
    created_at TIMESTAMP,
);

-- Human evaluations
CREATE TABLE human_evaluations (
    id STRING PRIMARY KEY,
    item_id STRING FOREIGN KEY,
    rater_id STRING FOREIGN KEY,
    tier STRING,
    score INT,
    reasoning STRING,
    time_taken_minutes INT,
    created_at TIMESTAMP,
);

-- Rater profiles
CREATE TABLE rater_profiles (
    rater_id STRING PRIMARY KEY,
    tier STRING,
    accuracy FLOAT,
    bias FLOAT,
    n_items INT,
    last_updated TIMESTAMP,
);

-- Final scores
CREATE TABLE final_scores (
    item_id STRING PRIMARY KEY,
    final_score INT,
    confidence FLOAT,
    source STRING,  -- "llm_only", "human_only", "merged"
    audit_trail JSON,
    created_at TIMESTAMP,
);
```

### 6.6 Queue Architecture

```
Intake Queue (Kafka)
    ↓
Triage Topic
    ├─→ LLM Topic (fast, async)
    ├─→ Human Topic (by tier: tier_1, tier_2, tier_3)
    └─→ Calibration Topic (both LLM & human, 5% sample)
        ↓
    Aggregation Service (listen to all)
        ↓
    Output Topic → Database → API
```

---

## Edge Cases & Failure Modes

### 7.1 Unevaluable Items

If a human rater flags an item as unevaluable (e.g., corrupted submission, ambiguous rubric):

```python
def handle_unevaluable(item_id, reason: str):
    """What to do with items humans can't evaluate."""
    
    # Try reassigning to different rater tier
    try_reassign(item_id, alternative_tier)
    
    # If still unevaluable, escalate
    queue_for_manual_review(item_id, reason)
    
    # Fallback: use LLM ensemble consensus
    if all_humans_reject(item_id):
        fallback_score = run_llm_ensemble(item_id)
        log_confidence_low(item_id)
```

### 7.2 LLM Hallucination / Inconsistency

If LLM ensemble is inconsistent (all 3 models give different scores):

```python
if len(set([sonnet_score, gpt4_score, gemini_score])) == 3:
    # No consensus
    escalate_to_human(item_id)
    llm_confidence = 0.5  # very low
```

### 7.3 Human Unavailability

If tier-1 experts are offline:

```python
if tier_1_queue_time > 5_minutes:
    # Too slow; route to tier 2 instead
    downgrade_tier(item_id, "tier_2")
```

### 7.4 Budget Exhaustion

If we run out of budget mid-day:

```python
if llm_budget_remaining < total_remaining_items * 0.005:
    # Can't afford LLM for all; must queue for next batch
    queue_overflow_for_tomorrow(items)
    alert("Daily budget exhausted; overflow to tomorrow")
```

---

## Summary & Trade-offs

| Decision | Rationale | Alternative | Cost of Alternative |
|----------|-----------|-------------|---------------------|
| **LLM-first routing** | Fast, cheap; escalate if uncertain | Human-first | Quality improves to ρ=0.90, but cost blows to $0.30/item |
| **3-model ensemble** | Reduces hallucination | Single model | ρ drops to ~0.78; faster but less reliable |
| **Platt calibration** | Maps confidence to accuracy | Manual threshold tuning | Routing becomes unreliable; quality varies |
| **Skill tiers + accuracy tracking** | Efficient use of human budget | Uniform pay | Waste money on bad raters or overpay good ones |
| **5% calibration budget** | Continuous improvement | No calibration | Drift undetected; ρ degrades over time |
| **Greedy routing by confidence** | Maximizes ROI | Fixed 50/50 split | Misallocate resources; quality suffers |
| **Quarterly recalibration** | Detect drift without over-engineering | Monthly or no monitoring | Miss slow drift (monthly) or overspend on monitoring (daily) |

---

## Follow-Up Questions & Deeper Dives

### Q: How do you handle multi-lingual submissions?

**A:** Add a language detection step at triage. LLM evaluators work reasonably well cross-lingual. Human raters are matched by language capability (metadata in rater profile). For rare languages, may need to use specialized LLMs or accept longer latency.

### Q: What if the LLM models improve or degrade over time?

**A:** 
1. Track ρ weekly; if drops > 5%, trigger investigation
2. Quarterly: re-evaluate all 3 models on validation set; switch out if one degrades
3. Maintain a "model registry" with rolling validation scores

### Q: Can you explain the Platt scaling step more?

**A:** 
- LLM outputs "confidence" (0.0–1.0), but this doesn't map directly to accuracy
- Example: LLM says "95% confident" but is actually right 88% of the time → miscalibrated
- Platt scaling fits a logistic regression: `P(correct) = sigmoid(a * raw_confidence + b)`
- This gives us **true accuracy** for a given confidence level, which we use for routing

### Q: How do you prevent LLM-generated reasoning from being hallucinated?

**A:**
1. Use structured output with JSON schema (forces specific format)
2. Ensemble voting: if reasoning differs significantly across 3 models, flag as unreliable
3. Spot-check with humans: on calibration set, validate that reasoning matches score
4. Monitor for common hallucinations (e.g., "the submission mentions X" when X isn't present)

### Q: What's the cost breakdown if you hit the $0.10/item target?

**A:**
```
Assuming 86.4M items/day:

LLM-only (60M items): 60M × $0.005 = $300K
Human-only (20M items): 20M × $1.00 (avg) = $20M
Calibration both (6.4M items): 6.4M × $1.50 = $9.6M

Total: ~$29.9M/day ≈ $0.346/item (hmm, overbudget!)

Reality: Must tighten allocations:
- Reduce human % from 20M to 5M items
- Use more tier-3 raters ($0.50) instead of tier-2 ($1.00)
- Accept slightly lower quality: aim for ρ=0.82 instead of 0.85

Revised:
LLM-only (80M items): 80M × $0.005 = $400K
Human (5M items): 5M × $0.70 (mostly tier-3) = $3.5M
Calibration (1.4M items): 1.4M × $0.80 = $1.1M

Total: ~$4.9M/day ≈ $0.057/item ✓
```

### Q: How do you scale to multiple evaluation domains (essays, code, images)?

**A:**
1. Separate routing pipelines per domain (different LLM confidence thresholds)
2. Domain-specific Platt scaling (calibration set per domain)
3. Rater tiering also per domain (code experts ≠ essay experts)
4. Unified aggregation & final output layer

---

## Implementation Priorities (MVP → Full)

**MVP (Week 1–2):**
- Single LLM model (Claude 3.5 Sonnet) with fixed routing (confidence > 0.75 → human)
- Human pool as a FIFO queue (no tiering yet)
- Basic aggregation (majority vote if both available)

**Phase 2 (Week 3–4):**
- 3-model ensemble
- Platt calibration
- Skill tiers + accuracy tracking

**Phase 3 (Month 2):**
- Real-time budget optimization
- Quarterly recalibration & validation
- Rater bias detection

**Phase 4 (Month 3+):**
- Multi-domain support
- Automated retraining of LLM prompts
- Advanced anomaly detection
