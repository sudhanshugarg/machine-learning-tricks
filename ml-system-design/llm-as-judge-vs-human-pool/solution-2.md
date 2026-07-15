# Solution 2: Evaluation System - Operational Implementation

This document covers the engineering and operational perspectives on building the hybrid human + LLM evaluation platform. It complements [solution.md](solution.md) with focus on day-to-day operations, rater quality, calibration loops, and cost tracking.

---

## 1. System Architecture

### 1.1 Service Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     Submission Intake (API)                      │
│                  submission + rubric + metadata                  │
└────────────────────────────┬─────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│                      Routing Layer                               │
│  - compute routing score (LLM conf, rubric complexity, stakes)  │
│  - decide: LLM-only | human-only | both | critical-multi-human │
│  - check budget remaining, human capacity                       │
└────────────┬──────────────────────────┬────────────────┬────────┘
             ↓                          ↓                ↓
      ┌─────────────┐        ┌──────────────────┐  ┌──────────┐
      │  LLM Judge  │        │   Human Pool     │  │ Critical │
      │ (3 models)  │        │  (Tier 1/2/3)    │  │  Queue   │
      │             │        │                  │  │ (Multi)  │
      └─────────────┘        └──────────────────┘  └──────────┘
             ↓                          ↓                ↓
      Raw LLM outputs      Per-rater scores      Multi-rater cons.
             ↓                          ↓                ↓
      ┌─────────────────────────────────────────────────────────┐
      │          Score Aggregation Layer                       │
      │  - merge LLM + human (weighted by historical accuracy) │
      │  - apply calibration mapping                           │
      │  - compute confidence intervals                        │
      └────────────────────┬─────────────────────────────────────┘
                           ↓
      ┌─────────────────────────────────────────────────────────┐
      │         Quality Monitor & Sampling                     │
      │  - sample 5-10% of LLM-only items → route to humans    │
      │  - detect drift (LLM-human agreement < threshold)      │
      │  - trigger recalibration if needed                     │
      └────────────────────┬─────────────────────────────────────┘
                           ↓
      ┌─────────────────────────────────────────────────────────┐
      │         Result Store (DB + Event Log)                 │
      │  - final score + confidence                           │
      │  - audit trail (evaluator IDs, reasoning)            │
      │  - raw outputs (for replay)                          │
      └─────────────────────────────────────────────────────────┘
```

---

## 2. Routing Policy

**Decision Framework:**

```python
def compute_routing_decision(item, llm_eval, rubric_metadata):
    """
    Determine which evaluator pool(s) to send item to.
    
    Returns: routing_decision {
        "route": "llm_only" | "human_only" | "both" | "critical_multi",
        "reasoning": str,
        "estimated_cost": float,
        "estimated_latency": float
    }
    """
    
    # Factor 1: LLM Confidence
    llm_confidence = llm_eval['confidence']  # 0.0-1.0
    
    # Factor 2: Rubric Ambiguity (pre-computed per rubric)
    rubric_complexity = rubric_metadata['complexity_score']  # low/med/high
    
    # Factor 3: Stakes (inferred from batch metadata)
    is_critical = item['is_critical_evaluation']  # high-stakes, rare
    
    # Factor 4: Budget Status
    budget_remaining_pct = get_daily_budget_remaining() / total_daily_budget
    
    # Routing Logic
    if is_critical:
        # High-stakes items: always multi-human for consensus
        return {
            "route": "critical_multi",
            "n_humans": 3,
            "estimated_cost": 3.00,
            "estimated_latency": "30 min"
        }
    
    if llm_confidence >= 0.90 and rubric_complexity == "low":
        # Easy case: high confidence + simple rubric → LLM-only
        return {
            "route": "llm_only",
            "estimated_cost": 0.005,
            "estimated_latency": "200ms"
        }
    
    if llm_confidence >= 0.75 and rubric_complexity == "medium":
        # Medium case: decent confidence + moderate complexity
        if budget_remaining_pct > 0.30:
            return {
                "route": "both",
                "n_humans": 1,
                "estimated_cost": 1.005,
                "estimated_latency": "5-10 min"
            }
        else:
            # Budget tight: fallback to LLM-only
            return {
                "route": "llm_only",
                "estimated_cost": 0.005,
                "estimated_latency": "200ms"
            }
    
    if llm_confidence < 0.70 or rubric_complexity == "high":
        # Hard case: low confidence or complex rubric → human-only
        return {
            "route": "human_only",
            "n_humans": 1,
            "estimated_cost": 1.00,
            "estimated_latency": "10-20 min"
        }
    
    # Default fallback
    return {
        "route": "human_only",
        "estimated_cost": 1.00,
        "estimated_latency": "10-20 min"
    }
```

**Routing Policy by Case:**

| Case | LLM Conf | Rubric Complexity | Decision | Cost | Latency |
|------|----------|-------------------|----------|------|---------|
| Easy | ≥ 0.90 | Low | LLM-only | $0.005 | 200ms |
| Medium | 0.75–0.90 | Medium | Both (1 human) | $1.005 | 5–10 min |
| Hard | < 0.70 | High | Human-only | $1.00 | 10–20 min |
| Critical | Any | Any | Multi-human (3) | $3.00 | 30 min |

**Tuning:** The thresholds (0.90, 0.75, 0.70) and routing split are tuned per-rubric to hit the cost target ($0.10/item) while maintaining quality (Spearman's ρ ≥ 0.85).

---

## 3. LLM-as-Judge Implementation

### 3.1 Prompting & Output Schema

**System Prompt:**
```
You are an expert evaluator. Your job is to grade submissions against a rubric.

Key principles:
- Focus on the work quality, not the source or author identity
- Be consistent: same quality should get same score
- Provide reasoning that justifies your score
- Output JSON with: score (1-5), confidence (0.0-1.0), reasoning (str)

Bias controls:
- Option order is randomized below (avoid position bias)
- Author identity is anonymized (avoid known-model bias)
- You will be asked the same question multiple times; each answer should be independent
```

**User Prompt Template:**
```
Submission to grade:
[ANONYMIZED SUBMISSION TEXT]

Rubric:
[RUBRIC]

Requested output:
{
  "score": <int 1-5>,
  "confidence": <float 0.0-1.0>,
  "reasoning": <string, 2-3 sentences justifying the score>
}
```

### 3.2 Multi-Call Aggregation

Each item is evaluated by 3 LLM models in parallel:

```python
def evaluate_with_llm_ensemble(submission, rubric, n_calls=3):
    """
    Run LLM judge multiple times and aggregate.
    
    n_calls: 3 (Claude Sonnet, GPT-4o, Gemini 2.0)
    Returns: aggregated score + confidence
    """
    
    models = ["claude-3-5-sonnet", "gpt-4o", "gemini-2-0-pro"]
    results = []
    
    for model in models:
        # Randomize option order to avoid position bias
        shuffled_submission = shuffle_options(submission)
        
        # Call LLM
        response = llm_call(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=format_prompt(shuffled_submission, rubric),
            temperature=0.5  # slight randomness for self-consistency
        )
        
        # Parse JSON
        parsed = parse_json(response)
        results.append({
            "model": model,
            "score": parsed['score'],
            "confidence": parsed['confidence'],
            "reasoning": parsed['reasoning']
        })
    
    # Aggregate via majority vote + confidence
    scores = [r['score'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Majority vote on score
    final_score = statistics.mode(scores)  # or median if no consensus
    
    # Aggregate confidence: take median (robust to outliers)
    final_confidence = statistics.median(confidences)
    
    # Self-consistency bonus: if all 3 models agree, boost confidence
    if len(set(scores)) == 1:  # all agree
        final_confidence = min(1.0, final_confidence * 1.2)
    
    return {
        "score": final_score,
        "raw_confidence": final_confidence,
        "num_models_agree": len([s for s in scores if s == final_score]),
        "all_results": results  # for audit trail
    }
```

### 3.3 Bias Controls

**1. Option Order Randomization:**
```python
def shuffle_options(submission):
    """
    If submission contains options (e.g., multiple choice, rank-order),
    shuffle them to avoid position bias.
    """
    # Before: "A. Option 1 B. Option 2 C. Option 3"
    # After: "C. Option 3 A. Option 1 B. Option 2"
    # Remember mapping for unshuffling
    pass
```

**2. Source Anonymization:**
```python
def anonymize_submission(submission):
    """
    Remove author identity, model family, timestamps, etc.
    Prevent judge from favoring known sources.
    """
    # Remove: names, emails, model IDs, dates
    # Keep: only the content to be evaluated
    anonymized = re.sub(r"(by|from|model|author)\s+\S+", "[REDACTED]", submission)
    return anonymized
```

**3. Self-Consistency:**
Run temperature > 0 to get varied outputs from the same model, then take the mode. This helps detect hallucinations or unstable reasoning.

---

## 4. Human Pool Management

### 4.1 Per-Rater Accuracy Tracking

Maintain a **confusion matrix** per rater against gold standard items:

```python
class RaterProfile:
    rater_id: str
    tier: str  # "tier_1" (expert), "tier_2", "tier_3" (general)
    
    # Accuracy metrics (updated daily)
    accuracy: float  # P(rater_score == gold_score)
    confusion_matrix: np.ndarray  # 5x5 for Likert
    bias: float  # mean(rater_score - gold_score)
    inter_rater_kappa: float  # Cohen's κ vs other raters
    
    # Workload
    items_evaluated_today: int
    items_evaluated_week: int
    items_evaluated_lifetime: int
    
    # Status
    is_active: bool
    last_evaluated: datetime
    tier_last_reviewed: datetime
    
    def accuracy_vs_gold(self, gold_set: List[Item]) -> float:
        """Compute accuracy on gold standard items."""
        n_correct = sum(1 for item in gold_set if self.score_on(item) == item.gold_score)
        return n_correct / len(gold_set)
    
    def check_tier_qualification(self, tier_threshold=0.90):
        """Verify rater meets tier requirements."""
        return self.accuracy >= tier_threshold
```

### 4.2 Gold Set & Evaluation

Maintain a **gold set per rubric** (~500 items) with consensus ground truth:

```python
class GoldSet:
    rubric_id: str
    items: List[GoldItem]  # 500 items per rubric
    
    def __init__(self, rubric_id, n_items=500):
        # Curate: diverse items covering the rubric's range
        self.items = curate_diverse_items(rubric_id, n_items)
        
        # Evaluate: 3+ expert raters, take majority vote
        for item in self.items:
            expert_raters = get_tier_1_experts(n=3)
            scores = [r.evaluate(item) for r in expert_raters]
            item.gold_score = statistics.mode(scores)
            item.gold_kappa = compute_inter_rater_kappa(scores)
    
    def evaluate_rater(self, rater_id, sample_size=20):
        """
        Insert gold items at low frequency (1-5% of rater's work).
        Track accuracy without the rater knowing.
        """
        sample = random.sample(self.items, sample_size)
        correct = sum(1 for item in sample if rater.evaluate(item) == item.gold_score)
        accuracy = correct / sample_size
        
        # Flag raters who fail too many
        if accuracy < 0.80:
            log.warning(f"Rater {rater_id} accuracy {accuracy:.0%} below threshold")
            escalate_for_retraining(rater_id)
        
        return accuracy
```

### 4.3 Spam Detection

Insert known-answer items at low frequency and flag raters who systematically fail:

```python
def detect_spam_raters(rater_pool, gold_set, detection_rate=0.05):
    """
    Periodically check raters against gold set.
    Flag those who fail too often (likely not trying).
    """
    for rater in rater_pool:
        # Sample 1-5% of rater's assignments
        if random.random() < detection_rate:
            accuracy = gold_set.evaluate_rater(rater.id, sample_size=20)
            
            if accuracy < 0.70:
                log.alert(f"SPAM: Rater {rater.id} accuracy {accuracy:.0%}")
                # Actions:
                # 1. Suspend rater from accepting new work
                # 2. Audit their recent submissions for quality
                # 3. Re-onboard or terminate
                suspend_rater(rater.id)
```

---

## 5. Calibration Loop

### 5.1 Weekly Calibration Pipeline

Every Monday morning, re-fit the calibration mapping:

```python
def weekly_calibration():
    """
    Run every Monday 2am. Re-fit Platt scaling / isotonic regression.
    """
    
    for rubric in all_rubrics:
        # 1. Collect calibration data from past week
        #    - Items evaluated by both LLM and human
        #    - Use human score as ground truth
        cal_data = collect_calibration_samples(
            rubric=rubric,
            lookback_days=7,
            min_samples=500
        )
        
        # 2. Fit calibration mapping
        llm_confidences = np.array([d['llm_confidence'] for d in cal_data])
        human_scores = np.array([d['human_score'] for d in cal_data])
        
        # Measure: was LLM correct?
        llm_scores = np.array([d['llm_score'] for d in cal_data])
        is_correct = (llm_scores == human_scores).astype(int)
        
        # Fit Platt scaling
        platt = LogisticRegression()
        platt.fit(llm_confidences.reshape(-1, 1), is_correct)
        
        # 3. Evaluate calibration quality
        platt_pred = platt.predict_proba(llm_confidences.reshape(-1, 1))[:, 1]
        brier_loss = brier_score_loss(is_correct, platt_pred)
        
        # 4. Compare to previous week
        prev_brier = load_metric(f"{rubric}_brier_loss_prev_week")
        if brier_loss > prev_brier + 0.05:  # degraded by 5%
            log.alert(f"DRIFT: {rubric} calibration degraded. Brier {prev_brier:.3f} → {brier_loss:.3f}")
            trigger_drift_investigation(rubric)
        
        # 5. Deploy new calibration
        save_calibration(rubric, platt)
        log.info(f"Updated calibration for {rubric}. Brier loss: {brier_loss:.4f}")
```

### 5.2 Drift Alarm

Track LLM ↔ human agreement as the production metric:

```python
def monitor_llm_human_agreement():
    """
    Continuous monitoring: sample items evaluated by both LLM and human.
    Compute agreement rate and alert if it drops.
    """
    
    AGREEMENT_THRESHOLD = 0.85  # target: ρ ≥ 0.85
    ALERT_THRESHOLD = 0.80  # alarm if < 0.80
    
    for rubric in all_rubrics:
        # Sample 100 items evaluated by both in the last 24h
        recent_both = sample_items_with_both_evals(rubric, lookback_hours=24, n=100)
        
        if len(recent_both) < 50:
            log.debug(f"Not enough recent items for {rubric}. Skipping.")
            continue
        
        # Compute Spearman's ρ
        llm_scores = [item['llm_score'] for item in recent_both]
        human_scores = [item['human_score'] for item in recent_both]
        
        rho, p_value = spearmanr(llm_scores, human_scores)
        
        # Log metric
        metrics_db.append({
            "rubric": rubric,
            "timestamp": now(),
            "spearman_rho": rho,
            "n_samples": len(recent_both),
            "agreement_pct": sum(1 for i in range(len(llm_scores)) if llm_scores[i] == human_scores[i]) / len(llm_scores)
        })
        
        # Alert if degraded
        if rho < ALERT_THRESHOLD:
            log.alert(f"DRIFT ALARM: {rubric} Spearman ρ = {rho:.3f} (< {ALERT_THRESHOLD})")
            escalate_to_on_call(rubric, rho)
```

### 5.3 Quality Monitor Sampling

Periodically re-route LLM-only items to humans for validation:

```python
def quality_monitor_sample():
    """
    Every hour: sample 5-10% of LLM-only items from the past hour.
    Send them to humans for re-evaluation (without revealing LLM score).
    """
    
    SAMPLE_RATE = 0.05  # sample 5% of LLM-only
    
    for rubric in all_rubrics:
        # Get LLM-only items from past hour
        llm_only_items = get_items_by_route(
            rubric=rubric,
            route="llm_only",
            lookback_hours=1
        )
        
        # Sample a fraction
        to_validate = random.sample(
            llm_only_items,
            k=int(len(llm_only_items) * SAMPLE_RATE)
        )
        
        # Queue for human re-evaluation
        for item in to_validate:
            # Important: don't show the LLM score to the human
            queue_for_human_evaluation(
                item=item,
                hide_llm_score=True,
                priority="normal"
            )
            
            # When human completes, compare scores
            def on_human_complete(human_score):
                llm_score = item['llm_score']
                agree = (llm_score == human_score)
                
                log.metric(f"{rubric}_llm_only_agreement", int(agree))
                
                if not agree:
                    log.debug(f"LLM-only disagreement: {item.id} LLM={llm_score} Human={human_score}")
```

---

## 6. Score Aggregation

### 6.1 Multiple Human Raters

When an item is evaluated by multiple humans (e.g., for high-stakes items):

```python
def aggregate_human_scores(human_scores: List[int], rater_profiles: List[RaterProfile]) -> Tuple[int, float]:
    """
    Combine multiple human scores using accuracy-weighted aggregation.
    
    Args:
        human_scores: [3, 4, 3] from three raters
        rater_profiles: profiles with accuracy metrics
    
    Returns:
        (final_score, confidence)
    """
    
    # Weight by each rater's historical accuracy on this rubric
    weights = np.array([profile.accuracy for profile in rater_profiles])
    weights = weights / weights.sum()
    
    # Compute weighted mean
    weighted_avg = np.average(human_scores, weights=weights)
    
    # Round to nearest integer, but prefer mode for ties
    mode_score = statistics.mode(human_scores) if len(set(human_scores)) <= 2 else round(weighted_avg)
    
    # Confidence: if raters strongly agree, high confidence
    if len(set(human_scores)) == 1:
        confidence = 0.95  # unanimous
    else:
        agreement_ratio = max(human_scores.count(mode_score) / len(human_scores))
        confidence = 0.7 + (agreement_ratio * 0.25)  # 0.7-0.95
    
    return mode_score, confidence
```

### 6.2 LLM + Human Aggregation

When both LLM and human evaluate the same item:

```python
def aggregate_llm_and_human(
    llm_score: int,
    llm_confidence_raw: float,
    human_scores: List[int],
    rater_profiles: List[RaterProfile],
    calibrator: IsotonicRegression  # pre-fitted
) -> Tuple[int, float, str]:
    """
    Merge LLM and human signals.
    """
    
    # Step 1: Calibrate LLM confidence
    llm_confidence_calibrated = calibrator.predict([llm_confidence_raw])[0]
    
    # Step 2: Aggregate human scores
    human_score_agg, human_confidence = aggregate_human_scores(human_scores, rater_profiles)
    
    # Step 3: Detect disagreement
    if abs(llm_score - human_score_agg) > 1.5:
        # Large disagreement: flag and prefer human
        return human_score_agg, 0.6, "disagreement_flagged"
    
    # Step 4: Weight by calibrated confidence
    total_weight = llm_confidence_calibrated + human_confidence
    final_score = (
        llm_score * llm_confidence_calibrated +
        human_score_agg * human_confidence
    ) / total_weight
    
    final_confidence = min(llm_confidence_calibrated, human_confidence)
    
    return int(round(final_score)), final_confidence, "llm_human_merged"
```

---

## 7. Cost Model

### 7.1 Per-Item Cost Breakdown

```python
def compute_item_cost(routing_decision):
    """
    Break down the cost of evaluating one item.
    """
    
    # LLM evaluation cost
    if "llm" in routing_decision['route']:
        n_judges = 3  # 3 models in ensemble
        cost_per_judge = 0.001  # varies by model
        llm_cost = n_judges * cost_per_judge
    else:
        llm_cost = 0
    
    # Human evaluation cost
    if "human" in routing_decision['route']:
        if routing_decision['route'] == "critical_multi":
            n_humans = 3
            cost_per_human = 2.00  # tier-1 expert
        else:
            n_humans = 1
            # Cost varies by tier
            tier = route_to_tier(routing_decision)
            cost_per_human = {"tier_1": 2.00, "tier_2": 1.00, "tier_3": 0.50}[tier]
        
        human_cost = n_humans * cost_per_human
    else:
        human_cost = 0
    
    total_cost = llm_cost + human_cost
    
    return {
        "llm_cost": llm_cost,
        "human_cost": human_cost,
        "total_cost": total_cost,
        "route": routing_decision['route']
    }

def track_daily_costs():
    """
    Monitor daily spend against budget.
    """
    
    DAILY_BUDGET = 8.64e6 * 0.10  # $864K for 86.4M items at $0.10 avg
    
    today_spend = sum_costs_today()
    items_evaluated = count_items_today()
    avg_cost_per_item = today_spend / items_evaluated if items_evaluated > 0 else 0
    
    projected_spend = today_spend * (24 / hours_elapsed())
    
    metrics.log({
        "today_spend": today_spend,
        "projected_daily_spend": projected_spend,
        "avg_cost_per_item": avg_cost_per_item,
        "budget_remaining": DAILY_BUDGET - projected_spend,
        "budget_utilization": projected_spend / DAILY_BUDGET
    })
    
    if projected_spend > DAILY_BUDGET * 1.10:
        log.alert(f"BUDGET OVERRUN: projected ${projected_spend:.0f} vs ${DAILY_BUDGET:.0f}")
        # Tighten routing: increase LLM-only threshold
        tighten_routing_threshold(confidence_threshold=0.85)
```

### 7.2 Cost Equation

```
per_item_cost = p_human · cost_human + (1 − p_human) · cost_LLM · n_judges

where:
  p_human = fraction of items sent to human pool
  cost_human = avg. cost per human eval (mix of tier-1/2/3)
  cost_LLM = cost per LLM call (~$0.001 per call)
  n_judges = num. LLM models in ensemble (typically 3)

Example:
  70% LLM-only, 30% human
  cost = 0.30 · $1.00 + 0.70 · $0.003 · 3
       = $0.30 + $0.0063
       = $0.3063/item

To hit $0.10 target:
  0.10 = p_human · $1.00 + (1 − p_human) · $0.003
  0.10 = $1.00 · p_human + $0.003 · (1 − p_human)
  p_human ≈ 0.095 ≈ 10%

So: ~10% to humans, ~90% LLM-only achieves $0.10 avg cost.
```

Tune the routing thresholds (LLM confidence cutoffs) to hit the target split.

---

## 8. Audit & Replay

### 8.1 Storing Raw Evaluator Outputs

```python
class EvaluationRecord:
    """Store everything needed to replay or audit."""
    
    item_id: str
    rubric_id: str
    created_at: datetime
    
    # Submission
    submission: str  # raw or anonymized version
    
    # Routing decision
    routing_decision: dict  # {route, reasoning, cost}
    
    # LLM evaluations
    llm_evaluations: List[dict]
    # [
    #   {"model": "claude", "score": 4, "confidence": 0.92, "reasoning": "..."},
    #   {"model": "gpt4o", "score": 4, "confidence": 0.88, "reasoning": "..."},
    #   {"model": "gemini", "score": 3, "confidence": 0.75, "reasoning": "..."}
    # ]
    
    # Human evaluations
    human_evaluations: List[dict]
    # [
    #   {"rater_id": "rater_123", "tier": "tier_1", "score": 4, "time_taken_sec": 120},
    #   {"rater_id": "rater_456", "tier": "tier_2", "score": 4, "time_taken_sec": 90}
    # ]
    
    # Aggregation
    final_score: int
    final_confidence: float
    aggregation_method: str  # "llm_only", "human_only", "llm_human_merged"
    
    # Calibration (at evaluation time)
    calibration_version: int  # which calibration mapping was used
    llm_confidence_raw: float
    llm_confidence_calibrated: float
```

**Database Schema:**
```sql
CREATE TABLE evaluation_records (
    item_id STRING PRIMARY KEY,
    rubric_id STRING,
    created_at TIMESTAMP,
    submission BLOB,
    routing_decision JSON,
    llm_evaluations JSON,  -- array of evaluator outputs
    human_evaluations JSON,  -- array of rater results
    final_score INT,
    final_confidence FLOAT,
    aggregation_method STRING,
    calibration_version INT,
    INDEX (rubric_id, created_at)
);
```

### 8.2 Replay on Rubric Change

When a rubric is updated, re-evaluate historical items without re-collecting:

```python
def replay_with_new_rubric(old_rubric_id, new_rubric_id, start_date, end_date):
    """
    Re-evaluate items from the past using the new rubric.
    Use stored raw LLM/human outputs, re-run through new aggregation pipeline.
    """
    
    # 1. Fetch all evaluation records for date range
    old_records = query_evaluation_records(
        rubric_id=old_rubric_id,
        created_at_gte=start_date,
        created_at_lte=end_date
    )
    
    # 2. For each record, re-run aggregation with new rubric
    new_records = []
    for record in old_records:
        # Use the raw LLM and human outputs from before
        llm_evals = record.llm_evaluations
        human_evals = record.human_evaluations
        
        # Re-aggregate with new calibration mapping
        new_calibrator = load_calibration(new_rubric_id)
        
        new_score, new_conf, method = aggregate_llm_and_human(
            llm_score=llm_evals[0]['score'],
            llm_confidence_raw=llm_evals[0]['confidence'],
            human_scores=[h['score'] for h in human_evals],
            rater_profiles=[...],
            calibrator=new_calibrator
        )
        
        new_record = EvaluationRecord(
            item_id=record.item_id,
            rubric_id=new_rubric_id,
            final_score=new_score,
            final_confidence=new_conf,
            aggregation_method=method,
            llm_evaluations=llm_evals,  # reuse
            human_evaluations=human_evals  # reuse
        )
        
        new_records.append(new_record)
    
    # 3. Store replayed results
    bulk_insert_evaluation_records(new_records)
    
    log.info(f"Replayed {len(new_records)} items from {old_rubric_id} to {new_rubric_id}")
```

**Benefits:**
- No need to re-collect human judgments
- Fast iteration on rubric changes
- Audit trail shows what changed and why

---

## 9. Real-World Rubric Examples

### Example 1: Text Summarization Quality

```json
{
  "rubric_id": "summarization_v2",
  "task": "Score the quality of a summary (0-500 words) of a long article",
  "complexity_score": "high",
  "scoring_criteria": [
    {
      "score": 5,
      "description": "Summary captures all key points, is accurate, concise, and well-organized"
    },
    {
      "score": 4,
      "description": "Summary captures most key points, mostly accurate, slightly verbose"
    },
    {
      "score": 3,
      "description": "Summary captures some key points, minor inaccuracies, longer than needed"
    },
    {
      "score": 2,
      "description": "Summary misses important points, some inaccuracies, unclear organization"
    },
    {
      "score": 1,
      "description": "Summary is poor quality: missing key points, inaccurate, incoherent"
    }
  ]
}
```

**Routing in practice:**
- **Easy case:** Summary is clearly excellent or terrible → LLM-only (high confidence)
- **Hard case:** Summary is borderline (score 2-4) → human review (ambiguity is high)
- **Critical case:** Summary is for a safety-critical domain (medical, legal) → multi-human consensus

### Example 2: Factual Correctness

```json
{
  "rubric_id": "factual_correctness_v1",
  "task": "Score the factual accuracy of a statement against a reference",
  "complexity_score": "medium",
  "scoring_criteria": [
    {
      "score": 5,
      "description": "All claims are factually correct and well-supported"
    },
    {
      "score": 4,
      "description": "Most claims correct; 1 minor factual issue or unsupported claim"
    },
    {
      "score": 3,
      "description": "Mixed; some claims correct, some incorrect or unsupported"
    },
    {
      "score": 2,
      "description": "Most claims are incorrect or unsupported"
    },
    {
      "score": 1,
      "description": "Severely inaccurate; most claims are false"
    }
  ]
}
```

**Calibration observations:**
- LLM judges excel at this (high confidence ≥ 0.85)
- Humans sometimes miss subtle errors (lower agreement on nuanced cases)
- Gold set includes edge cases (claims that are technically true but misleading)

---

## 10. Operational Dashboard

**Key metrics to surface:**

```python
metrics_dashboard = {
    "cost": {
        "daily_spend": "$450K",
        "avg_cost_per_item": "$0.105",
        "budget_target": "$864K",
        "utilization": "52%"
    },
    "quality": {
        "llm_human_agreement": {
            "summarization": 0.84,  # Spearman ρ
            "factual_correctness": 0.88,
            "overall": 0.86
        },
        "rater_accuracy": {
            "tier_1": 0.93,
            "tier_2": 0.82,
            "tier_3": 0.71
        },
        "inter_rater_kappa": 0.72
    },
    "routing": {
        "llm_only_pct": 88,
        "human_only_pct": 10,
        "both_pct": 2,
        "p99_latency_llm_only": "250ms",
        "p99_latency_human": "12 min"
    },
    "alerts": [
        {
            "rubric": "summarization_v2",
            "alert": "DRIFT: Spearman ρ dropped from 0.87 to 0.82",
            "action": "Investigating LLM model drift. Triggered recalibration."
        }
    ]
}
```

---

## 11. Week-by-Week Operational Checklist

| Day | Task | Owner | Frequency |
|-----|------|-------|-----------|
| Mon 2am | Weekly calibration: refit Platt scaling for each rubric | Data Eng | Weekly |
| Mon 8am | Calibration review: surface drift alarms, changes | ML Eng | Weekly |
| Daily 6am | Cost report: daily spend, utilization, projections | Finance | Daily |
| Daily 12pm | Quality report: LLM-human agreement by rubric | ML Eng | Daily |
| Daily | Rater onboarding: test on gold set, gate by tier | Ops | Continuous |
| Weekly | Gold set audits: ensure diversity, update for drift | Data Eng | Weekly |
| Monthly | Rater performance reviews: recompute accuracy, flag demotions | Ops | Monthly |
| Quarterly | Rubric health reviews: are evals still meaningful? | Product | Quarterly |

---

## Summary: Interview Talking Points

**Routing Policy:**
"Easy cases go LLM-only (high confidence + simple rubric), hard cases go to humans, critical cases always get multi-human consensus. Tune the thresholds per-rubric to hit the cost target."

**Calibration Loop:**
"We maintain a 500-item gold set per rubric. Every week, we refit Platt scaling from items evaluated by both LLM and human. We monitor Spearman's ρ between LLM and human scores; if it drops below 0.80, we sound an alarm and investigate drift."

**Rater Quality:**
"Each rater has an accuracy profile. We insert gold items at 1-5% frequency to catch spam. We compute inter-rater κ to spot outliers. Bad raters get retrained or terminated."

**Audit & Replay:**
"Every evaluation record stores the raw LLM completions and individual human scores. When a rubric changes, we replay without re-collecting by running the same evaluations through the new calibration mapping."

**Cost Modeling:**
"Per-item cost ≈ p_human · $1.00 + (1 − p_human) · $0.003. To hit $0.10 average, we need ~10% human and ~90% LLM-only. We monitor daily spend and tighten routing in real-time if overbudget."
