# Curbside Dropoff Spot Ranking — FAQ

## Question Log

| # | Date | Category | Question | Status |
|---|------|----------|----------|--------|
| 1 | - | Terminology | What is "calibration" and why do we need isotonic regression? | [ANSWERED](#q1-what-is-calibration-and-why-do-we-need-isotonic-regression-answered) |
| 2 | - | Architecture | Why LambdaMART over linear / neural networks? | [ANSWERED](#q2-why-lambdamart-over-linear--neural-networks-answered) |
| 3 | - | Tradeoffs | How much should we trust historical success rates at a micro-location? | [ANSWERED](#q3-how-much-should-we-trust-historical-success-rates-at-a-micro-location-answered) |
| 4 | - | Failure | What if the perception system says a spot is occupied, but it's a false positive? | [ANSWERED](#q4-what-if-the-perception-system-says-a-spot-is-occupied-but-its-a-false-positive-answered) |
| 5 | - | Follow-up | How do you prevent the model from gaming passenger preferences (e.g., always left side)? | [ANSWERED](#q5-how-do-you-prevent-the-model-from-gaming-passenger-preferences-answered) |
| 6 | - | Terminology | What does "calibration" mean in the context of LambdaMART scores? | [ANSWERED](#q6-what-does-calibration-mean-in-the-context-of-lambdamart-scores-answered) |
| 7 | - | Math | Walk through an isotonic regression example step-by-step. | [ANSWERED](#q7-walk-through-an-isotonic-regression-example-step-by-step-answered) |
| 8 | - | Architecture | How do you avoid the ranker overriding safety rules? | [ANSWERED](#q8-how-do-you-avoid-the-ranker-overriding-safety-rules-answered) |

---

## Detailed Answers

### Q1: What is "calibration" and why do we need isotonic regression? `[ANSWERED]`

**A:**

**Calibration** means ensuring that the model's **predicted probability matches reality**. 

**Example:**
- If the model predicts "this drop-off has 80% chance of success (no replan)", then in a batch of 100 such drop-offs, roughly 80 should succeed without replan, and 20 should replan.
- If in reality only 60 succeed, the model is **uncalibrated** — it's overconfident.

**Why it matters:**
1. **Planner usability**: The planner wants to know: "Is this the best candidate (confident = 0.9) or am I picking from mediocre options (all scores ~0.51)?" Calibrated scores answer this.
2. **Online learning**: We use the calibrated score as a prior in our multi-armed bandit. If the score is miscalibrated, the bandit explores the wrong arms.
3. **Fairness**: If the model is miscalibrated for certain passenger groups (e.g., underconfident for accessibility cases), fairness audits will catch it.

**Why isotonic regression?**

LambdaMART outputs **ranking scores, not probabilities**. A LambdaMART score of 0.7 doesn't directly mean "70% success rate" — it just means "this candidate is better than others."

Isotonic regression maps the raw scores to a calibrated probability:

```
Raw score → Isotonic regression → Calibrated probability
   0.3    →        iso_reg        →      0.42
   0.5    →        iso_reg        →      0.70
   0.7    →        iso_reg        →      0.82
   0.9    →        iso_reg        →      0.95
```

**Why isotonic instead of alternatives?**

| Method | Pros | Cons | When to use |
|--------|------|------|------------|
| **Isotonic Regression** | Non-parametric; handles any shape; robust | Requires separate validation set | General case; **recommended** |
| **Platt Scaling** (sigmoid) | Simple; few params | Assumes sigmoid shape (not always true) | When you have <1K validation samples |
| **Temperature Scaling** | Fast; one parameter | Assumes linear scale shift | Quick baseline |

We use isotonic regression because LambdaMART outputs have complex, non-sigmoid shapes that isotonic handles well.

*Pointer:* [solution.md](solution.md), Section 3.3 — Calibration method

---

### Q2: Why LambdaMART over linear / neural networks? `[ANSWERED]`

**A:**

This is a **ranking problem**, not a classification problem. We want to **order candidates**, not predict an absolute score.

| Model | Ranking Quality | Inference Speed | Interpretability | Why (Not) Use |
|-------|-----------------|-----------------|------------------|--------------|
| **Linear (Logistic Regression)** | OK | <1 ms | Excellent | Too simple; can't learn nonlinear interactions (e.g., walking_distance × side_preference) |
| **Neural Network (MLP)** | Excellent | 50+ ms | Poor | Can represent any function, but too slow for <2 s latency + hard to explain to ops |
| **LambdaMART (GBDT)** | Excellent | <10 ms | Excellent | **Best tradeoff** for this problem |

**Why LambdaMART specifically?**

LambdaMART is a **listwise ranking loss** that directly optimizes NDCG (Normalized Discounted Cumulative Gain). Instead of predicting "candidate A is good/bad," it learns "candidate A should rank higher than candidate B."

```
Standard loss (pointwise):  L = sum(|y_true - y_pred|²)
LambdaMART loss (listwise): L = sum(lambda_ij * (score_i - score_j)²)
                            where lambda_ij accounts for whether swapping ranks i,j improves NDCG
```

This means LambdaMART naturally learns to rank-order candidates, which is exactly what we need.

**Speed advantage:**
- Feature extraction: ~50 ms (deterministic)
- GBDT inference: ~10 ms (tree traversal)
- Isotonic calibration: ~1 ms (lookup table)
- **Total: ~60 ms** (well within 2 s SLA)

By comparison:
- Neural network: 50 ms (feature extraction) + 50 ms (inference) = 100 ms (1.7× slower)
- More importantly, explaining a neural network decision to ops ("why was this spot ranked 2nd?") is hard; GBDT gives feature importance

**Interpretability example:**
```
Top feature importances:
1. walking_distance_m (30%)       ← candidate closer to destination
2. historical_success_rate (20%)  ← this spot rarely gets replanned
3. re_merge_cost_seconds (15%)    ← quick merge back into traffic
4. passenger_side_preference (12%)
5. lighting_quality (8%)
...
```

Ops can immediately see: "We prioritize walking distance, then reliability of the spot, then traffic impact." This is auditable and debuggable.

*Pointer:* [solution.md](solution.md), Section 3.2 — Model Architecture

---

### Q3: How much should we trust historical success rates at a micro-location? `[ANSWERED]`

**A:**

**Rule of thumb**: Only trust a historical success rate if you have ≥100 samples at that micro-location.

**Why 100?**

The historical success rate is noisy. If a 10 m × 10 m grid cell has only 10 samples, observing 9 successes (90% rate) doesn't mean it's truly 90% safe — it could be random variance.

With 100 samples, the 95% confidence interval is tighter:
- Observed: 85/100 successes (85% rate)
- 95% CI: [76%, 92%] (reasonable precision)

With 10 samples:
- Observed: 9/10 successes (90% rate)
- 95% CI: [56%, 99%] (very wide!)

**Handling sparse micro-locations:**

```python
def historical_success_rate(location_id, min_samples=100):
    count, successes = fetch_from_db(location_id)
    if count < min_samples:
        # Return default prior (city-wide average success rate)
        return 0.85  # e.g., 85% baseline
    else:
        return successes / count
```

**Alternative: Hierarchical Bayesian approach**

If you have a cold-start problem (new locations with <10 samples):

```
Prior: City-wide success rate = 0.85
Observed: Location A has 3 successes in 5 samples
Posterior: Beta(85*10, 15*10) * Beta(3+1, 5-3+1)
        = Weighted average of prior + observed
        ≈ 0.75 (closer to prior because of sparse data)
```

This is more principled than hard-thresholding at 100 samples, but more complex. For this problem, hard-threshold at 100 samples is fine.

**Empirical note:**
In our dataset, ~95% of grid cells have ≥100 samples (given 5,000 vehicles × 1000 drop-offs per vehicle per month). Cold-start is rare.

*Pointer:* [solution.md](solution.md), Section 3.1 — Features (historical_success_rate)

---

### Q4: What if the perception system says a spot is occupied, but it's a false positive? `[ANSWERED]`

**A:**

**Scenario**: Hard filter rejects a spot because occupancy grid shows 80% confidence of an obstacle, but there's nothing there.

**Consequences:**
- Best case: We miss a good candidate (candidate pool shrinks from 100 → 99)
- Worst case: No candidates pass hard filters, we fall back to rule-based (rare)

**How to handle false positives:**

1. **Conservative threshold**: Hard filter only rejects if occupancy confidence > 70% (not 50%)
   - Trade-off: Higher false positive rate (miss some real obstacles) vs. lower false negative rate (don't reject safe spots)
   - **Decision**: Set to 70% (ops prefer missing a spot over not seeing an actual obstacle)

2. **Recheck at execution time**: 
   - Hard filter rejects at t=0 (feature extraction time)
   - Motion planner checks again at t=0.5 s (execution time)
   - If occupancy has cleared (false positive decayed), planner may accept it
   - This is automatic; no manual intervention needed

3. **Learn from false positives**:
   - If a spot is hard-filtered but then never actually has an obstacle, log it
   - Over time, LambdaMART learns: "If this spot is hard-filtered only for occupancy (not legality), it's probably a false positive; other candidates are safer"
   - Soft penalize spots with high false-positive history

4. **Perception team tunes LiDAR**:
   - Reduce false positive rate in the occupancy grid (currently ~2%)
   - More aggressive temporal filtering (use 5 s history to confirm obstacles)

**Empirically:**
- Occupancy false positive rate: ~2% (measured against ground truth from human review)
- False negatives (missing real obstacles): <0.5% (we're conservative)
- Net effect: Hard filters reject ~0.1% of good candidates due to false positives

This is acceptable because candidate pool is large (~200 per block).

*Pointer:* [solution.md](solution.md), Section 2.2 — Safety (hard constraints)

---

### Q5: How do you prevent the model from gaming passenger preferences? `[ANSWERED]`

**A:**

**Problem**: If the ranker learns "passengers prefer left side," it might always recommend left, even when right is objectively better (closer, better lighting).

**Solution: Explicit constraints on feature weight**

LambdaMART learns feature weights automatically, but we can constrain them:

```python
# Hard constraint in LambdaMART training:
# The 'side_preference' feature can only reduce the score by 5%
# (i.e., don't let preference override objective quality)

constraints = {
    'side_preference': {'max_weight': 0.05},  # 5% influence
    'walking_distance_m': {'max_weight': 0.30},  # 30% influence
    're_merge_cost_seconds': {'max_weight': 0.20},
    ...
}
```

**Why this works:**
- We're saying: "You can learn that side preference matters, but don't over-optimize for it."
- This prevents the model from becoming a side-preference oracle and forgetting safety/quality

**Alternative: Explicit trade-off surfacing**

Instead of constraining weights, we **show passengers the trade-off**:

```
Recommended: Right side, 12 m walk, excellent lighting
Your preference: Left side
Alternative: Left side, 45 m walk, poor lighting

Would you like the recommended spot or your preferred spot?
```

This respects passenger autonomy while not gamifying the system.

**Fairness audit:**
- Monthly check: Group drop-offs by side preference (left-requesters vs. right-requesters vs. no-preference)
- Measure: Mean walking distance per group (should be equal, ±5 m)
- If disparity: Retrain with fairness weighting or adjust constraints

*Pointer:* [solution.md](solution.md), Section 7 — Personalization & Fairness

---

### Q6: What does "calibration" mean in the context of LambdaMART scores? `[ANSWERED]`

**A:**

A **calibrated score** is one where the predicted probability matches empirical reality.

**Example:**

```
LambdaMART outputs: raw_score = 0.75 for candidate X

WITHOUT calibration:
  → Planner interprets 0.75 as "75% likelihood of success"
  → But in reality, candidates with similar scores only succeed 60% of the time
  → Planner is overconfident; makes bad decisions

WITH isotonic calibration:
  iso_reg(0.75) = 0.60
  → Planner correctly interprets: "60% likelihood of success"
  → Planner is calibrated; makes better decisions
```

**Mathematical definition:**

A score is **calibrated** if:
```
E[actual_success | predicted_prob = p] = p

Example:
  Predicted prob = 0.7
  Among 1000 candidates with score ≈ 0.7:
    - 700 succeed (no replan)
    - 300 fail (replan)
  → This is perfectly calibrated
```

**Why it matters for this problem:**

1. **Confidence ranking**: With calibrated scores, we know which candidate to try first. If score 0.95 vs. 0.51, we know 0.95 is much better.
2. **Fallback decisions**: If all top candidates have scores < 0.65, we know to fall back to a rule-based default (not trustworthy enough).
3. **Online learning**: Multi-armed bandit uses the score as a prior. If miscalibrated, bandit explores the wrong arms.

*Pointer:* [solution.md](solution.md), Section 3.3 — Score Normalization & Calibration

---

### Q7: Walk through an isotonic regression example step-by-step. `[ANSWERED]`

**A:**

**Setup:**

You have a validation set of 1000 drop-offs with:
- Raw LambdaMART score for each candidate
- Ground truth: whether the candidate led to a replan (0 = no replan, 1 = replan)

**Goal:** Find a mapping from raw scores to calibrated probabilities of success (1 - replan).

**Step 1: Collect raw scores and outcomes**

```
Candidate 1: raw_score = 0.45, outcome = 0 (no replan, success = 1)
Candidate 2: raw_score = 0.52, outcome = 0 (no replan, success = 1)
Candidate 3: raw_score = 0.48, outcome = 1 (replan, success = 0)
...
Candidate 1000: raw_score = 0.91, outcome = 0 (no replan, success = 1)
```

**Step 2: Sort by raw score**

```
raw_score | success
0.40      | 1
0.41      | 0
0.42      | 1
0.43      | 1
0.44      | 1
...
0.90      | 1
0.91      | 1
0.92      | 1
```

**Step 3: Fit isotonic regression**

Isotonic regression finds a **monotonic (non-decreasing) mapping** from raw scores to calibrated probabilities:

```
iso_reg(0.40) = 0.50  (among candidates with score ≈0.40, 50% succeeded)
iso_reg(0.45) = 0.68  (among candidates with score ≈0.45, 68% succeeded)
iso_reg(0.50) = 0.72  (among candidates with score ≈0.50, 72% succeeded)
iso_reg(0.70) = 0.85  (among candidates with score ≈0.70, 85% succeeded)
iso_reg(0.90) = 0.96  (among candidates with score ≈0.90, 96% succeeded)
```

**Key property**: The mapping is **monotonic** — as raw score increases, calibrated probability must not decrease.

```
iso_reg(0.50) >= iso_reg(0.45)  ✓ (0.72 >= 0.68)
iso_reg(0.70) >= iso_reg(0.50)  ✓ (0.85 >= 0.72)
```

**Step 4: Apply at inference time**

On a new drop-off:
```
raw_score = ranker.predict(features)  # e.g., 0.63
calibrated_prob = iso_reg(0.63)      # e.g., 0.78

Interpretation: "This candidate has ~78% chance of success"
```

**Step 5: Validate on a test set**

Check that the calibrated scores actually match reality:
```
Test set:
  - 100 candidates with calibrated_prob = 0.70 (± 0.05)
  - Expected successes: ~70
  - Actual successes: 71  ✓ (calibrated!)

  - 100 candidates with calibrated_prob = 0.50 (± 0.05)
  - Expected successes: ~50
  - Actual successes: 48  ✓ (calibrated!)
```

**Code example (Python):**

```python
from sklearn.isotonic import IsotonicRegression
import numpy as np

# Validation set
X_val = np.array([0.45, 0.52, 0.48, ...])  # raw scores
y_val = np.array([1, 1, 0, ...])           # success (1 = no replan, 0 = replan)

# Fit isotonic regression
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(X_val, y_val)

# Inference
raw_score = 0.65
calibrated_prob = iso_reg.predict([raw_score])  # e.g., [0.78]
```

**Why isotonic and not a simpler function?**

Isotonic regression doesn't assume any functional form (like sigmoid). It learns directly from data:

```
Simple assumption (sigmoid): P(success | score) = 1 / (1 + exp(-score))
  → Assumes smooth S-shape

Isotonic (non-parametric): P(success | score) = whatever fits the data
  → Handles bumps, plateaus, anything monotonic
```

LambdaMART outputs often have complex shapes, so isotonic is more robust.

*Pointer:* [solution.md](solution.md), Section 3.3 — Score Normalization & Calibration

---

### Q8: How do you avoid the ranker overriding safety rules? `[ANSWERED]`

**A:**

**Key design principle**: Hard filters run **before** the ranker. The ranker only sees feasible candidates.

```
Architecture:

1. Generate candidates (200 per block)
   ↓
2. HARD FILTERS (legality + safety)
   ├─ Map query: no-stopping zones, fire hydrants, intersections
   ├─ Perception query: obstacles, occupied spots
   └─ Output: 50–100 feasible candidates
   ↓
3. SOFT RANKER (learned model)
   ├─ Input: only feasible candidates
   ├─ Output: ranked list of top-5
   └─ Ranker cannot override hard filters; only sees vetted candidates
```

**Why this separation is critical:**

- **Hard filters are deterministic rules** (no fire hydrant within 15 ft — period)
- **Ranker is a learned model** that can be fooled by outliers or distribution shift
- By filtering first, we ensure the ranker can never recommend an unsafe spot

**What if the ranker tries to override?**

Example: Ranker wants to score a spot that's in a no-stopping zone (should have been filtered).

```
Scenario 1: Bug in hard filter (spot slipped through)
  → Planner rejects it → log the error → fix hard filter

Scenario 2: Ranker scores a feasible spot extremely high
  → Planner uses it (safe, by hard-filter guarantee)
  → No override; ranker was never asked to override
```

**Validation**: Monthly safety audit

```
Sample 100 spots that hard filters rejected:
  1. Manually inspect (are they truly unsafe?)
  2. False positive rate: < 1% (we're rejecting safe spots)
  3. False negative rate: < 0.1% (we're missing unsafe spots)

If either threshold breached → investigate hard filter logic
```

*Pointer:* [solution.md](solution.md), Section 2 — Hard Constraint Filter

---

## Common Interview Follow-Ups

### Follow-Up 1: "Walk me through a failure case end-to-end."

**Setup**: A passenger arrives at a destination in downtown SF. It's rush hour (4 PM). The street is busy.

**Play-by-play**:

1. **Candidate generation**: System generates 200 candidates along the 4-block perimeter around destination
2. **Hard filter**:
   - Map query: "Which spots are legal?" → 180 pass (no fire hydrants, no no-stopping zones)
   - Perception query: "Which spots are occupied?" → 150 pass (some cars parked)
   - **Output**: 150 feasible candidates
3. **Ranking**:
   - Top candidate: Spot A (walking_distance=8m, success_rate=0.92, re_merge_cost=2s)
   - Score: 0.87 → calibrated_prob = 0.94
4. **Planner executes**:
   - Planner re-checks occupancy at execution time (t=0.5 s)
   - **Occupancy has changed**: A car parked in spot A (false negative from Step 2, or just-parked)
   - Planner aborts, requests rerank
5. **Reranking**:
   - Ranker scores remaining 149 candidates (~20 ms)
   - Top candidate: Spot B (walking_distance=12m, success_rate=0.88, re_merge_cost=3s)
   - Score: 0.83 → calibrated_prob = 0.91
6. **Planner executes spot B**: Success (no replan)
7. **Outcome**:
   - Passenger walks 12 m to destination (acceptable)
   - Log this failure to training data: spot A had false occupancy detection
   - Next retraining: LambdaMART learns to weight spots with high "false positive history" lower

**Discussion**:
- Hard filters caught the occupancy, but it was a false negative (didn't detect the newly-parked car at time 0)
- Reranking was fast enough (20 ms) to not disrupt the passenger
- System gracefully degraded to spot B (3× longer than ideal, but still good)
- Failure captured for future learning

### Follow-Up 2: "How do you handle the case where there are zero legal spots?"

**Scenario**: Entire destination block is in a no-stopping zone (rare, but happens near airports, government buildings).

**Solution**:

1. **Expand search radius** to ±100 m
2. **Re-generate + filter candidates** on the next block
3. **Return best-effort** + alert passenger:
   ```
   "Your preferred destination has no available drop-off spots.
    We're stopping 150 m ahead (2-minute walk) at [location].
    Expected fare adjustment: -$1.50"
   ```
4. **Cap retries**: If after expanding to ±200 m still <3 candidates, request ops human review

**Frequency**: <0.1% of drop-offs (cities have abundant curb space)

**Data**: Log these fallbacks. If a location consistently has zero candidates, flag it for manual inspection (might indicate a missing map annotation or a real constraint we didn't account for).

### Follow-Up 3: "Your model recommends spot A, but the passenger visually prefers spot B. How does that play into your design?"

**Answer**:

1. **Passenger has agency**: If passenger has a strong preference, they can override (tap on map)
2. **Override logging**: Each override is logged with reason code
   - Codes: "closer_to_door", "better_lighting", "left_side_preference", "accessibility", "other"
3. **Feedback loop**: Overrides become training data (weight = 2.0) for the next retraining
   - LambdaMART learns: "I recommended spot A, but passengers prefer spot B; what did I miss? Maybe walking_distance alone isn't enough — I should also weight [lighting quality or curb comfort]"
4. **Monthly fairness audit**: If override rate on side-preference > 5%, adjust the feature weight or retrain

**Tradeoff**:
- **Pro**: Passengers feel in control; system learns from feedback
- **Con**: If override rate is high, it signals the ranker is systematically wrong (e.g., ignoring a feature passengers care about)

This is a sign to investigate the model, not to over-rely on passenger overrides.

---

## Deployment & Calibration Deep Dive

### "How do you know if the model has drifted in production?"

**Monitoring dashboard (real-time)**:

```
Metric: Replan rate (% of top-1 candidates rejected by planner)
Baseline: 5%
Alert: If 1-hour rolling average > 7%, page on-call

Why replan rate matters:
  - High replan rate = model is recommending bad spots (drifted, or distribution shift)
  - Low replan rate = model is good (or being too conservative)
  - Target: 5% (some failure is expected; aim for low but not zero)
```

**Weekly retraining**:
- Retrain on past 7 days of data
- A/B test: New model vs. current model on holdout test set
- If new model NDCG@5 > current + 0.02, auto-deploy (canary, then full rollout)

**Trigger-based retraining**:
- If replan rate > 7% for 2 consecutive hours, investigate and retrain immediately
- If ranker-human disagreement rate > 2%, pull the model and debug

### "What does an isotonic regression plot look like?"

**X-axis**: Raw LambdaMART score (0 to 1)
**Y-axis**: Calibrated probability of success (0 to 1)

```
Calibrated Prob
      ^
    1 |           /
      |          /
  0.8 |        /
      |      /
  0.6 |    /    ← Isotonic regression line
      |  /
  0.4 | /
      |/
  0.2 /
      |_______________> Raw Score
      0    0.2  0.4  0.6  0.8  1.0
```

**Interpretation**:
- If the line is flat or non-monotonic → isotonic regression failed; model may be miscalibrated
- If the line is close to diagonal (y=x) → model was already well-calibrated before isotonic (LambdaMART did a good job)
- If the line is below diagonal early (e.g., 0.5 raw → 0.3 calibrated) → model was overconfident on low scores

---

## Edge Cases & Gotchas

### Edge Case 1: Passenger gets out on the wrong side

**Scenario**: Passenger requested left side, but ranker recommended right side. Passenger gets out on the wrong side.

**Problem**: Passenger walks in the wrong direction, realizes mistake, takes an Uber home.

**Solution**:
- Explicit UI warning: "You requested left side, but the model recommends right (12 m closer). Top choice: right. Confirm?"
- Surface the trade-off: Let passengers trade off their preference against objective quality
- Log the choice: If passenger chooses left despite recommendation, it's not an override failure — it's informed choice

### Edge Case 2: Spot A has high confidence, but spot B has low confidence and better location

**Scenario**:
- Spot A: calibrated_prob = 0.95, walking_distance = 50 m
- Spot B: calibrated_prob = 0.70, walking_distance = 8 m

Should we recommend spot A (confident) or spot B (better UX but risky)?

**Answer**: Return both in top-2. Let the planner decide based on real-time dynamics:
- If lane is congested, planner picks spot A (avoid adding delay)
- If lane is clear, planner picks spot B (better UX, acceptable risk)

This is the planner's job, not the ranker's.

### Edge Case 3: Regulatory rules change mid-month

**Scenario**: City announces construction; certain curb sections become no-parking zones.

**Solution**:
1. **Immediate**: Ops team manually marks zone in the map database (bypass model)
2. **Retraining cycle**: LambdaMART learns to avoid zones with "active_construction_permit=true" feature
3. **Next month**: Model automatically learns the pattern; no ops intervention needed

**Key**: Don't wait for retraining cycle to deploy safety fixes. Use hard filters (map annotations) for immediate deployment.

