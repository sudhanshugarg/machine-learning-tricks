# Curbside Dropoff Spot Ranking — Solution

---

## Clarifying Questions (Interviewer Lens)

Before diving into the architecture, confirm these assumptions with the interviewer:

1. **Scale & latency**: <2 s for candidate generation + ranking?
2. **Historical data**: Can we assume millions of past trips with annotations (success, replan, override)?
3. **Regulatory scope**: Do we need to handle jurisdiction-specific rules, or assume a single city?
4. **Passenger feedback**: Do we have post-trip satisfaction ratings or just operational signals (replan, second-dropoff)?
5. **Graceful degradation**: If no legal candidates exist, do we replan to the next block, or return an "unsafe" recommendation for human review?

---

## Goals & Constraints

### Goals (in priority order)
1. **Safety**: Never recommend a spot that violates hard constraints (no-stopping zones, fire hydrants, obstacles, intersection proximity)
2. **Passenger Experience**: Minimize walking distance to destination + maximize comfort (lighting, weather exposure, accessibility)
3. **Operational Efficiency**: Minimize re-merge cost (delay to get back into traffic), avoid replans
4. **Regulatory Compliance**: Respect jurisdiction-specific parking rules and accessibility requirements

### Hard Constraints (Non-Negotiable)
- **Legality**: No-stopping zones, fire hydrants, parking-prohibited areas, loading zones, bus stops
- **Safety**: Clear lane visibility (no obstacles within 2 m), minimum distance from intersections (15 m), not in an active lane
- **Vehicle Geometry**: Enough space to fit the vehicle (~5 m length) without overhanging a curb cut

### Soft Objectives (Scored by Ranker)
- Walking distance to destination door
- Historical success rate at this micro-location
- Passenger preference signal (side, accessibility)
- Lane congestion / re-merge cost
- Environmental comfort (lighting, weather exposure, accessible sidewalk)

---

## Architecture Overview

```
Candidate Generation
    ↓
Hard Constraint Filter (Legality + Safety)
    ↓
Soft Ranker (Learned Model)
    ↓
Ranking + Deduplication
    ↓
Top-K Candidates → Motion Planner
```

---

## 1. Candidate Generation

### Sampling Strategy

**Sample candidate stop poses every 0.5 m along the destination block face**, up to ±50 m from the route-planned drop-off location. This gives ~200 candidates per block.

```
Input: Destination block location (lat/lon), route heading
Output: List of (x, y, yaw) poses along the curb

For each 0.5 m interval:
  1. Compute position along the block edge (using map centerline)
  2. Orient pose perpendicular to the curb (standard curbside orientation)
  3. Check if pose is within the destination block or adjacent blocks
  4. Yield (x, y, yaw, block_id, segment_id)
```

**Why 0.5 m intervals?**
- Fine-grained enough to find a spot in dense urban areas
- Coarse enough to keep candidate count tractable (<300 per block)
- Empirically, 0.5 m vs 0.25 m doesn't improve UX but doubles compute

**Why ±50 m range?**
- Passenger walking budget: ~200 m = ~2.5 min walk (typical tolerance)
- Fallback to next block if no candidates here (see Failure Modes)

---

## 2. Hard Constraint Filter (Legality & Safety)

This is where we **enforce non-negotiable rules**. Only candidates that pass all hard filters are sent to the ranker.

### 2.1 Legality (From Map Data)

**Data source**: OpenStreetMap (OSM) + internal parking regulation database + Google Maps curb data

```
For each candidate (x, y):
  1. Query map: Is this location in a no-stopping zone? 
     → Query: OSM way tags (highway:no_stopping, parking:restriction)
     → Lookup: Curb database (parked cars, loading zones, bus stops)
  2. Query map: Distance to fire hydrants?
     → Threshold: >= 15 ft (jurisdictionally specific)
  3. Query map: Distance to intersections?
     → Threshold: >= 15 m (min safe distance from intersection)
  4. Query map: Curb side accessible (not blocked by building overhang)?
     → Check: Sidewalk clearance >= 1.5 m (ADA)
  5. Query map: Jurisdiction-specific rule (e.g., "no parking 9–12 AM on Tuesdays")?
     → Lookup: City parking reg API
  
  If ANY check fails → filter out candidate
```

**How to handle map latency**: Pre-cache curb data (fire hydrants, intersections, no-parking zones) in a spatial index (e.g., H3 cells at resolution 11 ~ 100 m coverage). Query the cache; refresh if cache age > 1 day or <5 candidates remain.

### 2.2 Safety (From Real-Time Perception)

**Data source**: Real-time occupancy grid from perception module (LiDAR + camera fusion)

```
For each candidate (x, y):
  1. Query occupancy grid: Is the spot occupied by a parked car or moving object?
     → Occupancy confidence > 70% → filter out
  2. Query occupancy grid: Are there obstacles 1 m in front of the spot?
     → Pedestrians, cyclists, double-parked cars → filter out
  3. Query motion planning feasibility: Can the planner safely execute a stop here?
     → Ask planner: "Can you pull into (x, y)?" 
     → Planner returns: feasible (yes/no) + estimated execution time
     → Threshold: execution time < 30 s (else too disruptive to traffic)
  
  If ANY check fails → filter out candidate
```

**Occupancy grid format**:
- Resolution: 0.2 m cells (fine enough for a 5 m vehicle)
- Update rate: 100 Hz (real-time)
- Retention: 5 s history (to detect transient obstacles)

### 2.3 Fallback if No Candidates Pass

**If fewer than 3 candidates pass hard filters:**

1. **Relax map thresholds slightly** (e.g., 12 ft from hydrant instead of 15 ft) — but only if this is explicitly approved by the safety/ops team
2. **Expand search radius** to ±100 m (next block over)
3. **Fall back to rule-based default**: Closest legal spot to the destination + alert ops/passenger ("Your preferred stop isn't available; pulling over 50 m ahead")
4. **Request human override** if the vehicle is within 200 m of the destination and has <2 candidates

**Empirical**: In our dataset, this fallback happens <0.1% of the time (urban areas have abundant curb space).

---

## 3. Soft Ranker (Learned Scoring Model)

Once hard filters are passed, the ranker **scores candidates on soft objectives** using a learned model. The top-K candidates form a shortlist for the motion planner.

### 3.1 Features

**Top 10 features (by importance in our LambdaMART model):**

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| **walking_distance_m** | Continuous | Map | Distance from candidate spot to destination door |
| **historical_success_rate** | Continuous | ML logs | % of past drop-offs at this spot that didn't trigger replan |
| **curb_occupancy** | Categorical | Perception | None / parked car (rare post-filter) / unclear |
| **passenger_side_preference** | Categorical | Passenger | Left / right / none |
| **passenger_accessibility_need** | Binary | Passenger | Requires ADA-accessible curb cut + smooth sidewalk |
| **re_merge_cost_seconds** | Continuous | Route planner | Time delay to merge back into traffic from this spot |
| **lighting_quality** | Continuous | External data | Streetlight coverage (0–1), from city database or satellite imagery |
| **sidewalk_width_m** | Continuous | Map | Sidewalk width; wider → more comfortable |
| **weather_exposure** | Continuous | Weather API | Wind speed, precipitation (passenger comfort) |
| **hour_of_day** | Categorical | System time | Peak hours have higher congestion / replan rates |

**Feature engineering notes**:
- **walking_distance_m**: Actual path distance (not Euclidean) via map routing engine; capped at 200 m
- **historical_success_rate**: Aggregated per 10 m × 10 m grid cell (microlocation). Updated daily; minimum 100 samples to be considered, else default to 0.85
- **re_merge_cost_seconds**: Computed by motion planner as part of feasibility check; correlates with lane congestion and time-of-day
- **lighting_quality, sidewalk_width**: Cached at dataset build time; refreshed monthly
- **passenger_side_preference**: Binary (left/right) or None. If None, we don't penalize either side

### 3.2 Model Architecture

**LambdaMART (gradient boosted decision trees with listwise ranking loss)**

Why LambdaMART over linear/neural?
1. **Speed**: Tree-based inference is <10 ms (important for <2 s SLA)
2. **Interpretability**: Feature importance is transparent (debuggable for ops)
3. **Robustness**: Handles categorical features natively; less prone to overfitting on small data
4. **Ranking-aware loss**: LambdaMART optimizes pairwise ranking, not pointwise scores

**Model training**:
- **Training data**: 1M+ drop-offs per month, labeled with:
  - `y_label = passenger_satisfaction_1_to_5` (if available) or `y_label = 0` if replan occurred, `y_label = 1` if no replan (binary signal)
  - Features extracted at time of drop-off
- **Loss function**: LambdaMART with pairwise ranking loss (XGBoost's `rank:ndcg`)
- **Cross-validation**: Temporal split (train on month 1–11, validate on month 12) to avoid data leakage
- **Hyperparameters**: max_depth=5, learning_rate=0.1, num_leaves=100 (tuned on validation set)

**Inference**:
```python
# Pseudo-code
candidates = hard_filter(candidates)  # 50–100 candidates remain
features = extract_features(candidates)  # shape: (N, 10)
scores = ranker_model.predict(features)  # shape: (N,)
ranked = candidates[np.argsort(-scores)]  # sort by descending score
return ranked[:5]  # top-5 candidates
```

**Latency breakdown**:
- Feature extraction: ~50 ms
- Ranking inference: ~10 ms
- **Total**: ~60 ms (well within 2 s budget)

### 3.3 Score Normalization & Calibration

**Do we need calibration?**

Yes, for two reasons:
1. **Planner usability**: The planner needs to know if the top score is a confident recommendation (0.9) or uncertain (0.51)
2. **Online learning**: We use scores as priors in our multi-armed bandit (see Evaluation)

**Calibration method: Isotonic Regression**

```
1. Hold out a validation set of 100K drop-offs
2. For each drop-off, compute:
   - pred_score = ranker_model.predict(features)
   - actual_replan = 1 if replan occurred, 0 otherwise
3. Fit isotonic regression: calibrated_prob = iso_reg.fit(pred_score, 1 - actual_replan)
4. Apply at inference time:
   - raw_score = ranker_model.predict(features)
   - calibrated_score = iso_reg.transform(raw_score)  # now in [0, 1], interpretable as "P(no replan)"
```

**Why isotonic instead of Platt scaling?**
- Isotonic is non-parametric; handles arbitrary nonlinearities in the LambdaMART output
- Platt assumes a sigmoid shape, which doesn't always hold for tree models

**Empirical calibration curve**:
```
Raw score 0.5 → calibrated prob 0.7 (the top candidate has ~70% chance of success)
Raw score 0.7 → calibrated prob 0.85
Raw score 0.9 → calibrated prob 0.95
```

---

## 4. Ranking & Deduplication

### Output Format

```json
{
  "candidates": [
    {
      "id": "cand_1",
      "x": 40.7128,
      "y": -74.0060,
      "yaw": 0.5,
      "raw_score": 0.875,
      "calibrated_score": 0.92,
      "walking_distance_m": 12.5,
      "reason": "Close to destination, high historical success, good lighting"
    },
    {
      "id": "cand_2",
      "x": 40.7125,
      "y": -74.0055,
      "yaw": 0.5,
      "raw_score": 0.820,
      "calibrated_score": 0.88,
      "walking_distance_m": 18.0,
      "reason": "Further from destination, but better re-merge cost"
    },
    ...
  ],
  "fallback_reason": null,
  "generated_at_ms": 1234567890,
  "latency_ms": 62
}
```

### Deduplication Logic

**Problem**: Two candidates very close to each other (e.g., 0.5 m apart) with nearly identical scores are redundant.

**Solution**: Cluster candidates by location; keep only the highest-scoring candidate per cluster.

```python
def deduplicate(candidates, cluster_radius_m=2.0):
    sorted_cands = sorted(candidates, key=lambda c: c['calibrated_score'], reverse=True)
    deduped = []
    for cand in sorted_cands:
        if not any(dist(cand, c) < cluster_radius_m for c in deduped):
            deduped.append(cand)
    return deduped[:5]  # return top-5 clusters
```

---

## 5. Human & Regulatory Override

### Where Humans Come In

1. **Passenger preference override**: "I want to be dropped on the left side" → soft feature, not hard filter
2. **Regulatory exception**: City issues a permit for construction; planner needs to know which spots are blocked → human annotation in map data
3. **Ops incident response**: A spot caused repeated replans → ops team marks it as "avoid until fix" (temporary rule)
4. **Fairness/accessibility**: Passenger with mobility aid needs a specific curb cut → hard constraint, not soft score

### Learning from Overrides

**Capture failure modes via labeling pipeline:**

```
When a rider overrides the recommended spot:
  1. Log: (recommended_rank, passenger_override_choice, reason_code)
     Reason codes: "closer_to_door", "better_lighting", "left_side_preference", "accessibility", "other"
  2. Label: Add (features, override_choice) as a training example with label weight 2.0 (higher importance)
  3. Retrain monthly: LambdaMART learns to avoid recommending spots that get overridden
  4. Analyze: If override rate > 5% on a spot, flag it for inspection (may be perception blind spot or latent safety issue)
```

**Example**:
- Spot A is ranked #1 by model, but passenger always walks 100 m further to Spot D
- We label (features_A, label=0, weight=2.0) to penalize similar spots in future
- LambdaMART learns: "walking_distance is important, but so are other factors like lighting"

### Long-Tail Scenarios

**Construction zone**:
- Historical data: Construction announced → spots in that zone are repeatedly avoided by model
- Solution: Ops team marks zone as "avoid_until_date" → hard-filter it out (bypass model)
- Alternative: Retrain model to learn the feature "has_active_construction_permit" from city API

**Special event (concert, parade)**:
- Predicted passenger demand spike; many riders arrive at same location simultaneously
- Solution: Increase cluster_radius to 5 m to spread vehicles across more curbs (reduce congestion)
- Alternative: Dynamically adjust feature weights (increase re_merge_cost_seconds weight) to favor less-congested blocks

**Passenger requests accessibility code (e.g., "building is locked; wait 5 minutes")**:
- Model can't predict this
- Solution: Passenger provides override instruction → log it (label for future) + relay to ops
- Long-term: Train a separate language model to extract accessibility instructions from ride notes; use as a soft feature

---

## 6. Evaluation

### Offline Metrics (Before Deployment)

**Ranking quality:**
- **NDCG@5** (Normalized Discounted Cumulative Gain): Are the top-5 candidates closer to where passengers actually want to be?
  - Target: NDCG@5 > 0.85 (vs. rule-based baseline)
- **Spearman correlation**: How well does the model's ranking correlate with passenger satisfaction?
  - Target: Spearman ρ > 0.7 (moderate-to-strong correlation)

**Safety**:
- **Hard filter coverage**: % of drop-offs where >=3 candidates pass hard filters
  - Target: >= 99.9% (only 0.1% fallback rate)
- **False positive rate**: % of hard-filtered candidates that later have occupancy issues (at execution time)
  - Target: < 1% (hard filters are conservative but not overly so)

**Fairness**:
- **Accessibility override rate**: % of riders with accessibility needs who override the recommended spot
  - Target: < 10% (model is satisfying accessibility needs)
- **Side-preference accuracy**: % of riders requesting left/right side who get the preferred side in top-3
  - Target: > 90%

### Online Metrics (After Deployment)

**User satisfaction**:
- **Passenger satisfaction rating** (1–5 stars, optional post-ride survey)
  - Target: >= 4.2 average (baseline: 4.0)
- **Willingness-to-rate**: % of riders who rate the drop-off experience
  - Target: > 20% (high engagement signal)

**Operational efficiency**:
- **Replan rate**: % of drop-offs where the planner rejected the top-1 candidate and re-ranked
  - Target: < 5% (top-1 is usually usable)
- **Second-pullover rate**: % of riders who are dropped off twice (once replan + once successful)
  - Target: < 1%
- **Time-to-spot**: Wall-clock time from "destination reached" to "vehicle stopped"
  - Target: < 15 s (median; includes perceptual latency, planning, execution)

**Model health**:
- **Ranker-vs-ops-override rate**: % of drop-offs where ops manually flagged a spot (fallback, blocked, safety)
  - Target: < 0.5% (model is rarely overridden)
- **Feature drift**: Monthly drift in feature distributions (e.g., historical_success_rate shifting due to map changes)
  - Target: KL divergence < 0.1 between months

### Online Learning (Multi-Armed Bandit)

We want to **balance exploitation** (trust the learned ranker) with **exploration** (try new candidate-generation strategies).

**Thompson Sampling approach:**

```
For each drop-off:
  1. Rank candidates using LambdaMART
  2. With prob epsilon (e.g., 0.1), instead sample a candidate uniformly from top-5
  3. Execute the selected candidate
  4. Observe outcome: (replan=yes/no, satisfaction=1-5)
  5. Update posterior on candidate quality
  6. Decrease epsilon over time (exploitation increases)

Advantage: Naturally balances exploration/exploitation without manual tuning
```

**Alternative: Contextual bandit**

If we want to vary exploration by context (time-of-day, location density, passenger type):

```
For each drop-off:
  1. Compute context vector: (hour, location_density, passenger_segment)
  2. Rank candidates using LambdaMART
  3. Sample candidate with probability: P(candidate) ∝ exp(beta * score + context_bonus)
  4. Observe outcome
  5. Update posterior per context
```

---

## 7. Personalization & Fairness

### Passenger Preferences

**In-scope**:
- **Side preference** (left/right curb): Strong signal; should be a feature
- **Walking distance budget**: Accessible riders may have lower tolerance
- **Accessibility requirement** (ADA curb cut, smooth sidewalk): Hard constraint

**Out-of-scope** (privacy risk):
- Passenger age, gender, or other demographic (fairness: don't profile)
- Passenger history of where they *always* prefer to be dropped (memorization risk)

### Implementation

```python
features = {
    'walking_distance_m': compute_distance(candidate, destination),
    'side_preference_match': 1 if (passenger_side == 'left' and candidate_on_left) else 0,
    'accessibility_curb_cut': 1 if candidate.has_curb_cut and passenger.needs_accessibility else 0,
    ...
}
```

**Fairness audit (monthly)**:
- Group drop-offs by accessibility status (needs vs. doesn't need)
- Compare metrics:
  - Mean walking distance: should be equal (±5 m)
  - Replan rate: should be equal (±2%)
  - Passenger satisfaction: should be equal (±0.2 stars)
- If disparity detected, investigate and retrain with fairness-aware weighting

---

## 8. Failure Modes & Graceful Degradation

### Mode 1: Spot Becomes Occupied Between Generation & Execution

**Problem**: Hard filters validate spot at t=0, but by t=1 s (execution time), a car parks there.

**Solution**:
1. **Planner checks again** before executing stop (real-time re-query of occupancy grid)
2. **If occupied**: Planner rejects candidate + asks ranker for next best
3. **Re-ranking**: Ranker re-scores remaining candidates (fast, ~20 ms) and returns rank 2
4. **Max retries**: 3 retries (rank 1, 2, 3); if all fail, fallback to rule-based

**Latency**: 0.5 s for 3 retries; acceptable

### Mode 2: No Legal Spots on Destination Block

**Problem**: All candidates filtered by hard constraints; fewer than 3 remain.

**Solution**:
1. Expand search radius to ±100 m (next block)
2. Re-generate + filter candidates
3. Return best-effort recommendation + alert passenger: "Your preferred stop isn't available; pulling over 50 m ahead"
4. If still <3 candidates: Request ops review (rare; should be <0.1%)

### Mode 3: Spot Blocked by Oncoming Vehicle (Execution Failure)

**Problem**: Motion planner executes stop, but a vehicle is in the way (bike, scooter, pedestrian).

**Solution**:
1. Planner detects collision risk and aborts
2. Ranker re-ranks remaining candidates (same set, re-sorted)
3. If top candidates all fail: Fallback to "drive 50 m further, re-plan"

**Prevention**: Perception team tunes LiDAR false-positive rate (currently ~2% false positives = obstacle detected but not real). Ranker learns to penalize spots with higher false-positive history.

### Mode 4: Passenger Wants a Spot Not Recommended

**Problem**: Passenger overrides and points to a spot (e.g., "stop there, by that building").

**Solution**:
1. **High-confidence override** (passenger explicitly points): Ops accepts it; log as training example (weight=2.0)
2. **Safety check**: Run hard filters on passenger's choice. If it fails (e.g., fire hydrant), reject with explanation
3. **Retrain**: LambdaMART learns to avoid spots that passengers later override

**Frequency**: <5% of drop-offs (most passengers are satisfied with model recommendations)

---

## 9. Deployment & Monitoring

### Deployment Strategy

**Canary deployment**:
1. Deploy ranker to 5% of fleet
2. Monitor: replan rate, passenger satisfaction, ops overrides (should be similar to baseline)
3. If metrics green: Ramp to 25%, then 100%
4. Rollback threshold: If replan rate > 8% (vs. baseline 5%), auto-rollback

**A/B test** (optional):
- Control: Rule-based ranker (simple distance-based heuristic)
- Treatment: LambdaMART ranker
- Duration: 2 weeks (1M drop-offs)
- Success criteria: Passenger satisfaction +0.3 stars, replan rate -2%

### Monitoring Dashboard

**Real-time alerts:**
- Replan rate > 7% for past 1 hour → page on-call
- Ranker latency > 500 ms → investigate feature extraction bottleneck
- Hard filter coverage < 99% → map data may be stale or perception issues

**Daily report:**
- Replan rate trend (should be stable ±2%)
- Top 5 spots with highest replan rate (debug signal)
- Ops override frequency by reason (construction, accessibility, other)
- Model-vs-human disagreement rate (should be <1%)

### Retraining Schedule

- **Weekly**: Retrain on past 7 days of data (captures short-term distribution shifts)
- **Monthly**: Full retraining on past 90 days (removes old data, adapts to seasonal trends)
- **Trigger-based**: If replan rate spikes or fairness audit detects disparity, retrain immediately

---

## 10. Common Follow-Up Questions (Internal)

**Q: Why not just use a neural network ranker?**

**A**: Neural networks are slower (~50 ms inference) and less interpretable. LambdaMART (GBDTs) are <10 ms and give feature importance rankings, which are crucial for ops debugging. If we need more expressive power, we'd use a lightweight NN (e.g., 2-layer MLP) with offline distillation from the GBDT.

---

**Q: How do you handle time-of-day shifts (rush hour vs. off-peak)?**

**A**: Time-of-day is a feature in the ranker. LambdaMART learns that during rush hours, re_merge_cost_seconds is more important than walking distance. We could also train separate models per time-of-day, but that adds complexity and data fragmentation; one model with time-of-day feature is simpler and works well.

---

**Q: What if the map data is wrong (e.g., fire hydrant location is outdated)?**

**A**: Hard filters can filter out safe spots. This is a data quality issue, not a modeling issue. Solutions:
1. **Regular map audits**: Ground-truth check fire hydrant locations quarterly
2. **Feedback loop**: If a drop-off is flagged as safe by ops but hard filters rejected it, investigate the map
3. **Soften constraints slightly**: Allow 12 ft from hydrant instead of 15 ft if map data is known to be stale

---

**Q: Can you personalize more aggressively (e.g., learn per-rider preferences)?**

**A**: Risky. Memorizing individual rider preferences can lead to:
- **Privacy concern**: We're learning intimate patterns (e.g., "this rider always goes to building X")
- **Fairness**: If rider A has mobility issues, personalizing to their preference may systematize inaccessible spots

Better approach: Capture explicit preferences (side, accessibility) in features, but don't learn per-rider models.

---

**Q: How do you validate that hard filters are actually safe?**

**A**: 
1. **Offline audit**: Sample 100 candidates that hard filters rejected; manually inspect (are they truly unsafe?)
2. **Online audit**: If a candidate that passed hard filters causes a replan/accident, we have a false negative (hard filter was too lenient)
3. **Adversarial examples**: Regularly test edge cases (e.g., candidate 10 m from intersection, 15 ft from hydrant)

Target: < 1 false negative per 100K drop-offs.

---

**Q: What's the backup plan if the ML ranker goes down?**

**A**: 
1. Graceful degradation: Fall back to rule-based ranker (distance + congestion heuristic)
2. Rule-based ranker is always deployed alongside ML ranker (zero overhead)
3. RTO < 30 s: If ML ranker times out or errors, ops automatically routes to rule-based within seconds
