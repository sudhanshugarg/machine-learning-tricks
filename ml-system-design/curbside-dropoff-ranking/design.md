# Curbside Dropoff Spot Ranking — Design Question

## Context

You're building the **drop-off ranking system** for a fully-autonomous vehicle (like Waymo). When a passenger arrives at their destination, the vehicle must decide where exactly to pull over on the curbside to safely and conveniently drop off the passenger.

This is a **real-time ranking problem** that sits between the route planner (which delivers you to the destination block) and the motion planner (which executes the stopping maneuver). It runs in <2 seconds and must balance **legality, safety, passenger experience, and operational cost**.

---

## Problem Statement

**Design an ML ranking system that:**

1. **Generates candidate stop poses** along the destination block (e.g., every 0.5 m)
2. **Filters for hard constraints** (legality, safety)
3. **Scores and ranks** remaining candidates using learned features
4. **Returns a ranked list** to the motion planner, which picks the first feasible candidate

**Scale:**
- 5,000+ autonomous vehicles deployed
- Peak of ~50–100 simultaneous drop-offs in urban areas (e.g., downtown SF/LA)
- Candidate generation / ranking must complete in <2 seconds (hard real-time SLA)
- Historical drop-off data available: millions of past trips with success/failure signals

**Inputs you have:**
- Ego vehicle state (position, velocity, heading)
- Real-time perception (LiDAR, camera, radar occupancy grids)
- Map data (lane markings, parking regulations, building entrances, fire hydrants, no-stopping zones)
- Passenger preferences (door choice — left/right, accessibility requirements, walking budget)
- Historical data at this location (success rate, how often replan occurred, passenger feedback)
- Regulatory constraints per jurisdiction (e.g., 15 ft minimum from hydrant, curb law variance by city)

**Output:**
- Ranked list of 3–5 candidate stop poses with confidence scores
- Reason codes for why each candidate was filtered or ranked (for ops debugging)

---

## What Your Answer Should Cover

1. **Candidate Generation**: How do you sample candidate stop poses? What makes them feasible candidates?

2. **Hard Constraints (Legality & Safety)**:
   - Which constraints are non-negotiable? (e.g., no-stopping zones, fire hydrants, occupied spaces)
   - How do you encode map annotations (parking regulations) and real-time perception (obstacles)?
   - What happens if **zero candidates** are legal? (fallback strategy)

3. **Soft Scoring (Learned Ranker)**:
   - What are the top 5–10 features the ranker should use?
   - How do you combine them? (linear score, neural net, decision tree, LambdaMART?)
   - Why those features, and what tradeoff are you making?

4. **Calibration & Ranking**:
   - Do you need to calibrate the ranker output (temperature scaling, isotonic regression)?
   - How do you ensure the top-K candidates are actually usable for the planner?

5. **Human & Regulatory Override**:
   - Where does human decision-making come in? (ops override, passenger request, regulatory exception)
   - How do you capture rare or policy-driven failures in training data?
   - What's your strategy for long-tail scenarios (construction, special events, access codes)?

6. **Evaluation (Offline & Online)**:
   - How do you measure ranker quality offline before deployment?
   - What online metrics matter? (passenger satisfaction, replan rate, second-pull-over rate)
   - How do you detect performance degradation?

7. **Personalization**:
   - How much should passenger preferences influence ranking?
   - Privacy/fairness concerns: How do you avoid passenger profiling or accessibility discrimination?

8. **Failure Modes & Graceful Degradation**:
   - Spot becomes occupied between candidate generation and execution (re-plan in <2 s)
   - No legal spots on the destination block (fallback to nearby block)
   - Spot is blocked by an oncoming vehicle (motion planner aborts, ranker retries)
   - What does fallback look like? Rule-based default? Passive bandit?

---

## Clarifying Questions You Might Ask

- Does the passenger have a strong preference for left vs. right curbside? (Yes, capture it as a feature; big driver of satisfaction)
- Do we have real-time occupancy data (i.e., do we know which spots are currently taken)? (Yes, LiDAR + inference pipeline)
- What's the consequence of a wrong ranking? (Poor UX, replan cost ~0.5–2 s, second drop-off, ops overhead; rarely safety-critical if hard filters work)
- How much historical drop-off data do we have per location? (Millions of global trips; hotspots like airports have 10K+ per week)
- Are there regulatory differences by city/state? (Yes; fire hydrant rules, curb laws, ADA requirements vary)

---

## Common Follow-up Questions to Expect

- **"Your top feature is walking distance — isn't that deterministic from the map? Why learn it?"**
  - Answer: Yes, it's a known feature, but the ranker learns the *nonlinear* trade-off between distance and other factors, and handles edge cases (inaccessible doors, blocked sidewalks).

- **"How do you avoid the model overriding a safety rule?"**
  - Answer: Hard filters run *before* the ranker. Only feasible candidates are scored. The ranker never sees infeasible poses.

- **"What if two candidates have nearly identical scores?"**
  - Answer: Return both in top-K and let the planner pick based on real-time dynamics (lane occupancy, re-merge cost).

- **"How do you handle distribution shift (new construction, weather, time-of-day)?"**
  - Answer: Retrain monthly on recent data; online bandit to explore new candidate-generation strategies; anomaly detection on ranker disagreement with human overrides.

- **"Passenger says 'I want the left side' — do you hard-constrain it or soft-penalize?"**
  - Answer: It depends on confidence. High-confidence preference → filter; uncertain → soft penalty. Surface trade-off: "you requested left, but the right side is 10 m closer and has better lighting."
