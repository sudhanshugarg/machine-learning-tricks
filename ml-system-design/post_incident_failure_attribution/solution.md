# Solution: Post-Incident Root-Cause Attribution for Autonomous Vehicle Failures

## Step 1: Clarifying Questions & Requirements

### Questions to Ask the Interviewer

**What counts as an "incident"?**
- Just true collisions, or the full severity spectrum (collisions, near-misses, disengagements, harsh braking, abrupt replans)? (Assume: the full spectrum, with severity tiers — the rare severe cases need the highest-confidence analysis, but the frequent low-severity cases are the main *volume* the automated system must handle.)
- Is there an existing trigger/detector that flags "something incident-worthy happened," or does this system also need to do that detection itself? (Assume triggers already exist — e.g., harsh braking/steering thresholds, safety-driver disengagement flags, collision sensors — and this system starts from "an incident window has been flagged.")

**What data is available per incident?**
- Full raw sensor logs (camera, LiDAR, radar)? Every sub-system's intermediate outputs (detections, tracks, predicted trajectories, planned trajectory, executed controls)? (Assume yes to both — this is what makes root-cause attribution possible at all.)
- Is there any post-hoc ground truth (e.g., a human-reviewed "what actually happened" annotation) for at least some past incidents to bootstrap supervised learning?

**Latency requirements**
- Does triage need to happen within minutes (e.g., to catch a fleet-wide issue fast) or is same-day/next-day acceptable for most incidents? (Assume: severe incidents need same-day turnaround; bulk near-miss triage can be nightly batch.)
- Is there a separate real-time on-vehicle monitoring requirement, or is this purely an offline, post-hoc, data-center system? (Assume offline/post-hoc for this system; note the real-time on-vehicle case as a related-but-distinct problem in follow-ups.)

**Ground truth & labeling budget**
- How many past incidents have human-adjudicated root-cause labels today? (Assume: a small, high-quality set from safety review, but far too small to train a robust classifier alone — the design must lean heavily on counterfactual replay and weak supervision.)
- Is there a simulation/replay environment where sub-system outputs can be substituted with ground truth to test counterfactuals? (Assume yes — this is a common capability in AV development stacks and is central to the design.)

**Consumers & success criteria**
- Is the primary consumer safety engineers doing incident review, or also the Perception/Prediction/Planning teams needing routed bug reports? (Assume both.)
- What does "success" look like — faster triage, more consistent attribution, catching things humans miss, or all three?

---

## Step 2: Goals, Requirements & Assumptions

**Assumed Scale**
- Thousands of vehicles; incident triggers (harsh braking, abrupt replans, disengagements) fire on the order of thousands of times per day fleet-wide; true collisions are extremely rare (a handful per year, in line with real-world AV safety records); severe near-misses are more frequent but still rare relative to routine driving.
- Each incident log spans a window of raw sensor data plus every pipeline stage's intermediate outputs for some lead-in period before the trigger (e.g., 10-30 seconds) through the trigger event itself.

**Functional Requirements**
- FR1: Given an incident's full log, produce a root-cause attribution: which sub-system(s) (Perception / Prediction / Planning / sensor-hardware / none-ML-i.e.-unavoidable) most likely caused or most contributed to the incident.
- FR2: Distinguish **root cause** from **downstream symptom** when multiple sub-systems show anomalies in the same incident (e.g., a degraded Perception confidence cascading into a bad Prediction — Perception is the root cause, Prediction's error is a symptom).
- FR3: Provide **supporting evidence** alongside the attribution (which specific signals/anomalies drove the conclusion) so an engineer can quickly verify rather than blindly trust it.
- FR4: Route high-confidence attributions to the owning ML team with a structured failure report; route low-confidence or ambiguous cases to human review.
- FR5: Continuously monitor each sub-system's health in production (not just reactively after an incident) via an observability layer, to catch degradations early.
- FR6: Feed confirmed failure cases back into the long-tail training-data mining pipeline (see [driving_scene_search](../driving_scene_search/solution.md)) so the underlying scenario gets represented in future training/eval sets.

**Non-Functional Requirements**
- Turnaround: severe incidents (collision, severe near-miss) triaged same-day; bulk lower-severity incidents processed in nightly batch.
- Precision on root-cause attribution matters *more* than recall for routing (a wrong attribution sends a fix to the wrong team and wastes engineering time and, worse, leaves the real fault unaddressed) — but the system should never silently suppress an incident; low-confidence cases must surface for human review rather than being auto-closed.
- Observability monitoring: near-real-time (minutes-level lag) for fleet-wide health dashboards, so a systemic regression (e.g., a bad model rollout) is caught fast.
- Explainability: every automated attribution must come with evidence a human can audit — this is a safety system, not a black box classifier we blindly trust.

**Non-Goals (assumed, confirm with interviewer)**
- Real-time on-vehicle failure prediction/intervention is out of scope — this system operates on logs after the fact (though its anomaly detectors may later inform an on-vehicle monitor; see follow-ups).
- Fully autonomous "auto-fix" is out of scope — the system attributes and routes; humans still design and validate the fix.

---

## Step 3: System Architecture (High-Level)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    INCIDENT TRIGGER (existing upstream systems)        │
│  collision sensor / disengagement flag / harsh-braking or              │
│  abrupt-replan threshold / safety-driver takeover                      │
└──────────────────────────────────┬─────────────────────────────────-─┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LOG RETRIEVAL & RECONSTRUCTION                                        │
│  - Pull full raw sensor + per-stage intermediate outputs for the        │
│    incident window (lead-in + event)                                   │
│  - Time-align sensor, perception, prediction, planning, control logs    │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  PER-STAGE ANOMALY DETECTION   (Step 4A)                               │
│  - Sensor-level anomaly detectors (occlusion, dropout, noise)           │
│  - Perception-level (confidence, track stability, misses/misclass)      │
│  - Prediction-level (trajectory error vs. what actually happened)       │
│  - Planning-level (replan frequency, safety-margin, jerk)               │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  COUNTERFACTUAL REPLAY ENGINE   (Step 4B)                              │
│  - Re-run downstream stages with ground-truth/idealized upstream        │
│    inputs substituted in, to isolate which stage's error actually       │
│    changed the outcome                                                  │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  ROOT-CAUSE ATTRIBUTION CLASSIFIER   (Step 4B)                         │
│  - Combines per-stage anomaly signals + counterfactual deltas +         │
│    weak/heuristic labels + human-verified labels                        │
│  - Outputs: sub-system(s), failure type, confidence, evidence           │
└─────────────────┬─────────────────────────────┬───────────────────────┘
                  │ high confidence               │ low confidence /
                  ▼                                │ disagreement
┌──────────────────────────────┐      ┌────────────▼────────────────────┐
│  AUTOMATED ROUTING             │      │  HUMAN REVIEW QUEUE              │
│  - Structured failure report   │      │  - Engineer confirms/corrects    │
│    to owning ML team           │      │    attribution                   │
└────────────────┬───────────────┘      └────────────────┬─────────────-─┘
                 │                                        │
                 └───────────────┬────────────────────────┘
                                 ▼
                ┌───────────────────────────────────────┐
                │  FEEDBACK LOOP                          │
                │  - Confirmed labels retrain classifier  │
                │  - Confirmed scenarios feed into         │
                │    long-tail training-data mining        │
                │    (see driving_scene_search)             │
                └───────────────────────────────────────┘

                          (in parallel, always-on)
┌───────────────────────────────────────────────────────────────────────┐
│  MODEL OBSERVABILITY PLATFORM   (Step 4C)                              │
│  - Fleet-wide per-sub-system health dashboards                          │
│  - Drift detectors on input distributions & output behavior             │
│  - Alerting on threshold breaches, independent of any single incident   │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Step 4A: Anomaly Detection in Sensor & Pipeline Telemetry

### Why anomaly detection, not just "look at the final crash"

A single outcome (the vehicle braked hard) doesn't tell you *where* in the pipeline things went wrong. The key idea is to instrument **every stage** of the pipeline with its own notion of "nominal" behavior, so we can pinpoint the *earliest* stage at which something deviated from normal — that's the strongest signal for root cause, since errors typically cascade downstream from wherever they first occur.

### Defining "nominal" per stage

For each stage, establish a baseline distribution of its outputs/behavior under normal (non-incident) driving, so incident-time behavior can be compared against it (e.g., via a z-score, a learned normal-behavior model, or a percentile-rank against the fleet-wide distribution for similar conditions):

**Sensor level**
- Signal dropout/gaps, unexpected noise levels, calibration drift, lens occlusion/glare detectors (e.g., a lightweight classifier on raw frames flagging blur/occlusion/glare).
- *Anomaly signal:* sensor health score drops sharply right before the incident window.

**Perception level**
- Detection confidence trajectory for the relevant agent(s) over time (does confidence stay low or fluctuate right up until the incident, suggesting Perception was struggling?).
- Track stability: ID switches, track drops/re-acquisitions, bounding-box jitter.
- Detection *latency*: time from an agent entering sensor range to first confident detection (a late first-detection on the agent involved in the incident is a strong Perception-failure signal).
- Classification changes: an agent's class flips between frames (e.g., "unknown object" → "pedestrian" very late).

**Prediction level**
- **Trajectory error against realized outcome**: once we know what the agent *actually* did (available post-hoc, after the incident, from later frames/tracks), compute the error between what Prediction forecasted at each time step and what actually happened. A large, sustained error for the incident-relevant agent, especially right before the event, is the core Prediction-anomaly signal.
- Predicted-intent flips (e.g., predicted "continue straight" until very late, then abruptly switches to "turning/swerving").
- Multi-modal prediction calibration: if Prediction outputs a distribution over possible trajectories, was the realized trajectory in a low-probability mode? (a well-calibrated model should rarely be blindsided; frequent blindsiding on a scenario type flags a systematic Prediction gap.)

**Planning level**
- Replan frequency/magnitude: how often and how drastically the planned trajectory changed frame-to-frame in the lead-up (frequent large replans indicate Planning was reacting to a fast-changing/uncertain world model, which may itself stem from upstream instability, or indicate Planning itself thrashing given stable inputs).
- Safety-margin metrics: minimum time-to-collision, minimum predicted clearance to other agents along the planned trajectory, at each planning cycle.
- Control smoothness: jerk/deceleration magnitude of the *executed* trajectory (a very late, very hard braking maneuver, even if it avoided a collision, indicates Planning had little margin — a "near-miss" worth flagging even absent perception/prediction anomalies).

### Distinguishing a genuine fault from "the world was just hard"

This is the trickiest part — heavy legitimate traffic or genuinely ambiguous scenarios can look anomalous without any sub-system being *wrong*. Approach:
- Compare the incident's anomaly signature against a **peer group** of similar scenarios (same rough scenario type, e.g. via the embedding/retrieval system from [driving_scene_search](../driving_scene_search/solution.md) — "find other clips embedding-similar to this incident") that did **not** result in an incident. If Perception's confidence was similarly low in many peer clips that resolved safely, that's evidence of a hard-but-handled scenario, not necessarily a fault — whereas if peer clips show high, stable confidence and this one doesn't, that's stronger evidence of an actual anomaly specific to this case.
- Use **counterfactual replay** (Step 4B) as the definitive check: an anomaly only "matters" causally if correcting it would have changed the outcome.

---

## Step 4B: Automated Labeling via Counterfactual Replay + Learned Attribution

### The core technique: counterfactual replay

The single most powerful tool for root-cause attribution is **replaying the incident with one stage's output substituted for an idealized/ground-truth version**, and checking whether the downstream outcome changes:

```
Counterfactual 1 (isolate Perception):
  Re-run Prediction + Planning using GROUND-TRUTH object detections/tracks
  (reconstructed post-hoc from later frames, other sensors, or offline
  higher-compute perception models) instead of the real-time Perception
  output.
  → If Planning now avoids the incident: Perception's error was causally
    responsible (a Perception failure).
  → If the incident still occurs: Perception wasn't the (sole) root cause;
    look downstream.

Counterfactual 2 (isolate Prediction):
  Feed Perception's REAL (possibly already-correct) detections into
  Prediction, but replace Prediction's output with the agent's ACTUAL
  realized trajectory (known post-hoc), then re-run Planning.
  → If Planning now avoids the incident: Prediction's forecast error was
    causally responsible (a Prediction failure).

Counterfactual 3 (isolate Planning):
  Feed Planning the REAL Perception and REAL Prediction outputs (i.e. no
  substitution), and compare Planning's chosen trajectory against a safer
  alternative trajectory that was feasible given those same inputs
  (verified via the vehicle dynamics/safety-margin model).
  → If a safer feasible trajectory existed given accurate upstream info
    and Planning didn't take it: a Planning failure.
```

This directly answers the "cascading failure" problem from the design doc: if Counterfactual 1 shows fixing Perception resolves the incident, Perception is the **root cause**, even if Prediction's output also looked anomalous (Prediction was just correctly forecasting a bad detection — a symptom, not a root cause). Running all three counterfactuals and observing *which one flips the outcome* is what separates root cause from downstream symptom.

### From counterfactuals to a labeled training set

Counterfactual replay alone gives a **rule-based / simulation-derived label** for each incident (not a learned prediction) — but it's not free: it requires a simulation/replay environment and reconstructing high-quality "ground truth" (harder for Perception ground truth than for Prediction, where the realized trajectory is directly observable). This is exactly the automated-labeling engine:

1. **Weak/heuristic labeling (cheap, scalable):** run the anomaly detectors (Step 4A) and simple counterfactual replays automatically on *every* incident (including the high-volume, low-severity near-misses), producing a candidate root-cause label with a confidence score, with no human involved. This alone can triage the bulk of low-severity events.
2. **Human verification (expensive, targeted):** route a sample of the weakly-labeled incidents — prioritizing high-severity incidents, low-confidence weak labels, and disagreements between the anomaly-detector signal and the counterfactual-replay signal — to safety engineers for confirmation or correction. This produces a smaller set of **high-quality human-verified labels**.
3. **Supervised attribution classifier:** train a classifier (e.g., a gradient-boosted tree or small neural net over the structured anomaly-signal features from Step 4A, plus counterfactual-replay deltas as features) on the growing human-verified label set, to predict root-cause attribution **directly from the anomaly/counterfactual features** — this lets the system get faster and cheaper over time as it learns which combinations of signals humans tend to confirm as a given failure type, rather than requiring a full (compute-expensive) counterfactual replay chain on every single incident.
4. **Active learning loop:** incidents where the learned classifier disagrees with the counterfactual-replay "ground truth," or where its confidence is low, are prioritized for human review — the same active-learning pattern used for long-tail scenario mining in [driving_scene_search](../driving_scene_search/solution.md) Step 4C, closing the loop between "confirmed labels improve the classifier" and "the classifier's uncertainty tells you what to review next."

### Handling the extreme label scarcity for true collisions

True collisions are far too rare to train a supervised classifier on directly. This design routes around that:
- The classifier is trained primarily on the much larger population of **near-misses and lower-severity incidents**, which share the same underlying anomaly signatures and can be counterfactually replayed the same way — a Perception-failure-driven near-miss and a Perception-failure-driven collision look structurally similar in the anomaly/counterfactual feature space, just differing in how narrowly the outcome was avoided.
- For the rare true collisions, always route to human review regardless of classifier confidence — the automated system's job for these is to **accelerate** the human's analysis (pre-computed anomaly signals, pre-run counterfactuals, evidence assembled), not to replace human judgment.

### What the output looks like

```
Incident #48213
Trigger: harsh braking (peak decel: 0.6g), no collision
Attribution: PREDICTION FAILURE (confidence: 0.87)
Evidence:
  - Perception: cyclist detected at t-4.2s, confidence stable at 0.95+
    throughout → no Perception anomaly.
  - Prediction: forecast "continue straight" with 0.9 probability from
    t-4.2s to t-1.1s; cyclist actually swerved left at t-1.3s → trajectory
    error spiked to 3.2m at t-0.8s (fleet p99 for this scenario type: 0.9m).
  - Planning: replanned hard brake at t-0.9s, immediately after Prediction's
    error became visible → consistent with Planning reacting correctly to
    a late/wrong Prediction, not an independent Planning fault.
  - Counterfactual: substituting the realized trajectory for Prediction's
    forecast and re-running Planning avoids the harsh-braking event
    entirely → confirms Prediction as causal root cause.
Routed to: Prediction team
Flagged for training-data mining: yes → cyclist-swerve scenario added to
  long-tail candidate pool (see driving_scene_search).
```

---

## Step 4C: Model Observability

### Why observability, separate from per-incident analysis

Per-incident attribution is inherently reactive — it only runs after something already went wrong. **Observability** is the always-on layer that tries to catch sub-system degradation *before* it contributes to an incident, by continuously tracking each sub-system's health against its own historical baseline.

### Per-sub-system health dashboards

For each sub-system, track fleet-wide aggregates of the same signals used in per-incident anomaly detection (Step 4A), but as continuous production metrics rather than incident-triggered:

| Sub-system | Key observability metrics |
|---|---|
| Perception | detection confidence distribution, track-stability rate (ID switches per hour), miss-rate proxy (agents detected late relative to sensor range), per-class precision/recall on any held-out labeled sample |
| Prediction | trajectory error distribution vs. realized outcomes (computed with a lag, once ground truth is observable), calibration of probabilistic forecasts, prediction-intent-flip rate |
| Planning | replan frequency/magnitude distribution, safety-margin distribution (time-to-collision, clearance), control smoothness (jerk) distribution, disengagement rate |
| Cross-cutting | end-to-end incident-trigger rate (harsh braking/replans per 1000 miles), segmented by city/weather/road-type to catch localized regressions |

### Drift detection

- Monitor each sub-system's **input distribution** (e.g., the distribution of scenes/scenarios it sees, via the same embedding space from [driving_scene_search](../driving_scene_search/solution.md)) and **output distribution** (e.g., detection confidence histograms, predicted-trajectory distributions) over time using standard drift statistics (e.g., population stability index, KL-divergence between recent and baseline windows).
- Segment drift monitoring by known confounders (new sensor hardware revision, new city/region rollout, seasonal weather shift) so a real regression isn't confused with an expected distribution shift from fleet expansion.
- Trigger alerts when drift exceeds a threshold *before* it necessarily manifests in a visible incident spike — e.g., a subtle miscalibration in Perception's confidence after a model update should be caught by drift monitoring, not by waiting for the first bad incident.

### Alerting design (avoiding alert fatigue)

- Tiered alerting: hard thresholds (e.g., a sudden spike in disengagement rate) page on-call immediately; softer drift signals (e.g., gradual confidence-distribution shift) roll into a daily digest for the owning team rather than paging.
- Every alert should link directly to supporting evidence (e.g., a handful of example incidents/segments exhibiting the drift, retrieved via the search system) so responders aren't starting from a bare metric — this mirrors the same "evidence alongside attribution" principle from Step 4B.
- Track alert precision over time (what fraction of alerts led to a confirmed real issue vs. were noise) and use it to tune thresholds — treat the alerting system itself as something to continuously calibrate, the same way we calibrate the root-cause classifier.

---

## Step 5: Evaluation

### Offline (Attribution Quality)
- Curate a benchmark of human-adjudicated incidents spanning all root-cause categories (Perception / Prediction / Planning / sensor / multi-cause / none).
- **Confusion matrix** across categories — pay particular attention to the cost-asymmetric errors (e.g., attributing a real Perception failure to Planning sends the fix to the wrong team and leaves the real issue live).
- **Counterfactual-replay agreement rate**: how often does the learned classifier's output match the more-expensive, simulation-grounded counterfactual replay result? (This validates that the cheap classifier is a good proxy, not just correlated with human labels.)

### Online / Operational
- Triage turnaround time (time from incident trigger to routed, actionable report), split by severity tier.
- Fraction of incidents auto-triaged at high confidence vs. escalated to human review, and how that fraction improves over time as the classifier retrains on more verified labels.
- Alert precision/recall for the observability layer (fraction of alerts confirmed as real issues; fraction of known real issues caught by an alert before/without needing a per-incident trigger).
- **Ultimate outcome metric:** for incidents where a fix was routed and shipped, does the recurrence rate of similar incidents (measured via the embedding-similarity peer-group technique from Step 4A) actually drop afterward? This is the real test of whether attribution is correct and actionable, not just plausible.

---

## Step 6: Tradeoffs & Design Decisions

### Tradeoff 1: Precision vs. Recall in Root-Cause Attribution
**Decision:** Optimize for high precision on automated (non-human-reviewed) attributions; never silently suppress a low-confidence incident — route it to human review instead of guessing.
**Tradeoff:** A recall-maximizing system would flag more true issues automatically but at the cost of more false attributions sent directly to teams, wasting engineering time and eroding trust in the system (teams start ignoring reports). Precision-first with a human-review fallback is slower for ambiguous cases but preserves trust and safety.

### Tradeoff 2: Rule-Based Counterfactual Replay vs. End-to-End Learned Classifier
**Decision:** Use counterfactual replay as the grounding/label-generation mechanism, and a learned classifier as a **cheap approximation** of it for the bulk of triage, reserving full replay for high-severity or classifier-disagreement cases.
**Tradeoff:** Pure counterfactual replay for every incident is the most rigorous approach (it's causally grounded, not just correlational) but is computationally expensive (re-running simulation/downstream models per incident) and doesn't scale to thousands of daily low-severity events. A pure learned classifier without any replay grounding would be fast but risks learning spurious correlations, especially given how few true-collision labels exist. The hybrid captures most of the rigor at a fraction of the cost.

### Tradeoff 3: Automated Attribution vs. Human Review for Severe Incidents
**Decision:** For true collisions and severe near-misses, always route to human review; the automated system's role there is to accelerate (pre-compute anomalies, run counterfactuals, assemble evidence), not replace, human judgment.
**Tradeoff:** This adds latency and human labor cost to the highest-severity cases, but an automated-only decision on a true collision carries too much liability, safety, and trust risk if wrong — the cost asymmetry strongly favors keeping a human in the loop exactly where stakes are highest, even though it's the opposite of where automation offers the most efficiency gain.

### Tradeoff 4: Real-Time Monitoring vs. Deep Offline Analysis
**Decision:** Keep these as two systems sharing infrastructure (the same anomaly-detection signal definitions and embedding space) rather than one unified real-time pipeline: observability dashboards run near-real-time on lightweight aggregate metrics; full root-cause attribution (including counterfactual replay) runs offline/batch.
**Tradeoff:** A single unified real-time system would give faster root-cause turnaround, but counterfactual replay is inherently too compute-heavy to run at real-time/streaming latency at fleet scale — splitting the systems lets each be designed for its actual latency budget instead of compromising both to fit one architecture.

### Tradeoff 5: Single Root Cause vs. Modeling Cascading Failures
**Decision:** The counterfactual-replay chain (Step 4B) explicitly checks each stage in isolation, so the system can report **either** a single root cause **or** an explicit "cascading: Perception→Prediction" attribution when multiple counterfactuals show partial effects, rather than forcing every incident into exactly one category.
**Tradeoff:** Supporting multi-cause/cascading output is more complex to build, evaluate, and communicate to engineers (a bug report with two owning teams is messier to route) than forcing single-label attribution — but forcing a single label on a genuinely cascading failure risks under-fixing the problem (only one team fixes their piece, and a similar incident recurs because the interaction wasn't addressed).

### Tradeoff 6: Peer-Group Comparison vs. Absolute Thresholds for "Nominal" Behavior
**Decision:** Compare an incident's anomaly signature against a peer group of similar scenarios (via embedding similarity) rather than relying solely on fixed absolute thresholds.
**Tradeoff:** Fixed thresholds are simpler to implement, explain, and audit, but a single global threshold either over-flags genuinely hard scenarios (dense urban traffic, heavy rain) as anomalous, or under-flags real faults in easy scenarios where even a small deviation matters. Peer-group-relative comparison is more accurate but requires the embedding/retrieval infrastructure from [driving_scene_search](../driving_scene_search/solution.md) to already exist and be reliable — a real dependency between these two systems.

---

## Step 7: Common Follow-up Questions

### Q: How do you handle an incident where the sensors themselves were at fault, not any ML sub-system?
**A:** The sensor-level anomaly detectors in Step 4A (occlusion, dropout, glare, calibration drift) run as the *first* stage of attribution, before Perception is even considered — if a camera was occluded or a LiDAR channel dropped out, that's flagged as a hardware/sensor issue and routed to the hardware/sensor-ops team rather than any ML team. Counterfactual replay can also confirm this: substituting clean sensor data (from an unaffected redundant sensor, if available) and re-running the full pipeline isolates whether the fault was upstream of any ML model entirely.

### Q: How do you build a labeled training set for the root-cause classifier when true collisions are so rare?
**A:** See Step 4B — train primarily on the much larger population of near-misses and lower-severity incidents (which share the same anomaly/counterfactual feature signatures), using counterfactual replay as a scalable weak-labeling mechanism, and reserve human-verified labels for calibrating and validating rather than being the sole source of training signal.

### Q: How would you validate that a proposed fix actually addresses the root cause before it ships?
**A:** Re-run the counterfactual replay for the original incident (and its peer group of similar scenarios) with the *candidate fixed model* substituted in for the failing sub-system — if the new model no longer produces the anomalous output/incident outcome on that scenario and its peers, that's strong offline evidence the fix addresses the identified root cause, before committing to a full fleet rollout and A/B evaluation.

### Q: How do you handle disagreement between the automated attribution and a human reviewer?
**A:** Every human correction is logged as a new labeled example and fed back into the classifier's training set (Step 4B's active-learning loop); disagreements are also periodically reviewed in aggregate to check for systematic classifier blind spots (e.g., consistently misattributing a certain scenario type), which may reveal a need for a new anomaly signal/feature rather than just more labels.

### Q: How would this system change to also flag near-term risk in real time, on-vehicle?
**A:** That's a distinct system with a very different latency/compute budget (milliseconds, embedded hardware) — it would reuse the same *definitions* of nominal/anomalous behavior per stage (Step 4A) but as lightweight on-vehicle monitors rather than the full offline counterfactual-replay pipeline, likely only flagging "elevated risk, consider conservative fallback behavior" rather than doing full root-cause attribution on-the-fly.

### Q: How do you avoid alert fatigue from the observability layer?
**A:** Tiered alerting (hard thresholds page immediately, soft drift signals digest daily), always attaching concrete evidence/examples to every alert, and continuously tracking alert precision to retune thresholds — treating threshold-tuning as an ongoing calibration problem rather than a one-time setup (Step 4C).

### Q: How does this connect back into the training-data mining pipeline?
**A:** Every confirmed failure case (whether Perception, Prediction, or Planning) is exactly the kind of safety-critical long-tail scenario the [driving_scene_search](../driving_scene_search/solution.md) mining pipeline is designed to surface more examples of — the confirmed incident (and its embedding) becomes a **seed** for that system's retrieve→diversify→verify loop, so the archive is mined for similar-but-previously-unflagged scenarios to enrich the training/eval set for whichever sub-system needs the fix.
