# System Design: Post-Incident Root-Cause Attribution for Autonomous Vehicle Failures

## Problem Statement

You are an ML engineer at **Waymo**. The self-driving stack is composed of a pipeline of ML sub-systems that run continuously on every drive:

- **Perception** — detects and classifies agents (pedestrians, cyclists, vehicles, etc.) and static objects from raw sensor input (camera, LiDAR, radar).
- **Prediction** — forecasts the future trajectories of the agents perception identified.
- **Planning** — decides the vehicle's own trajectory/behavior given the predicted motion of everything else on the road.

When a **collision or a near-miss** (e.g. hard braking, an emergency swerve, a safety-driver disengagement, a planner replanning abruptly) occurs, safety engineers need to know **which sub-system actually failed** and why — did Perception miss or misclassify an object? Did Prediction forecast the wrong trajectory for an agent Perception correctly identified? Or did Planning make a poor decision despite receiving correct upstream information?

Today, this root-cause analysis is done **manually**: an engineer pulls the raw logs for an incident and painstakingly steps through each pipeline stage's outputs frame-by-frame to figure out where things went wrong. This does not scale — the fleet generates far more "interesting" events (harsh braking, disengagements, near-misses) per day than engineers can manually triage, and the true collisions/severe near-misses that most need fast, reliable root-cause analysis are exactly the events where getting the diagnosis right matters most.

You are asked to design an **automated post-incident analysis system** that ingests the full sensor + pipeline log for an incident and outputs a **root-cause attribution**: which sub-system(s) (Perception, Prediction, and/or Planning) failed, with supporting evidence, so safety engineers can triage incidents quickly and route the right fix to the right team.

**Illustrative cases the system must handle:**
- A cyclist swerves in front of the vehicle and the car brakes hard. Perception detected the cyclist correctly and early — Prediction failed to anticipate the swerve.
- A pedestrian partially occluded by a parked truck steps into the road. Perception never detected the pedestrian until very late — a Perception failure.
- Perception and Prediction were both correct (a car was correctly identified as merging into the lane with an accurate predicted trajectory), but Planning chose a trajectory that didn't leave enough buffer — a Planning failure.
- Multiple sub-systems degrade together (e.g., heavy rain degrades Perception confidence, which cascades into a worse Prediction, which Planning acts on) — the system should be able to say which stage was the *root* cause vs. a downstream symptom.

---

## Scale & Context (assume unless told otherwise, confirm with interviewer)

- Fleet of thousands of vehicles, generating a continuous stream of full sensor + pipeline logs (raw sensor data plus every sub-system's intermediate outputs: detections, tracks, predicted trajectories, planned trajectories, control commands).
- "Incidents" span a severity spectrum: true collisions (extremely rare), safety-driver disengagements / takeovers (rare), and softer signals like harsh braking, abrupt replanning, or large last-second trajectory changes (much more frequent — thousands per day across the fleet).
- Existing state: raw logs are retained for some window; there is no automated per-subsystem fault attribution today — only coarse, mostly manual review.
- Consumers: safety/incident-review engineers (need fast, explainable root-cause output per incident), the Perception/Prediction/Planning ML teams (need routed, actionable failure reports to prioritize fixes), and the training-data pipeline (confirmed failure cases are exactly the kind of long-tail, safety-critical examples worth mining into future training/eval sets — see the related [driving_scene_search](../driving_scene_search/design.md) system).

---

## What You Should Cover

1. **Clarifying Questions** — what counts as an "incident" (severity thresholds), what logs/signals are available per sub-system, whether ground-truth (e.g. what actually happened, post-hoc-labeled) is available for training, latency requirements (real-time triage vs. offline batch analysis), and who consumes the output.

2. **Goals, Requirements & Constraints**
   - Functional: given an incident's full log, output a root-cause attribution across Perception/Prediction/Planning (including "multiple sub-systems" and "root cause vs. downstream symptom" cases), with supporting evidence an engineer can quickly verify.
   - Non-functional: turnaround time for triage, precision/recall expectations (a missed or misattributed root cause on a real collision is very costly), throughput to keep up with the volume of lower-severity near-miss events.

3. **Anomaly Detection in Sensor & Pipeline Telemetry**
   - How do you define "nominal" behavior for each sub-system's outputs (detection confidence, track stability, prediction error, planning replan frequency, control smoothness) so you can detect *deviations* automatically?
   - How do you detect anomalies at each stage of the pipeline (sensor level, perception level, prediction level, planning level) rather than only looking at the final outcome?
   - How do you distinguish a genuine sub-system fault from an anomaly that's simply a hard-but-correctly-handled scenario (e.g., legitimately dense traffic)?

4. **Automated Labeling for Root-Cause Attribution**
   - Given a raw incident log, how do you automatically generate a candidate root-cause label (which sub-system, what kind of failure) without requiring a human to review every single incident from scratch?
   - What role does **counterfactual replay/simulation** play (e.g., "if Perception had produced the ground-truth detections instead, would Planning still have made the same decision?")?
   - How do you bootstrap and continuously improve an automated attribution *classifier* from a growing set of human-verified incidents, and route low-confidence cases to human review?

5. **Model Observability**
   - How do you continuously monitor each sub-system's health in production (not just after an incident is flagged) so degradations are caught before they contribute to a collision?
   - What does a per-sub-system "dashboard" look like — what metrics, what alerting thresholds?
   - How do you detect **drift** in each sub-system's input distribution or output behavior over time (e.g., a new sensor hardware revision subtly changing Perception's confidence calibration)?

6. **System Architecture** — end-to-end data flow from incident detection/triggering → log retrieval and reconstruction → per-stage anomaly detection → counterfactual replay → root-cause classification → human review/verification → routing to the owning ML team → feedback into training-data mining.

7. **Evaluation**
   - Offline: attribution accuracy against a curated set of human-adjudicated incidents (confusion matrix across Perception/Prediction/Planning/multi-cause).
   - Online: triage turnaround time, reduction in manual review effort, fraction of incidents auto-triaged with high confidence vs. escalated to humans, and — ultimately — whether fixes routed by this system measurably reduce the recurrence of similar incidents.

8. **Tradeoffs** — discuss explicitly:
   - **Precision vs. recall** in flagging a sub-system as the root cause (a false attribution sends a fix to the wrong team; a missed attribution means a real fault goes unaddressed).
   - **Rule-based heuristics vs. learned classifiers** for root-cause attribution, especially given how few true collisions exist to train on.
   - **Automated attribution vs. human review** — how much to trust the automated output for the highest-severity incidents.
   - **Real-time monitoring vs. deep offline post-incident analysis** — these have very different latency/compute budgets and may need different systems that share infrastructure.
   - **Attributing a single root cause vs. modeling cascading/compounding failures** across sub-systems.

---

## Common Follow-up Questions to Expect

- How would you handle an incident where the *sensors themselves* were at fault (e.g., a dirty/occluded camera lens) rather than any ML sub-system?
- How do you build a labeled training set for the root-cause classifier when true collisions are extremely rare?
- How would you validate that a proposed fix to Perception/Prediction/Planning actually addresses the root cause identified, before it ships?
- How do you handle disagreement between the automated attribution and a human reviewer's judgment — how does that feedback improve the system?
- How would this system change if it needed to also flag *near-term* risk in real time on-vehicle, not just analyze incidents after the fact?
- How do you avoid the observability system generating so many low-value alerts that engineers start ignoring it (alert fatigue)?
- How does this system's output connect back into the training-data mining pipeline (e.g., the [driving scene search](../driving_scene_search/design.md) system) to close the loop on fixing the failure?
