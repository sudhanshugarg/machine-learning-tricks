# FAQ: Simulation Log Collection and Result Aggregation

## Question Log

| # | Question | Category | Status | Asked |
|----|----------|----------|--------|-------|
| — | (Questions logged as raised during prep/interview) | — | — | — |

---

## Terminology & Definitions

### Q: What's a "scenario cluster"?
**A:** A logical grouping of similar driving scenarios. Examples: `urban_intersection`, `highway_merging`, `parking_lot_dense`, `rain_heavy`. Each cluster contains 10–1000 individual scenario variations (different seeds, weather, traffic density). Aggregating by cluster balances granularity (catch model regressions per scenario type) with statistical power (enough runs per cluster to compute tight confidence intervals).

*Pointer:* [solution.md § 2. Schema & Versioning](solution.md) — run grouping; [design.md](design.md) — challenge 2.

---

### Q: Why do we track simulator version and scenario library version separately from model version?
**A:** Because they evolve independently. A model update (e.g., retrain the decision network) doesn't invalidate simulator runs; a simulator bug fix doesn't break all model versions. By encoding all three as first-class fields, you can:
- Re-aggregate sim runs grouped by `simulator_version` when a bug is found
- Compare "model X on scenario library v1" vs. "model X on scenario library v2" to measure scenario coverage improvement
- Audit exactly which sim/scenario versions a model was evaluated against before promotion

This is critical for reproducibility and debugging.

*Pointer:* [solution.md § 2. Schema & Versioning](solution.md) — core design rationale.

---

### Q: What's a "summary record"?
**A:** A lightweight JSON blob (few KB) emitted per simulation run. Contains aggregate statistics from the run (collision count, latency percentiles, success flag) plus metadata (model version, simulator version, scenario ID). Summaries are sent to Kafka/Pub/Sub to drive the real-time pipeline, while full logs (10 MB–1 GB) stay in object storage for cold access / forensics.

*Pointer:* [solution.md § 1. Ingestion & Summarization](solution.md) — worker-side.

---

### Q: Why bootstrap instead of a closed-form confidence interval?
**A:** Metric distributions are often non-normal. Collision counts follow a Poisson-ish distribution (skewed, heavy tail); latencies are often log-normal. Parametric CI methods (t-test, z-score) assume normality and underestimate uncertainty. Bootstrap makes no distributional assumption and is robust to any shape.

*Pointer:* [solution.md § 3. Batch ETL § Aggregation](solution.md); [§ 9. Tradeoffs & Alternatives](solution.md).

---

### Q: What does "importance weight" mean in the context of sim-vs-real blending?
**A:** When you have 10,000 sim runs for a scenario cluster but only 5 real runs, a naive average of their metrics drowns the real signal in confidence-interval noise. Importance weighting adjusts the effective sample size: if sim data is much more abundant than real data, we weight sim lower (because it's less informative per run). Concretely:

```
importance[cluster] = (n_real + 1) / (n_sim + 1)
```

This ensures rare, real-world events are not masked by abundant but noisy simulation. The Laplace smoothing term (+1) prevents division by zero.

*Pointer:* [solution.md § 3. Batch ETL § Cluster Weighting](solution.md); [§ 5. Sim-vs-Real Reconciliation](solution.md).

---

### Q: What does "poisoned run" mean?
**A:** A run whose results are invalidated by a discovered bug in the simulator (or scenario library). Example: simulator has a physics bug affecting collision detection; all runs using that simulator binary are marked `poisoned: true`. These runs are excluded from aggregates (via `WHERE NOT poisoned`) until they can be re-run with the patched simulator.

*Pointer:* [solution.md § 6. Replay & Poisoning](solution.md).

---

## Architecture & Design

### Q: Why split ingestion into object storage + streaming bus instead of streaming everything?
**A:** **Decision**: Kafka for summaries only; full logs to object storage.

**Tradeoff**: 
- Streaming everything to Kafka: simple architecture, but 1 GB per run × 10,000 runs/day = 10 TB/day of Kafka traffic (expensive, hard to scale, high operational overhead).
- Streaming summaries + cold logs: higher operational complexity (two data paths), but Kafka carries only 1 KB summaries (10 GB/day, easily manageable).

**Why we chose split**:
- Cost: object storage is 10–100x cheaper than Kafka scale
- Forensics: full logs stay durable in object storage; you can drill-down to exact decision timestep
- Real-time aggregation: 1 KB summaries give you everything needed for real-time dashboards (you don't need 1 GB of detailed logs to compute daily collision rate)

*Pointer:* [solution.md § 1. Ingestion](solution.md); [§ 9. Tradeoffs](solution.md).

---

### Q: Why batch ETL (Spark) instead of streaming aggregation (Flink)?
**A:** **Decision**: Batch ETL every 5 minutes.

**Tradeoff**:
- Stream aggregation (Flink): 30–60s latency, always-on cluster (high cost).
- Batch every 5 min: 5-minute latency, scale-down between runs (lower cost).

**Why we chose batch**:
- 5-minute latency is fine for evaluation dashboards (not real-time control)
- Exactly-once semantics are trivial with batch (re-read, re-compute, idempotent write)
- Cost is 3–5x lower (shut down cluster between batches)
- Replay/debugging is easier (re-run a batch from a known time)

**Alternative**: If ranking feedback must update < 1s, add optional streaming fast-path for pre-aggregates; merge with batch results hourly.

*Pointer:* [solution.md § 9. Tradeoffs & Alternatives](solution.md).

---

### Q: Why three tiers of storage (hot / warm / cold) instead of just one?
**A:** Different query patterns have different latency/cost tradeoffs.

- **Hot** (0–14 days, SSD): dashboards need < 1s, users expect "latest metrics now"
- **Warm** (14–90 days, compressed): post-mortems can tolerate 1–5 min decompression; "what happened 30 days ago?"
- **Cold** (90+ days, Glacier): forensic investigation only, rehydration overnight is OK; regulatory backups

Cost difference: SSD costs ~50x more per GB than Glacier; you don't want all data hot forever.

*Pointer:* [solution.md § 4. Storage Tiering](solution.md).

---

### Q: How does the promotion gate actually prevent a bad model from going live?
**A:** **Three-part check** (all must pass):

1. **Sim metrics pass**: collision rate < threshold, latency < threshold, **across all tracked scenario clusters** (simulation is broad)
2. **Real metrics pass**: same thresholds on real-world test corpus (real world is truth)
3. **Gap is aligned**: for each scenario in real corpus, |sim_metric - real_metric| < 1.5σ (sim and real agree)

If any check fails, promotion is blocked. Example:
- Sim passes, real fails → model doesn't work in real world, **block**
- Both pass but gap is 3σ → sim is wildly optimistic, **block** (simulator is poorly calibrated)

This prevents both false negatives (unsafe models) and false positives (overly pessimistic gates).

*Pointer:* [solution.md § 5. Sim-vs-Real Reconciliation](solution.md).

---

### Q: What happens if real-world data arrives 24 hours late?
**A:** You need a **staged promotion tier**:

1. **Sim-ready**: model passes all sim tests; flagged for deployment monitoring
2. **Real data pending**: model is deployed behind a canary or feature flag; real-world metric collection begins
3. **Production-ready**: after 24–48 hours, real metrics confirm safety; model goes to 100% traffic

Warehouse supports this by storing `real_metrics.arrival_timestamp`; the promotion gate checks `current_time - arrival_timestamp < MAX_STALENESS` before greenlight.

*Pointer:* [solution.md § 7. Operational Instrumentation](solution.md) — real-world lag; [§ 10. Failure Modes](solution.md).

---

### Q: How do you prevent real fleet data from being drowned out by simulation?
**A:** **Weighted blending + inverse abundance**.

Without weights:
- sim: 10,000 runs, CI = ±1%
- real: 5 runs, CI = ±15%
- **naive blend**: CI ≈ ±1.5% (sim dominates)

With inverse-abundance weighting:
- importance[cluster] = (5 + 1) / (10,000 + 1) ≈ 0.0006
- sim contribution is downweighted by this factor
- real contribution is upweighted (weight = 1.0)
- **weighted blend**: CI ≈ ±4% (real signal survives)

Additionally, the promotion gate **requires real metrics to pass independently**; it's not just a blend. So even if weighted blending favors sim, a model must still clear real-world thresholds to ship.

*Pointer:* [solution.md § 3. Batch ETL § Cluster Weighting](solution.md); [§ 5. Sim-vs-Real Reconciliation](solution.md).

---

## Tradeoffs & Decisions

### Q: Why delete poisoned data instead of keeping a "before/after" version?
**A:** **Decision**: Delete + materialized replay log.

**Tradeoff**:
- **Keep poisoned runs, flag as "bad"**: Maintains full audit trail in warehouse; users can see history. But warehouse is now complex (must remember to filter `WHERE NOT poisoned`); careless queries break.
- **Delete + replay log**: Clean warehouse (truth is always canonical); lineage table (separate from metrics) preserves audit. But requires strict data governance; if replay log gets out of sync, you lose history.

**Why we chose delete**: Metrics are used for real-time decision-making (promotion gates, dashboards). Users expect metrics to be clean ground truth, not a historical artifact log. Lineage table is a separate audit concern; it's checked only during forensics, not on every query.

*Pointer:* [solution.md § 6. Replay & Poisoning](solution.md); [§ 9. Tradeoffs](solution.md).

---

### Q: What if a model regresses on a rare scenario (low sample count)?
**A:** **Confidence intervals widen, but don't disappear.**

Example: cluster `blizzard_highway_merge` has only 20 real runs / day.
- CI on collision rate = ±5% (vs. ±0.5% for common clusters)
- If sim/real agree, gate is skeptical but passes (wide CI = low confidence, but not negative evidence)
- If sim/real **disagree**, gap > 1.5σ → gate blocks (high variance means agreement is crucial)

**Mitigation**: For rare scenarios, gate requires tighter sim-vs-real alignment (e.g., gap < 1σ instead of 1.5σ) to compensate for fewer real samples.

*Pointer:* [solution.md § 5. Sim-vs-Real Reconciliation](solution.md); [§ 10. Failure Modes](solution.md).

---

### Q: What's the cost of this system, ballpark?
**A:** Rough estimate (1000 sim runs/day, millions of data points):

| Component | Cost | Notes |
|-----------|------|-------|
| **Object storage (hot 14d)** | $100–200/mo | SSD-backed, 14 TB/mo retention |
| **Object storage (warm 14–90d)** | $20–50/mo | Compressed archive |
| **Kafka / Pub/Sub** | $50–100/mo | 10 GB/day throughput |
| **Spark batch cluster (5-min cadence)** | $200–400/mo | Pre-emptible instances, scale-to-zero |
| **Warehouse (Trino/BigQuery)** | $500–1000/mo | Query costs + storage |
| **Monitoring / alerting** | $100–200/mo | Datadog / Prometheus |
| **Total** | **~$1000–2000/mo** | Scales linearly with 10x throughput |

Sim runs themselves are the dominant cost; this pipeline is ~5–10% of total simulation cost.

*Pointer:* [solution.md § 7. Operational Instrumentation](solution.md).

---

## Deep-Dives & Edge Cases

### Q: Can you explain the sim-to-real gap metric with a concrete example? `[ANSWERED]`

**A:** The sim-to-real gap measures how well your simulator matches the real world. Small gaps are good (simulator is well-calibrated); large gaps indicate distribution shift and require investigation.

**Definition**:
```
gap[cluster] = |sim_metric[cluster] - real_metric[cluster]| / std(real_metric[cluster])
```

This is a **standardized gap** — you measure the difference in units of real-world standard deviation. Why? Because a 0.5% collision rate difference matters more on a rare scenario (high variance) than a common one (low variance).

#### Example 1: Urban Intersection (Good Calibration)

**Real-world data** (from 150 on-road runs in urban intersections):
- Collision rate: 0.8%
- Standard deviation: 0.2%
- Latency (p99): 95 ms

**Simulated data** (from 5,000 sim runs, same scenario cluster):
- Collision rate: 0.85%
- Standard deviation: 0.1%
- Latency (p99): 93 ms

**Calculate gap**:
```
gap[urban_intersection] = |0.85% - 0.8%| / 0.2% = 0.05% / 0.2% = 0.25σ
```

**Interpretation**: The simulator predicts 0.25 standard deviations higher collision rate than reality. This is **excellent agreement**. Why? Because:
- The difference (0.05%) is tiny in absolute terms
- It's only 1/4 of a standard deviation — well within natural noise
- Model is safe to promote (simulator is slightly conservative, which is safer)

**Decision**: ✅ **Pass the alignment check.** Simulator generalizes well to real-world urban intersections.

---

#### Example 2: Highway Merging (Distribution Mismatch)

**Real-world data** (from 120 on-road runs, highway merging):
- Collision rate: 0.6%
- Standard deviation: 0.3%
- Latency (p99): 110 ms

**Simulated data** (from 8,000 sim runs, same scenario cluster):
- Collision rate: 0.3%
- Standard deviation: 0.1%
- Latency (p99): 75 ms

**Calculate gap**:
```
gap[highway_merging] = |0.3% - 0.6%| / 0.3% = 0.3% / 0.3% = 1.0σ
```

**Interpretation**: Simulator predicts **1 standard deviation lower** collision rate than reality. This is moderate concern:
- Real world: 0.6% collisions
- Sim world: 0.3% collisions
- Gap is 100% of real variance — not negligible

**Why does this happen?**
- Highway merging has complex multi-agent behavior (vehicles cutting in, lane changes)
- Simulator likely uses simplified traffic models (pre-canned trajectories, not reactive agents)
- Possible: poor calibration of sensor noise (highway merging is more sensitive to perception latency)

**Decision**: ⚠️ **Investigate, but not automatically blocked.** Promotion gate checks:
- Does sim metric pass safety threshold? 0.3% < 0.5% → ✅ Yes
- Does real metric pass safety threshold? 0.6% < 0.5% → ❌ No
- Is gap < 1.5σ? 1.0σ < 1.5σ → ✅ Yes (just barely)

**Mitigation**: Either (a) lower sim collision rate threshold for highway scenarios to account for the gap, or (b) investigate & recalibrate the simulator's highway merging model before promoting.

---

#### Example 3: Rain on Urban Street (Misaligned Distribution)

**Real-world data** (from 45 on-road runs, heavy rain on urban streets):
- Collision rate: 2.1%
- Standard deviation: 0.5%
- Latency (p99): 250 ms (perception is slower in rain)

**Simulated data** (from 3,000 sim runs, heavy rain scenario):
- Collision rate: 0.4%
- Standard deviation: 0.15%
- Latency (p99): 100 ms

**Calculate gap**:
```
gap[rain_urban] = |0.4% - 2.1%| / 0.5% = 1.7% / 0.5% = 3.4σ
```

**Interpretation**: Simulator is **3.4 standard deviations too optimistic**. This is a **critical mismatch**:
- Real world: 2.1% collision rate
- Sim world: 0.4% collision rate
- Difference: 1.7% — **more than 3σ away**

**Why?**
- Simulator's rain rendering doesn't degrade sensor realism enough
- Real cameras have real rain drops on lens; sim sensors don't
- Sim perception latency (100ms) is much lower than real (250ms)
- Simulator hasn't been trained / validated on rainy conditions

**Decision**: 🚨 **Block promotion, investigate simulator.**

The gate fails the alignment check: gap > 1.5σ. Before deploying this model, you must:
1. Improve rain simulation (add photorealistic rain drops, degrade sensor output)
2. Increase simulated perception latency to match real systems
3. Re-run evaluation; recompute gap
4. Only promote after gap < 1.5σ

---

#### Summary: Gap Interpretation

| Gap (σ) | Real-world Match | Action |
|---------|------------------|--------|
| < 0.5σ | Excellent calibration | ✅ Safe to promote |
| 0.5–1.5σ | Good match, monitor | ✅ Promote with notes; monitor real metrics |
| 1.5–2.5σ | Concerning drift | ⚠️ Investigate simulator; may block |
| > 2.5σ | Severe mismatch | 🚨 Block promotion; recalibrate |

**Key insight**: You're not asking "is the simulator perfect?" (it never is). You're asking "**is the simulator consistently wrong in a predictable way?**" If real and sim agree (gap is small), your models will generalize. If they diverge (gap is large), your simulator is useless as a safety signal.

*Pointer:* [solution.md § 5. Sim-vs-Real Reconciliation](solution.md); [§ 9. Failure Modes — sim-to-real gap alarm](solution.md).

---

### Q: What if the scenario library gets updated? Do old sim runs become invalid?
**A:** Not automatically. A scenario library update (e.g., new intersection definition) only invalidates runs that used the old definition **if the change is breaking** (e.g., intersection geometry changed fundamentally).

**Strategy**:
- Encode `scenario_library_version` in every run
- When a scenario library update is released, mark the new version explicitly
- In the warehouse, you can now compare:
  - Model X on scenario-lib v1 (all old runs)
  - Model X on scenario-lib v2 (new runs)
  - Measure coverage improvement / regression due to updated scenarios
- For promotion, require model to pass on **current** scenario library version; older versions are historical reference

*Pointer:* [solution.md § 2. Schema & Versioning](solution.md).

---

### Q: How do you handle out-of-order arrival of summaries (e.g., delayed Kafka message)?
**A:** **Assume eventual consistency, use a long watermark.**

Batch ETL aggregates runs arriving in the last 24 hours, not just the last 5 minutes. This way:
- A summary delayed by 2 hours is picked up by the next batch
- Late-arriving real-world data (24+ hour delay) is handled separately (see staged promotion tiers)

In the warehouse, add `watermark_timestamp` to each row; queries filter by `watermark_timestamp >= cutoff` to exclude incomplete aggregates.

*Pointer:* [solution.md § 3. Batch ETL](solution.md).

---

### Q: What if you discover a systematic bias in the simulator (e.g., collision detection is always 10% too conservative)?
**A:** This is a **calibration** issue, not a poisoning issue. Don't delete runs; instead:

1. **Track the bias** as a first-class metric: `sim_to_real_gap[collision_detection] = 0.1`
2. **Adjust promotion thresholds** for the biased component: if sim says collision_rate < 0.5%, real threshold is < 0.4% (accounting for the 10% conservative bias)
3. **Recompute blended metrics** with the correction factor

Example:
```
sim_collision_rate_biased = 0.45%
bias_correction = 0.1
sim_collision_rate_corrected = 0.45% - 0.1 = 0.35%
gate_threshold = 0.4%
sim_collision_rate_corrected < gate_threshold → pass
```

This is less disruptive than poisoning and preserves data integrity.

*Pointer:* [solution.md § 5. Sim-vs-Real Reconciliation](solution.md) — calibration & gap tracking.

---

## Questions Awaiting Answers

(None yet — add as they arise during prep or interview.)
