# Solution: Simulation Log Collection and Result Aggregation

## Clarifying Questions & Assumptions

Before diving into architecture, align on:

1. **Simulator scale**: are all runs on the same simulator binary, or do different workers run different versions?
   - Assume: simulator version is an explicit field on every log; we track runs per (model, simulator, scenario library) triple
2. **Real-world data**: how much on-road data arrives per day vs. simulation?
   - Assume: 100–1,000 real fleet runs per day (much smaller), with 4–48 hour latency; sim runs have ~1–5 minute latency
3. **Metric grain**: what's the smallest meaningful aggregation unit?
   - Assume: scenario cluster (e.g., "urban intersection," "highway merging"); fine-grained enough to catch model regressions, coarse enough to have 50+ runs per cluster
4. **Replay scope**: how often do you discover simulator bugs?
   - Assume: monthly discovery; affects ~5–20% of runs in the pipeline; re-run cost is acceptable

## Goals & Constraints

| Goal | Rationale |
|------|-----------|
| **No data loss** | Bugs in simulation must be traceable; audit trail required |
| **Sub-minute ingestion latency** | Dashboards should reflect new runs near real-time |
| **Second-scale query latency** (hot) | On-call engineers need fast feedback on regressions |
| **Sim signal doesn't swallow real signal** | Real-world metrics must remain discoverable despite 1000:1 data imbalance |
| **Reproducibility** | poisoned runs marked, replaced runs traced, lineage auditable |

| Constraint | Impact |
|------------|--------|
| **Cost** | cold storage must be cheap; archive after 90 days |
| **Backward compatibility** | can't break existing dashboards during schema migrations |
| **Latency heterogeneity** | real fleet data arrives late; must support backfill without breaking aggregates |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    INGESTION & SUMMARIZATION                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Sim Workers                     Real Fleet Vehicles              │
│      │                                 │                          │
│      ├─→ [Object Storage]         ├─→ [Object Storage]           │
│      │   (logs/{date}/{scenario}/ │   (onroad/{date}/{vehicle}/) │
│      │    partitioned by cluster) │                              │
│      │                             │                              │
│      └─→ [Kafka / PubSub] ────────┴─→ Summary Stream             │
│          {model_v, sim_v,           (metric snapshot,             │
│           scenario_id, run_id,       model_v, sim_v,              │
│           success_flag, metrics}     scenario_id, success_flag}   │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              [Dashboard]      [Batch ETL]     [Real-time Aggregates]
              (hot queries)    (Spark/Beam)    (optional low-latency path)
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
         ┌──────────────────────────┴─────────────────────────┐
         │                                                      │
         v                                                      v
    [Hot Metric Warehouse]                     [Sim-vs-Real Reconciliation]
    (14 days, SSD-backed)                      (comparison, gap tracking,
    - Aggregates by                           confidence score blending)
      (scenario_cluster,
       model_v, sim_v)
    - Percentiles, counts,
      confidence intervals
                                │
                                v
                    ┌─────────────────────┐
                    │  Promotion Gate     │
                    │  - Sim metrics ✓    │
                    │  - On-road metrics✓ │
                    │  - Gap aligned? ✓   │
                    └─────────────────────┘
                                │
                        ┌───────┴───────┐
                        v               v
                   [Approve]       [Block/Investigate]
```

---

## 1. Ingestion & Summarization

### Worker-side (in simulation harness)

Each simulation worker writes:

**Full logs** → Object storage (cold, for forensics)
```
s3://sim-logs/{YYYY-MM-DD}/{scenario_cluster}/{run_id}/
  ├── logs.pb.gz         # raw simulation trace (protobuf)
  ├── metadata.json      # run-level metadata (model_v, sim_v, scenario_v, seed, etc.)
  └── manifest.json      # file checksums, line counts for validation
```

**Lightweight summary** → Kafka / Pub/Sub (drives real-time pipeline)
```json
{
  "run_id": "sim-20260714-001234",
  "timestamp": "2026-07-14T10:05:30Z",
  "model_version": "av-model-v3.2.1",
  "simulator_version": "carla-0.9.13-patched",
  "scenario_library_version": "scenarios-2026-07-01",
  "scenario_id": "urban_intersection_1_seed_42",
  "scenario_cluster": "urban_intersection",
  "success_flag": true,
  "collision_count": 0,
  "path_deviation_m": 0.23,
  "planning_latency_ms": [12, 14, 13, 15, ...],
  "perception_latency_ms": [8, 9, 8, 10, ...],
  "run_duration_s": 45.2,
  "checksum": "abc123def456"
}
```

**Decision**: Ship full logs to object storage, not Kafka. Reason: logs are 10 MB–1 GB; Kafka can handle it but storage is cheaper and more durable for cold access. Use Kafka only for summaries to avoid overwhelming the streaming bus.

### Real fleet side

Each on-road vehicle emits logs (aggregated per 5–10 minute window to reduce volume):
```json
{
  "vehicle_id": "fleet-waymo-123",
  "timestamp": "2026-07-14T10:05:30Z",
  "model_version": "av-model-v3.2.1",
  "scenario_id": "real_urban_intersection",
  "scenario_cluster": "urban_intersection",
  "success_flag": true,
  "collision_count": 0,
  "path_deviation_m": 0.18,
  "planning_latency_ms": [11, 13, 12, 14, ...],
  "perception_latency_ms": [9, 10, 9, 11, ...],
  "window_duration_s": 600,
  "checksum": "xyz789uvw012"
}
```

**Same schema for both sim and real**, with explicit source field in warehouse to distinguish them.

---

## 2. Schema & Versioning

### Core Design: Version Everything Independently

Each summary record carries three version dimensions as **first-class fields**:

| Field | Purpose | Example |
|-------|---------|---------|
| `model_version` | AV decision model | `av-model-v3.2.1` |
| `simulator_version` | Simulator binary + config | `carla-0.9.13-patched` |
| `scenario_library_version` | Scenario definitions | `scenarios-2026-07-01` |

**Why**: each can be updated independently. If the scenario library gets a new intersection definition, old sim runs remain valid for the old library; you can re-aggregate sim runs grouped by `scenario_library_version`. If the simulator gets a bug fix, you can mark runs by `simulator_version` and replay only those with the buggy version.

### Protobuf / Avro for raw logs

Raw logs use versioned Protobuf (or Avro) to handle field additions and deprecations:

```protobuf
syntax = "proto3";

message SimulationLog {
  string run_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  string model_version = 3;
  string simulator_version = 4;
  string scenario_library_version = 5;
  
  // Per-timestamp telemetry
  repeated TimestepSnapshot snapshots = 6;
  
  // Reserved for future fields (prevents conflicts)
  reserved 100 to 200;
}

message TimestepSnapshot {
  double sim_time = 1;
  VehicleState ego_state = 2;
  repeated VehicleState agent_states = 3;
  PerceptionOutput perception = 4;
  PlannerOutput planning = 5;
  CollisionFlags collisions = 6;
}
```

**Decision**: use Protobuf over Avro. Reason: cleaner backward compatibility via `reserved` fields; native support for nested messages; widely used in AV pipelines.

---

## 3. Batch ETL: Aggregation Pipeline

A Spark / Beam job runs every 5 minutes (or hourly, depending on latency tolerance):

### Input
Consume from Kafka / Pub/Sub the last N hours of summary records (both sim and real).

### Aggregation

```python
# Pseudocode
aggregates = (
    summaries
    .filter(lambda r: r['timestamp'] > cutoff_time)
    .groupBy(['scenario_cluster', 'model_version', 'simulator_version', 'source'])
    .agg({
        'collision_count': ['sum', 'mean'],
        'path_deviation_m': ['percentile_approx(50)', 'percentile_approx(95)'],
        'planning_latency_ms': ['mean', 'stddev'],
        'run_id': 'count'  # number of runs
    })
)

# Bootstrap confidence intervals
for (cluster, model_v, sim_v, source), group in aggregates.groupby([...]):
    runs = group['run_id'].values
    metrics_samples = [
        group[runs_sample]['collision_count'].mean()
        for runs_sample in bootstrap_samples(runs, n_iterations=1000)
    ]
    ci_lower, ci_upper = percentile(metrics_samples, [2.5, 97.5])
```

**Decision**: use bootstrap over closed-form confidence intervals. Reason: metric distributions are often non-normal (collision_count is count data; latency is long-tailed); bootstrap is robust and doesn't assume a parametric form.

### Cluster Weighting (Sim-vs-Real Reconciliation)

**Problem**: simulation has 10,000 runs in cluster X, real fleet has 5 runs. Naive aggregation drowns the real signal.

**Solution**: weight simulation confidence intervals by scenario importance:

```python
# Per-cluster importance weight
importance[cluster] = (
    (n_real_runs[cluster] + 1) /  # Laplace smoothing
    (n_sim_runs[cluster] + 1)
)  # Sim weight is inverse proportional to abundance

# Blend metrics
sim_ci = bootstrap_ci(sim_runs[cluster], weights=importance[cluster])
real_ci = bootstrap_ci(real_runs[cluster], weights=1.0)

blended_score = (
    0.7 * normalized(sim_ci) +  # sim is strong prior
    0.3 * normalized(real_ci)    # real is truth signal
)
```

**Why 70/30 split?** Sim is broader coverage (catch regressions), but has distribution shift. Real is sparse but accurate. Tune the ratio based on incident history: if simulator bugs blind-side you, increase real weight.

### Output: Materialized Metric Warehouse

Metrics are written to an OLAP warehouse (Trino, BigQuery, ClickHouse):

```sql
CREATE TABLE metrics_agg (
  date DATE,
  scenario_cluster STRING,
  model_version STRING,
  simulator_version STRING,
  scenario_library_version STRING,
  source STRING,  -- 'sim' or 'real'
  
  n_runs INT,
  collision_rate FLOAT,
  collision_rate_ci_lower FLOAT,
  collision_rate_ci_upper FLOAT,
  
  path_deviation_p50 FLOAT,
  path_deviation_p95 FLOAT,
  planning_latency_mean FLOAT,
  planning_latency_p99 FLOAT,
  
  blended_confidence_score FLOAT,
  last_updated TIMESTAMP
)
PARTITION BY date
```

---

## 4. Storage Tiering

### Hot Tier (0–14 days)
- **Storage**: SSD-backed object storage (S3 with SSD, GCS with high-performance class)
- **Access**: fast queries; full logs accessible for dashboard drill-downs
- **Cost**: ~$0.02–0.05 per GB
- **Retention**: 14 days; roll to warm after

### Warm Tier (14–90 days)
- **Storage**: compressed archive (gzip, zstd); moved to cheaper object class
- **Access**: 1–5 minute latency (requires decompression); used for post-mortems, historical comparisons
- **Cost**: ~$0.004 per GB
- **Retention**: 90 days; roll to cold after

### Cold Tier (90+ days)
- **Storage**: Glacier, Deep Archive, or equivalent
- **Access**: hours to days latency (on-demand rehydration); forensic investigation only
- **Cost**: ~$0.001–0.004 per GB
- **Retention**: 7 years (regulatory requirement for safety-critical events)

**Decision**: three-tier. Reason: cost scales as 10x between tiers, but query latency acceptability jumps. Hot queries (dashboard, last 2 weeks) use SSD; incident investigations (warm, last 90 days) tolerate decompression; regulatory backups (cold, 7 years) are fire-and-forget.

---

## 5. Sim-vs-Real Reconciliation

### The Gap Problem

Simulation has systematic biases: unrealistic sensor noise, simplified weather, perfect maps, no multi-agent interactions beyond pre-canned trajectories. You must **track and alarm on the sim-to-real gap**.

### Per-Scenario Gap Metric

For each scenario cluster, compute:

```
gap[cluster] = (
    |sim_metric[cluster] - real_metric[cluster]| /
    std(real_metric[cluster])
)
```

Examples:
- `gap[urban_intersection]` = 0.3σ → model generalizes well
- `gap[highway_merging]` = 2.1σ → **alarm** → investigate

### Alignment Before Promotion

The promotion gate requires **both**:

1. **Sim metrics pass**: all tracked scenarios within tolerance (e.g., collision rate < 0.1%, latency < 100ms p99)
2. **Real metrics pass**: on the real-world test corpus (smaller, curated set of 50–200 real drives), metrics pass the same thresholds
3. **Gap is aligned**: for each scenario in the real-world corpus, sim-to-real gap < 1.5σ (or your risk tolerance)

**Pseudocode**:
```python
def can_promote(model_v):
    # Check sim
    for cluster in sim_metrics[model_v]:
        if not all_within_bounds(sim_metrics[model_v][cluster]):
            return False, f"Sim fails {cluster}"
    
    # Check real
    for cluster in real_metrics[model_v]:
        if not all_within_bounds(real_metrics[model_v][cluster]):
            return False, f"Real fails {cluster}"
        
        # Check gap
        gap = compute_gap(sim_metrics[model_v][cluster], real_metrics[model_v][cluster])
        if gap > ALIGNMENT_THRESHOLD:
            return False, f"Gap misaligned {cluster}: {gap}σ"
    
    return True, "Approved"
```

---

## 6. Replay & Poisoning

### When a Simulator Bug is Found

1. **Mark affected runs**: tag all runs with the buggy simulator version as `poisoned: true`
   ```sql
   UPDATE metrics_agg 
   SET poisoned = true 
   WHERE simulator_version = 'carla-0.9.13-buggy' AND date >= '2026-06-20';
   ```

2. **Re-run scenarios**: re-simulate the affected scenario classes with the patched simulator
   ```
   scenario_classes_to_rerun = (
       SELECT DISTINCT scenario_cluster 
       FROM metrics_agg 
       WHERE poisoned = true
   )
   # Batch job: for each scenario_cluster, re-run N representative seeds
   ```

3. **Merge patched results**: replace old entries with new ones
   ```sql
   -- Delete old poisoned aggregates
   DELETE FROM metrics_agg 
   WHERE poisoned = true AND date >= '2026-06-20';
   
   -- Insert new aggregates from patched runs
   INSERT INTO metrics_agg 
   SELECT * FROM metrics_agg_patched WHERE ...;
   ```

4. **Lineage tracking**: maintain a **replay log** table
   ```sql
   CREATE TABLE replay_log (
       original_run_ids ARRAY<STRING>,
       replay_id STRING,
       reason STRING,
       patched_simulator_version STRING,
       created_at TIMESTAMP
   )
   ```
   This allows auditing which metrics are first-generation vs. replayed.

**Decision**: mark-and-delete, not versioned history. Reason: metrics should reflect ground truth; users should see the best version of the data. Lineage table preserves audit trail without cluttering the main warehouse.

---

## 7. Operational Instrumentation

### Key Metrics to Monitor

| Metric | Alert Threshold | Purpose |
|--------|-----------------|---------|
| **Worker queue depth** | > 1000 pending runs | detect worker stalls |
| **Ingestion latency** (p99) | > 5 min | detect streaming lag |
| **ETL latency** (end-to-end) | > 10 min | detect aggregation bottleneck |
| **Sim-to-real gap** (per cluster) | > 1.5σ | detect distribution shift |
| **Cost per run** | > $0.50 (tune to baseline) | detect sim config inflation |
| **Poisoned run %** | > 5% of daily volume | detect simulator instability |

### Dashboards

1. **Worker health**: per-cluster throughput, error rates, queue depth, cost/run trend
2. **Pipeline health**: ingestion latency, ETL latency, record count, poisoned %
3. **Metric health**: sim-to-real gap per cluster, confidence intervals over time
4. **Cost breakdown**: per-cluster, per-simulator-version, storage tiering costs

---

## 8. Querying & Analysis

### Hot Path: Pre-computed Dashboards

Top 20 most-watched metrics are materialized every 5 minutes as views:

```sql
CREATE MATERIALIZED VIEW dashboard_latest_metrics AS
SELECT
  scenario_cluster,
  model_version,
  simulator_version,
  n_runs,
  collision_rate,
  collision_rate_ci_lower,
  collision_rate_ci_upper,
  planning_latency_p99,
  last_updated
FROM metrics_agg
WHERE date = CURRENT_DATE
  AND source IN ('sim', 'blended')
ORDER BY scenario_cluster, model_version DESC;
```

**Latency**: < 1 second for dashboard queries.

### Cold Path: Notebook-Driven Analysis

For ad-hoc queries (comparing model versions, deep-diving into a scenario), use Trino / BigQuery:

```sql
-- Example: compare model-v3.2.0 vs v3.2.1 on highway scenarios
SELECT
  scenario_cluster,
  'v3.2.0' as model,
  collision_rate as collision_rate_old,
  collision_rate_ci_lower as ci_lower_old
FROM metrics_agg
WHERE model_version = 'av-model-v3.2.0'
  AND source = 'sim'
  AND scenario_cluster LIKE 'highway%'
  AND date = CURRENT_DATE
UNION ALL
SELECT
  scenario_cluster,
  'v3.2.1' as model,
  collision_rate as collision_rate_new,
  collision_rate_ci_lower as ci_lower_new
FROM metrics_agg
WHERE model_version = 'av-model-v3.2.1'
  AND source = 'sim'
  AND scenario_cluster LIKE 'highway%'
  AND date = CURRENT_DATE
```

**Latency**: 5–60 seconds (acceptable for exploration).

---

## 9. Tradeoffs & Alternatives

### Tradeoff 1: Stream-only vs. Batch + Stream

| Approach | Latency | Cost | Operability |
|----------|---------|------|-------------|
| **Stream-only** (Kafka → Flink/Spark Streaming) | 30–60s agg latency | Higher (always-on cluster) | Complex: exactly-once is hard |
| **Batch + Stream** (Kafka → Spark batch every 5 min) | 5 min agg latency | Lower (batch clusters scale down) | Simpler: exactly-once via batch checkpoints |

**Decision**: Batch + Stream. Reason: 5-minute latency is acceptable for evaluation dashboards; batch is easier to debug and replay; cost is 3–5x lower; exactly-once is enforced by batch semantics.

**Alternative**: if real-time ranking feedback is critical, add an optional fast-path stream aggregator (Flink) for pre-aggregates only; merge with batch every hour.

### Tradeoff 2: Cluster Weight Strategy

| Strategy | Pros | Cons |
|----------|------|------|
| **Inverse abundance** (weight by n_real / n_sim) | Real signal not drowned | High variance for rare clusters |
| **Fixed weight** (0.7 sim / 0.3 real always) | Stable, predictable | Ignores data imbalance; doesn't adapt |
| **Learned weight** (optimize promotion gate accuracy) | Empirically optimal | Requires historical incidents; harder to explain |

**Decision**: Inverse abundance (+ Laplace smoothing). Reason: intuitive; adapts as real data arrives; explainable to stakeholders. Monitor promotion gate false negative rate; if > 2%, switch to learned weights.

### Tradeoff 3: Confidence Intervals (Bootstrap vs. Parametric)

| Method | Assumes | CPU Cost | Robustness |
|--------|---------|----------|-----------|
| **Bootstrap** | None | ~1000 resamples per metric | High; works for any distribution |
| **T-test CI** | Normal | ~1 calculation | Low; fails on skewed data (latencies) |
| **Poisson CI** (for counts) | Poisson | ~1 calculation | Medium; collision_count often is count data, but underestimates variance |

**Decision**: Bootstrap. Reason: collision_count and latency distributions are non-normal; bootstrap is robust; 1000 resamples is fast enough (milliseconds) on modern hardware.

### Tradeoff 4: Poisoning Strategy (Delete vs. Flag)

| Strategy | Pros | Cons |
|----------|------|------|
| **Mark poisoned, keep in warehouse** | Audit trail in main table; users can filter | Warehouse bloat; careless queries see bad data |
| **Delete old, materialized replay log** | Clean warehouse; force fresh aggregation | Requires lineage to stay in sync |

**Decision**: Delete + replay log. Reason: users expect metrics to be canonical ground truth; lineage table preserves audit without cluttering the main interface.

---

## 10. Failure Modes & Mitigations

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| **Simulator crash/segfault mid-run** | Incomplete logs; worker stalls | Heartbeat monitoring; timeout logs as "incomplete"; mark for re-run |
| **Schema backward-incompatibility** | Old logs unreadable; queries fail | Protobuf `reserved` fields; strict schema review; read-path compatibility layer |
| **Metric pipeline lag (> 30 min)** | Stale dashboards; delayed incident detection | Alert on ETL latency p99; auto-scale batch cluster; prioritize hot-tier queries |
| **Sim-to-real gap suddenly widens** | Silent promotion of unsafe models | Alarm gate checks gap < 1.5σ; require human review on large gaps |
| **Real fleet data arrives late (4–24h delay)** | Promotion gate blocks waiting for real-world confirmation | Separate promotion tiers: "sim-ready" → monitor real data → "production-ready" |
| **Poisoned data accidentally queried** | Contaminated metrics; wrong decisions | Flag as `poisoned` in schema; add `WHERE NOT poisoned` to default views; audit trail |

---

## Summary

**Architecture**: Workers emit full logs to object storage + lightweight summaries to Kafka. Batch ETL (Spark, 5-min cadence) aggregates by (scenario, model, sim version). Output materialized into metric warehouse (Trino/BigQuery). Sim-vs-real reconciliation via weighted blending and gap tracking. Promotion gate checks sim ✓ + real ✓ + gap aligned ✓. Replay log + poisoning flags handle simulator bugs without rewriting history.

**Key insight**: **Treat versioning as a first-class design dimension.** Independently version model, simulator, and scenario library. This buys you reproducibility and the ability to slice metrics across any version triple without re-running the world.

**Key insight**: **Real signal must survive the sim flood.** Inverse-abundance weighting prevents real-world metrics from being drowned in confidence intervals; gap tracking forces alignment checks before promotion.
