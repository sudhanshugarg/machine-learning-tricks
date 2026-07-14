# Solution: ML Experiment Tracking & Analysis Platform

## Step 1: Clarifying Questions & Requirements

### Questions to Ask the Interviewer

**What are the core entities — run vs. experiment vs. project?**
- Do we agree on a hierarchy: a **project** (a modeling effort, e.g. "ranking-v2") contains many **runs** (a single execution of a training script), and runs can be grouped into **experiments/sweeps** (a set of runs sharing a purpose, e.g. one hyperparameter sweep)? (Assume yes — `project → run`, with sweeps/tags as ways to group runs.)
- Is a **distributed training job** (many workers) one logical run or many? (Assume one logical run with per-worker sub-streams that we aggregate.)

**Real-time or post-hoc?**
- Do users need **live dashboards** updating while a job trains (to kill bad runs early), or is it acceptable to see results only after the run finishes? (Assume live is a hard requirement — early-stopping a bad sweep run is a primary use case.)

**What's the logging throughput?**
- Roughly how many metric points per second across the fleet? (Assume peaks of **hundreds of thousands to low-millions of metric points/sec** platform-wide, from tens of thousands of concurrent runs.)

**Who owns large artifacts?**
- Does this platform store model weights/datasets itself, or store **metadata + a pointer** into existing blob storage? (Assume metadata + pointer — we never put multi-GB blobs in our own DB; we integrate with S3/GCS.)

**Tracking only, or orchestration too?**
- Do we also **launch and schedule** sweeps (own the compute), or only **track** runs launched by an existing scheduler? (Assume: tracking is the core; we provide a *thin* sweep-orchestration layer that emits run configs and consumes tracked metrics for early-stopping, but the actual compute runs on the existing training platform.)

**What does "reproducibility" mean here?**
- Bit-for-bit re-execution, or "capture enough (code commit + config + data snapshot + environment) that a human can re-run and get a statistically equivalent result"? (Assume the latter — bit-for-bit is often infeasible with GPU nondeterminism; we aim for *captured and re-runnable*.)

**Multi-tenancy / access control?**
- Per-team isolation with the ability to share? Confidential projects? (Assume yes — RBAC at the project level, with a discoverable public-within-company default and a private option.)

---

## Step 2: Goals, Requirements & Assumptions

**Assumed Scale**
- ~500 users; **50k–100k runs/month**; tens of thousands of concurrent runs at peak.
- **Hundreds of millions to billions of metric points/day**; a single long run can emit **millions of points**.
- Artifacts from KB to **hundreds of GB/run**; stored in blob storage, referenced by metadata.
- Interactive dashboards must feel instant: **p95 query latency < ~500 ms** for run-list filtering and metric-plot loads.

**Functional Requirements**
- FR1: **Client SDK** — a few lines of code auto-captures params, code version, dataset version, environment, and streams metrics/system-stats/artifact-metadata, with minimal boilerplate and minimal overhead.
- FR2: **Ingestion** — sustain high-throughput metric writes without blocking the training job.
- FR3: **Single-run view** — live and post-hoc: metric curves, system utilization, logs, config, artifacts.
- FR4: **Multi-run comparison** — overlay curves, parallel-coordinates over hyperparameters, grouped/aggregated views, hyperparameter-importance analysis.
- FR5: **Search/query** — filter/sort across all runs by params, metrics, tags, status, author, date.
- FR6: **Sweeps** — launch/track hyperparameter sweeps, early-stop underperformers, surface winners.
- FR7: **Reproducibility + registry** — capture enough to re-run; promote a run's model to a model registry with a durable back-link to the producing experiment.

**Non-Functional Requirements**
- NFR1: **Client is non-blocking and fault-tolerant** — logging never slows down or crashes training; a backend outage must not lose data or halt the job (local buffering + retry + offline mode).
- NFR2: **High write throughput + fast interactive reads** — the write path (append-heavy time series) and read path (filtered queries, downsampled plots) have very different profiles and are optimized separately.
- NFR3: **Durability** — a lost metric history is a lost experiment; acknowledged writes must survive.
- NFR4: **Multi-tenancy & RBAC** — project-level access control; isolation with opt-in sharing.
- NFR5: **Cost-bounded** — retention/downsampling/tiered storage keep growth sustainable.

**Non-Goals (assumed, confirm)**
- Not a **feature store**, not a **model-serving/inference** system, not a production **model-monitoring** system — though it links to all of them (a promoted model carries its experiment lineage into serving/monitoring).
- Not the **compute scheduler** — jobs run on the existing training platform.

---

## Step 3: System Architecture (High-Level)

```
┌────────────────────────────────────────────────────────────────────────┐
│  TRAINING JOB (on existing compute platform; may be distributed)         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  CLIENT SDK  (in-process, Step 4A)                                │   │
│  │   - auto-capture: git commit, deps/env, hyperparams, dataset ver  │   │
│  │   - in-memory ring buffer  →  batch  →  async background flush     │   │
│  │   - local disk WAL (offline/outage buffer)  →  retry              │   │
│  │   - large artifacts uploaded directly to blob storage (pointer    │   │
│  │     + checksum sent to backend)                                    │   │
│  └───────────────────────────────┬──────────────────────────────────┘   │
└──────────────────────────────────┼──────────────────────────────────────┘
                                    │ batched gRPC/HTTP (metrics, params, meta)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  INGESTION / WRITE PATH (Step 4B)                                        │
│   - stateless ingest API (load-balanced, autoscaled)                     │
│   - validates + writes params/metadata → Metadata DB                     │
│   - appends metric points → durable log / stream (Kafka)                 │
│         │                                                                 │
│         ├──► stream consumers: write raw points to Time-Series store      │
│         └──► rollup workers: precompute downsampled resolutions           │
└───────────────┬───────────────────────────────┬────────────────────────┘
                │                                 │
                ▼                                 ▼
   ┌──────────────────────┐          ┌──────────────────────────┐
   │  METADATA DB          │          │  TIME-SERIES STORE        │
   │  (relational, e.g.    │          │  (columnar/TSDB:          │
   │   Postgres)           │          │   metrics-over-time +     │
   │  projects, runs,      │          │   system metrics,         │
   │  params, tags, status,│          │   multi-resolution        │
   │  artifact pointers    │          │   rollups)                │
   └──────────┬───────────┘          └───────────┬──────────────┘
              │                                    │
              │       ┌──────────────────┐         │
              │       │  BLOB STORAGE     │         │
              │       │  (S3/GCS:         │         │
              │       │  weights, data,   │         │
              │       │  plots, media)    │         │
              │       └──────────────────┘         │
              ▼                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│  QUERY / SERVING LAYER (Step 4C)                                         │
│   - run-list search & filter (Metadata DB, + search index for scale)     │
│   - metric-history reads (Time-Series store, resolution-aware)           │
│   - comparison/analysis endpoints (multi-run aggregation, HP importance) │
│   - caching for hot dashboards                                           │
└───────────────┬───────────────────────────────┬────────────────────────┘
                │                                 │
                ▼                                 ▼
     ┌────────────────────┐          ┌──────────────────────────────┐
     │  DASHBOARDS / UI    │          │  SWEEP ORCHESTRATOR (Step 4D) │
     │  single-run, multi- │◄────────►│  emits configs, reads live    │
     │  run compare, sweeps│          │  metrics, early-stops runs     │
     └────────────────────┘          └──────────────────────────────┘
                │
                ▼
     ┌────────────────────────────────────────────────────────────┐
     │  MODEL REGISTRY + CI/CD PROMOTION (Step 4E)                  │
     │  promote run → registered model version (lineage back-link)  │
     └────────────────────────────────────────────────────────────┘
```

---

## Step 4A: Client SDK Design (the write path & fault tolerance)

The SDK is the product's front door — if it's heavy, awkward, or ever crashes a training job, adoption dies. Two properties dominate: **ergonomics** (a few lines) and **it must never harm the job it instruments**.

### Ergonomics — minimal instrumentation

```python
import exptrack

run = exptrack.init(project="ranking-v2", config=hyperparams)  # auto-captures env
for step in range(num_steps):
    ...
    run.log({"train_loss": loss, "val_auc": auc}, step=step)   # non-blocking
run.log_artifact("model.pt", type="model")                      # → blob storage
run.finish()
```

`init()` auto-captures, with no extra user code:
- **Code version**: current git commit hash + dirty-diff (warn/store the uncommitted diff so a "dirty" run is still reconstructable).
- **Environment**: Python version, pip/conda freeze, CUDA/driver versions, container image digest.
- **Hyperparameters**: the `config` dict (and optionally CLI args / config files).
- **Dataset version**: a dataset ID/snapshot hash from the data-versioning system (or the SDK hashes the input manifest).
- **Hardware**: node/GPU type, world size for distributed jobs.

**Decision:** auto-capture as much as possible rather than relying on users to log it manually.
**Tradeoff:** auto-capture can miss context the user knows and adds a little startup cost; but manual capture is the #1 reason experiments turn out irreproducible — people forget. Auto-first with manual overrides is the sweet spot.

### The write path — non-blocking, batched, buffered

The training loop may call `log()` thousands of times per second. Doing a synchronous network round-trip per call would wreck throughput. Design:

1. **In-memory ring buffer**: `log()` just appends to an in-process buffer and returns immediately (sub-microsecond). No network on the hot path.
2. **Background flush thread/process**: batches buffered points and ships them every ~1–5 s or every N points, over a compact batched gRPC/HTTP call.
3. **Local disk WAL (write-ahead log)**: batches are also (or first) persisted to a local append-only file before/while being sent, so a crash or backend outage doesn't lose acknowledged data.
4. **Backpressure policy**: if the buffer fills because the backend is slow, prefer **dropping/aggregating low-priority high-frequency metrics** (e.g., downsample per-step system stats) over ever blocking the training loop. Critical events (run start/finish, final metrics, artifacts) are never dropped.

**Decision (the key NFR):** on failure, **never block and never silently lose** — buffer to local disk and retry; if the buffer is truly exhausted, degrade fidelity (sample) rather than stall training.
**Tradeoff:** the alternative extremes are both bad — *blocking* until the backend recovers can stall or OOM a multi-GPU job (catastrophic cost); *fire-and-forget* silently loses data. The disk-WAL + retry + graceful-degradation path costs some local disk and complexity but honors both halves of NFR1.

### Handling outages & resume (offline mode)

- If the backend is unreachable, the SDK keeps appending to the local WAL and retries with exponential backoff. A **fully offline mode** lets a run log entirely to disk and **sync later** (`exptrack sync ./run-dir`), which also covers air-gapped clusters.
- Each metric point carries `(run_id, metric_name, step, timestamp, value)` and writes are **idempotent** on `(run_id, metric_name, step)`, so replayed/retried batches after an outage don't create duplicates.

### Distributed runs

- All workers share one `run_id`; each tags its stream with a `rank`. **Rank-0 aggregation** is the default (only the chief logs training metrics to avoid N× duplication), but per-worker system metrics are kept separate so you can spot a straggler/hot GPU. The backend presents them as **one logical run** with drill-down per worker.

---

## Step 4B: Ingestion / Write Path (backend)

The ingest tier is **stateless and autoscaled** behind a load balancer; it does three things: authenticate/authorize, write run **metadata/params** to the Metadata DB, and append **metric points** to a durable stream.

**Decision:** put a durable log/stream (**Kafka** or equivalent) between ingest and storage, rather than writing metric points straight to the DB synchronously.
**Tradeoff:** it adds a component and a few seconds of end-to-end lag, but it (a) absorbs bursty write spikes from thousands of concurrent runs, (b) decouples ingest availability from storage availability, and (c) lets multiple consumers fan out — one writes raw points to the time-series store, another computes rollups, another can feed the sweep early-stopper — all from the same stream. A synchronous direct-to-DB write path would be simpler but couples the client's success to DB write capacity and makes bursts dangerous.

**Rollup workers** consume the stream and precompute **downsampled resolutions** (e.g., raw, 1-in-10, 1-in-100, 1-in-1000) as data arrives, so the read path never has to scan millions of raw points to draw a plot (see 4C).

---

## Step 4C: Storage & Data Model (the read path)

### Polyglot persistence — three stores for three access patterns

**Decision:** use **specialized stores** rather than forcing everything into one database.

| Data | Access pattern | Store |
|---|---|---|
| projects, runs, **params**, tags, status, authors, artifact **pointers** | filtered/sorted **queries**, joins, transactional updates | **Relational DB** (e.g., Postgres) |
| **metrics-over-time**, system metrics | append-heavy writes, range reads, downsampling | **Time-series / columnar store** (TSDB or columnar OLAP) |
| model weights, datasets, plots, media | large blobs, immutable | **Blob storage** (S3/GCS), referenced by pointer |
| run-list free-text/faceted search at scale | flexible filtering across many fields | **Search index** (e.g., Elasticsearch), fed from the Metadata DB |

**Tradeoff:** polyglot persistence means more moving parts and cross-store consistency to manage (a run's params live in Postgres, its curves in the TSDB, its weights in S3). But the access patterns are genuinely different — cramming billions of append-only metric points into a relational DB kills it on write volume and index bloat, while a TSDB is poor at the rich relational filtering the run-list needs. Matching store to pattern is worth the operational complexity at this scale.

### Data model (Metadata DB)

```
Project(project_id, name, owner_team, visibility, ...)
Run(run_id, project_id, author, status, created_at, finished_at,
    git_commit, env_hash, dataset_version, sweep_id?, ...)
Param(run_id, key, value)          -- hyperparameters (indexed for filtering)
Tag(run_id, key, value)
SummaryMetric(run_id, key, value)  -- final/best value per metric (for fast leaderboards)
Artifact(run_id, name, type, blob_uri, checksum, size)
```

`SummaryMetric` is a deliberate **denormalization**: leaderboard/run-list queries ("top 20 by best val_auc") read one indexed row per run instead of scanning each run's full metric history in the TSDB.

### Serving fast reads

- **Run-list filter/sort** ("top 20 runs where lr<0.01 and dataset=v3, by val_auc") → Metadata DB with proper indexes, or the search index once cardinality is high. Sub-second because it touches summary rows, not time series.
- **Metric-history plots** → **resolution-aware reads** from the TSDB: a plot has only ~1–2k pixels of width, so serve the appropriate precomputed rollup (Step 4B) instead of millions of raw points. Zooming into a time window fetches finer resolution on demand. This is what keeps a 10-million-step curve rendering instantly.
- **Caching**: hot dashboards (a live sweep everyone's watching) are cached with short TTLs; finished runs are immutable and cache indefinitely.

---

## Step 4D: Analysis, Comparison & Sweeps

### Multi-run comparison

- **Overlaid metric curves**: many runs' `val_auc` on one chart, aligned by step or wall-clock, with grouping/aggregation (mean±std band across seeds).
- **Parallel-coordinates plot**: each run is a line threading through hyperparameter axes and a final-metric axis — the workhorse for spotting "high lr + small batch → bad" patterns across hundreds of runs at a glance.
- **Hyperparameter importance / sensitivity**: fit a quick surrogate (e.g., gradient-boosted tree, or fANOVA) predicting the target metric from the hyperparameters, and report per-parameter importance — tells the user *which knobs actually mattered*, not just which run won.

### Hyperparameter sweeps

**Decision:** provide a **thin orchestrator** — it owns the *search strategy* and *early-stopping*, but delegates *compute* to the existing training platform.
- The user declares a search space + strategy (grid / random / **Bayesian optimization**). The orchestrator emits run configs; the compute platform runs them; each run reports metrics through the normal SDK path.
- **Early stopping** (**ASHA / Hyperband**): the orchestrator consumes live metrics off the same stream and kills the bottom fraction of runs at successive rungs, reallocating budget to promising configs — often a 3–10× efficiency win on large sweeps.

**Tradeoff (tracking-only vs. owning orchestration):** owning orchestration makes the product stickier and enables early-stopping (a big cost saver), but it also means owning a scheduler-adjacent system and its failure modes. Keeping it *thin* (search + early-stop decisions, not compute) captures most of the value while staying composable with whatever scheduler teams already use.

### Statistically honest comparison

A platform that makes it trivial to sort 500 runs by "best val metric" actively encourages bad science. Guardrails:
- **Same-eval-set enforcement**: comparisons warn/block if runs were evaluated on different data splits/versions (the platform knows the `dataset_version`).
- **Seed variance**: encourage multi-seed runs and show **mean ± std / confidence intervals**, not a single cherry-picked number; flag when a "win" falls within the seed-noise band.
- **Many-comparisons / selection bias**: when a user picks the best of N runs, the reported best is an **optimistic** estimate. Nudge toward reporting on a **held-out test set** distinct from the selection metric, and surface that "best-of-500 on val" ≠ expected production performance.

**Tradeoff:** these guardrails add friction to a workflow users want to be frictionless. But silent support for p-hacking-by-dashboard produces confident-but-wrong "improvements" that don't replicate in production — the friction buys trustworthy conclusions.

---

## Step 4E: Reproducibility, Model Registry & Production Path

### What makes a run reproducible

Capture, at `init()` time, the four pillars — **code, config, data, environment** — plus the outputs:
- **Code**: git commit + uncommitted diff.
- **Config**: full hyperparameters + resolved config files/CLI args.
- **Data**: dataset version/snapshot hash (relies on a data-versioning system; if none, hash the input manifest).
- **Environment**: dependency lockfile + container image digest + hardware/driver versions.
- **Outputs**: final weights + metric history as artifacts.

**Handling what's hard to pin down:**
- **Nondeterminism** (GPU atomics, async dataloaders): capture the seed and determinism flags, but be honest that bit-for-bit reproduction often isn't achievable — the goal is *statistically equivalent* re-runs, and multi-seed runs quantify the residual noise.
- **Data drift / external services**: pinning the dataset snapshot handles training data; runs that call live external services are flagged as **not fully reproducible** and that limitation is recorded on the run.

### Model registry & production path

- A promising run is **promoted** to a **model registry** entry: a named model with versioned stages (`staging → production → archived`). The registry entry stores a **durable back-link to the producing run** (its code/config/data/env/metrics lineage).
- **CI/CD promotion**: promotion can trigger validation gates (eval on a frozen test set, fairness/slice checks, latency/size budgets) before a version advances to `production`.
- **Closing the loop**: when a production model regresses, on-call pulls the registry entry → the exact experiment → its full lineage, turning a multi-hour archaeology exercise into a click. This lineage is exactly what governance/audit needs, too.

---

## Step 5: Evaluation & Operations

**Adoption & value**
- Fraction of training jobs instrumented (adoption); weekly active users; number of runs tracked.
- **Reproducibility rate**: fraction of runs that have all four capture pillars complete (a proxy for "could actually be re-run").
- **Time-to-insight**: time from run start to a usable comparison dashboard.

**Reliability & performance (SLOs)**
- **Dropped-metric rate**: fraction of logged points that never reach durable storage — should be ~0; this is the core trust metric.
- Ingest availability and end-to-end logging lag (log call → visible on live dashboard).
- Dashboard **p95 query latency** (< ~500 ms target) for run-list and plot loads.

**Cost**
- Storage growth vs. retention policy effectiveness; cost per tracked run; TSDB and blob spend.

---

## Step 6: Tradeoffs & Design Decisions

### Tradeoff 1: Real-time streaming ingestion vs. batch upload
**Decision:** streaming (batched every few seconds) via a durable log, giving near-live dashboards.
**Tradeoff:** streaming raises backend load and client complexity vs. a simple "upload a file when the run finishes." But live dashboards + early-stopping of sweeps (a major cost saver) require freshness; batch-only upload would forfeit the platform's most valuable interactive use cases.

### Tradeoff 2: One general-purpose store vs. specialized stores
**Decision:** polyglot — relational (metadata/params) + time-series (metrics) + blob (artifacts) + search index.
**Tradeoff:** more components and cross-store consistency to manage, but the write/read profiles are too different for one store to serve well at this scale (see Step 4C).

### Tradeoff 3: Client-side aggregation/sampling vs. logging everything raw
**Decision:** log raw by default up to a budget; degrade to sampling/aggregation only under backpressure or for very high-frequency, low-value metrics (e.g., per-step system stats), and always precompute rollups server-side.
**Tradeoff:** sampling loses fidelity (you might miss a transient spike), but logging every raw point from millions-of-steps runs at fleet scale is expensive to store and slow to query. Rollups give cheap fast plots while raw (within budget/retention) stays available for zoom-in.

### Tradeoff 4: Blocking vs. non-blocking logging on failure
**Decision:** never block the training loop; buffer to a local disk WAL, retry, and degrade fidelity before ever stalling. Never silently drop acknowledged data.
**Tradeoff:** local buffering + idempotent replay is more complex and uses local disk, but blocking can stall/OOM a hundred-GPU job (catastrophic cost) and fire-and-forget loses experiments — the WAL path is the only option honoring both halves of "non-blocking *and* no data loss."

### Tradeoff 5: Tracking-only vs. also owning orchestration
**Decision:** a **thin** orchestration layer (search strategy + early-stopping) that delegates compute to the existing platform.
**Tradeoff:** owning orchestration adds scheduler-adjacent complexity and failure modes, but enables early-stopping (large cost savings) and a stickier product. Keeping it thin captures the value without re-implementing a scheduler.

### Tradeoff 6: Store everything forever vs. retention & downsampling
**Decision:** tiered retention — keep raw metrics for a recent window, keep rollups/summaries long-term; keep registered/production-linked runs and their lineage indefinitely; expire raw data for stale, unpromoted runs.
**Tradeoff:** downsampling old runs loses the ability to zoom into their fine detail later, but storing every raw point for years is unbounded cost. Tying retention to *importance* (promoted/audited runs kept in full) targets spend where it matters.

---

## Step 7: Common Follow-up Questions

### Q: A run logs a metric every 10 steps for 10M steps — how do you render a responsive plot?
**A:** Don't ship raw points to the browser. Server-side, precompute **multi-resolution rollups** as data arrives (Step 4B); a plot is ~1–2k pixels wide, so serve the resolution matching the requested time window (e.g., 1-in-1000 for the full curve). Zooming fetches finer resolution for that window on demand. The client never handles millions of points.

### Q: The backend goes down for 20 minutes during a 3-day distributed run — what happens?
**A:** The SDK keeps appending to its **local disk WAL** and retries with backoff (Step 4A); the training job is unaffected because logging is non-blocking. When the backend recovers, buffered batches sync; **idempotent writes keyed on `(run_id, metric, step)`** prevent duplicates. Worst case (buffer exhausted), high-frequency system metrics are sampled while critical events are preserved.

### Q: How do you treat a distributed job as one logical run?
**A:** All workers share one `run_id` tagged by `rank` (Step 4A). Training metrics are logged by rank-0 by default (avoiding N× duplicates); per-worker system metrics are kept separate so you can spot a straggler. The UI aggregates them into one run with per-worker drill-down.

### Q: How do you make a run reproducible when training is nondeterministic?
**A:** Capture the four pillars (code/config/data/env) plus seed and determinism flags (Step 4E), but be explicit that bit-for-bit isn't guaranteed on GPU. The target is *statistically equivalent* re-runs; multi-seed runs quantify residual variance, and runs depending on live external services are flagged as not fully reproducible.

### Q: Someone reports the single best val number from a 1,000-run sweep — what's wrong, and how does the platform help?
**A:** Best-of-N on the selection metric is an **optimistically biased** estimate (selection bias / many-comparisons). The platform nudges toward reporting on a **held-out test set** distinct from the selection metric, shows **mean±CI across seeds** rather than a lone max, and flags wins inside the noise band (Step 4D).

### Q: How do you enforce multi-tenancy without killing discovery?
**A:** Project-level RBAC with a **discoverable-by-default-within-company** visibility and a **private** option for confidential work (Step 2). Metadata is searchable across teams by default so people can learn from each other; sensitive projects opt into isolation. Artifact access inherits the project's ACL.

### Q: How do you keep storage costs bounded over years?
**A:** Tiered retention (Step 6, Tradeoff 6): raw metrics for a recent window, rollups long-term, promoted/audited runs kept in full indefinitely, stale unpromoted raw data expired. Artifacts follow blob-storage lifecycle policies (transition to cold storage, then expire) keyed on run importance.

### Q: Where does this platform end and feature store / serving / monitoring begin?
**A:** This platform owns **experiment-time** lineage and analysis; it hands a promoted model (with lineage) to the **model registry**, which serving/monitoring consume. The **feature store** supplies training data (whose version this platform records) but isn't owned here. Production **monitoring** is a separate system — but the experiment lineage captured here is exactly what it links back to when diagnosing a production regression.
