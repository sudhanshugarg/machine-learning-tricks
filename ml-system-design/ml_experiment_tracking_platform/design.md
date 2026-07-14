# System Design: ML Experiment Tracking & Analysis Platform

## Problem Statement

You are an ML platform engineer at a company with **hundreds of ML engineers and researchers** spread across many teams (recommendations, ads, search, computer vision, LLMs, forecasting). Today, every team tracks their experiments in an ad-hoc way — some log metrics to text files, some to spreadsheets, some print to stdout and scroll their terminal, some keep results in a personal Jupyter notebook. As a result:

- **Nobody can reproduce anyone else's results.** A model that "worked last quarter" can't be re-run because the exact code version, hyperparameters, and dataset snapshot weren't captured together.
- **There is no shared record of what's been tried.** Two engineers on the same team unknowingly run the same failed experiment. A new hire has no way to see the last six months of tuning history for a model they're inheriting.
- **Comparing runs is painful.** Deciding "is model B actually better than model A" means manually copying numbers into a spreadsheet, with no consistent notion of which metric, which data split, or whether the difference is even statistically meaningful.
- **The path from experiment to production is broken.** When a model is promoted to production, there's no reliable link back to the exact experiment (code + config + data + weights) that produced it, which makes debugging production regressions and audits very hard.

You are asked to design an **ML experiment tracking and analysis platform** — an internal product (think MLflow / Weights & Biases / Comet / Neptune, built in-house) that ML engineers instrument their training code with, so that **every experiment run is automatically captured, comparable, reproducible, and discoverable**, and so that the best runs can be reliably promoted toward production.

**Illustrative usage the platform must support:**
- An engineer adds a few lines to their training script; from then on, every run automatically logs its hyperparameters, code version, dataset version, environment, metrics-over-time (loss/accuracy per step), system utilization (GPU/CPU/memory), and final artifacts (model weights, plots, sample predictions).
- A researcher launches a **hyperparameter sweep** of 500 runs and wants a live dashboard to compare them, kill underperforming runs early, and find the best configuration.
- A team lead opens a dashboard and compares this week's candidate model against the current production model across a dozen metrics and data slices, and wants to know whether the improvement is real or noise.
- Six months later, an on-call engineer debugging a production regression needs to pull up the exact experiment that produced the deployed model — its code commit, config, training data snapshot, and full metric history.

---

## Scale & Context (assume unless told otherwise, confirm with interviewer)

- **~500 ML engineers/researchers**, running on the order of **50,000–100,000 experiment runs per month** (a single sweep can be hundreds or thousands of runs). Runs range from 30-second smoke tests to multi-week distributed training jobs on hundreds of GPUs.
- **Metric logging volume**: a long training run logs metrics every few steps — potentially **millions of metric data points per run**, and hundreds of millions to billions of metric points written per day platform-wide, arriving as a high-throughput stream from many concurrent jobs.
- **Artifacts** (model checkpoints, datasets, plots, media) range from kilobytes to **hundreds of gigabytes per run**; the platform stores metadata about them and integrates with blob storage rather than storing large blobs in its own database.
- **Consumers**: individual engineers (live monitoring + post-hoc analysis of their own runs), teams (shared project dashboards, leaderboards, comparisons), platform/infra teams (cost & utilization reporting), and downstream systems (model registry, CI/CD for model promotion, governance/audit).
- **Existing state**: assume there is a compute/training platform (Kubernetes or a scheduler) that runs the jobs, blob storage for artifacts, and a data-versioning story for datasets — this platform must **integrate** with those, not replace them.

---

## What You Should Cover

1. **Clarifying Questions** — what exactly counts as a "run" vs. "experiment" vs. "project"; whether tracking must be real-time (live dashboards while a job runs) or post-hoc is acceptable; the write throughput of metric logging; how large artifacts are handled; multi-tenancy/access-control needs; whether the platform also owns hyperparameter sweep *orchestration* or just *tracking* of externally-launched runs; and what "reproducibility" concretely means here.

2. **Goals, Requirements & Constraints**
   - Functional: an ergonomic **client SDK** that instruments training code with minimal boilerplate; automatic capture of config/code/data/environment; high-throughput logging of metrics-over-time, system metrics, and artifact metadata; live and post-hoc **dashboards** for a single run and for comparing many runs; **search/query** across all runs; and a path to **promote** a run's model toward production.
   - Non-functional: the client must be **low-overhead and non-blocking** (it must never slow down or crash the training job it's instrumenting); the backend must sustain very high write throughput while serving fast interactive reads for dashboards; strong durability (a lost metric history means a lost experiment); and multi-tenant isolation and access control.

3. **Client SDK Design (the write path & fault tolerance)**
   - How does the SDK log millions of metric points from inside a training loop without adding meaningful latency? (buffering, batching, async flush, sampling/aggregation).
   - What happens when the network is flaky or the tracking backend is briefly down — how do you avoid **either** blocking the training job **or** silently losing data? (local buffering, retries, offline mode, resume).
   - How do you capture code version, dependency environment, hyperparameters, and dataset version automatically and reliably enough that a run is actually reproducible?

4. **Storage & Data Model (the read path)**
   - What is the data model — how do you represent projects → experiments → runs → (params, metrics-timeseries, artifacts, system-metrics, tags)?
   - Metric histories are **append-heavy time series**; run metadata/params are **queried and filtered** interactively. Would you use one store or several (e.g., a relational/metadata DB + a time-series store + blob storage for artifacts)? Why?
   - How do you serve fast interactive queries — "show me the top 20 runs in this project by validation AUC where lr < 0.01 and dataset = v3" — over tens of millions of runs, and fast metric-history reads for plotting long runs (downsampling/rollups)?

5. **Analysis & Comparison Layer**
   - How do you let users **compare many runs** at once (parallel-coordinates over hyperparameters, overlaid metric curves, grouped/aggregated views) and surface which hyperparameters actually mattered (importance/sensitivity analysis)?
   - How do you support **hyperparameter sweeps** — launching, monitoring, early-stopping bad runs (e.g., Hyperband/ASHA-style), and picking winners — and where does the boundary sit between *orchestrating* sweeps vs. just *tracking* them?
   - How do you make comparisons **statistically honest** — same eval set, controlling for seed variance, flagging when a "win" is within noise, and guarding against the many-comparisons problem when someone sifts 500 runs for the best?

6. **Reproducibility, Model Registry & Production Path**
   - What must be captured so that any run can be **re-executed** and (ideally) reproduce its result? How do you handle the parts that are hard to pin down (nondeterminism, data drift, external services)?
   - How does a promising run get promoted to a **model registry** and then to production, with a durable link back to the experiment that produced it, for debugging and audit later?

7. **System Architecture** — end-to-end: client SDK → ingestion/write path (buffering, batching, streaming) → storage tiers (metadata DB, time-series store, blob storage) → query/serving layer → dashboards/analysis UI → integration with sweep orchestration, model registry, and CI/CD.

8. **Evaluation & Operations**
   - How do you measure whether the platform is **succeeding** (adoption, reproducibility rate, time-to-insight, dropped-metric rate, dashboard latency)?
   - How do you keep costs bounded as metric/artifact volume grows (retention/downsampling policies, tiered storage)?

9. **Tradeoffs** — discuss explicitly:
   - **Real-time streaming ingestion vs. batch upload** of metrics (freshness of live dashboards vs. backend load and client complexity).
   - **One general-purpose store vs. specialized stores** (relational + time-series + blob) for the very different access patterns of params, metric time series, and artifacts.
   - **Client-side aggregation/sampling vs. logging everything raw** (write cost & query speed vs. fidelity of the metric history).
   - **Blocking vs. non-blocking logging** on failure (never lose data vs. never slow/kill the training job).
   - **Tracking-only vs. also owning orchestration** (sweep scheduling, early stopping) — a bigger, stickier product vs. a focused, composable one.
   - **Storing everything forever vs. retention/downsampling** (full historical fidelity & audit vs. cost).

---

## Common Follow-up Questions to Expect

- A training job logs a metric every 10 steps for 10 million steps — how do you store that and still render a responsive plot instantly? (downsampling, rollups, resolution tiers)
- The tracking backend goes down for 20 minutes during a 3-day distributed training run — what happens to the metrics logged during that window?
- How do you attribute and compare runs from a **distributed** training job (many workers) as a single logical run?
- How do you make a run truly reproducible when training is nondeterministic (GPU nondeterminism, async data loading, changing upstream data)?
- Someone runs a 1,000-run sweep and reports the single best validation number as their result — what's wrong with that, and how does the platform help do it right?
- How do you enforce access control and multi-tenancy (team A shouldn't see team B's confidential runs) without making cross-team discovery useless?
- How do you keep storage costs bounded as the platform accumulates years of runs and hundreds of TB of artifact metadata?
- Where does this platform end and the feature store / model serving / monitoring systems begin?
