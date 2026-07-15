# System Design: Simulation Log Collection and Result Aggregation

## Context

You are building the metric pipeline for an autonomous vehicle (AV) platform. The platform runs thousands to millions of simulation runs per day on a fleet of workers. Each simulation produces structured logs: vehicle state, perception outputs, planner decisions, collision flags, etc. Your job is to ingest these logs, convert them into evaluation metrics, and surface insights—while **handling an asymmetry: simulation produces orders of magnitude more data than the real fleet**.

## Scale Assumptions

- **Throughput**: 1,000–10,000 simulation runs per day
- **Data per run**: 10 MB to 1 GB (structured logs, one folder per run)
- **Metadata**: lightweight summary record per run (metric snapshot, scenario ID, model version, success flag)
- **Ingestion latency**: logs arrive within minutes of simulation completion
- **Query latency**: dashboard queries should return within seconds to minutes (hot tier); deep-dive drill-downs can tolerate hours (cold tier)
- **Retention**: hot tier 14 days, warm tier 90 days, cold/archive beyond

## Key Challenges

1. **Volume**: billions of log records per day, but simulation should not drown out real-world fleet data
2. **Schema evolution**: model versions, simulator versions, and scenario library versions all change independently
3. **Reconciliation**: simulation metrics are a strong prior, on-road metrics are ground truth—how do you blend them without letting sim swallow the signal?
4. **Reproducibility**: when a simulator bug is found, how do you replay only affected runs and merge patched results?
5. **Operational visibility**: detect stalled workers, metric-pipeline lag, cost anomalies

## What a Strong Answer Should Cover

- **Ingestion architecture**: how logs flow from workers to durable storage, what triggers downstream processing
- **Schema design**: versioning strategy for logs, summary records, and aggregated metrics
- **Aggregation & storage**: how you compute metrics, where they live, and how you query them
- **Sim-vs-real reconciliation**: explicit strategy for treating simulation as prior and on-road data as truth
- **Replay & poisoning**: how you handle simulator bugs and re-run affected scenarios
- **Operational instrumentation**: monitoring worker health, pipeline latency, and cost
- **Tradeoff decisions**: why you chose object storage vs. streaming-only, why bootstrap confidence intervals, etc.

## Common Follow-Up Questions to Expect

- How do you prevent a simulator version bug from poisoning weeks of metrics?
- What happens when a scenario library update invalidates old sim runs?
- How do you weight simulation confidence intervals so rare scenarios don't dominate?
- How do you surface sim-to-real gaps so incidents are caught early?
- What does the promotion gate look like when you have asymmetric data?
