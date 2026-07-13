# Solution: Semantic Search & Retrieval over Autonomous Driving Video Data

## Step 1: Clarifying Questions & Requirements

### Questions to Ask the Interviewer

**Scale & Data**
- How many vehicles, how many hours of video recorded per day? (Assume: thousands of vehicles, millions of driving-hours/year, hundreds of millions of clips accumulated.)
- What sensors are available besides camera — LiDAR, radar, IMU/vehicle telemetry, existing perception-stack outputs (detected objects/tracks)?
- Is there any existing metadata (GPS, weather API, disengagement flags, auto-labels from the perception stack)?

**Query Types**
- Free-text natural language queries only, or also query-by-example (give a clip, find similar ones) and structured filters (weather=rain AND time=night)?
- Do queries need to localize *where in a long clip* the event occurs, or is clip-level retrieval enough?

**Users & Latency**
- Interactive search for an engineer (seconds-level latency) vs. large offline batch mining jobs (minutes/hours acceptable)? Likely **both** — design must support both.
- Who curates ground truth / relevance judgments for evaluation?

**Long-Tail Mining**
- Is the goal purely retrieval (given a known scenario, find examples), or also **discovery** of previously-unknown rare scenario types?
- Is there a labeling budget / human-in-the-loop review pipeline downstream?

**Constraints**
- Storage/compute budget for embedding hundreds of millions of clips?
- Freshness requirement — must new drives become searchable within minutes, hours, or is next-day acceptable?
- Privacy/compliance: faces, license plates must be blurred/anonymized before broader engineering access.

---

## Step 2: Goals, Requirements & Assumptions

**Assumed Scale**
- 5,000 vehicles, each driving ~8 hrs/day → ~40,000 vehicle-hours/day.
- Segmented into ~10-second event windows → ~14M new indexable segments/day, ~5B/year accumulated (assume a rolling multi-year retained archive of ~2-5B segments after filtering/dedup).
- Multiple camera views per vehicle (front, side, rear) + LiDAR + telemetry.

**Functional Requirements**
- FR1: Free-text natural-language search ("rainy night, low visibility").
- FR2: Query-by-example (embedding of a reference clip as the query).
- FR3: Structured/metadata filtering composable with semantic search (city, weather, time-of-day, road type).
- FR4: Bulk export of top-N matches for a query, for training-set construction.
- FR5: Long-tail scenario mining workflow: given a rare-category description (or a small labeled seed set), surface a large, diverse pool of candidates for human review and training-set inclusion.

**Non-Functional Requirements**
- Interactive search: p95 latency < 1-2s for top-K retrieval over the full archive.
- Offline mining jobs: can scan/score billions of segments within hours, not required to be real-time.
- Indexing throughput: keep up with ~14M new segments/day without falling behind (near-real-time freshness target: new drives searchable within a few hours).
- Precision: for safety validation use cases, prioritize **high recall** (missing a real jaywalking event is worse than a few false positives an engineer has to skim past).
- Cost: embedding every frame of every camera at full resolution for the entire archive is prohibitive — must be judicious about sampling.

**Non-Goals (assumed, confirm with interviewer)**
- Real-time on-vehicle scenario detection is out of scope (assume this is an offline/data-center system operating on already-uploaded logs).
- Perfect natural-language understanding of arbitrarily complex compositional queries is out of scope for v1 — start with strong single-concept and simple-conjunction queries.

---

## Step 3: System Architecture (High-Level)

```
┌────────────────────────────────────────────────────────────────────┐
│                     FLEET UPLOAD (raw sensor logs)                  │
│   multi-camera video, LiDAR, radar, telemetry, perception outputs   │
└───────────────────────────────┬───────────────────────────────────-┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  PREPROCESSING & SEGMENTATION                                     │
│  - Decode, sync multi-sensor streams                               │
│  - PII redaction (blur faces/plates)                                │
│  - Segment continuous drive into indexable units (Step 4B)          │
│  - Attach weak metadata (GPS→city, weather API, time-of-day)        │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│  MULTIMODAL EMBEDDING SERVICE  (Step 4A)                           │
│  - Video encoder (+ LiDAR/telemetry fusion) → scene embedding       │
│  - Shared video-text embedding space (contrastive pretraining)      │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
    ┌───────────────────────────┐  ┌───────────────────────────────┐
    │  VECTOR INDEX (ANN)        │  │  METADATA STORE (structured)   │
    │  Sharded HNSW/IVF-PQ        │  │  city, weather, time, tags,    │
    │  index, incrementally       │  │  perception-stack detections   │
    │  updated  (Step 4C)         │  │                                 │
    └─────────────┬───────────────┘  └───────────────┬─────────────────┘
                  │                                  │
                  └────────────────┬─────────────────┘
                                   ▼
                    ┌─────────────────────────────────┐
                    │   QUERY SERVICE                  │
                    │  - Text query → text encoder      │
                    │  - Example query → video encoder  │
                    │  - ANN search + metadata filter   │
                    │    (pre- or post-filtering)       │
                    │  - Re-ranking (cross-attention)   │
                    └─────────────┬─────────────────────┘
                                  │
                ┌─────────────────┼──────────────────────┐
                ▼                                        ▼
   ┌─────────────────────────┐              ┌─────────────────────────────┐
   │ INTERACTIVE SEARCH UI     │              │ LONG-TAIL MINING PIPELINE    │
   │ (engineer, seconds-level) │              │ (batch, Step 4D)             │
   └─────────────────────────┘              │  - active-learning loop      │
                                              │  - diversity-aware sampling  │
                                              │  - human review queue        │
                                              └──────────────┬───────────────┘
                                                             ▼
                                              ┌─────────────────────────────┐
                                              │ CURATED TRAINING / EVAL SETS │
                                              └─────────────────────────────┘
```

---

## Step 4A: Multimodal Video/Text Embeddings

### Goal
Represent every indexable segment (Step 4B) as a dense vector such that:
- Segments that are **semantically similar** (same scenario type) are close together.
- A **text description** of a scenario embeds close to matching video segments (enables free-text search).

### Architecture: Dual/Multi-Tower Contrastive Embedding (CLIP-style, video-adapted)

```
Video Tower:
  Multi-frame clip (sampled at ~2-4 fps) + optional LiDAR BEV summary
      → Spatio-temporal video encoder (e.g., a ViT-based video transformer
        such as ViViT/VideoMAE-style backbone, or a frame-level image
        encoder + temporal transformer aggregator)
      → mean/attention-pool over time → clip embedding (d=512-1024)

Text Tower:
  Natural language description
      → Pretrained text transformer (e.g., a CLIP/BERT-style text encoder)
      → text embedding (d=512-1024, same space as video tower)

Training objective:
  Contrastive loss (InfoNCE / CLIP loss) pulling matched (video, text)
  pairs together and pushing mismatched pairs apart, using large in-batch
  negatives.
```

**Where does training data (video, text) come from?**
- **Auto-generated captions** from existing perception-stack outputs — e.g., turn tracked-object attributes into templated captions ("pedestrian crossing mid-block, no crosswalk, nighttime") to bootstrap millions of weakly-labeled (video, text) pairs cheaply.
- **Human-written captions** on a smaller curated subset for higher-quality supervision (especially for nuanced/rare scenarios).
- **General-domain video-language pretraining** (large public video-caption corpora) as a starting checkpoint, then **domain-adapt/fine-tune** on driving-specific data — driving scenes have very different statistics (dashboard camera perspective, road-specific vocabulary) than generic web video.

**Multimodal fusion beyond RGB video:**
- Encode LiDAR as a bird's-eye-view (BEV) occupancy/intensity map through a small CNN, and fuse (concatenate or cross-attend) with the video embedding before the shared projection layer — this helps distinguish geometry-dependent scenarios (e.g., "unprotected left turn," lane geometry) that are ambiguous from RGB alone.
- Optionally fuse telemetry (harsh braking, steering angle spikes) as auxiliary signal — useful for flagging "interesting" moments (e.g., near-miss events) even before semantic labeling.
- Existing perception-stack detections (bounding boxes, tracks, classified agents) can be embedded as a structured side-input (e.g., a small transformer over detected-object tokens) and fused in — this gives the model an explicit, higher-precision signal about *what agents are present*, complementing the raw-pixel signal.

### Segmentation Strategy (what unit do we embed?)

- **Fixed sliding windows** (e.g., 10s, 50% overlap) — simple, uniform, but can split events awkwardly and wastes compute on uneventful stretches.
- **Event-based / saliency-triggered segmentation (recommended)** — use cheap, lightweight signals to find candidate "interesting" boundaries first, then embed those windows at higher fidelity:
  - Telemetry-based triggers: braking/steering/acceleration anomalies.
  - Perception-stack triggers: new agent enters near path, agent class change, sudden track velocity change.
  - This lets us **skip embedding long stretches of uneventful highway driving at full cost**, focusing embedding compute on segments likely to matter — critical for cost control at this scale (see Step 4C on cost).
- Store embeddings at **multiple granularities**: a coarse embedding per long segment for fast filtering, plus finer sub-window embeddings for precise localization within a matched clip.

### Keeping Embeddings Comparable Over Time
- Embedding models will be periodically retrained/improved. Maintain a **model version tag** on every stored embedding, and support **incremental re-embedding** of the archive (prioritized: re-embed recently-queried or high-value segments first, backfill the rest over time) rather than requiring an all-at-once full re-index, which would be prohibitively expensive at this scale.

---

## Step 4B: Fast Vector Similarity Search at Scale

### Scale of the Problem
- Billions of embedding vectors (segments), each d=512-1024 floats.
- Interactive queries need top-K results in ~1-2s; mining jobs can tolerate longer batch scans.
- Continuous ingestion of ~14M new segments/day requires the index to support incremental updates, not just periodic full rebuilds.

### Index Choice
- **Approximate Nearest Neighbor (ANN) index**, not exact search — exact k-NN over billions of high-dim vectors is computationally infeasible at interactive latency.
- Candidate structures:
  - **HNSW (Hierarchical Navigable Small World graphs):** excellent recall/latency tradeoff, supports incremental inserts, but memory-hungry (full graph + vectors in RAM) — good for a "hot" subset (e.g., most recent N months, or most-queried data).
  - **IVF-PQ (Inverted File + Product Quantization, e.g., FAISS):** compresses vectors (quantization) to fit far more vectors per unit of memory/disk, trades some recall for massive scale — good for the "cold"/full historical archive.
  - **Recommended hybrid:** tiered index — HNSW for a hot recent tier (fast, high recall, frequently queried), IVF-PQ for the full cold archive (scalable, cheaper, acceptable recall loss for exploratory/bulk mining queries).

### Sharding & Distribution
- Shard the index horizontally (e.g., by time range and/or geographic region) across many machines; a query fans out to relevant shards in parallel and merges top-K results.
- Sharding by time is natural here: most interactive queries skew toward recent data (e.g., "find recent examples for this week's triage"), while long-tail mining jobs scan the full historical range in batch — time-based shards let us route interactive queries to a small hot set of shards.

### Metadata Filtering + Vector Search
- Two approaches:
  - **Pre-filtering:** restrict the candidate set by metadata (weather=rain AND time=night) *before* running ANN search on the filtered subset. Efficient when the filter is very selective, but naive pre-filtering can break ANN graph structure (HNSW graphs assume searching the full index).
  - **Post-filtering:** run ANN search over the full index, then filter results by metadata. Simple, but risks returning too few results if the filter is highly selective and matches are sparse in the top-K.
  - **Recommended:** use ANN libraries with native **filtered search support** (e.g., HNSW variants with attribute-aware graph traversal, or IVF-PQ combined with per-partition metadata bucketing) so filtering happens jointly with the graph/cluster traversal rather than as a separate pre/post step — this avoids both the recall collapse of naive pre-filtering and the wasted-work problem of post-filtering.

### Incremental Index Updates
- New segments arrive continuously (~14M/day). Rebuilding a billion-scale ANN index from scratch daily is infeasible.
- Use an index structure that supports **online insertion** (HNSW natively supports this) for the hot tier.
- For the IVF-PQ cold tier, batch new segments into periodic (e.g., hourly/daily) micro-index builds ("segments"), and merge/compact them into the main index on a slower cadence (similar to LSM-tree compaction) — query time fans out across the main index + small number of recent micro-indexes, merging results.
- Periodically (e.g., weekly) re-balance/re-cluster the IVF-PQ index as data distribution shifts (new cities, new sensor hardware).

### Re-ranking
- ANN retrieval (top-1000) trades precision for speed; add a **re-ranking stage** using a heavier cross-attention model (jointly encodes query + candidate rather than independent embeddings) over the top-K candidates to improve final precision before returning results to the engineer — mirrors retrieval+ranking patterns from other large-scale search/recsys systems.

### Serving Latency Budget (interactive path)
```
Text/example query encoding:      ~20-50ms
ANN search (sharded, parallel):   ~200-500ms
Metadata-filtered merge:          ~50-100ms
Re-ranking top-200 → top-20:      ~200-400ms
─────────────────────────────────────────────
Total:                            ~1-1.5s (within budget)
```

---

## Step 4C: Long-Tail Data Sampling for Training Sets

### The Core Challenge
Safety-critical scenarios (jaywalking mid-block, emergency-vehicle interactions, unusual double-parking, erratic cyclists) are, by definition, **rare relative to routine driving**. A random sample of the archive will be dominated by uneventful highway/street driving, so naive random sampling for training-set construction will produce highly imbalanced, low-value datasets. We need a **targeted mining pipeline**.

### Approach: Seed → Retrieve → Diversify → Human-Verify → Expand (active-learning loop)

**1. Seed a scenario definition**
- Start from a natural-language description ("cyclist swerving into the lane") and/or a small handful of known example clips (from an incident report, a simulation team request, or a previous mining round).

**2. Coarse retrieval (recall-oriented, cheap)**
- Use the semantic embedding index (Step 4B) to retrieve a **large candidate pool** (e.g., top-50K) via text and/or example-based query — deliberately over-retrieve since the goal here is *recall*, not precision; false positives are cheap (filtered later), false negatives mean we permanently miss real long-tail examples.
- Optionally combine with cheap heuristic filters (perception-stack tags like "cyclist detected," telemetry anomaly flags) to narrow the pool further without needing embedding search alone to do all the work.

**3. Diversity-aware sub-sampling (avoid near-duplicate over-representation)**
- Raw archives contain highly correlated near-duplicates (the same intersection recorded by many vehicles on similar days). Including thousands of near-identical examples wastes labeling budget and biases the trained model toward over-represented conditions (e.g., one specific intersection).
- Techniques:
  - **Embedding-space clustering** (e.g., k-means or greedy farthest-point sampling within the candidate pool's embedding space) to select a diverse, representative subset rather than the raw top-K by similarity score alone.
  - **Deduplication** via near-neighbor thresholding (discard candidates whose embedding is within ε of an already-selected example).
  - Stratify sampling across metadata dimensions we care about generalizing over (city, weather, time-of-day, road type) so the mined set isn't accidentally concentrated in one condition.

**4. Human-in-the-loop verification**
- Route the diversified candidate pool to human labelers/safety engineers to confirm true positives, correct boundary/timing of the event, and add fine-grained labels (severity, novel sub-variants).
- Prioritize labeler time using a **confidence-ranked queue** — e.g., rank by embedding-similarity score combined with a lightweight classifier's confidence, so labelers see the highest-value (most likely to be a genuine, useful long-tail example) items first, and lowest-confidence "interesting but uncertain" items are also sampled deliberately to catch what pure similarity search would miss.

**5. Active learning loop to expand recall over rounds**
- Confirmed true positives (and hard negatives — near-misses that turned out *not* to be the target scenario) are fed back to:
  - Fine-tune/refine the embedding model or a lightweight scenario-specific classifier on top of embeddings, improving future retrieval precision/recall for this scenario type.
  - Re-query the full archive with the improved model to catch examples the first round missed (embedding search is imperfect; iterating closes the recall gap).
- Repeat for several rounds until new-example yield plateaus (diminishing returns signal it's time to stop mining this scenario).

**6. Discovering *unknown* long-tail scenarios (not just retrieving known ones)**
- For scenario types nobody has thought to query for yet: 
  - **Anomaly/novelty detection** in embedding space — flag segments that are outliers relative to dense clusters of "routine driving," since rare/unusual scenarios tend to be embedding-space outliers.
  - **Disengagement/intervention-triggered mining** — segments immediately preceding a human safety-driver intervention or planner disagreement are disproportionately likely to contain novel edge cases; prioritize mining around these signals.
  - **Clustering the outlier pool** to surface emergent scenario *categories* (rather than one-off outliers) for engineers to name, define, and turn into new seed queries — closing the loop back into step 1.

### Measuring Success of the Mining Pipeline
- **Coverage:** number of distinct, verified scenario sub-types represented in the curated set (not just raw count).
- **Diversity:** embedding-space spread / cluster coverage of the mined set relative to the full candidate pool.
- **Yield rate:** fraction of retrieved candidates confirmed as true positives by human review (tracks precision of the retrieval+ranking stage over time).
- **Downstream signal:** does a model trained/evaluated with the newly mined long-tail examples show measurable improvement (or newly-caught regressions) on held-out safety-critical benchmarks?

---

## Step 5: Evaluation

### Offline (Retrieval Quality)
- Curate a benchmark set of (query → relevant segment IDs) pairs, ideally including genuinely rare scenarios, not just common ones.
- **Recall@K** (critical for safety use cases — did we surface all known relevant clips within the first K results?).
- **Precision@K / mAP** for interactive search quality.
- **Embedding retrieval vs. re-ranked retrieval** — measure the lift from the cross-attention re-ranking stage to justify its added latency cost.

### Online / Operational
- Interactive search: query latency (p50/p95/p99), engineer task-success rate (did they find what they needed within N queries), search abandonment rate.
- Mining pipeline: labeler yield rate (% of surfaced candidates confirmed relevant), time-to-curate-N-examples for a new scenario type, diversity metrics of curated sets over time.
- Indexing pipeline: ingestion lag (time from raw upload to searchable), index freshness SLA adherence.

---

## Step 6: Tradeoffs & Design Decisions

### Tradeoff 1: Recall vs. Latency
**Decision:** Two distinct paths — a fast, latency-bounded ANN path for interactive search (accepts some recall loss), and a slower, more exhaustive/iterative path for long-tail mining where recall matters far more than speed. Don't force one architecture to serve both needs equally well.

### Tradeoff 2: Embedding Granularity (whole-clip vs. fixed-window vs. event-triggered)
**Decision:** Event/saliency-triggered segmentation with multi-granularity embeddings (coarse + fine). 
**Tradeoff:** Uniform fixed windows are simpler to implement and reason about, but waste embedding compute on uneventful driving and can split an event across window boundaries, hurting retrieval. Event-triggered segmentation is more complex (relies on upstream heuristics/telemetry signals that could themselves miss subtle events) but is far more cost-efficient at this scale and improves localization precision.

### Tradeoff 3: Index Freshness vs. Rebuild Cost
**Decision:** Tiered index (hot HNSW with online inserts + cold IVF-PQ with periodic compaction), rather than full daily rebuilds.
**Tradeoff:** A single always-consistent global index would be simpler to reason about but computationally infeasible to rebuild daily at billions-of-vectors scale; the tiered/LSM-like approach adds operational complexity (merge/compaction logic, querying across tiers) in exchange for sustainable ingestion throughput.

### Tradeoff 4: Automated Mining vs. Human Review
**Decision:** Always keep a human-in-the-loop verification step for safety-critical training data, using automation purely to *prioritize* what humans review, not to fully replace review.
**Tradeoff:** Fully automated labeling (trusting embedding similarity scores directly) would be far cheaper and faster, but false positives/negatives in safety-critical training data are unacceptably costly — mislabeled rare scenarios can silently degrade the planner's behavior in exactly the situations that matter most.

### Tradeoff 5: Storage/Compute Cost vs. Coverage
**Decision:** Don't embed every frame of every camera at full resolution for the entire archive; use saliency-triggered segmentation and lower frame sampling rates for routine stretches, reserving dense/high-fidelity embedding for triggered "interesting" windows.
**Tradeoff:** This risks missing subtle events that don't trip any cheap trigger heuristic (a pure recall-vs-cost tradeoff) — mitigate by periodically re-processing older data at higher fidelity as compute budget allows, and by using the anomaly-detection/disengagement-triggered mining (Step 4C.6) as a complementary net that doesn't solely rely on the upfront triggers.

### Tradeoff 6: General-Purpose vs. Domain-Adapted Embedding Model
**Decision:** Start from large-scale general video-language pretraining, then fine-tune/domain-adapt on driving-specific (video, text) pairs (auto-generated + human-curated).
**Tradeoff:** Training a driving-specific embedding model from scratch would better fit the domain but requires far more curated paired data than is initially available; general pretraining + domain adaptation gets useful embeddings faster and can be iteratively improved as more in-domain captioned data is collected via the mining loop itself.

---

## Step 7: Common Follow-up Questions

### Q: How would you support multi-camera / 360° context?
**A:** Encode each camera view with a shared video encoder, then fuse per-camera embeddings with a small cross-view attention/aggregation layer before the final projection, so the scene embedding reflects the full surround context (e.g., a jaywalker seen transitioning from a side camera into the front camera). Store per-camera embeddings too, to support camera-specific queries when needed.

### Q: How do you fuse LiDAR/radar, not just video?
**A:** See Step 4A — encode LiDAR as a BEV map through a CNN and fuse with the video tower before the shared embedding space; radar can contribute a lightweight feature vector (relative velocity/range of nearby agents) fused similarly. This is especially valuable for scenarios where geometry/motion (not appearance) is the defining signal, like "unprotected left turn with oncoming traffic."

### Q: How do you avoid missing safety-critical scenarios (false negatives)?
**A:** Bias the mining pipeline toward recall at every stage (over-retrieve, use multiple complementary triggers — semantic search, perception heuristics, disengagement-triggered mining, anomaly detection), track recall against a held-out benchmark of known scenarios, and treat any confirmed "missed by the system, found some other way" case as a signal to retrain/improve the embedding model or add a new triggering heuristic.

### Q: How do you discover a brand-new scenario type nobody has queried for?
**A:** Combine embedding-space anomaly/outlier detection with disengagement/intervention-triggered mining (Step 4C.6) to surface unusual segments proactively, then cluster the outlier pool so engineers can identify and name emergent categories, turning them into new seed queries for the standard retrieval loop.

### Q: How do you handle temporal/compositional queries spanning multiple sub-events?
**A:** v1 handles single-concept and simple-conjunction queries well via the shared embedding space plus metadata filters. For genuinely sequential queries ("jaywalker appears right after exiting a tunnel"), decompose the query into sub-events, retrieve segment sequences independently, then verify temporal ordering/adjacency as a post-retrieval join over each clip's timeline — this is a natural v2 extension once the core embedding/index infrastructure is in place.

### Q: How would the design change for on-vehicle, real-time scenario flagging?
**A:** That's a fundamentally different latency/compute regime (milliseconds, on embedded hardware) — would require a much smaller, distilled embedding model running on-device, likely only flagging candidate windows for later upload/indexing rather than doing full semantic search on-vehicle. This system as designed is the offline/data-center counterpart that such a system would feed into.

### Q: How do you keep embeddings comparable as you upgrade the embedding model?
**A:** Version-tag every embedding with the model version that produced it; support mixed-version search during migration (either by projecting old embeddings into the new space with a learned adapter, or by prioritized incremental re-embedding of the archive — recent/high-value data first); avoid requiring a full synchronized re-index cutover, which is infeasible at this scale (see Step 4A).
