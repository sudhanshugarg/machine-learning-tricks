# System Design: Semantic Search & Retrieval over Autonomous Driving Video Data

## Problem Statement

You are an ML engineer at **Waymo**. The fleet's self-driving vehicles continuously record sensor data (multi-camera video, LiDAR, radar, vehicle telemetry) during every drive. Over years of operation this has produced a **massive archive of driving footage** — hundreds of millions of short clips, spanning many cities, weather conditions, and times of day.

Today, finding specific scenarios in this archive is a manual, painful process: engineers rely on sparse metadata tags (city, date, disengagement flags) and ad-hoc scripts. This severely limits two critical workflows:

1. **Safety & validation engineers** need to pull every clip matching a specific real-world scenario to test whether the current planning/perception stack handles it correctly — e.g., *"find all clips with a pedestrian jaywalking mid-block at night,"* or *"find unprotected left turns in heavy rain with oncoming traffic."*
2. **ML training teams** need to mine the archive for **rare, safety-critical "long-tail" events** (e.g., cyclists suddenly swerving, double-parked delivery trucks blocking a lane, emergency vehicles with sirens) to build balanced training and evaluation sets, since these events are naturally very rare relative to routine driving.

You are asked to design an end-to-end system that lets engineers **search and retrieve specific driving scenes from this video database using free-text or example-based queries**, and that also supports **mining and curating long-tail examples** for model training.

**Illustrative queries the system must support:**
- "Rainy conditions at night with low visibility"
- "Pedestrian jaywalking mid-block"
- "Cyclist swerving into the lane"
- "Unprotected left turn with oncoming traffic"
- "Double-parked vehicle blocking the lane"
- "Find clips visually/semantically similar to *this* clip" (query-by-example)

---

## Scale & Context (assume unless told otherwise, confirm with interviewer)

- Fleet of thousands of vehicles, each recording multiple synchronized camera streams (plus LiDAR/radar) continuously during operation.
- Archive size: hundreds of millions of clips (billions of frames), growing by many terabytes per day.
- Clips range from a few seconds to a couple of minutes; most interesting "events" occur in a short sub-window of a longer drive segment.
- Existing weak metadata: GPS/route, timestamp, weather API data, coarse disengagement/intervention flags, some auto-generated object-detection tags.
- Consumers: safety/validation engineers (ad hoc, exploratory queries), ML training/data engineers (bulk retrieval + mining), simulation team (scenario extraction).

---

## What You Should Cover

1. **Clarifying Questions** — scale, data modalities available, query types (text vs. example-based vs. structured filters), latency/throughput expectations (interactive search vs. offline batch mining), who the users are, and what "success" looks like for each user type.

2. **Goals, Requirements & Constraints**
   - Functional: free-text search, query-by-example, compositional/structured filters (weather + time of day + maneuver type), long-tail mining for dataset curation.
   - Non-functional: search latency for interactive use, indexing throughput for continuously arriving data, cost of storing/indexing at this scale, precision/recall expectations for safety-relevant queries.

3. **Data & Representation**
   - How do you represent a clip (or sub-clip/event window) so it's searchable by natural language and by similarity?
   - What **multimodal embedding** approach would you use to jointly encode video (and optionally LiDAR/telemetry) and text into a shared space?
   - How do you segment continuous drive logs into indexable units (fixed windows vs. event-based segmentation)?

4. **Indexing & Retrieval at Scale**
   - How do you build and maintain a **vector similarity search** index over hundreds of millions to billions of embeddings?
   - How do you keep the index fresh as new drives stream in continuously, without full rebuilds?
   - How do you combine vector similarity with structured/metadata filtering (e.g., "rainy AND night AND jaywalking")?

5. **Long-Tail Data Sampling for Training Sets**
   - Rare/safety-critical events are a tiny fraction of the archive. How do you efficiently discover, sample, and curate a **balanced, diverse training/eval set** of long-tail scenarios without manually reviewing the entire archive?
   - How do you avoid near-duplicate over-representation and measure diversity/coverage of the mined set?
   - How do you close the loop as new rare-event types are discovered (active learning / human-in-the-loop labeling)?

6. **System Architecture** — end-to-end data flow from raw fleet uploads → preprocessing/segmentation → embedding generation → indexing → serving the search API → downstream consumption by training pipelines.

7. **Evaluation**
   - Offline: retrieval metrics (recall@K, precision@K, mAP) against a curated benchmark of query → relevant-clip pairs.
   - Online: engineer satisfaction/task success, query latency, coverage of long-tail categories in resulting training sets, downstream model improvement from mined data.

8. **Tradeoffs** — discuss explicitly:
   - **Recall vs. latency** for interactive search vs. exhaustive offline mining.
   - **Embedding granularity**: whole-clip vs. fixed-window vs. dynamically segmented events.
   - **Index freshness vs. rebuild cost** as the fleet continuously generates new data.
   - **Automated mining vs. human review** for confirming true positives on safety-critical long-tail scenarios.
   - **Storage/compute cost** of embedding and indexing every frame vs. sparse/keyframe sampling.

---

## Common Follow-up Questions to Expect

- How would you extend this to support **multi-camera / 360° context** rather than a single camera feed?
- How would you fuse LiDAR and radar signals into the embedding, not just video?
- How do you validate that the system doesn't miss safety-critical scenarios (false negatives are costly)?
- How would you detect and mine for a **brand-new** rare scenario type that has never been explicitly labeled before?
- How do you handle **temporal queries** spanning multiple events within a clip (e.g., "a jaywalker appears right after the car exits a tunnel")?
- How would this system's design change if it needed to run **on-vehicle** for real-time scenario flagging vs. purely offline in a data center?
- How do you keep embeddings valid/comparable as you periodically upgrade the embedding model over time?
