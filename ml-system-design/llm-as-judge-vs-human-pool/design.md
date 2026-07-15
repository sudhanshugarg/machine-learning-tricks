# System Design: Evaluation System with Human + LLM Evaluators

## Context

You're building an evaluation platform for an AI training company. The platform grades candidate outputs (e.g., essay responses, code solutions, creative work) against a detailed rubric. Your system must handle **two evaluator pools simultaneously**:

1. **Human raters**: Trained annotators with skill tiers, geographic/timezone coverage, and throughput constraints
2. **LLM judges**: Automated scoring via language models (cheaper, faster, but less reliable)

The business constraint: **minimize cost while maintaining evaluation quality**. You can't simply send everything to humans (too expensive), and you can't trust LLM judges blindly (accuracy suffers). Your system must intelligently route items to the right evaluator type and combine their signals.

---

## Example: RLHF Feedback Platform

To make this concrete, consider **a company training a large language model** (e.g., Claude, GPT) and collecting human feedback for reinforcement learning from human feedback (RLHF).

**The scenario:**
- The model generates responses to thousands of prompts per day
- Each response must be rated (quality, safety, instruction-following) on a 5-point scale
- Hiring enough expert annotators to evaluate everything is prohibitively expensive (~$20M/day at current costs)
- Using an LLM to grade responses is tempting (cheap, instant) but unreliable (gets tricked, has biases, doesn't match human judgment)

**Why the hybrid system is essential:**
- **Easy responses** (clearly good or bad): LLM grades them instantly for $0.001 per item
- **Borderline responses** (score 2–4): Route to expert humans ($2/item) because these cases most affect model learning
- **Validation**: Continuously measure whether LLM grades correlate with human ground truth; retrain the LLM judge if it drifts

**Business impact:**
- Cost: $0.10/item on average (vs. $2.00 if all-human)
- Quality: Feedback used for training is as good as human-only eval (Spearman's ρ ≥ 0.85)
- Speed: 95% of items scored within seconds (vs. 30 minutes for humans)

---

## Scale Assumptions

- **Throughput**: 1,000 items/second (86.4M items/day)
- **Human capacity**: ~1,000 items/day per annotator (includes annotation time, breaks, QA)
- **LLM latency**: ~200ms per item
- **Human latency**: 5–30 minutes per item (variable by tier and complexity)
- **Rubric**: 5-point Likert scale (1–5) + optional reasoning/calibration feedback
- **Cost model**:
  - Human rater: $0.50–$2.00 per item (varies by tier and complexity)
  - LLM call: $0.001–$0.01 per item (varies by model)

---

## Functional Requirements

1. **Score every item** submitted to the system with a per-item score (1–5) and confidence interval
2. **Provide an audit trail** showing:
   - Which evaluator(s) graded the item
   - Final score and reasoning
   - Any human-LLM disagreement or calibration steps
3. **Support multi-evaluator consensus** for high-stakes items
4. **Enable ongoing LLM-to-human calibration** so the LLM judge improves over time

---

## Non-Functional Requirements

1. **Cost control**: Operate within a budget per evaluation batch (e.g., $0.10 avg cost/item)
2. **Latency**: p99 latency ≤ 2 seconds for initial score, full audit trail within 5 minutes
3. **Quality**: Maintain correlation with gold-standard human-only evaluation (e.g., Spearman's ρ ≥ 0.85)
4. **Operational agility**: Dynamically re-route traffic between humans and LLMs based on:
   - Remaining budget for the batch
   - Human rater availability (time zone, skill tier)
   - LLM judge confidence (per item)
5. **Observability**: Track per-rater accuracy, LLM drift, and inter-rater reliability (Cohen's κ)

---

## What a Strong Answer Should Cover

1. **Routing Strategy**
   - How do you decide: LLM-only, human-only, or both?
   - When do you escalate edge cases (confidence < threshold)?

2. **LLM Judge Calibration**
   - How do you tune LLM scores against human gold labels?
   - What structure/prompt techniques improve LLM reliability?

3. **Human Pool Management**
   - How do you track rater accuracy and bias?
   - How do you handle skill tiers and workload balancing?

4. **Score Aggregation & Consensus**
   - How do you combine LLM + human scores when both evaluate the same item?
   - Majority voting, weighted averaging, Bayesian fusion?

5. **Cost Optimization**
   - How do you allocate your human budget across batches?
   - When do you reject an item as unevaluable, and how?

6. **Evaluation & Feedback Loop**
   - How do you measure end-to-end quality (correlation with ground truth)?
   - How do you retrain/prompt-tune the LLM judge over time?

---

## Common Follow-Up Questions to Expect

- How do you handle **data drift** in LLM judge performance (e.g., new evaluation domains)?
- What if a human rater is consistently biased? How do you detect and correct for it?
- How do you structure the LLM prompt to reduce hallucination or reasoning errors?
- What happens when the human pool is overloaded? Do you degrade to LLM-only mode?
- How do you validate that your LLM judge is really measuring what humans measure?
