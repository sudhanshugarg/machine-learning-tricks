# FAQ: Evaluation System with Human + LLM Evaluators

## Question Log

| Date | Category | Question | Status |
|------|----------|----------|--------|
| 2026-07-14 | LLM Calibration | Why calibrate confidence (not score)? How do you group calibration data? | [ANSWERED] |
| 2026-07-14 | Evaluation & Metrics | How do you compute Spearman's correlation coefficient? | [ANSWERED] |

---

## Answers by Category

### LLM Calibration

#### Q: Why calibrate confidence instead of the score itself? How do you actually collect calibration data? `[ANSWERED]`

**A:** 

**The distinction:**
- **Score** = the actual rating output (1, 2, 3, 4, or 5 on Likert scale). This is already valid.
- **Confidence** = the LLM's claimed certainty about that score (e.g., 0.95 means "95% sure"). This is what needs calibration.

**The problem:** LLM confidence doesn't match reality. For example, when an LLM claims confidence=0.95, it might actually be correct only 92% of the time—it's overconfident.

**How to collect calibration data:**

1. Run LLM on N calibration items (e.g., 10,000), each producing: `(score, confidence)`
2. Get human ground truth for the same N items
3. **Group LLM outputs by claimed confidence level:**
   ```
   Confidence [0.90–1.00]:
     Item 1 (LLM conf=0.95): correct ✓
     Item 4 (LLM conf=0.95): correct ✓
     Item 5 (LLM conf=0.92): wrong ✗
     ... (50 items in this bucket)
     → Actual accuracy: 46/50 = 92%
   
   Confidence [0.80–0.90):
     Item 2 (LLM conf=0.87): correct ✓
     Item 11 (LLM conf=0.84): correct ✓
     ... (120 items in this bucket)
     → Actual accuracy: 96/120 = 80%
   
   Confidence [0.70–0.80):
     ... (200 items)
     → Actual accuracy: 75%
   ```

4. **Create a calibration table:**
   ```
   LLM Confidence | Actual Accuracy
   --------------|----------------
   0.95          | 0.92
   0.85          | 0.80
   0.75          | 0.75
   0.60          | 0.60
   ```

5. **Fit Platt scaling (logistic regression)** to get a smooth curve:
   ```python
   platt = LogisticRegression()
   platt.fit(X=confidences, y=accuracies)
   
   # For a new item with confidence 0.89:
   true_accuracy = platt.predict_proba([[0.89]])[0][1]  # → ~0.81
   ```

**Why Platt scaling?** LLM confidence is continuous (0.0–1.0). You can't bucket every possible value. Platt scaling interpolates between your observed data points so you can predict accuracy for *any* confidence value.

**Why this matters for routing:**
- Without calibration: you'd naively route based on raw confidence (risky if LLM is overconfident)
- With calibration: you know the LLM's true accuracy for each confidence level, so you can route intelligently
  - "Route items with calibrated accuracy > 0.90 to LLM-only"
  - "Route items with calibrated accuracy < 0.75 to human"

*Pointer:* [solution.md](solution.md), Section 2.3 "Confidence Calibration via Platt Scaling" — discusses the technical implementation

---

### Routing Strategy

*(Answers go here)*

### Human Pool Management

*(Answers go here)*

### Score Aggregation & Consensus

*(Answers go here)*

### Cost Optimization

*(Answers go here)*

### Evaluation & Feedback Loop

#### Q: How do you compute Spearman's correlation coefficient? `[ANSWERED]`

**A:**

**What it is:** Spearman's correlation (ρ, rho) measures how well two ranked lists correlate. It checks: "do items that rank high in one list also rank high in the other?"

**The process:**
1. Take two lists of scores (e.g., LLM scores and human ground-truth scores for the same items)
2. Rank them separately (highest score = rank 1, next highest = rank 2, etc.)
3. Compute the correlation between the two rank lists

**Example:**

```
Item | LLM Score | Human Score | LLM Rank | Human Rank | Rank Diff²
-----|-----------|-------------|----------|------------|----------
  1  |     5     |      5      |    1     |     1      |     0
  2  |     4     |      3      |    2     |     3      |     1
  3  |     2     |      2      |    4     |     4      |     0
  4  |     3     |      4      |    3     |     2      |     1
  5  |     1     |      1      |    5     |     5      |     0
                                                     Sum = 2

ρ = 1 - (6 × Σ(d²)) / (n(n²-1))
  = 1 - (6 × 2) / (5 × 24)
  = 1 - 0.10
  = 0.90
```

**Code (using scipy):**

```python
from scipy.stats import spearmanr

llm_scores = [5, 4, 2, 3, 1]
human_scores = [5, 3, 2, 4, 1]

rho, p_value = spearmanr(llm_scores, human_scores)

print(f"Spearman's ρ: {rho:.3f}")   # → 0.900
print(f"p-value: {p_value:.4f}")     # → 0.0368
```

**Interpretation:**

| ρ Value | Meaning |
|---------|---------|
| 1.0 | Perfect agreement (identical rankings) |
| 0.85+ | Strong agreement ← **Target in this design** |
| 0.5 | Moderate agreement |
| 0.0 | No correlation |
| -1.0 | Perfect inverse (opposite rankings) |

The **p-value** indicates statistical significance (p < 0.05 = statistically significant).

**In the design context:**

From [design.md](design.md), the quality requirement is **ρ ≥ 0.85**. You'd compute this quarterly on a 10K-item validation set:

```python
# Validation set: 10,000 items evaluated by LLM + human consensus
llm_scores_validation = [...]  # 10,000 items
gold_scores_validation = [...]  # human consensus (ground truth)

rho, p_value = spearmanr(llm_scores_validation, gold_scores_validation)

if rho < 0.85:
    alert(f"Quality degraded: ρ = {rho:.3f}. Investigating LLM drift...")
    # Take action: retrain Platt scaler, adjust prompts, etc.
```

This is tracked in the [solution.md](solution.md) quarterly recalibration loop (Section 6.2).

*Pointer:* [solution.md](solution.md), Section 6.1 "Quality Metrics" and Section 6.2 "Quarterly Recalibration"
