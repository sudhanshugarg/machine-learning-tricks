# FAQ: Evaluation System with Human + LLM Evaluators

## Question Log

| Date | Category | Question | Status |
|------|----------|----------|--------|
| 2026-07-14 | LLM Calibration | Why calibrate confidence (not score)? How do you group calibration data? | [ANSWERED] |
| 2026-07-14 | Evaluation & Metrics | How do you compute Spearman's correlation coefficient? | [ANSWERED] |
| 2026-07-14 | Evaluation & Metrics | Why is there a 6 in the Spearman formula? | [ANSWERED] |
| 2026-07-14 | Routing Strategy | How does the initial confidence get computed? When does LLM run? | [ANSWERED] |
| 2026-07-14 | Score Aggregation | If LLM gave 0.5 confidence and human gave 0.9, what's the final confidence? | [ANSWERED] |

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

#### Q: How does the initial confidence get computed? When does the LLM judge run? `[ANSWERED]`

**A:**

**Items arrive with NO confidence.** The LLM judge runs immediately on every item, and its output confidence is what drives the routing decision.

**The actual flow:**

```
Item intake (essay, code, etc.)
    ↓
LLM Judge runs immediately [ALWAYS]
  Input: submission + rubric
  Output:
    {
      "score": 4,
      "confidence": 0.95,
      "reasoning": "..."
    }
    ↓
Router uses LLM confidence to decide
  ├─ conf ≥ 0.90 → LLM-only (return score immediately)
  ├─ conf 0.70-0.90 → Send to human pool (high-value uncertain cases)
  └─ conf < 0.70 → Send to human + run ensemble (risky cases)
    ↓
[If human also evaluates]
    Aggregate LLM + human scores
    ↓
Final score + audit trail
```

**Example timeline:**

```
T=0ms:    Item 1 arrives
T=0-200ms: LLM judge evaluates (3 models in parallel)
T=200ms:   LLM outputs: score=4, confidence=0.95
T=201ms:   Router decision: confidence 0.95 > 0.90 → LLM-only
T=202ms:   Return final score to client (latency: 202ms)

Cost: $0.005
Humans: not involved
```

vs.

```
T=0ms:    Item 2 arrives
T=0-200ms: LLM judge evaluates
T=200ms:   LLM outputs: score=3, confidence=0.68
T=201ms:   Router decision: confidence 0.68 in [0.70, 0.90) → send to human
T=202ms:   Queue to human pool (FIFO by tier)
T=5-30min: Human evaluates; outputs: score=3
T=5-30min: Aggregate: LLM (3, conf 0.68) + Human (3, conf 0.95) → Final: 3
T=5-30min: Return final score to client (latency: 5-30 min)

Cost: $0.005 (LLM) + $1.00 (human) = $1.005
Humans: involved
```

**Where does the LLM's confidence number come from?**

The LLM outputs it directly via **structured decoding**:

```python
system_prompt = """
You are an expert evaluator. Grade submissions on a 1-5 scale.
Output JSON with score, confidence (0.0-1.0), and reasoning.
Confidence: how sure are you about this score?
"""

user_prompt = """
Submission: [essay text]
Rubric: [detailed rubric]

Output JSON:
{
  "score": <int 1-5>,
  "confidence": <float 0.0-1.0>,
  "reasoning": <string>
}
"""

# LLM response
{
  "score": 4,
  "confidence": 0.95,
  "reasoning": "Clear thesis with strong evidence..."
}
```

The LLM generates a confidence value (0.0-1.0) as part of its output. This is the model's own estimate of certainty based on its training.

**But is this confidence accurate?**

**No.** That's the whole point of Platt scaling (from the earlier FAQ Q&A):

1. Raw LLM confidence (0.95) is often miscalibrated
2. Platt scaling maps it to true accuracy (e.g., 0.95 → 0.92)
3. Routing uses the calibrated confidence, not the raw confidence

**Key insight:** The LLM runs on ALL items. Its 200ms latency is unavoidable. But this fast 200ms evaluation lets us route intelligently: 95% of easy items go LLM-only (instant), 5% of hard items go to humans (for quality).

*Pointer:* [solution.md](solution.md), Section 1 "Router & Triage" and Section 2.2 "Structured Output"

### Human Pool Management

*(Answers go here)*

### Score Aggregation & Consensus

#### Q: If LLM gave 0.5 confidence and human gave 0.9, what's the final confidence sent to the user? `[ANSWERED]`

**A:**

**Short answer:** With the current solution, it would be **min(0.5, 0.9) = 0.5** (conservative approach).

But first, clarify: **what does "human confidence" mean?** There are a few options:

---

## Option 1: Human Confidence from Rater Accuracy (Current Solution)

Each human rater has a historical accuracy profile:

```python
class RaterProfile:
    rater_id: str
    tier: str
    accuracy: float  # P(rater's score == gold standard)
    # e.g., tier-1 expert = 0.95 accuracy
```

So "human confidence" = the rater's historical accuracy, not something they output per item.

**Example:**
```
Item scored by tier-1 expert (accuracy = 0.95)
LLM gave score=3, confidence=0.5
Human gave score=3, confidence=0.95 (their tier accuracy)

Aggregation:
  llm_weight = 0.5
  human_weight = 0.95
  
  # Scores agree (both 3)
  final_score = weighted_avg(3, 3) = 3
  final_confidence = min(0.5, 0.95) = 0.5
```

**Why use min()?** The idea is: "our final answer is only as confident as our least confident signal." If the LLM is uncertain (0.5), we can't be more confident than that, even if the human agrees.

---

## Option 2: Human Outputs Confidence (Better)

You could ask humans to output confidence like LLM does:

```python
human_output = {
    "score": 3,
    "confidence": 0.9,  # human's own uncertainty estimate
    "reasoning": "..."
}
```

Then aggregate the same way:
```
final_confidence = min(llm_confidence, human_confidence)
                 = min(0.5, 0.9) = 0.5
```

**Pros:** More granular (humans rate their own certainty per item)
**Cons:** Humans often output overconfident or underconfident numbers (just like LLMs!)

---

## Option 3: Inter-Rater Agreement (Better for Consensus)

If you send the item to **multiple humans** (e.g., 2-3 raters):

```python
def consensus_score(rater_scores: List[int]) -> Tuple[int, float]:
    """Majority vote with confidence based on agreement."""
    mode = statistics.mode(rater_scores)
    agreement_ratio = rater_scores.count(mode) / len(rater_scores)
    confidence = agreement_ratio  # high agreement → high confidence
    return mode, confidence

# Example:
rater_scores = [3, 3, 2]  # 2 raters agree on 3, 1 disagrees
final_score = 3
human_confidence = 2/3 ≈ 0.67  # agreement ratio
```

Then aggregate with LLM:
```
final_confidence = min(0.5, 0.67) = 0.5
```

---

## The Problem with min()

Using `min()` is conservative but feels wrong in some cases:

```
Case 1: LLM and human agree
  LLM: score=3, confidence=0.5
  Human: score=3, confidence=0.95
  Final: score=3, confidence=0.5 ← feels wrong! They agree, so we should be confident

Case 2: LLM and human disagree
  LLM: score=3, confidence=0.95
  Human: score=2, confidence=0.90
  Final: score=2.5 (weighted avg), confidence=0.90 ← should be lower due to disagreement
```

---

## Better Aggregation Strategy

**The solution code actually has this logic:**

```python
# If they strongly disagree, flag and lower confidence
if abs(llm_score - human_avg) > 1.5:
    return int(human_avg), 0.6, "disagreement_flagged"

# If they agree, use weighted average
final = (llm_score * llm_weight + human_avg * human_weight) / total_weight
confidence = min(llm_weight, human_weight)
```

**Better approach:** Consider agreement as a signal:

```python
# If they agree → boost confidence
# If they disagree → reduce confidence
agreement_bonus = 0.0 if abs(llm_score - human_avg) > 0.5 else 0.1

final_confidence = min(llm_weight, human_weight) + agreement_bonus
final_confidence = min(final_confidence, 0.99)  # cap at 0.99
```

**Example with better logic:**
```
LLM: score=3, confidence=0.5
Human: score=3, confidence=0.95

They agree (diff = 0)
final_score = 3
final_confidence = min(0.5, 0.95) + 0.1 = 0.6 ← boosted!
```

---

## Summary

| Approach | Final Confidence | Pro | Con |
|----------|------------------|-----|-----|
| min(LLM, human) | 0.5 | Conservative | Doesn't reward agreement |
| max(LLM, human) | 0.9 | Optimistic | Too trusting of either signal |
| weighted_avg(LLM, human) | 0.72 | Balanced | Ignores disagreement |
| min() + agreement_bonus | 0.6 | Smart | Complex logic |

*Pointer:* [solution.md](solution.md), Section 4.1 "Aggregation Rules" — shows the actual code used

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

#### Q: Why is there a 6 in the Spearman formula? `[ANSWERED]`

**A:**

The formula is:
```
ρ = 1 - (6 × Σ(d²)) / (n(n²-1))
```

The **6 comes from algebra**. Here's why:

**Spearman's ρ is Pearson's correlation applied to ranks.** 

Pearson's general formula is:
```
r = Σ((x - mean_x)(y - mean_y)) / sqrt(Σ(x - mean_x)² × Σ(y - mean_y)²)
```

But when you apply this to **ranks** (which are always 1, 2, 3, ..., n), something special happens:

1. **The mean of ranks is always (n+1)/2**
   - For n=5: mean = 3
   - For n=100: mean = 50.5

2. **The sum of squared deviations from mean is always n(n²-1)/12**
   - For n=5: Σ(rank - 3)² = 5(25-1)/12 = 10

3. **When you substitute these into Pearson's formula and simplify the algebra, the 6 falls out naturally.**

**Where does 6 come from mathematically?**

Through algebraic manipulation:
```
Pearson on ranks:
r = [numerator] / [n(n²-1)/12]

Rearranging:
ρ = 1 - (6 × Σ(d²)) / (n(n²-1))
```

The 6 = 2 × 3 emerges from the specific form of this simplification. It's not arbitrary—it's what the math produces when you specialize Pearson's formula to ranks.

**Concrete example:**

```
n = 5 items
Σ(d²) = 2 (rank differences squared)

ρ = 1 - (6 × 2) / (5 × (25 - 1))
  = 1 - 12 / 120
  = 1 - 0.10
  = 0.90

(If 6 weren't there, you'd get the wrong answer!)
```

**Why use this formula?**

It's a **shortcut formula** specific to ranks. Instead of computing full Pearson correlation (means, standard deviations, etc.), you can:
1. Compute rank differences (d)
2. Square and sum them
3. Plug into one formula

**Much faster than Pearson from scratch.**

**When does this formula NOT work?**

Only when **there are tied rankings**. Example:
- Two items both score 4 → they'd both get rank 2.5 (average of 2 and 3)
- The formula above assumes no ties

In that case, you need to use **Pearson correlation on the ranks** directly. 

That's why `scipy.spearmanr()` is smart—it detects ties and auto-adjusts:
```python
from scipy.stats import spearmanr

llm_scores = [5, 4, 4, 3, 1]    # two 4's (tied)
human_scores = [5, 3, 4, 2, 1]

rho, p_value = spearmanr(llm_scores, human_scores)
# scipy handles the tie automatically
```
