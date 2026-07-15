# FAQ: Evaluation System with Human + LLM Evaluators

## Quick Navigation

**By Category:**
- [LLM Calibration](#llm-calibration)
- [Routing Strategy](#routing-strategy)
- [Score Aggregation & Consensus](#score-aggregation--consensus)
- [Evaluation & Metrics](#evaluation--metrics)

**All Questions:**
1. [Why calibrate confidence instead of the score itself?](#q-why-calibrate-confidence-instead-of-the-score-itself-how-do-you-actually-collect-calibration-data-answered)
2. [How does the initial confidence get computed?](#q-how-does-the-initial-confidence-get-computed-when-does-the-llm-judge-run-answered)
3. [If LLM gave 0.5 confidence and human gave 0.9, what's the final confidence?](#q-if-llm-gave-05-confidence-and-human-gave-09-whats-the-final-confidence-sent-to-the-user-answered)
4. [How do you compute Spearman's correlation coefficient?](#q-how-do-you-compute-spearmans-correlation-coefficient-answered)
5. [Why is there a 6 in the Spearman formula?](#q-why-is-there-a-6-in-the-spearman-formula-answered)
6. [When would I use Platt scaling vs isotonic regression?](#q-when-would-i-use-platt-scaling-vs-isotonic-regression-for-calibration-answered)
7. [Can you walk me through isotonic regression with a concrete example?](#q-can-you-walk-me-through-isotonic-regression-with-a-concrete-example-answered)
8. [What are BLEU and ROUGE scores, and when would you use them instead of LLM judging?](#q-what-are-bleu-and-rouge-scores-and-when-would-you-use-them-instead-of-llm-judging-answered)

---

## Question Log

| Date | Category | Question | Status |
|------|----------|----------|--------|
| 2026-07-14 | LLM Calibration | Why calibrate confidence (not score)? How do you group calibration data? | [ANSWERED] |
| 2026-07-14 | Evaluation & Metrics | How do you compute Spearman's correlation coefficient? | [ANSWERED] |
| 2026-07-14 | Evaluation & Metrics | Why is there a 6 in the Spearman formula? | [ANSWERED] |
| 2026-07-14 | Routing Strategy | How does the initial confidence get computed? When does LLM run? | [ANSWERED] |
| 2026-07-14 | Score Aggregation | If LLM gave 0.5 confidence and human gave 0.9, what's the final confidence? | [ANSWERED] |
| 2026-07-14 | LLM Calibration | When would I use Platt scaling vs isotonic regression? | [ANSWERED] |
| 2026-07-14 | LLM Calibration | Can you walk me through isotonic regression with a concrete example? | [ANSWERED] |
| 2026-07-15 | Evaluation & Metrics | What are BLEU and ROUGE scores? When use instead of LLM judge? | [ANSWERED] |

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

#### Q: When would I use Platt scaling vs isotonic regression for calibration? `[ANSWERED]`

**A:**

Both are used to calibrate confidence → accuracy mapping, but they make different trade-offs. Here's the comparison:

---

## Platt Scaling

**What it is:** Fit a logistic function to the data.

```python
# Fit
from sklearn.linear_model import LogisticRegression

platt = LogisticRegression()
platt.fit(X=raw_confidences.reshape(-1, 1), y=is_correct)

# Predict
calibrated_accuracy = platt.predict_proba([[0.95]])[0][1]
```

**Formula:**
```
P(correct) = sigmoid(a * confidence + b)
           = 1 / (1 + exp(-(a*confidence + b)))
```

**Pros:**
- **Parametric:** Only 2 parameters (a, b) → low overfitting risk
- **Fast:** Simple matrix operations
- **Interpretable:** Clear sigmoid shape
- **Small data OK:** Works well with ~1000 calibration samples
- **Well-behaved:** Assumes smooth, monotonic relationship (usually true for confidence)

**Cons:**
- **Rigid shape:** Assumes sigmoid. If true curve is different (e.g., S-curve flipped), it won't fit well
- **Worse fit:** May underfit if true relationship is complex
- **Less flexible:** Can't learn bumpy/non-standard curves

**Good for:**
- Small to medium calibration sets (< 10K samples)
- Well-behaved confidence (monotonic, no weird jumps)
- Production systems where interpretability matters
- Fast iteration and low compute

---

## Isotonic Regression

**What it is:** Fit a non-parametric monotonic curve to the data.

```python
from sklearn.isotonic import IsotonicRegression

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(X=raw_confidences, y=is_correct)

calibrated_accuracy = iso.predict([0.95])
```

**How it works:**
- Finds a monotonically increasing function that best fits the data
- No shape assumption (not sigmoid, not linear—whatever the data shows)
- Piecewise-linear (connects calibration points with straight lines)

**Pros:**
- **Non-parametric:** Fits any monotonic curve (no shape assumption)
- **Flexible:** Can learn complex relationships (S-curves, flat regions, steep jumps)
- **Better fit:** Lower calibration error when relationship is non-sigmoid
- **Large data OK:** Works well with 10K+ calibration samples

**Cons:**
- **More parameters:** Requires more calibration data to avoid overfitting
- **Slower:** More complex computation
- **Less interpretable:** Can produce odd-looking curves
- **Overfitting risk:** With small data, can fit noise instead of signal
- **Edge case handling:** Needs care for out-of-range predictions

**Good for:**
- Large calibration sets (10K+ samples)
- Unknown/complex relationship between confidence and accuracy
- When you have time to experiment with hyperparameters
- Research/offline settings where compute isn't constrained

---

## Visual Comparison

```
Calibration data: (confidence, accuracy) pairs

Data points:
0.95 → 0.92
0.85 → 0.80
0.75 → 0.75
0.60 → 0.60
0.50 → 0.55


Platt Scaling:          Isotonic Regression:
(smooth sigmoid curve)  (piecewise linear curve)

  1.0 ┌─────────         1.0 ┌─────────
      │    ╱╱╱              │╱╱
  0.9 │   ╱                │╱
      │  ╱                │
  0.8 │ ╱                 │•──
      │╱                  │   •──
  0.7 ├────────           │       •──
      │•                  │           •─
  0.6 │ •                 │             •
      │  •                │
  0.5 │   •               │
      │    •              │
  0.4 └────────           └─────────
      0.5  0.75  1.0     0.5  0.75  1.0
```

Platt is smooth; isotonic follows data more closely.

---

## Decision Matrix

| Factor | Platt | Isotonic |
|--------|-------|----------|
| **Calibration data size** | < 10K ✓ | > 10K ✓ |
| **Compute cost** | Fast ✓ | Slower |
| **Interpretability** | Clear ✓ | Opaque |
| **Overfitting risk** | Low ✓ | High (needs tuning) |
| **Fit quality** | Good | Excellent ✓ |
| **Well-behaved confidence** | ✓✓ | ✓ |
| **Unknown/complex relationship** | ✓ | ✓✓ |
| **Production ready** | ✓✓ | ✓ |

---

## Practical Recommendation for This Design

**Start with Platt scaling:**
- 10K-item calibration set is small-to-medium
- LLM confidence is usually well-behaved (monotonic)
- Production system needs speed and interpretability
- 2-parameter model = easy to monitor, retrain monthly

**Switch to isotonic if:**
- You notice Platt's calibration error is > 10%
- Your data analysis reveals non-sigmoid relationship (e.g., bimodal)
- You have 50K+ calibration items and want lower error
- You can afford compute + complexity overhead

---

## Code Example: Comparing Both

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# Calibration set: (confidence, is_correct) pairs
X_cal = np.array([0.95, 0.85, 0.75, 0.60, 0.50, ...])  # 10K items
y_cal = np.array([0.92, 0.80, 0.75, 0.60, 0.55, ...])  # actual accuracy

# Platt Scaling
platt = LogisticRegression()
platt.fit(X_cal.reshape(-1, 1), y_cal)
platt_pred = platt.predict_proba(X_cal.reshape(-1, 1))[:, 1]
platt_error = brier_score_loss(y_cal, platt_pred)

# Isotonic Regression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(X_cal, y_cal)
iso_pred = iso.predict(X_cal)
iso_error = brier_score_loss(y_cal, iso_pred)

print(f"Platt error: {platt_error:.4f}")
print(f"Isotonic error: {iso_error:.4f}")

# Choose the better one for production
if iso_error < platt_error - 0.01:  # isotonic better by >1%
    calibrator = iso
else:
    calibrator = platt  # stick with Platt (simpler)
```

---

## Summary

- **Platt**: Simple, fast, robust for small data, assumes sigmoid shape
- **Isotonic**: Flexible, better fit for large data, no shape assumption, more complex

For this evaluation system: **Start with Platt, monitor quarterly, switch to isotonic if calibration error degrades.**

*Pointer:* [solution.md](solution.md), Section 2.3 "Confidence Calibration via Platt Scaling"

#### Q: What are BLEU and ROUGE scores, and when would you use them instead of LLM judging? `[ANSWERED]`

**A:** BLEU and ROUGE are **reference-based automatic evaluation metrics** — they compare a generated text (e.g., LLM output) against a reference text (e.g., human gold standard) using n-gram overlap. They're fast, deterministic, and cost-free, but limited to tasks where a single "correct answer" exists.

---

## BLEU Score (Bilingual Evaluation Understudy)

**What it is:** Measures n-gram precision — how many n-grams in the generated text appear in the reference text.

**Formula:**
```
BLEU = BP × exp(Σ w_n log(p_n))

where:
  BP = brevity penalty (penalizes short outputs)
  p_n = precision of n-grams of size n
  w_n = weight for n-gram size (typically 0.25 for each of 1-4)
```

**Example:**

```
Reference (gold standard):
  "The quick brown fox jumps over the lazy dog"

Generated (LLM output):
  "The quick brown fox jumps over a lazy dog"

N-gram overlap:
  Unigrams (1-word):   "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"
                       → 8/9 = 89% match
  Bigrams (2-word):    "The quick", "brown fox", "fox jumps", "over a" (WRONG), "a lazy", "lazy dog"
                       → 5/8 = 63% match (missing "over the")
  Trigrams (3-word):   "The quick brown", "quick brown fox", "brown fox jumps", ...
                       → similar degradation
  4-grams:             4-word phrases, even lower overlap

BLEU-4 = BP × exp(0.25 × (log(0.89) + log(0.63) + ...) + ...)
       ≈ 0.72 (out of 1.0)
```

**Pros:**
- ✅ Fast (milliseconds per comparison)
- ✅ Cost-free (no API calls)
- ✅ Deterministic (same input → same score)
- ✅ Works for machine translation, summarization, question answering
- ✅ Correlates reasonably with human judgment on these tasks

**Cons:**
- ❌ Penalizes paraphrases: "fast canine" vs. "quick fox" = 0% match despite being synonymous
- ❌ Doesn't capture semantic meaning
- ❌ Requires reference text (doesn't work for open-ended generation)
- ❌ Unreliable on short outputs (brevity penalty is harsh)
- ❌ No credit for partial correctness

**When to use BLEU:**
- **Machine translation**: "Translate Spanish → English" (one correct translation structure)
- **Summarization**: "Generate 3-sentence summary of article" (similar condensing patterns)
- **Question answering**: "Answer: What is capital of France?" (expected answer: "Paris")
- **Data-to-text**: "Generate sentence from table row" (structured inputs, templated outputs)

**Code example:**
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = "The quick brown fox jumps over the lazy dog".split()
candidate = "The quick brown fox jumps over a lazy dog".split()

# BLEU-4 (weights for 1, 2, 3, 4-grams)
weights = (0.25, 0.25, 0.25, 0.25)
smoothing = SmoothingFunction().method1

score = sentence_bleu(
    [reference],
    candidate,
    weights=weights,
    smoothing_function=smoothing
)
print(f"BLEU-4: {score:.3f}")  # → 0.714
```

---

## ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

**What it is:** Measures n-gram **recall** — what fraction of the reference's n-grams appear in the generated text. (BLEU uses precision; ROUGE uses recall.)

**Three main variants:**

### ROUGE-N: N-gram Overlap

```
ROUGE-N = (# of n-grams in reference that appear in generated) / (total n-grams in reference)
```

**Example:**

```
Reference:
  "The quick brown fox jumps over the lazy dog"
  Unigrams: {The, quick, brown, fox, jumps, over, the, lazy, dog}  [9 total]

Generated:
  "The quick brown fox jumps over a lazy dog"
  Unigrams: {The, quick, brown, fox, jumps, over, a, lazy, dog}

Matching unigrams: {The, quick, brown, fox, jumps, over, lazy, dog}  [8 match]

ROUGE-1 = 8 / 9 = 0.889
```

### ROUGE-L: Longest Common Subsequence

Measures longest substring that appears in both texts (allows gaps, doesn't require contiguity).

```
ROUGE-L = (longest common subsequence length) / (reference length)

Example:
Reference: "The quick brown fox"      [4 words]
Generated: "The brown quick fox"      [4 words]

Longest common subsequence: "The", "brown", "fox"  [3 words, different order]

ROUGE-L = 3 / 4 = 0.75
```

This is better at capturing reorderings/paraphrases than ROUGE-N.

### ROUGE-S: Skip-Bigram (Unigram Pairs)

Bigrams where words don't have to be adjacent.

```
Reference:  "The quick brown fox"
Bigrams: (The, quick), (The, brown), (The, fox), (quick, brown), (quick, fox), (brown, fox)

Generated: "The brown quick fox"
Bigrams: (The, brown), (The, quick), (The, fox), (brown, quick), (brown, fox), (quick, fox)

ROUGE-S matches all 6 bigrams → perfect 1.0 score (even though word order differs)
```

**Pros:**
- ✅ Fast (milliseconds)
- ✅ Cost-free
- ✅ ROUGE-L and ROUGE-S handle paraphrasing better than BLEU
- ✅ Works for any task with reference text
- ✅ Recall-based (penalizes missing content from reference, not hallucination)

**Cons:**
- ❌ Still penalizes legitimate paraphrases (e.g., "smart canine" vs. "intelligent dog")
- ❌ Doesn't measure semantic correctness
- ❌ Requires reference text
- ❌ Can't detect factual errors (if generated text matches reference structure but has wrong facts)

**When to use ROUGE:**
- **Summarization**: (Reference summary exists; compare system output)
- **Machine translation**: (Reference translation given)
- **Paraphrase detection**: (Especially ROUGE-L and ROUGE-S)
- **Document retrieval**: (ROUGE can match documents to queries)

**Code example:**
```python
from rouge_score import rouge_scorer

reference = "The quick brown fox jumps over the lazy dog"
generated = "The quick brown fox jumps over a lazy dog"

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, generated)

print(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.3f}")  # → 0.889
print(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.3f}")  # → 0.857
```

---

## BLEU vs ROUGE vs LLM Judge

| Metric | Speed | Cost | Semantic | Paraphrase | Open-Ended | Reference Required |
|--------|-------|------|----------|-----------|------------|-------------------|
| **BLEU** | 1ms | Free | ❌ No | ❌ Poor | ❌ No | ✅ Yes |
| **ROUGE** | 1ms | Free | ❌ No | ⚠️ Medium | ❌ No | ✅ Yes |
| **LLM Judge** | 200ms | $0.01 | ✅ Yes | ✅ Good | ✅ Yes | ❌ No |

---

## When To Use Each

### Use BLEU/ROUGE:
```
Task: "Summarize this article in 3 sentences"
Why: Reference summaries exist; paraphrasing OK; fast feedback needed

Task: "Translate English → Spanish"
Why: Structure-based task; reference translation available

Task: "Extract named entities"
Why: Exact matches expected; deterministic answers
```

### Use LLM Judge:
```
Task: "Grade essay quality (1-5)"
Why: No reference answer; requires semantic understanding; requires nuance

Task: "Does this explanation correctly answer the question?"
Why: Requires reasoning; many valid phrasings; needs judgment

Task: "Rate code quality (readability, efficiency, style)"
Why: Multiple valid approaches; subjective rubric; needs understanding
```

---

## In This LLM Judge Design

BLEU/ROUGE are **not used** as the evaluation mechanism because:

1. **Essays, code, explanations have no single "correct answer"** — you can't use BLEU (requires reference)
2. **Human judgment is needed** — graders score based on quality rubric, not n-gram overlap
3. **LLM judge replaces automatic metrics** — it reasons about the quality, not just string overlap

However, BLEU/ROUGE could be **auxiliary metrics** in the confidence calibration loop:

```python
# In the LLM judge prompt:
"Score this essay 1-5 based on the rubric.
Also compute ROUGE-L vs. a reference answer (if available).
Use both your semantic judgment + ROUGE as confidence signals."

# Then calibrate LLM confidence against:
#   1. Human ground truth (primary)
#   2. ROUGE score (secondary signal for certain items)
```

This is rare in essay grading but common in summarization/QA evaluation systems.

*Pointer:* [solution.md](solution.md), Section 2.2 "LLM Judge Prompt & Structured Output"

---

#### Q: Can you walk me through isotonic regression with a concrete example? `[ANSWERED]`

**A:**

Let me show you step-by-step how isotonic regression learns to map LLM confidence → actual accuracy using 10,000 evaluation items.

---

## The Data Setup

You have:
- 10,000 items evaluated by both LLM and human
- LLM output: score (1-5) + confidence (0.0-1.0)
- Human output: score (1-5) [ground truth]

```
Item  | LLM Score | LLM Conf | Human Score | Correct?
------|-----------|----------|-------------|----------
  1   |     4     |   0.95   |      4      |   YES (1)
  2   |     3     |   0.95   |      2      |   NO  (0)
  3   |     5     |   0.87   |      5      |   YES (1)
  4   |     2     |   0.68   |      2      |   YES (1)
  5   |     3     |   0.68   |      3      |   YES (1)
  ... | ...       | ...      | ...         | ...
10000 |     4     |   0.52   |      3      |   NO  (0)
```

## Step 1: Prepare Data for Calibration

Convert to (confidence, is_correct) pairs:

```python
import numpy as np

# Load 10,000 items
llm_confidences = np.array([0.95, 0.95, 0.87, 0.68, 0.68, ..., 0.52])  # shape: (10000,)
llm_scores = np.array([4, 3, 5, 2, 3, ..., 4])
human_scores = np.array([4, 2, 5, 2, 3, ..., 3])

# Create is_correct: did LLM match human?
is_correct = (llm_scores == human_scores).astype(int)  # shape: (10000,)
# [1, 0, 1, 1, 1, ..., 0]

print(f"LLM accuracy overall: {is_correct.mean():.2%}")  # e.g., 78%
```

## Step 2: Fit Isotonic Regression

```python
from sklearn.isotonic import IsotonicRegression

# Create and fit the calibrator
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(X=llm_confidences, y=is_correct)

# What happened:
# - iso learned a monotonic function: confidence → true_accuracy
# - It found the best-fit curve through all 10,000 points
```

**What isotonic does internally:**

1. Sorts the data by confidence (X)
2. Computes the mean accuracy (y) for each confidence bin
3. Applies PAV (Pool Adjacent Violators) algorithm to smooth into monotonic curve
4. Creates piecewise-linear function connecting the bins

```
Raw means by confidence bin:
Confidence [0.90-1.00]: 920/1000 correct → 92% accuracy
Confidence [0.80-0.90): 760/1000 correct → 76% accuracy  ← violates monotonicity!
Confidence [0.70-0.80): 750/1000 correct → 75% accuracy
Confidence [0.60-0.70): 580/1000 correct → 58% accuracy

After PAV smoothing (force monotonic):
Confidence [0.90-1.00]: 92% accuracy
Confidence [0.80-0.90): 76% accuracy  → becomes 80% (averaged with neighbor)
Confidence [0.70-0.80): 75% accuracy  → becomes 77% (adjusted up)
Confidence [0.60-0.70): 58% accuracy

Result: monotonic increasing curve
```

## Step 3: Use the Fitted Calibrator

For a new item with LLM confidence 0.87:

```python
# Get calibrated accuracy
new_confidence = 0.87
calibrated_accuracy = iso.predict([0.87])  # → e.g., 0.79

# This means: "when LLM claims 0.87 confidence, it's actually correct ~79% of the time"
```

## Step 4: Visualize What It Learned

```python
import matplotlib.pyplot as plt

# Generate curve
x_range = np.linspace(0, 1, 100)
y_calibrated = iso.predict(x_range)

plt.figure(figsize=(10, 6))
plt.scatter(llm_confidences, is_correct, alpha=0.1, label='Individual items')
plt.plot(x_range, y_calibrated, 'r-', linewidth=2, label='Isotonic fit')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
plt.xlabel('LLM Confidence')
plt.ylabel('True Accuracy')
plt.legend()
plt.grid()
plt.title('Isotonic Regression: LLM Confidence → True Accuracy')
plt.show()
```

**Output might look like:**

```
True Accuracy
    1.0 ├──────────
        │        ╱╱
    0.9 │      ╱╱
        │    ╱╱
    0.8 │  ╱╱•──────  ← here: conf=0.87 → acc=0.79
        │╱╱  •
    0.7 ├────•─
        │      •
    0.6 │       •
        │        •
    0.5 │         •
        │          •
    0.4 └─────────────
        0.5    0.75   1.0
        LLM Confidence
```

## Full Example with Routing Decision

```python
from sklearn.isotonic import IsotonicRegression

# Step 1: Fit on 10K calibration items
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(llm_confidences, is_correct)

# Step 2: New item arrives with LLM confidence 0.68
new_llm_confidence = 0.68
calibrated_accuracy = iso.predict([0.68])[0]  # → 0.62

# Step 3: Make routing decision
if calibrated_accuracy >= 0.90:
    decision = "LLM-only"
    cost = 0.005
elif calibrated_accuracy >= 0.75:
    decision = "Send to human"
    cost = 1.00
else:
    decision = "Human + secondary LLM"
    cost = 1.50

print(f"Raw LLM confidence: {new_llm_confidence}")
print(f"Calibrated accuracy: {calibrated_accuracy:.2%}")
print(f"Decision: {decision}")
print(f"Est. cost: ${cost}")

# Output:
# Raw LLM confidence: 0.68
# Calibrated accuracy: 62%
# Decision: Send to human
# Est. cost: $1.00
```

## Why Isotonic Learned This Mapping

Isotonic doesn't assume any particular shape. It just looks at the data:

```
"When LLM said 0.95:  920/1000 times correct (92%)"
"When LLM said 0.85:  760/1000 times correct (76%)"
"When LLM said 0.75:  750/1000 times correct (75%)"
"When LLM said 0.65:  580/1000 times correct (58%)"
"When LLM said 0.55:  420/1000 times correct (42%)"

→ Learns: "higher confidence → higher accuracy" (monotonic)
→ Creates piecewise-linear curve through these points
```

## Isotonic vs Platt on This Data

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Fit both
iso = IsotonicRegression()
iso.fit(llm_confidences, is_correct)

platt = LogisticRegression()
platt.fit(llm_confidences.reshape(-1, 1), is_correct)

# Evaluate on same data
iso_pred = iso.predict(llm_confidences)
platt_pred = platt.predict_proba(llm_confidences.reshape(-1, 1))[:, 1]

iso_error = brier_score_loss(is_correct, iso_pred)
platt_error = brier_score_loss(is_correct, platt_pred)

print(f"Isotonic Brier loss: {iso_error:.4f}")
print(f"Platt Brier loss:    {platt_error:.4f}")

# Example output:
# Isotonic Brier loss: 0.1823  ← lower = better fit
# Platt Brier loss:    0.1896  ← Platt is less flexible
```

## Key Takeaway

Isotonic regression:
1. **Learns from data**: "What's the actual accuracy when LLM says 0.68?"
2. **No shape assumption**: Fits whatever curve the data shows (no forced sigmoid)
3. **Piecewise-linear**: Connects calibration points with straight lines
4. **Monotonic guarantee**: Ensures accuracy never decreases with confidence
5. **Better fit**: Lower calibration error on the training data

The cost: needs more samples (10K+) and can overfit if you're not careful.

*Pointer:* [solution.md](solution.md), Section 2.3 "Confidence Calibration via Platt Scaling"
