# Solution: Understanding Conditional Distributions and Decision Boundaries

## Initial Question: How Would You Model x and y?

**Strong Answer:**

I would frame this as a **classification problem using conditional probability**:

1. **Estimate the class-conditional distributions:** Learn $P(x|y=0)$ and $P(x|y=1)$ from the data
   - These tell us: "Given we're in class 0, what does $x$ typically look like?"
   - Similarly for class 1

2. **Use Bayes' rule to get the posterior:**
   $$P(y=1|x) = \frac{P(x|y=1)P(y=1)}{P(x|y=1)P(y=1) + P(x|y=0)P(y=0)}$$

   - This gives us the probability that $x$ belongs to class 1 after observing it
   - $P(y=0|x) = 1 - P(y=1|x)$

3. **Make predictions via thresholding:**
   $$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \geq \tau \\ 0 & \text{otherwise} \end{cases}$$
   where $\tau$ is a decision threshold (default $\tau=0.5$)

**Why this framing matters:**
- It's theoretically grounded in probability
- It separates the problem of learning from the problem of decision-making
- The threshold $\tau$ is a knob we can turn based on business constraints
- It naturally leads to discussing Bayes error and the overlap

---

## Question 1: Are the Classes Linearly Separable?

**Answer: No, not perfectly. Here's why:**

### Definition of Linear Separability
In 1D, linear separability means finding a threshold $t$ such that:
- All (or nearly all) points from class 0 satisfy $x < t$
- All (or nearly all) points from class 1 satisfy $x \geq t$

### Why Overlap Prevents Separation
When the distributions overlap, there exist points where:
- A point from class 0 has a value in the overlap region (e.g., $x = 50$)
- A point from class 1 also has a value in that same region (e.g., $x = 50$)
- No single threshold can correctly classify both

### Visualizing the Problem
```
Class 0: X X X   X X X X X
              ▲
              │ Overlap region
              ▼
Class 1:       X X X X X X X

         Threshold here misclassifies some Class 1 samples
         Threshold there misclassifies some Class 0 samples
```

### Key Insight
- **Perfect linear separability** only exists if the distributions don't overlap at all
- **In practice:** We accept some misclassification and try to **minimize it** by placing the threshold optimally
- Overlap doesn't make classification impossible—it just makes it impossible to be 100% correct

---

## Question 2: Is a Single Threshold Reasonable?

**Answer: Yes, a single threshold is very reasonable. Here's why:**

### Why Thresholding Works
1. **Simple and interpretable:** Easy to explain to stakeholders
2. **Computationally efficient:** $O(1)$ prediction time
3. **Matches the data-generating process:** If $P(y=1|x)$ is monotonic (smooth increasing/decreasing), a threshold is optimal

### Optimal Threshold Placement
The **Bayes-optimal threshold** is where the two posterior probabilities are equal:

$$\tau^* = \text{argmax}_\tau \quad P(\text{correct}) = P(y=1|x=\tau) P(y=1|x=\tau) + P(y=0|x=\tau)P(y=0|x=\tau)$$

**Simplified:** At the intersection point where $P(x|y=0)P(y=0) = P(x|y=1)P(y=1)$, the expected loss is minimized.

### Example
```
If at x=50:
  P(y=1|x=50) = 0.6  (60% sure it's class 1)
  P(y=0|x=50) = 0.4  (40% sure it's class 0)

Predict class 1 (go with the majority probability)
```

### When a Single Threshold Fails
A single threshold may be inadequate if:
- The relationship is non-monotonic (rare in 1D)
- You have multimodal class distributions (multiple peaks)
- Multiple features interact in complex ways (solution: use multiple features)

---

## Question 3: Bayes Error and Unavoidable Misclassification

**Answer: This is the deepest insight. Here's the full explanation:**

### What is Bayes Error?

**Definition:** The minimum classification error achievable by *any* classifier, given the data distribution.

It's a **fundamental limit** set by nature, not by your choice of algorithm or model capacity.

### Mathematical Definition

$$\text{Bayes Error} = 1 - \int_{-\infty}^{\infty} \max(P(x|y=0)P(y=0), P(x|y=1)P(y=1)) \, dx$$

Or equivalently:

$$\text{Bayes Error} = \int_{\text{Overlap Region}} \min(P(x|y=0)P(y=0), P(x|y=1)P(y=1)) \, dx$$

### Intuition

Even with a **perfect classifier** that knows the true distributions, points in the overlap region are inherently ambiguous:
- A point at $x=50$ might belong to either class
- The true probability is $P(y=1|x=50) = 0.6$
- Even with perfect knowledge, we can't do better than being wrong 40% of the time on that point

### Visualizing Bayes Error
```
     P(x|y=0)       P(x|y=1)
        ▲               ▲
        │              ╱╲
        │   ╱╲         ╱  ╲
        │  ╱  ╲       ╱    ╲
        │ ╱    ╲╲────╱╱    ╲
        │╱     ╱ ╲  ╱ ╲     ╲
        └──────────┼─────────────► x
                   │
         Optimal threshold
            ↓
        Shaded area = Bayes Error
        (Misclassified points even with optimal decision)
```

### Key Takeaways
1. **More overlap = Higher Bayes error**
   - Minimal overlap → Bayes error near 0%
   - Significant overlap → Bayes error can be 10-40%+

2. **No algorithm can beat Bayes error** (in expectation on infinite data)
   - More complex models (neural nets, SVMs) can't improve beyond this
   - Better features can reduce overlap → reduce Bayes error

3. **Bayes error is not a failure—it's reality**
   - It tells you the fundamental difficulty of the problem
   - Use it to set expectations with stakeholders: "Even perfect models will err this much"

### Practical Example
Imagine diagnosing a disease where symptoms overlap significantly:
- Healthy people: Some have symptom $x=5$
- Sick people: Some also have symptom $x=5$
- **You cannot diagnose perfectly** — some overlap is inevitable
- Bayes error quantifies this uncertainty

---

## Question 4: Threshold Selection in Practice

**Answer: The optimal threshold depends on your business constraints. Here are the strategies:**

### Strategy 1: Bayes-Optimal Threshold (Equal Costs)

**Use when:** False positives and false negatives are equally bad.

**Approach:**
$$\tau^* = \text{point where } P(x|y=0)P(y=0) = P(x|y=1)P(y=1)$$

Or equivalently:
$$\tau^* = \text{point where } P(y=1|x) = 0.5$$

**Why:** This minimizes total misclassification rate.

### Strategy 2: Cost-Weighted Threshold (Asymmetric Costs)

**Use when:** Costs of errors differ significantly.

Let:
- $c_{FP}$ = Cost of a false positive
- $c_{FN}$ = Cost of a false negative

The optimal threshold shifts to:
$$\tau^* = \text{point where } \frac{P(y=1|x)}{P(y=0|x)} = \frac{c_{FP}}{c_{FN}}$$

**Intuition:** If false negatives are 10× more costly than false positives, we lower the threshold to catch more of class 1 (at the cost of more false alarms).

### Strategy 3: Business-Driven Threshold

**Use when:** You have hard constraints.

**Example 1: Loan Approval**
- Goal: Approve as many good borrowers as possible
- Constraint: Default rate must be < 5%
- Solution: Find the threshold that keeps default rate at exactly 5%

**Example 2: Disease Screening**
- Goal: Catch all cases
- Constraint: We can only follow up on 20% of patients
- Solution: Set threshold so top 20% by risk score get screening

### Strategy 4: Cross-Validation on Validation Set

**Use when:** You don't know the true cost structure in advance.

1. Train your model on training data
2. For each possible threshold $\tau$ on validation data, compute:
   - Precision, Recall, F1, Accuracy
   - Cost = $c_{FP} \cdot FP + c_{FN} \cdot FN$
3. Pick $\tau$ that optimizes your chosen metric
4. Evaluate on held-out test set

### Visualization: ROC Curve

The **Receiver Operating Characteristic (ROC) curve** shows all possible operating points:

```
TPR (Recall)
  ▲
 1│                      ╱╱╱
  │                    ╱╱
  │                  ╱╱
  │                ╱╱
  │              ╱╱
  │   ╱╱╱╱╱╱╱╱╱╱
  │╱╱╱
  └─────────────────► FPR
  0               1
```

- Each point on the curve = a different threshold
- You choose the threshold (point) based on your cost structure
- ROC-AUC = Summary metric (1.0 = perfect, 0.5 = random)

---

## Question 5: Metrics and Cost Sensitivity

**Answer: Choose metrics that align with your business problem.**

### Metric Selection Guide

#### Scenario 1: Equal Cost / Balanced Problem
**When:** Both error types equally bad, classes roughly balanced

**Best metrics:**
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Balanced Accuracy:** $\frac{1}{2}(\text{TPR} + \text{TNR})$ — better if classes imbalanced
- **F1 Score:** Harmonic mean of precision and recall

**Example:** General-purpose classification, no clear business consequence

---

#### Scenario 2: False Positives are Costly
**When:** False alarms are expensive or harm users

- **Email spam filtering:** Rejecting legitimate email is worse than letting spam through
- **Medical screening:** False alarms cause unnecessary stress/procedures
- **Fraud detection (initial stage):** Manual review is expensive

**Best metrics:**
- **Precision:** $\frac{TP}{TP + FP}$ — "Of what we predicted as positive, how many are actually positive?"
- **Positive Predictive Value (PPV):** Same as precision
- **Specificity:** $\frac{TN}{TN + FP}$ — "Of actual negatives, how many did we correctly reject?"

**Action:**
- Raise the threshold $\tau$ (be more conservative about predicting class 1)
- Trade-off: Miss some true positives to avoid false alarms

**Example:**
```
Default threshold:    Predict 1 if P(y=1|x) ≥ 0.5
Cost-adjusted:       Predict 1 if P(y=1|x) ≥ 0.8
Result:              Higher precision, lower recall
```

---

#### Scenario 3: False Negatives are Costly
**When:** Missing a case is worse than false alarms

- **Disease detection:** Missing a diagnosis can be fatal
- **Security threat detection:** Missing a breach is catastrophic
- **Fraud detection (final stage):** Missing fraud is expensive

**Best metrics:**
- **Recall (Sensitivity, True Positive Rate):** $\frac{TP}{TP + FN}$ — "Of actual positives, how many did we catch?"
- **Negative Predictive Value (NPV):** $\frac{TN}{TN + FN}$ — "If we predict negative, how confident are we?"

**Action:**
- Lower the threshold $\tau$ (be more aggressive about predicting class 1)
- Trade-off: More false alarms to avoid missing true cases

**Example:**
```
Default threshold:    Predict 1 if P(y=1|x) ≥ 0.5
Cost-adjusted:       Predict 1 if P(y=1|x) ≥ 0.3
Result:              Higher recall, lower precision
```

---

### Scenario 4: Arbitrary Cost Structure

**When:** You have exact business costs for each error type.

**Approach:** Compute the expected cost for each threshold and pick the best:

$$\text{Cost}(\tau) = c_{FP} \cdot FP(\tau) + c_{FN} \cdot FN(\tau)$$

Example: Loan default prediction
- **Cost of FP:** $5,000 (lost opportunity from rejecting good borrower)
- **Cost of FN:** $50,000 (default on $250k loan, recover 80%)

Find $\tau$ that minimizes total cost.

---

### Summary: Metric Cheat Sheet

| Problem Type | Metric to Optimize | Action |
|---|---|---|
| Balanced classes, equal cost | Accuracy, F1 | Use default threshold (0.5) |
| FP costly (precision-critical) | Precision | Raise threshold |
| FN costly (recall-critical) | Recall | Lower threshold |
| Imbalanced but equal cost | Balanced Accuracy, F1-Macro | Use balanced threshold |
| Custom costs | Cost-weighted loss | Compute optimal threshold from cost matrix |

---

## Putting It All Together: A Complete Example

### Scenario: Email Spam Detection

**Given:**
- Feature $x$ = Some measure of "spam-likeness" (0-1 score)
- Class 0 = Legitimate email, Class 1 = Spam
- Data shows 5% spam, 95% legitimate
- False positive (marking legitimate as spam) is worse than false negative (missing spam)

### Your Answer Flow:

1. **Initial modeling:**
   - Estimate $P(x|y=0)$ and $P(x|y=1)$ from data
   - Compute $P(y=1|x)$ using Bayes' rule

2. **Discuss separability:**
   - "Spam and legitimate emails have overlapping features, so we can't separate perfectly"
   - "But we don't need perfect separation—we just need a good decision boundary"

3. **Discuss Bayes error:**
   - "The overlap means some emails are inherently ambiguous"
   - "Even a perfect classifier will misclassify some"
   - "This sets a lower bound on our error rate"

4. **Choose threshold:**
   - "Since false positives are costly (users will be angry if legitimate email goes to spam), I'd raise the threshold"
   - "Instead of predicting spam at $P(y=1|x) \geq 0.5$, I'd use $P(y=1|x) \geq 0.7$"
   - "This reduces false positives at the cost of missing some spam"

5. **Choose metrics:**
   - "I'd optimize for **precision** (avoid false positives)"
   - "I'd also track recall to ensure we're not missing too much spam"
   - "The product team would decide if 70% precision / 40% recall is acceptable"

---

## Why This Answer Impresses Interviewers

✓ Shows probabilistic thinking (not just memorizing algorithms)
✓ Understands fundamental limits (Bayes error)
✓ Connects theory to practice (threshold selection)
✓ Acknowledges trade-offs (false positive vs. false negative)
✓ Asks clarifying questions ("What are the costs?")
✓ Avoids over-engineering (simple threshold, not complex model)

