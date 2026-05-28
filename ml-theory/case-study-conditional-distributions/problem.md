# Case Study: Understanding Conditional Distributions and Decision Boundaries

## Problem Statement

You are given:
- A single feature $x$
- A binary target label $y \in \{0, 1\}$
- Historical data with observations from both classes

**Initial Question:** How would you model the relationship between $x$ and $y$?

---

## The Setup: Overlapping Class Distributions

After your initial answer, the interviewer presents a figure showing the class-conditional distributions:
- Curve for $P(x | y=0)$ (Class 0 distribution)
- Curve for $P(x | y=1)$ (Class 1 distribution)
- The two curves overlap significantly in the middle region

### Visual Intuition
```
     P(x|y=0)          P(x|y=1)
        ▲                  ▲
        │     ╱╲            │     ╱╲
        │    ╱  ╲           │    ╱  ╲
        │   ╱    ╲          │   ╱    ╲
        │  ╱      ╲╲──────╱╱  ╱      ╲
        │ ╱        ╱ ╲  ╱ ╲ ╱        ╲
        │╱________╱___╲╱___╲_________╲___► x
                    ↑
                 Overlap region
```

---

## Core Questions

### 1. Linear Separability
**Question:** Are the classes linearly separable? Why or why not?

**What to discuss:**
- In 1D, linear separability means finding a single threshold $t$ such that all (or most) of Class 0 is on one side and all (or most) of Class 1 is on the other
- The overlap region makes perfect separation impossible
- Mention the implications for decision boundaries

### 2. Single Threshold Reasonableness
**Question:** Is a simple threshold-based decision rule reasonable here?

**What to discuss:**
- A threshold at the intersection of the two curves minimizes misclassification on this data
- But reasonableness depends on business constraints and cost structure
- A threshold is still practical even with overlap (it just means some inevitable error)

### 3. Bayes Error and Unavoidable Misclassification
**Question:** What does the overlap tell us about Bayes error? Can we ever achieve zero error?

**What to discuss:**
- **Bayes error** = The minimum classification error achievable by any classifier (even with infinite data and infinite model complexity)
- In the overlap region, no classifier can be perfectly certain which class a point belongs to
- Mathematically: $P(\text{error}) = \int_{\text{overlap}} \min(P(x|y=0)P(y=0), P(x|y=1)P(y=1)) dx$
- The larger the overlap, the higher the Bayes error
- This is a fundamental limit, not a limitation of your model

### 4. Threshold Selection in Practice
**Question:** How would you choose where to place the decision threshold?

**What to discuss:**
- **Equal cost scenario:** Place threshold at the point where $P(x|y=0)P(y=0) = P(x|y=1)P(y=1)$ (Bayes optimal)
- **Unequal cost scenario:** Shift threshold to account for cost of different types of errors
- **Class imbalance:** If one class is much rarer, threshold should reflect that prior
- **Practical constraints:** Domain requirements may dictate tolerance for different error types

### 5. Metrics and Cost Sensitivity
**Question:** If false positives and false negatives have different costs, which metrics would you optimize?

**What to discuss:**
- **Symmetric costs (equal weight):** Use accuracy, balanced accuracy, or F1 score
- **False positives costly:** Optimize for precision (e.g., medical screening where false alarms cause unnecessary treatment)
- **False negatives costly:** Optimize for recall (e.g., disease detection where missing a case is worse than a false alarm)
- **Arbitrary costs:** Use cost-weighted variants like weighted F1, or tune threshold to maximize $\text{Benefit} = \text{TP} \cdot c_{TP} - \text{FP} \cdot c_{FP} - \text{FN} \cdot c_{FN}$
- **ROC-AUC and Precision-Recall curves:** Show how different thresholds trade off performance metrics

---

## What the Interviewer is Probing

1. **Probabilistic thinking:** Do you naturally think in terms of distributions and conditional probabilities?
2. **Understanding of fundamentals:** Do you know what Bayes error is and why overlap matters?
3. **Practical decision-making:** Can you connect theory (decision boundaries) to real-world choices (threshold selection)?
4. **Cost awareness:** Do you understand that optimal decisions depend on the business problem, not just the data?
5. **Communication:** Can you explain complex ideas clearly with intuition?

---

## Follow-up Branches

Depending on your answers, expect questions like:

- **"What if we had 2 features instead of 1?"** → How does dimensionality change the problem?
- **"How would regularization help here?"** → Connection to overfitting and model complexity
- **"The overlap is because the classes are inherently similar. How would you engineer better features?"** → Feature importance and domain knowledge
- **"What if the costs change mid-deployment?"** → How to make classifiers adaptable
- **"How would you estimate Bayes error on real data?"** → Cross-validation and performance bounds
- **"Can you ever be sure about a prediction?"** → Confidence intervals and calibration

---

## Expected Solution Structure

Your answer should demonstrate:
1. Clear mental model of $P(y|x)$ and how to estimate it
2. Understanding of why overlap creates unavoidable error
3. Practical knowledge of threshold selection given business constraints
4. Awareness of appropriate metrics for different cost structures
5. Ability to connect theory to implementation

