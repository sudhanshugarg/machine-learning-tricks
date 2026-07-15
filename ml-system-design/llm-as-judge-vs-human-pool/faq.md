# FAQ: Evaluation System with Human + LLM Evaluators

## Question Log

| Date | Category | Question | Status |
|------|----------|----------|--------|
| 2026-07-14 | LLM Calibration | Why calibrate confidence (not score)? How do you group calibration data? | [ANSWERED] |

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

*(Answers go here)*
