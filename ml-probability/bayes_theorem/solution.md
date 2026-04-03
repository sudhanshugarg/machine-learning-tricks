# Bayes' Theorem Solution

## Problem Setup

**Given:**
- $P(\text{Spam}) = 0.05$ (prior: 5% of emails are spam)
- $P(\text{Flagged} | \text{Spam}) = 0.95$ (likelihood: filter catches 95% of spam)
- $P(\text{Flagged} | \text{Legitimate}) = 0.02$ (false positive rate: 2% of legitimate flagged)

**Find:**
- $P(\text{Spam} | \text{Flagged}) = ?$ (posterior: probability email is spam given it's flagged)

## Solution Using Bayes' Theorem

**Bayes' Theorem:**
$$P(\text{Spam} | \text{Flagged}) = \frac{P(\text{Flagged} | \text{Spam}) \cdot P(\text{Spam})}{P(\text{Flagged})}$$

**Step 1: Calculate the marginal likelihood $P(\text{Flagged})$**

Using the law of total probability:
$$P(\text{Flagged}) = P(\text{Flagged} | \text{Spam}) \cdot P(\text{Spam}) + P(\text{Flagged} | \text{Legitimate}) \cdot P(\text{Legitimate})$$

$$P(\text{Flagged}) = 0.95 \times 0.05 + 0.02 \times 0.95$$
$$P(\text{Flagged}) = 0.0475 + 0.019 = 0.0665$$

**Step 2: Apply Bayes' theorem**
$$P(\text{Spam} | \text{Flagged}) = \frac{0.95 \times 0.05}{0.0665} = \frac{0.0475}{0.0665} = 0.714$$

## Result

**The probability that a flagged email is actually spam is ~71.4%**

## Intuitive Explanation

This result is often counterintuitive. Despite the filter being 95% accurate, when it flags an email as spam, there's only ~71% chance it's actually spam. Why?

1. **Prior is very strong**: Only 5% of emails are spam
2. **False positives are common**: With 95% legitimate emails × 2% false positive rate = 1.9% of all emails are false positives
3. **Competing hypotheses**: Flagged emails could be either:
   - Spam caught correctly: 4.75% of all emails
   - Legitimate email wrongly flagged: 1.9% of all emails
4. **Ratio**: 4.75% / (4.75% + 1.9%) ≈ 71.4%

## Visualization: Base Rate Fallacy

Out of 10,000 emails:

| | Spam | Legitimate | Total |
|---|------|-----------|-------|
| **Flagged** | 475 | 190 | **665** |
| **Not Flagged** | 25 | 9,310 | 9,335 |
| **Total** | **500** | **9,500** | **10,000** |

Of 665 flagged emails:
- 475 are actually spam ✓
- 190 are legitimate ✗

P(Spam | Flagged) = 475 / 665 = 0.714

## General Bayes' Theorem Formula

For multiple hypotheses:
$$P(H_i | E) = \frac{P(E | H_i) \cdot P(H_i)}{\sum_j P(E | H_j) \cdot P(H_j)}$$

Where:
- $H_i$ = hypothesis i
- $E$ = evidence
- Denominator = marginal likelihood (sum over all hypotheses)

## Key Insights

1. **Prior matters**: Low base rate makes false positives more likely to occur
2. **Sensitivity vs Specificity**: High sensitivity (catching true positives) doesn't guarantee low false positive rate
3. **Precision** = $P(\text{Spam} | \text{Flagged})$ (what we calculated)
4. **Recall** = $P(\text{Flagged} | \text{Spam})$ (sensitivity, 95% in this problem)

## Generalizations

### Multi-Class Problem

Extend to K classes:
$$P(C_k | \text{Evidence}) = \frac{P(\text{Evidence} | C_k) \cdot P(C_k)}{P(\text{Evidence})}$$

### Naive Bayes Classification

Assume features are independent given the class:
$$P(C | X_1, ..., X_n) \propto P(C) \prod_{i=1}^{n} P(X_i | C)$$

### Sequential Updates (Posterior becomes new prior)

After seeing one email is spam:
- New prior: updated belief about proportion of spam
- Observe more evidence
- Update again (online learning)

## Related Problems

- **Medical diagnosis**: Given positive test, what's probability of disease?
- **False positive paradox**: Why rare disease tests give many false positives
- **Prosecutor's fallacy**: Confusing $P(\text{Evidence} | \text{Innocent})$ with $P(\text{Innocent} | \text{Evidence})$
