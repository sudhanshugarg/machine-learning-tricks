# Designing a Model for a Severely Imbalanced Dataset

## The Scenario

You are building a **real-time credit-card fraud detection** model for a payment processor. The label is binary — a transaction is either fraudulent (`1`) or legitimate (`0`) — and the dataset is severely imbalanced: of roughly **1,000,000 transactions per day, only about 0.1% are fraudulent**.

You are given:

- **Tabular features** per transaction: amount, merchant category, time-of-day, country, currency, and a handful of aggregated "card history" features (number of transactions in the last 24h, average spend, distance from home, etc.).
- A **binary label** `y ∈ {0, 1}` from chargeback / dispute data, labelled with a ~60-day delay and with some label noise (not every dispute is actual fraud).
- A **cost structure**: a false negative (letting fraud through) costs the company the full transaction amount on average (~$120), while a false positive (blocking a legitimate transaction) costs roughly $8 in customer-support overhead and goodwill.

---

## The Single Prompt

> Walk us through how you would build this model end-to-end. In particular, **design the model architecture and the loss function** (with the math), explain how each choice addresses the class imbalance, and provide **working PyTorch code** for the model and at least one custom loss. Then explain how you would **train and evaluate** it, given that plain accuracy is nearly useless under a 0.1% prevalence.

You are allowed to make reasonable assumptions and state them. Treat this as a real design conversation — the interviewer will follow up on every choice you make.

---

## Your Task

1. **Architecture.** Propose a model architecture appropriate for tabular fraud data. Justify each choice (why this architecture, which layers, what regularization, why a single sigmoid output).
2. **Loss function.** Design a loss that handles severe class imbalance. Present at least two options (e.g., class-weighted BCE and focal loss) **with the mathematical derivation**, and explain the theoretical role of each hyperparameter.
3. **Evaluation.** Choose metrics that are meaningful under severe imbalance (not accuracy). Explain why each is appropriate and what its failure modes are.
4. **Training & deployment.** Address resampling, threshold selection, probability calibration, and the asymmetric cost structure. How do you turn a probability into a decision?
5. **Working code.** Provide runnable PyTorch code for the model architecture and at least one custom loss, and demonstrate the impact of the imbalance-handling choices on a synthetic dataset.

---

## Open-Ended Discussion Questions

After you present your design, expect follow-ups like:

1. **Why is accuracy a bad metric here?** If the model simply predicts "not fraud" for every transaction, what accuracy does it achieve, and what does that tell you about the Bayes-optimal strategy under 0-1 loss?

2. **Focal loss vs. class-weighted cross-entropy.** Both reweight the per-example loss. What does the $(1 - p_t)^{\gamma}$ modulating factor do that simple class weighting cannot? In what regime does focal loss help, and in what regime is it equivalent to (or worse than) plain class weighting?

3. **Resampling pitfalls.** Random oversampling of the minority class duplicates examples; random undersampling of the majority class throws away data. What are the failure modes of each for a neural network? Why can naive SMOTE hurt? Why is "oversample, then split into train/val" a data-leakage bug?

4. **The decision-theoretic view.** Given the cost structure ($c_{FN}$ vs $c_{FP}$), derive the Bayes-optimal decision threshold $\tau^*$. Show that the $0.5$ threshold is optimal only when the costs are symmetric and the classes are balanced.

5. **Calibration.** Why might a model trained with focal loss or heavy class weights produce poorly calibrated probabilities? Why does that matter (or not) if you only care about ranking? When does calibration actually matter for the fraud system?

6. **Label noise and delayed labels.** Disputes arrive up to 60 days late and not every dispute is fraud. How does label noise interact with class imbalance? What happens to the precision of your *labels* (not your model) at 0.1% prevalence?

7. **Streaming / concept drift.** Fraud patterns change weekly as attackers adapt. How would you detect concept drift, and how does it change your model architecture and retraining cadence?

---

## Deliverables

- A **model architecture** with a justification for each design choice.
- A **loss-function design** with the math for at least two imbalance-aware losses.
- An **evaluation plan** with the right metrics and their interpretations.
- A **training + deployment plan** covering resampling, thresholding, calibration, and cost.
- **Working PyTorch code** (model + custom loss) that demonstrates the impact of these choices on a synthetic imbalanced dataset.
