# Follow-up Discussion Topics

This document outlines common branching questions the interviewer may ask based on your initial responses. Be prepared to discuss any of these areas in depth.

---

## Feature Selection

The interviewer may repeatedly ask how you would choose or create better features. Be ready to discuss:

- **Which raw features are likely predictive?**
  - How do you identify features that correlate with the target?
  - How do you think about domain knowledge vs. data-driven feature discovery?

- **How to identify leakage?**
  - What is data leakage and why is it dangerous?
  - How do you catch leakage before deploying a model?

- **How to handle missing values?**
  - When is it safe to drop missing data?
  - When should you impute? What are good imputation strategies?

- **Scaling or normalization when needed?**
  - When is scaling necessary? Which algorithms require it?
  - What's the difference between standardization and normalization?

- **Encoding for categorical features?**
  - When do you use one-hot encoding vs. ordinal encoding?
  - How do you handle high-cardinality categorical variables?

- **Whether interactions or nonlinear transforms could help?**
  - How would adding polynomial features ($x^2$, $x_1 \cdot x_2$) change the boundary?
  - When are interactions important in your domain?

- **How to evaluate feature importance or feature usefulness?**
  - What methods quantify which features matter most?
  - How do you communicate feature importance to stakeholders?

---

## Model Choice

You may be asked:

- **When logistic regression is sufficient?**
  - What are the assumptions of logistic regression?
  - When should you stop and use logistic regression instead of more complex models?

- **When tree-based models are a better fit?**
  - What problems are tree-based models especially good at?
  - What are the trade-offs (interpretability vs. performance)?

- **How to think about linear vs. nonlinear boundaries?**
  - How do you detect if a nonlinear boundary is needed?
  - What's the cost of using a nonlinear model when linear suffices?

- **How to balance interpretability against predictive performance?**
  - When does interpretability matter most?
  - How do you explain a complex model's decisions?

---

## Evaluation

Expect discussion around:

- **Train/validation/test split?**
  - How should you split your data?
  - What can go wrong if you skip validation?

- **ROC-AUC vs PR-AUC for imbalanced data?**
  - When is each metric appropriate?
  - Why does ROC-AUC mislead on imbalanced problems?

- **Precision/recall/F1 trade-offs?**
  - How do you visualize and understand these trade-offs?
  - How do you pick the right balance for your application?

- **Calibration and probability interpretation?**
  - What does it mean for a model to be "calibrated"?
  - How do you improve calibration if it's poor?

---

## Cold Start

The round may end with cold start questions, such as:

- **How to make predictions for a new user with little or no history?**
  - What do you do if you have no historical data for a user?
  - How do you bootstrap initial recommendations?

- **How to handle a new item with limited interaction data?**
  - How are new items or products different from old ones?
  - What information can you use when you have no user interaction history?

- **What fallback features or priors you would use?**
  - When content or contextual features become crucial?
  - How do you leverage domain knowledge to fill gaps?

- **When heuristics, popularity baselines, or contextual features are appropriate before enough personalized data arrives?**
  - What's a good "baseline" system for cold start?
  - How do you transition from cold start heuristics to personalized models?

