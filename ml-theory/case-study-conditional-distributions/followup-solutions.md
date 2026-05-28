# Follow-up Solutions

Detailed answers to common branching questions that may arise during the interview.

---

## Feature Selection

### Question: Which raw features are likely predictive?

**Strong Answer:**

Start with domain intuition, then validate with data:

1. **Domain Knowledge Approach:**
   - Think about what factors causally influence the outcome
   - Example (loan default): Income, debt ratio, credit history intuitively relate to default risk
   - Avoid features that seem correlated but are just noise

2. **Data-Driven Approach:**
   - **Correlation with target:** Compute $\text{corr}(x_i, y)$ for each feature
   - **Mutual information:** $I(x_i; y)$ measures dependency (works for nonlinear relationships)
   - **Univariate statistical tests:** Chi-square (categorical), t-test (continuous)

3. **Visual Inspection:**
   - Plot feature distributions by class (as in the original problem)
   - Large separation = likely predictive
   - Heavy overlap = weak signal, but not useless

**Example Code:**
```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Compute correlation with target
correlations = df.corr()['target'].sort_values(ascending=False)
print(correlations)

# Compute mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
print(mi_df.sort_values('mi_score', ascending=False))

# Visualize distributions by class
import matplotlib.pyplot as plt
for col in df.columns:
    plt.figure()
    df[df['target']==0][col].hist(alpha=0.5, label='Class 0')
    df[df['target']==1][col].hist(alpha=0.5, label='Class 1')
    plt.legend()
    plt.title(f'{col} by Class')
    plt.show()
```

**Key Principle:** Features with strong class-conditional separation (as shown in the overlapping distributions problem) are predictive.

---

### Question: How to identify leakage?

**Strong Answer:**

Data leakage is when information "leaks" from the test set into the training set, causing artificially high performance.

**Types of Leakage:**

1. **Temporal Leakage:**
   - Using future information to predict the past
   - Example: Using next quarter's revenue to predict this quarter's default
   - **Fix:** Split by time. Train on past, test on future

2. **Label Leakage:**
   - Using information derived from the target itself
   - Example: Using "number of defaults in customer's peer group" when predicting default
   - **Fix:** Only use information available *before* the label is known

3. **Preprocessing Leakage:**
   - Computing statistics on full data before splitting
   - Example: Computing mean income on entire dataset, then using it to impute
   - **Fix:** Fit imputation/scaling only on training data, apply to test

**Detection Strategy:**

```python
# ❌ LEAKAGE: Fitting on full data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # FIT on full data!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# ✓ CORRECT: Fit on train, apply to test
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # FIT only on train
X_test_scaled = scaler.transform(X_test)        # APPLY to test
```

**Sanity Check:**
- If test performance is unreasonably high, suspect leakage
- Ask: "Is this information available at prediction time?"
- Look for features that seem too predictive

---

### Question: How to handle missing values?

**Strong Answer:**

The right approach depends on:
- Why is data missing (random vs. systematic)?
- How much is missing?
- What's the feature's importance?

**Strategy 1: Drop Missing Data**

When to use:
- Less than 5% missing
- Missing *completely at random* (MCAR)
- Not a critical feature

```python
df_clean = df.dropna()  # Remove rows with any missing values
df_clean = df.dropna(subset=['critical_features'])  # Remove only if key features missing
```

**Downsides:** Lose information, small sample size may become too small

**Strategy 2: Imputation**

**Mean/Median Imputation:**
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # For numerical
X_imputed = imputer.fit_transform(X_train)
```
- Simple, fast
- **Warning:** Reduces variance, underestimates uncertainty

**K-NN Imputation:**
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```
- Uses similarity to nearby points
- Better for nonlinear patterns

**Forward Fill (Time Series):**
```python
df = df.fillna(method='ffill')  # Last observation carried forward
```
- Good for time-series data

**Strategy 3: Create a Missing Indicator**

Sometimes, "missingness" is informative:
```python
# For feature 'income'
df['income_missing'] = df['income'].isna().astype(int)  # 1 if missing, 0 else
df['income'] = df['income'].fillna(df['income'].median())
```

**Principle:** If missing data correlates with the target (e.g., low-income applicants don't report income), treat missingness as a signal.

---

### Question: Scaling or normalization when needed?

**Strong Answer:**

**When to scale:**

| Algorithm | Needs Scaling? | Why |
|---|---|---|
| Logistic Regression | **YES** | Gradient descent sensitive to feature magnitude |
| Linear Regression | **YES** | Coefficient magnitude interpretation |
| SVM | **YES** | Distance-based, large features dominate |
| Tree-Based (RF, XGB) | **NO** | Splits are invariant to scaling |
| Neural Networks | **YES** | Initialization, gradient flow sensitive |
| K-NN | **YES** | Distance metric affected by scale |

**Two Main Approaches:**

1. **Standardization (Z-score):**
   $$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$
   - Result: Mean = 0, Std = 1
   - Use for: Logistic regression, SVM, neural networks
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_train)
   ```

2. **Normalization (Min-Max):**
   $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
   - Result: Range [0, 1]
   - Use for: Neural networks, when bounded range is important
   ```python
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler()
   X_scaled = scaler.fit_transform(X_train)
   ```

**Key Rule:** Always fit scaler on training data only, apply to test/val:
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### Question: Encoding for categorical features?

**Strong Answer:**

**Strategy 1: One-Hot Encoding**

Use when: Low-cardinality categorical features (< 20 unique values)

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X_train[['color', 'size']])
```

Example: `color = ['red', 'blue', 'green']` becomes:
```
[1, 0, 0]  # red
[0, 1, 0]  # blue
[0, 0, 1]  # green
```

**Advantages:** Works with all algorithms, interpretable
**Disadvantages:** Curse of dimensionality for high-cardinality features

**Strategy 2: Ordinal Encoding**

Use when: Ordinal categories with natural order (e.g., education level)

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['high school', 'bachelor', 'master', 'phd']])
X_encoded = encoder.fit_transform(X_train[['education']])
```

Result: `[0, 1, 2, 3]` representing order

**Advantages:** Preserves order, fewer dimensions
**Disadvantages:** Algorithm may misinterpret as numeric magnitude

**Strategy 3: Target Encoding**

Use when: High-cardinality categorical features (e.g., 1000+ categories)

Replace category with average target value in that category:

```python
# Example: Predicting loan default by city
city_target_means = df.groupby('city')['default'].mean()
df['city_encoded'] = df['city'].map(city_target_means)
```

**Advantages:** Dimensionality reduction, directly uses target info
**Risk:** Leakage if not careful. Must compute on train data only:

```python
# ✓ CORRECT
city_means = df_train.groupby('city')['default'].mean()
df_train['city_encoded'] = df_train['city'].map(city_means)
df_test['city_encoded'] = df_test['city'].map(city_means)  # Use train statistics
```

**Strategy 4: Frequency Encoding**

Use when: You just need to capture "how common" a category is

```python
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)
```

---

### Question: Whether interactions or nonlinear transforms could help?

**Strong Answer:**

This directly relates to the class-conditional distribution problem. If classes aren't linearly separable, you need nonlinearity.

**How to Detect Need for Nonlinearity:**

1. **Visualize the Decision Boundary:**
   - Fit logistic regression
   - Check if straight line separates classes well
   - If not, nonlinearity may help

2. **Check Model Performance:**
   - Linear model (logistic regression) gets 70% accuracy
   - Nonlinear model (random forest) gets 85% accuracy
   - Gap suggests nonlinear relationships

**Polynomial Features:**

```python
from sklearn.preprocessing import PolynomialFeatures

# For 2 features [x1, x2], create polynomial terms
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# Result: [1, x1, x2, x1^2, x1*x2, x2^2]
```

**Effect on Decision Boundary:**
- Linear: Straight line boundary
- Quadratic: Curved parabolic boundary
- Higher degree: More complex curves

**Visualization:**
```
Linear Boundary (degree=1):       Quadratic Boundary (degree=2):
     ▲                                 ▲
     │  • • •                          │  • • •
   1 │  • X • ─────────               │  • X ╱──
     │  • • • │ • X •                 │  • •╱ • X •
   0 │────────┼──────                 │  •╱───────
     │  • • • │                       │ ╱• • •
    -1│ • X •                         │╱ • X •
     └───────────────► x             └───────────► x
```

**Interaction Terms:**

Example: For credit score ($x_1$) and income ($x_2$):
```python
df['score_x_income'] = df['credit_score'] * df['income']
```

High income + high score might be more predictive than either alone.

**When to Use:**
- Domain knowledge suggests interactions (e.g., medication dosage × body weight)
- Exploratory analysis shows patterns
- You have sufficient data (more features = more data needed)

**Warning:** Adding too many polynomial features causes overfitting. Use:
- Regularization (L1/L2)
- Feature selection
- Validation to check if it actually helps

---

### Question: How to evaluate feature importance or feature usefulness?

**Strong Answer:**

**Method 1: Coefficient-Based (Linear Models)**

For logistic regression, inspect coefficients:

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

importances = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print(importances)
```

Larger absolute coefficient = more important

**Limitation:** Only works for linear models, requires scaling

**Method 2: Tree-Based Feature Importance**

For tree-based models (Random Forest, XGBoost):

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importances)
```

Measures how much each feature reduces impurity (Gini, entropy)

**Method 3: Permutation Importance**

Works with any model. Shuffle feature values and measure performance drop:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean
}).sort_values('importance', ascending=False)
```

High drop in performance = important feature

**Method 4: SHAP (SHapley Additive exPlanations)**

Most interpretable: Shows each feature's contribution to each prediction:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot feature importance across all samples
shap.summary_plot(shap_values, X_test)

# Detailed explanation for single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

**Method 5: Ablation Study**

Simple but informative: Train model without feature, compare performance:

```python
baseline_score = model.score(X_test, y_test)

for col in X.columns:
    X_test_ablated = X_test.drop(col, axis=1)
    ablated_score = model.score(X_test_ablated, y_test)
    drop = baseline_score - ablated_score
    print(f"{col}: {drop:.4f} drop in score")
```

---

## Model Choice

### Question: When logistic regression is sufficient?

**Strong Answer:**

Logistic regression is underrated. Use it when:

1. **Classes are roughly linearly separable**
   - From the original problem: if overlap is minimal, logistic regression is optimal
   - Trade-off: interpretability + efficiency vs. marginal accuracy gain

2. **You need interpretability**
   - Each coefficient tells you the impact of a feature
   - Easy to explain to non-technical stakeholders
   - Required in regulated industries (banking, healthcare)

3. **You have limited data**
   - Fewer parameters to estimate
   - Less likely to overfit

4. **Prediction time matters**
   - $O(d)$ to make prediction (d = number of features)
   - Compare: Random Forest is $O(d \times \text{num_trees})$

5. **Calibrated probabilities matter**
   - Logistic regression outputs well-calibrated probabilities
   - Tree models need post-hoc calibration

**When NOT to use:**
```
Model A (Logistic Regression): 85% accuracy
Model B (Random Forest): 92% accuracy
```
- If accuracy gap is large AND interpretability isn't critical, switch to B
- If gap is small (< 2-3%), stick with Logistic Regression

**Example Decision Flow:**
```
Is the problem linearly separable?
  ├─ YES + interpretability required? → Logistic Regression ✓
  ├─ YES + limited data? → Logistic Regression ✓
  ├─ NO + complex patterns? → Tree-based or Neural Network
  └─ UNSURE? → Try Logistic Regression first (baseline)
```

---

### Question: When tree-based models are a better fit?

**Strong Answer:**

Tree-based models (Random Forest, XGBoost, LightGBM) excel when:

1. **Nonlinear relationships are important**
   - Classes need curved decision boundaries
   - Features interact in complex ways
   - Example: Price depends on (Size × Location) in nonlinear way

2. **You have mixed feature types**
   - Numerical + categorical + text
   - Trees handle all without preprocessing
   - Contrast: Logistic regression needs encoding

3. **Feature importance is critical**
   - Built-in feature importance rankings
   - Easy to identify which features matter

4. **Training data is large**
   - More data justifies more parameters
   - Less overfitting risk

5. **You don't need probability calibration**
   - Tree probabilities are often miscalibrated
   - Fix with Platt scaling or isotonic regression

**Specific Advantages:**

| Aspect | Logistic Reg | Tree-Based |
|---|---|---|
| Handles nonlinearity | ❌ | ✓ |
| Interpretability | ✓ | Partial |
| Feature encoding | Need to encode | Automatic |
| Speed (training) | ✓ Fast | Slower |
| Speed (inference) | ✓ Fast | Moderate |
| Probability calibration | ✓ Good | Poor |
| Feature importance | Coefficients | Built-in |

**Hybrid Approach:**
```
1. Start with Logistic Regression (baseline)
   - Get feature importance from coefficients
   - Establish performance target

2. Try Tree-Based Model (Random Forest)
   - Compare accuracy gain
   - Extract feature importance

3. If accuracy gain > 5-10%:
   - Use tree model + post-hoc calibration

4. If accuracy gain < 2-3%:
   - Stick with logistic regression (simpler, faster)
```

---

### Question: How to think about linear vs. nonlinear boundaries?

**Strong Answer:**

This connects directly to the class distributions problem.

**Visual Intuition:**

```
Linear Boundary (Logistic Regression):
─────────────────────────────────────
     • • •     X X X
     • X •─────X X •          Straight line
     • • •     X X X           separates classes


Nonlinear Boundary (Decision Trees, Neural Networks):
──────────────────────────────────────────────────────
     • • •
     • X •  ╱╱╱╱     X X X     Curved boundary
     • • • ╱╱╱╱      X X •     adapts to data
           ╱╱╱╱
```

**How to Detect Nonlinearity:**

1. **Visual inspection:**
   - Plot both classes in 2D space
   - Can a straight line separate them? If no → nonlinear

2. **Model comparison:**
   ```python
   lr_score = LogisticRegression().fit(X_train, y_train).score(X_test, y_test)
   rf_score = RandomForestClassifier().fit(X_train, y_train).score(X_test, y_test)

   gap = rf_score - lr_score
   if gap > 0.05:  # > 5% improvement
       print("Nonlinear boundary likely needed")
   ```

3. **Decision boundary visualization:**
   ```python
   from sklearn.inspection import DecisionBoundaryDisplay

   # Plot decision boundaries for both models
   DecisionBoundaryDisplay.from_estimator(lr, X_train, response_method='predict')
   DecisionBoundaryDisplay.from_estimator(rf, X_train, response_method='predict')
   ```

**Cost of Overcomplicating:**

```
Accuracy on Training Data vs. Nonlinearity:
────────────────────────────────────────────
Train Accuracy
       │
    95%│     ╱╱───────────  Nonlinear model
       │    ╱╱ (overfitting)
    90%│───╱─────────────────  Linear model
       │  ╱   (appropriate complexity)
    85%│ ╱
       │
       └──────────────────────► Feature Complexity
           Linear    Poly    Trees  Neural Nets
```

**When Nonlinearity is Justified:**
- Test set gap between linear and nonlinear > 3-5%
- You have plenty of data (low overfitting risk)
- Complexity is worth the accuracy gain

**When to Stay Linear:**
- Gap < 2%
- Data is small
- Interpretability is critical
- Deployment constraints (fast, simple inference)

---

### Question: How to balance interpretability against predictive performance?

**Strong Answer:**

There's a fundamental trade-off. Choose based on context:

**High Interpretability:**
- Logistic Regression: Explain each coefficient
- Decision Trees: Show decision rules ("If income > $50k AND credit_score > 700, approve")
- Linear Models: Each feature's impact is transparent

**High Performance:**
- Neural Networks
- Ensemble Methods (Gradient Boosting)
- Complex nonlinear models

**Context Matters:**

| Context | Choose |
|---|---|
| **Regulated (finance, healthcare)** | Interpretable (accuracy slightly lower okay) |
| **High-stakes decisions** | Interpretable + auditable |
| **Competitive (Kaggle)** | Performance (complexity okay) |
| **Internal tools** | Balance both |
| **Consumer-facing** | Interpretable (users want to understand) |

**Practical Strategy:**

```
Step 1: Establish Baseline
   Linear Model (Logistic Regression)
   → Get baseline accuracy (e.g., 82%)
   → All coefficients interpretable

Step 2: Try Interpretable Improvements
   Decision Trees, Linear Model w/ engineered features
   → Can we reach 85-87% with interpretability?

Step 3: Explore Complex Models (if needed)
   Neural Networks, Gradient Boosting
   → If we need > 88%, accept loss of interpretability
   → Use SHAP/LIME to add back some interpretability

Step 4: Deploy with Explainability
   Even complex models can be explained via:
   - SHAP values (per-prediction explanation)
   - Feature importance (global understanding)
   - Decision rules from decision trees trained on predictions
```

**How to Explain Complex Models:**

```python
import shap

# Train any model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Add interpretability via SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For a specific prediction:
# "This loan was approved because:"
# - High income (+0.15)
# - Long employment history (+0.12)
# - Offset by recent credit inquiry (-0.08)
# Net decision: APPROVE
```

---

## Evaluation

### Question: Train/validation/test split?

**Strong Answer:**

The right split prevents overfitting and gives honest performance estimates.

**Standard Approach: 70-80% Train, 10-15% Validation, 10-15% Test**

```python
from sklearn.model_selection import train_test_split

# Split 1: Separate test set (hold out from start)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15)

# Split 2: Divide temp into train/val
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85)

# Result: 72% train, 13% val, 15% test
```

**Why Three Sets?**

| Set | Purpose | How Many Times |
|---|---|---|
| **Train** | Learn model parameters | Once (fit model) |
| **Validation** | Tune hyperparameters, choose between models | Many times (try different options) |
| **Test** | Final performance estimate | Once at the very end |

**What Goes Wrong:**

```
❌ MISTAKE 1: Tuning on test set
   - Try 100 hyperparameter combinations on test set
   - Pick the best
   - Report test accuracy
   → Test set is contaminated! Accuracy is too high

✓ CORRECT:
   - Try 100 combinations on validation set
   - Pick the best
   - Report accuracy on separate test set
```

```
❌ MISTAKE 2: Doing train/test split but no validation
   - Train on train set
   - Pick learning rate based on test accuracy
   - Report test accuracy
   → Still contaminating test set!

✓ CORRECT: Explicitly separate validation
```

**K-Fold Cross-Validation (Better for Small Data):**

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

Advantages:
- Uses all data for training (more learning signal)
- Gives estimate of variability
- Better for small datasets

**Stratified Split (For Imbalanced Data):**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y  # Maintain class ratios
)
```

Ensures train/val/test have same class distribution as full dataset.

---

### Question: ROC-AUC vs PR-AUC for imbalanced data?

**Strong Answer:**

This is critical for imbalanced datasets (e.g., 95% class 0, 5% class 1).

**ROC-AUC: The Problem**

ROC-AUC plots True Positive Rate vs. False Positive Rate:
- TPR = $\frac{TP}{TP+FN}$ (of actual positives, how many did we catch?)
- FPR = $\frac{FP}{FP+TN}$ (of actual negatives, how many did we falsely alarm?)

**Why it's misleading for imbalanced data:**

Imagine:
- 1000 samples: 50 positive, 950 negative
- Model predicts EVERYTHING as negative

Results:
```
TP = 0, FN = 50
TN = 950, FP = 0

TPR = 0/50 = 0%
FPR = 0/950 = 0%   ← FPR is 0 because there are SO MANY true negatives!

ROC-AUC = 0.5 (random classifier)  ← Misleading! We're catching 0% of positives!
```

**Precision-Recall (PR) Curve: Better for Imbalanced**

PR curve plots Recall vs. Precision:
- Recall = $\frac{TP}{TP+FN}$ (same as TPR)
- Precision = $\frac{TP}{TP+FP}$ (of what we predicted positive, how many are actually positive?)

Same example:
```
TP = 0, FP = 0
Precision = 0/0 = undefined
Recall = 0/50 = 0%

PR-AUC highlights the problem: We're not detecting any positives!
```

**Comparison:**

```
ROC-AUC:
- Works well when classes are balanced
- Can be misleading on imbalanced data
- Range: [0.5 (random), 1.0 (perfect)]

PR-AUC:
- Sensitive to class imbalance
- Better for imbalanced datasets
- Range: [class_prior (random), 1.0 (perfect)]
- Lower baseline makes differences more visible
```

**Visual:**

```
ROC Curve (Imbalanced):           PR Curve (Imbalanced):
TPR                              Precision
│        ╱╱╱                     │ ╲
│      ╱╱  ← Hard to see         │  ╲
│    ╱╱     difference           │   ╲  ← Easy to see
│  ╱╱╱╱     (many TN)            │    ╲   difference
│╱╱                              │     ╲
└─────────► FPR                  └──────► Recall
```

**Recommendation:**

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# For imbalanced data, prefer PR-AUC
if class_imbalance_ratio > 0.2:  # Less than 20% positive class
    auc_pr = average_precision_score(y_test, y_proba)
    print(f"PR-AUC: {auc_pr:.3f}")
else:
    auc_roc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc_roc:.3f}")

# Even better: Report both + confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(f"TP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}")
```

---

### Question: Precision/recall/F1 trade-offs?

**Strong Answer:**

These metrics are inherently in tension. The right choice depends on your business problem.

**Definitions:**

- **Precision:** $\frac{TP}{TP+FP}$ = "Of what we predicted positive, how many are correct?"
- **Recall:** $\frac{TP}{TP+FN}$ = "Of actual positives, how many did we catch?"
- **F1:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ = Harmonic mean

**Visualization: The Trade-off**

```
As we lower threshold (predict positive more often):
────────────────────────────────────────────────────

Threshold = 0.9:  Predict positive only for very confident cases
                  Precision = 95% (few false alarms)
                  Recall = 20% (miss many true positives)

Threshold = 0.5:  Balanced threshold
                  Precision = 70%
                  Recall = 65%
                  F1 = 67% (good balance)

Threshold = 0.1:  Predict positive liberally
                  Precision = 40% (many false alarms)
                  Recall = 95% (catch almost everything)
```

**Precision-Recall Curve:**

```
Precision
     ↑
  100│ •                    High threshold
     │  •                   (high precision, low recall)
  90 │   •
     │    •
  80 │     •••
     │        •••
  70 │           •••••
     │               •••••••
  60 │                     •••••
     │                           •••
  50 │                              •••
     │                                  ••
  40 │                                    ••
     └─────────────────────────────────────► Recall
      0  20  40  60  80  100

                     ↑
                Low threshold
         (low precision, high recall)
```

**When to Optimize What:**

**Case 1: False Positives are Costly (Optimize Precision)**

Example: Email spam filtering
- FP = Mark legitimate email as spam (user loses important email)
- FN = Let spam through (minor annoyance)

```python
# Raise threshold to reduce false alarms
threshold = 0.8  # Only predict spam if very confident

# Metrics to track:
# - Precision (avoid false alarms)
# - Recall (miss some spam, but that's okay)
```

**Case 2: False Negatives are Costly (Optimize Recall)**

Example: Disease diagnosis
- FN = Miss a diagnosis (patient doesn't get treatment, could be fatal)
- FP = False alarm (patient gets screened, no harm in investigation)

```python
# Lower threshold to catch more cases
threshold = 0.3  # Predict disease if any suspicion

# Metrics to track:
# - Recall (catch all or most cases)
# - Precision (expect false alarms, that's acceptable)
```

**Case 3: Both Matter Equally (F1 Score)**

Example: Customer churn prediction
- FP = Predict churn but customer stays (wasted retention effort)
- FN = Miss churn (lost customer)
- Both are costly

```python
# Optimize F1 score
from sklearn.metrics import f1_score

best_threshold = max(
    [(t, f1_score(y_test, y_proba > t)) for t in np.arange(0.1, 0.9, 0.1)],
    key=lambda x: x[1]
)
```

**ROC Curve as Threshold Visualization:**

```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr)
for i, thresh in enumerate(thresholds[::10]):
    plt.annotate(f'{thresh:.2f}', (fpr[i*10], tpr[i*10]))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (threshold increases along curve)')
plt.show()
```

Each point on ROC curve = a different threshold. You pick the threshold based on your cost structure.

---

### Question: Calibration and probability interpretation?

**Strong Answer:**

A calibrated model's predicted probabilities match actual frequencies.

**What is Calibration?**

If model predicts P(y=1|x) = 0.8 for 100 samples, roughly 80 should actually be positive class.

**Uncalibrated Model Example:**

```
Model predicts 60% probability for 100 samples
But in reality, 85% turn out positive

The model is "underconfident"
It should have predicted 85%, not 60%
```

**Why Calibration Matters:**

1. **Trust the probabilities:** If model says 70% confidence, you should believe it
2. **Set optimal thresholds:** Can't tune threshold properly without calibration
3. **Business decisions:** Confidence drives ROI calculations

**Detecting Poor Calibration:**

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.plot(prob_pred, prob_true, 'o-', label='Model')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.legend()
plt.show()
```

**Visualization:**

```
Perfect Calibration:        Poor Calibration:
Frac Positive              Frac Positive
       │                          │
     1 │     ╱                   1 │
       │    ╱                     │     ╱╱╱╱
     0.7│   ╱ (at pred=0.7,       │    ╱
       │  ╱   actual=0.7)       0.7│   ╱  (at pred=0.7,
     0.5│ ╱                        │  ╱    actual=0.5)
       │╱                         │╱
     0 │                        0 │
       └────────► Predicted      └────────► Predicted
        Prob 0.0   0.5   1.0       Prob 0.0   0.5   1.0
```

**How to Fix Poor Calibration:**

**Method 1: Platt Scaling (Simple)**

```python
from sklearn.calibration import CalibratedClassifierCV

# Wrap your model
calibrated = CalibratedClassifierCV(base_estimator=model, method='sigmoid')
calibrated.fit(X_train, y_train)

# Predictions are now calibrated
y_proba_cal = calibrated.predict_proba(X_test)
```

**Method 2: Isotonic Regression (More Flexible)**

```python
calibrated = CalibratedClassifierCV(base_estimator=model, method='isotonic')
calibrated.fit(X_val, y_val)  # Fit on validation set
y_proba_cal = calibrated.predict_proba(X_test)
```

**Models by Calibration Quality:**

| Model | Native Calibration | Notes |
|---|---|---|
| Logistic Regression | Excellent | Often well-calibrated out of box |
| SVM | Poor | Probabilities are not reliable |
| Random Forest | Moderate | Overconfident |
| Gradient Boosting | Poor | Very overconfident |
| Neural Networks | Moderate | Can be improved with temperature scaling |

**Best Practice:**

```python
# Always check calibration
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_val, y_proba_val)
calibration_error = np.abs(prob_true - prob_pred).mean()

if calibration_error > 0.05:  # If > 5% average error
    # Apply calibration
    model = CalibratedClassifierCV(model, method='sigmoid')
    model.fit(X_cal, y_cal)  # Fit on separate calibration set
```

---

## Cold Start

### Question: How to make predictions for a new user with little or no history?

**Strong Answer:**

Cold start is one of the hardest problems in ML. Different strategies for different levels of information:

**Scenario 1: Brand New User, No History**

Strategy: Use **population-level priors and content features**

```python
# If predicting user churn:
# Use base rate from existing users
base_churn_rate = df['churned'].mean()

# Plus any available features
# - User signup date (new users churn more?)
# - Subscription tier (premium users less likely to churn?)
# - Marketing channel (organic vs. paid signup?)

# Example prediction
prediction = base_churn_rate * 0.8  # Adjust slightly based on features
```

**Scenario 2: User Has Some Activity**

Strategy: Combine **historical patterns + user-specific signal**

```python
# Example: Movie recommendations for new user
# Week 1: Only 2 movies watched
# You don't have enough data for collaborative filtering

# Instead, use:
# 1. Content features (movie genre, director, rating)
# 2. User demographics (age, location, subscription type)
# 3. Popularity baseline (what are trending movies?)

# Blend these signals:
prediction = (
    0.3 * content_similarity +      # How similar to movies they watched
    0.3 * demographic_similarity +  # Users like them like X
    0.4 * popularity               # Popular movies work for new users
)
```

**Scenario 3: A/B Testing Approach**

For new users, randomize recommendations and track engagement:

```python
# Assign new users to different strategies
if user_is_cold_start:
    strategy = random.choice(['popular', 'demographic', 'content'])

    if strategy == 'popular':
        recommendations = top_trending_movies()
    elif strategy == 'demographic':
        recommendations = movies_for_similar_users()
    else:
        recommendations = genre_based_on_signup_flow()

# Track which strategy has best engagement
# After sufficient data, shift to personalized models
```

---

### Question: How to handle a new item with limited interaction data?

**Strong Answer:**

Similar to new users, but you have **item attributes** instead of user history.

**Scenario 1: New Product Launch**

Use **content features + popularity priors**:

```python
# Item: New smartphone
# Limited sales history, but rich attributes available

item_features = {
    'category': 'electronics',
    'brand': 'Samsung',
    'price': 799,
    'screen_size': 6.1,
    'processor': 'Snapdragon 8',
    'release_date': '2024-01'
}

# Find similar items that already have purchase history
similar_items = find_similar(item_features, existing_products)

# Estimate demand based on similar items
predicted_demand = average_demand(similar_items) * category_multiplier
```

**Scenario 2: New Movie in Recommendation System**

Use **metadata + cold-start embeddings**:

```python
movie_metadata = {
    'genre': ['action', 'sci-fi'],
    'director': 'Christopher Nolan',
    'cast': ['Timothée Chalamet', 'Matt Damon'],
    'IMDB_rating': 8.2,
    'budget': '$200M'
}

# Approach 1: Content-based similarity
similar_movies = find_by_genre_director_cast(movie_metadata)
predicted_rating = average_rating(similar_movies)

# Approach 2: Pre-trained embeddings
# Use embeddings trained on similar domains
movie_embedding = pretrained_model.embed(movie_metadata)
# Then use this embedding for recommendations
```

---

### Question: What fallback features or priors would you use?

**Strong Answer:**

**Hierarchy of Information (Use What You Have)**

```
Tier 1: Personalized behavior (Best)
   └─ User's past interactions
   └─ User's explicit preferences
   └─ Similar user's behaviors

Tier 2: Content features (Good)
   └─ Item metadata (category, price, author, etc.)
   └─ Textual content (review text, product description)
   └─ Item embeddings (pre-trained)

Tier 3: Population statistics (Okay)
   └─ Popularity (overall most popular items)
   └─ Category trends (popular in that category)
   └─ Seasonal patterns (summer vs. winter)

Tier 4: Demographics (Fallback)
   └─ Age-based preferences
   └─ Location-based preferences
   └─ Subscription tier

Tier 5: Random / Exploration (Last resort)
   └─ Random items (A/B testing new products)
   └─ Diverse items (avoid repeated recommendations)
```

**Practical Blending:**

```python
def predict_score(user_id, item_id):
    """Predict rating with graceful fallback"""

    # Tier 1: Personalized collaborative filtering
    if user_history(user_id).size > 100:
        return collaborative_filtering_score(user_id, item_id)

    # Tier 2: Content-based + user demographics
    if user_history(user_id).size > 10:
        content_score = content_similarity(user_id, item_id)
        demo_score = demographic_similarity(user_id, item_id)
        return 0.6 * content_score + 0.4 * demo_score

    # Tier 3: Popularity + category trends
    if user_history(user_id).size > 0:
        item_popularity = item_id.popularity
        category_trend = category_popularity(item_id.category)
        return 0.7 * item_popularity + 0.3 * category_trend

    # Tier 4: Demographics only
    if user_demographics_available(user_id):
        return demographic_preference(user_id, item_id)

    # Tier 5: Random exploration
    return random_score()
```

---

### Question: When heuristics, popularity baselines, or contextual features are appropriate before enough personalized data arrives?

**Strong Answer:**

**Phase-Based Approach:**

```
Phase 1: COLD START (Days 0-7)
══════════════════════════════════════════
User: Brand new, 0-10 interactions
Strategy: Heuristics + Popularity + Demographics
Code:
   def get_recommendation(user_id):
       user = get_user(user_id)
       if user.num_interactions < 10:
           return {
               'strategy': 'cold_start',
               'recommendations': popular_items_for_demographic(user.age, user.location),
               'personalization': 0%
           }

Example: Netflix recommends "Top 10 Now" to new users

────────────────────────────────────────────────

Phase 2: WARM-UP (Days 8-30)
════════════════════════════════════════════
User: 10-50 interactions
Strategy: Hybrid (30% popularity + 70% personalized)
Code:
   def get_recommendation(user_id):
       user = get_user(user_id)
       if 10 <= user.num_interactions < 50:
           popular = popular_items_in_category(user.favorite_category)
           personalized = cf_recommendation(user_id)
           return 0.3 * popular + 0.7 * personalized

Example: Netflix shows trending shows + some personalized picks

────────────────────────────────────────────────

Phase 3: RAMP UP (Days 31-60)
════════════════════════════════════════════
User: 50-200 interactions
Strategy: Mostly personalized (10% popularity + 90% personalized)
Code:
   def get_recommendation(user_id):
       user = get_user(user_id)
       if 50 <= user.num_interactions < 200:
           popular = popular_items_in_category(user.favorite_category)
           personalized = cf_recommendation(user_id)
           return 0.1 * popular + 0.9 * personalized

────────────────────────────────────────────────

Phase 4: MATURE (Days 60+)
════════════════════════════════════════════
User: 200+ interactions
Strategy: Fully personalized (0% popularity + 100% personalized)
Code:
   def get_recommendation(user_id):
       user = get_user(user_id)
       if user.num_interactions >= 200:
           return cf_recommendation(user_id)

Example: Netflix shows only personalized recommendations
```

**When Each Approach is Best:**

| Approach | Best For | Risk |
|---|---|---|
| **Heuristics** | New users, new items, fast decisions | May feel generic |
| **Popularity** | Exploration, when model uncertain | Creates "filter bubble" |
| **Content features** | New items with metadata | Misses collaborative signal |
| **Demographics** | Users with no history | Stereotyping |
| **Personalization** | Mature users with history | Requires sufficient data |

**Monitoring Strategy:**

```python
def monitor_cold_start_performance():
    """Track quality by data phase"""

    for phase in ['cold_start', 'warm_up', 'ramp_up', 'mature']:
        users_in_phase = filter_by_phase(users, phase)

        metrics = {
            'click_through_rate': compute_ctr(users_in_phase),
            'conversion_rate': compute_cvr(users_in_phase),
            'user_satisfaction': compute_rating(users_in_phase)
        }

        print(f"Phase: {phase}")
        print(f"  CTR: {metrics['ctr']:.3f}")
        print(f"  CVR: {metrics['cvr']:.3f}")
        print(f"  Rating: {metrics['rating']:.2f}/5.0")

        # If performance dips in phase, investigate blend ratios
```

**Best Practices:**

1. **Always have a baseline heuristic** - When ML model fails, fall back gracefully
2. **Track user journey** - Don't over-personalize too early (users feel creepy)
3. **Progressive personalization** - Gradually increase personalization as data arrives
4. **A/B test blend ratios** - "Is 30% popularity + 70% personalized better than 20%+80%?"
5. **Monitor diversity** - Don't let popularity dominate (users get bored seeing same items)

---

## Summary: Interview Flow

A strong candidate answers like this:

```
Interviewer: "How would you model x and y?"
You: "I'd think probabilistically: estimate P(x|y=0) and P(x|y=1),
     then use Bayes' rule to get P(y|x). For prediction,
     I'd threshold at P(y=1|x) ≥ τ."

Interviewer: "Here's a plot with overlapping distributions..."
You: "The overlap means classes aren't perfectly separable.
     No single threshold is perfect. That's the Bayes error—
     a fundamental limit, not a model limitation."

Interviewer: "How would you choose the threshold?"
You: "Depends on costs. If false positives are expensive,
     I'd raise the threshold. If false negatives are expensive,
     I'd lower it. Otherwise, I'd place it at the intersection
     where P(x|y=0)P(y=0) = P(x|y=1)P(y=1)."

Interviewer: "What metrics would you use?"
You: "That depends on the business. I'd ask: what's more costly—
     false positives or false negatives? If FP is worse,
     optimize precision. If FN is worse, optimize recall.
     If both matter equally, optimize F1 or cost-weighted loss."

Interviewer: (Follow-up) "What features would you engineer?"
You: "I'd start with domain knowledge—which factors causally
     affect the outcome? Then validate with correlation, MI,
     visualization. I'd also check for leakage and nonlinear
     relationships."

Interviewer: (Follow-up) "When would you use a simple logistic
regression instead of a complex model?"
You: "If the gap is small (< 2-3%), stick with logistic regression.
     It's faster, more interpretable, gives calibrated probabilities.
     Only switch to complex models if accuracy gain justifies the cost."
```

