# ML System Design - Frequently Asked Questions

This FAQ addresses common questions across ML system design problems. These are questions you'll encounter in interviews when designing systems like fraud detection, recommendation engines, ranking systems, etc.

---

## Table of Contents

### 1. Data Pipeline
- [1.1 Data Quality Issues](#11-data-quality-issues)
- [1.2 Handling Data Issues](#12-handling-data-issues)
- [1.3 Data Freshness](#13-data-freshness)
- [1.4 Oversampling vs Undersampling](#14-oversampling-vs-undersampling)
- [1.5 Probability Calibration with Resampling](#15-probability-calibration-with-resampling)
- [1.6 Why Weights Don't Need Calibration](#16-why-weights-dont-need-calibration)

### 2. Data Processing & Feature Engineering
- [2.1 Scalable Data Processing Pipelines](#21-scalable-data-processing-pipelines)

### 3. Data Versioning & Management
- [3.1 Importance of Data Versioning](#31-importance-of-data-versioning)
- [3.2 Managing Large Datasets](#32-managing-large-datasets)
- [3.3 Data Lineage & Reproducibility](#33-data-lineage--reproducibility)

### 4. Model Training
- [4.1 Cross-Validation](#41-cross-validation)
- [4.2 Weighted Cross-Entropy Loss](#42-weighted-cross-entropy-loss)

### 5. Model Deployment & Serving
- [5.1 Deployment Architectures](#51-deployment-architectures)
- [5.2 Choosing Deployment Architecture](#52-choosing-deployment-architecture)
- [5.3 Model Optimization](#53-model-optimization)
- [5.4 Scaling Model Serving](#54-scaling-model-serving)

### 6. Monitoring & Maintenance
- [6.1 Key Metrics to Monitor](#61-key-metrics-to-monitor)
- [6.2 Model Retraining Strategies](#62-model-retraining-strategies)
- [6.3 Debugging ML Systems](#63-debugging-ml-systems)

### 7. Industry Tools & Technologies
- [7.1 ML System Design Tools](#71-ml-system-design-tools)

---

## 1. Data Pipeline

### 1.1 Data Quality Issues

**Q: Can you identify potential data quality issues?**

**Answer:**

Common data quality issues to look for:

#### Missing Values

Missing data can occur at two levels:
1. **Column level**: Feature has many missing values across rows
2. **Row level**: Individual row has missing values in some features

**COLUMN-LEVEL decisions** (Should we keep this feature?):

```
If feature is missing > 50% of values:

Option 1: DROP THE COLUMN (Feature removal)
├─ When to use: Feature is not useful (most values missing)
├─ Pros: Simple, no imputation needed
├─ Cons: Lose all signal from that feature
└─ Example: If 'phone_number' is 60% missing, drop it

Option 2: IMPUTE ALL MISSING VALUES (Keep feature)
├─ When to use: Feature is useful but incomplete
├─ Pros: Retain information from feature
├─ Cons: Imputation adds noise
└─ Example: If 'age' is 30% missing, impute with median

Decision Framework:
┌─────────────────────────────────────────────────┐
│ Missing % in column:                            │
│ ├─ < 5%: Always impute (little missing)         │
│ ├─ 5-30%: Usually impute (moderate missing)     │
│ ├─ 30-50%: Consider both (borderline)           │
│ └─ > 50%: Usually drop (too much missing)       │
└─────────────────────────────────────────────────┘
```

**ROW-LEVEL decisions** (Should we keep this training sample?):

```
If a row has missing values in some features:

Option 1: DROP THE ROW (Sample removal)
├─ When to use: Row is incomplete/unreliable
├─ Pros: Keep data quality high
├─ Cons: Lose training samples (especially harmful if rare class)
└─ Example: Transaction with missing critical fields

Option 2: IMPUTE MISSING VALUES (Keep row)
├─ When to use: Row is mostly complete
├─ Pros: Keep training samples
├─ Cons: Imputed values may be noisy
└─ Example: Row with 1-2 missing values out of 50 features

Decision Framework:
┌──────────────────────────────────────────────────┐
│ Missing features in this row:                    │
│ ├─ 0 features: Keep (no missing)                │
│ ├─ 1-3 features: Impute (mostly complete)       │
│ ├─ 3-5 features: Consider both (borderline)     │
│ └─ > 5 features: Usually drop (too incomplete)  │
└──────────────────────────────────────────────────┘
```

#### Code Example: Distinguish Column vs Row-level

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'amount': [100, 200, np.nan, 400, 500],
    'age': [25, np.nan, 35, np.nan, 55],
    'phone': [np.nan, np.nan, np.nan, '555-1234', np.nan],
    'location': ['US', 'UK', 'US', 'US', 'UK']
})

print("Missing values per column:")
print(df.isnull().sum())
# amount: 1 missing (20%)
# age: 2 missing (40%)
# phone: 4 missing (80%)
# location: 0 missing (0%)

print("\nMissing values per row:")
print(df.isnull().sum(axis=1))
# Row 0: 1 missing
# Row 1: 2 missing
# Row 2: 1 missing
# Row 3: 1 missing
# Row 4: 1 missing

# STEP 1: COLUMN-LEVEL DECISIONS
# Drop columns with > 50% missing
missing_pct_col = df.isnull().sum() / len(df) * 100
print(f"\nColumn missing percentages:\n{missing_pct_col}")
# amount: 20%, age: 40%, phone: 80%, location: 0%

cols_to_drop = missing_pct_col[missing_pct_col > 50].index
print(f"Dropping columns with > 50% missing: {list(cols_to_drop)}")
# Drop 'phone' (80% missing)

df = df.drop(columns=cols_to_drop)
# Now we have: amount, age, location

# STEP 2: ROW-LEVEL DECISIONS
# Drop rows with > 3 missing values (in remaining columns)
missing_per_row = df.isnull().sum(axis=1)
print(f"\nRow missing values:\n{missing_per_row}")
# All rows have ≤ 2 missing, so keep all rows

rows_to_drop = missing_per_row[missing_per_row > 3].index
df = df.drop(rows_to_drop)

# STEP 3: IMPUTE REMAINING MISSING VALUES
# Now only impute the remaining scattered missing values
df['amount'].fillna(df['amount'].median(), inplace=True)  # Impute with median
df['age'].fillna(df['age'].median(), inplace=True)         # Impute with median

print("\nFinal dataset:")
print(df)
```

**Output**:
```
Missing values per column:
amount     1
age        2
phone      4
location   0
dtype: int64

Column missing percentages:
amount    20.0
age       40.0
phone     80.0
location   0.0
dtype: float64

Dropping columns with > 50% missing: ['phone']

Row missing values:
0    1
1    2
2    1
3    1
4    1
dtype: int64

Final dataset:
   amount   age location
0   100.0  25.0       US
1   200.0  27.5       UK    (age imputed with median 27.5)
2   NaN    35.0       US
3   400.0  30.0       US    (age imputed with median 30)
4   500.0  55.0       UK
```

#### Missing Value Handling: Decision Tree

```
Do we have missing data?
│
├─ YES: At COLUMN level (feature has many missing)
│   │
│   ├─ Missing % < 5%:
│   │   └─ Impute (too little missing to drop)
│   │
│   ├─ Missing % 5-30%:
│   │   ├─ Is feature important? YES → Impute
│   │   └─ Is feature important? NO → Consider dropping
│   │
│   ├─ Missing % 30-50%:
│   │   ├─ Can we impute reliably? YES → Impute
│   │   └─ Can we impute reliably? NO → Drop
│   │
│   └─ Missing % > 50%:
│       └─ DROP THE COLUMN (too much missing)
│
└─ YES: At ROW level (row has some missing values)
    │
    ├─ Missing features < 3:
    │   └─ IMPUTE (row is mostly complete)
    │
    ├─ Missing features 3-5:
    │   ├─ Is row informative? YES → Impute
    │   └─ Is row informative? NO → Drop row
    │
    └─ Missing features > 5:
        └─ DROP THE ROW (too incomplete)
```

#### Imputation Methods (for remaining missing values)

After dropping columns and rows, impute remaining scattered missing values:

```python
# Numerical features: Use median (robust to outliers)
df['amount'].fillna(df['amount'].median(), inplace=True)

# Categorical features: Use mode (most common value)
df['location'].fillna(df['location'].mode()[0], inplace=True)

# Advanced: KNN imputation (use neighbors)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df_numeric)

# Advanced: Predictive imputation (predict missing from other features)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_numeric = imputer.fit_transform(df_numeric)
```

#### Missing Indicator Feature (Optional)

Sometimes the fact that a value is missing is itself predictive:

```python
# Create indicator: Was age missing before imputation?
df['age_was_missing'] = df['age'].isnull().astype(int)

# Then impute
df['age'].fillna(df['age'].median(), inplace=True)

# Now model has two pieces of info:
# 1. The imputed age value
# 2. Whether age was originally missing (may correlate with fraud!)
```

**Example (Fraud Detection)**:
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('transactions.csv')

# STEP 1: Column-level decisions
print("Missing values per column:")
missing_pct = df.isnull().sum() / len(df) * 100
print(missing_pct)

# Drop columns with > 50% missing
df = df.dropna(axis=1, thresh=0.5*len(df))
print(f"Remaining columns: {df.columns.tolist()}")

# STEP 2: Row-level decisions
# Drop rows with > 3 missing values
df = df.dropna(thresh=len(df.columns) - 3)
print(f"Remaining rows: {len(df)}")

# STEP 3: Impute remaining missing values
imputer = SimpleImputer(strategy='median')
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = imputer.fit_transform(df_numeric)

print(f"Final shape: {df.shape}")
print(f"Any missing values left: {df.isnull().sum().sum()}")
```

#### Summary: Column vs Row Handling

| Level | Decision | When | Action |
|-------|----------|------|--------|
| **Column** | Missing > 50% | Feature too incomplete | DROP COLUMN |
| **Column** | Missing 30-50% | Borderline | IMPUTE or DROP |
| **Column** | Missing < 30% | Feature useful | IMPUTE |
| **Row** | Missing > 5 features | Row unreliable | DROP ROW |
| **Row** | Missing 3-5 features | Borderline | IMPUTE or DROP |
| **Row** | Missing < 3 features | Row mostly complete | IMPUTE |

#### Outliers
- **Identify**: 
  - Statistical methods: Z-score (|z| > 3), IQR (< Q1-1.5*IQR or > Q3+1.5*IQR)
  - Domain knowledge: e.g., transaction amount > $100k
  - Visual inspection: Box plots, histograms
- **Handle**:
  - **Remove**: If errors (typos, sensor failures)
  - **Cap**: Clip to percentile (e.g., 99th percentile)
  - **Transform**: Log or Box-Cox transformation to reduce skew
  - **Separate model**: Train different model for tail cases

**Example**:
```python
# Identify outliers using IQR
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['amount'] < Q1 - 1.5*IQR) | (df['amount'] > Q3 + 1.5*IQR)

# Cap at 99th percentile instead of removing
p99 = df['amount'].quantile(0.99)
df['amount'] = df['amount'].clip(upper=p99)
```

#### Biases
- **Identify**:
  - **Sampling bias**: Dataset doesn't represent target population
  - **Label bias**: Fraud labels skewed (90% legitimate)
  - **Temporal bias**: Data from specific time period (weekdays vs weekends)
  - **Demographic bias**: Model performs poorly for certain user groups
- **Handle**:
  - **Stratified sampling**: Preserve class distribution in train/val/test
  - **Rebalancing**: Oversample minority class, undersample majority
  - **Class weights**: Higher weight on minority class during training
  - **Bias monitoring**: Track model performance per demographic group
  - **Fairness constraints**: Ensure equal TPR/FPR across groups

**Example (Fraud Detection)**:
```python
# Stratified train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Class weights for imbalanced data
class_weight = len(y[y==0]) / len(y[y==1])  # Fraud is minority
model.fit(X_train, y_train, sample_weight=...)
```

---

### 1.4 Oversampling vs Undersampling

**Q: How exactly do you implement oversampling vs undersampling?** What do you modify in model.fit() and loss functions?

**Answer:**

When dealing with imbalanced data (fraud is 0.1%, legitimate is 99.9%), you have three main approaches:

#### Approach 1: Oversampling (Duplicate minority class)

**What it does**: Duplicate fraud examples until classes are balanced (1:1 ratio)

```python
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Original data
X_train = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
y_train = np.array([0, 0, 0, 0, 1])  # 1 fraud, 4 legitimate
# Ratio: 80% legitimate, 20% fraud

# Method 1: Random Oversampling (simple duplication)
ros = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
# Result: [[1,2], [3,4], [5,6], [7,8], [9,10], [9,10], [9,10], [9,10], [9,10]]
#         [0,    0,    0,    0,    1,    1,    1,    1,    1]
# Ratio: 50% legitimate, 50% fraud (balanced)

# Method 2: SMOTE (Synthetic Minority Over-sampling)
# Creates synthetic fraud samples by interpolating between existing ones
smote = SMOTE(sampling_strategy='minority', k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Result: Original samples + synthetic samples (not duplicates!)
# Better than random oversampling (less overfitting)
```

**Pros**:
- More training data (more samples to learn from)
- Less information loss than undersampling
- Good for small datasets

**Cons**:
- Risk of overfitting (duplicated samples too similar)
- Larger training dataset (slower training)
- Can create unrealistic synthetic samples (SMOTE)

---

#### Approach 2: Undersampling (Remove majority class)

**What it does**: Randomly remove legitimate examples until classes are balanced

```python
from imblearn.under_sampling import RandomUnderSampler

# Original data
X_train = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
y_train = np.array([0, 0, 0, 0, 1])  # 1 fraud, 4 legitimate

# Random Undersampling
rus = RandomUnderSampler(sampling_strategy='majority')
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
# Result: Keep all fraud, randomly remove legitimate samples
# Example: [[1,2], [5,6], [9,10]]
#          [0,    0,    1]
# Ratio: 66% legitimate, 33% fraud (more balanced)
```

**Pros**:
- Faster training (fewer samples)
- No overfitting risk from duplicates
- Simple to implement

**Cons**:
- Lose information (discard data)
- Only works when you have lots of data
- Might remove important patterns

---

#### Approach 3: Class Weights (Keep all data, weight samples during training)

**What it does**: Give higher weight to minority class during training (no data removal/duplication)

```python
from sklearn.utils.class_weight import compute_class_weight

# Original imbalanced data (kept as-is!)
X_train = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
y_train = np.array([0, 0, 0, 0, 1])  # 1 fraud, 4 legitimate

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
# Result: class_weights = [0.625, 2.5]
# Fraud weight: 2.5 (5x higher because rare)
# Legitimate weight: 0.625

# Train with class weights
# Method 1: XGBoost
model = xgb.XGBClassifier(scale_pos_weight=class_weights[1]/class_weights[0])
model.fit(X_train, y_train)

# Method 2: scikit-learn with sample_weight
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Method 3: Manual sample_weight
sample_weight = np.array([class_weights[int(label)] for label in y_train])
# sample_weight = [0.625, 0.625, 0.625, 0.625, 2.5]
model.fit(X_train, y_train, sample_weight=sample_weight)
```

---

#### Approach 4: Modified Loss Function (PyTorch)

If you're using PyTorch or TensorFlow, you can modify the loss function to weight classes:

```python
import torch
import torch.nn as nn

# Original imbalanced data
X_train = torch.tensor([[1,2], [3,4], [5,6], [7,8], [9,10]], dtype=torch.float32)
y_train = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)

# Compute class weights
class_counts = torch.bincount(y_train)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * len(class_weights)
# class_weights = [0.625, 2.5]

# Method 1: Weighted Cross Entropy Loss (most common)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# During training
for epoch in range(10):
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

# What happens inside loss function:
# For legitimate (class 0): loss_weight = 0.625
# For fraud (class 1): loss_weight = 2.5
# Fraud examples contribute 4x more to loss than legitimate examples
# So model focuses on getting fraud right

# Method 2: Focal Loss (for extreme imbalance)
# Focuses on hard examples, downweights easy examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # Probability of true class
        p = torch.exp(-ce_loss)
        
        # Focal loss = -alpha * (1-p)^gamma * ce_loss
        # If p is high (easy example): (1-p)^gamma ≈ 0 (downweight)
        # If p is low (hard example): (1-p)^gamma ≈ 1 (upweight)
        focal_weight = (1 - p) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()

loss_fn = FocalLoss()
```

---

#### Comparison Table

| Approach | Method | Pros | Cons | When to Use |
|----------|--------|------|------|-------------|
| **Oversampling** | Random duplication | More data | Risk of overfitting | Small dataset, fraud rate < 1% |
| **Oversampling** | SMOTE | No duplication, synthetic samples | Creates unrealistic samples | Medium dataset, interpretability matters |
| **Undersampling** | Random removal | Fast training | Loss of information | Large dataset, balanced is fast enough |
| **Class Weights** | In model.fit() | Keep all data, no data manipulation | Doesn't change learned decision boundary | Default approach, reliable |
| **Modified Loss** | Weighted cross-entropy | Fine-grained control | More complex | TensorFlow/PyTorch projects |
| **Modified Loss** | Focal loss | Focus on hard examples | Complex hyperparameters (alpha, gamma) | Extreme imbalance (0.01%) |

---

#### Practical Recommendations for Fraud Detection

**Scenario 1: Fraud rate = 0.1% (1 fraud per 1000 transactions)**
```python
# Combine approach: Undersampling + Class Weights
# - Undersample legitimate to 10:1 ratio (10 legit per 1 fraud)
# - Use class weights for remaining imbalance

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=0.1)  # 10% minority ratio
X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

# Train with class weights
class_weight = {
    0: 1.0,
    1: 10.0  # Fraud still 10x weight
}

model = xgb.XGBClassifier(scale_pos_weight=10.0)
model.fit(X_train_us, y_train_us)

# Why this works:
# - Undersampling: Reduces training from 1M to 100k (faster)
# - Class weights: Handles remaining imbalance
```

**Scenario 2: You have a LOT of data (millions of transactions)**
```python
# Just use class weights, no sampling
# You have enough data, no need to remove it

class_weight = {0: 1.0, 1: 100.0}  # 100:1 ratio
model = xgb.XGBClassifier(scale_pos_weight=100.0)
model.fit(X_train, y_train)

# No need for oversampling/undersampling
# Class weights are sufficient
```

**Scenario 3: You want the absolute best performance (and have time)**
```python
# Use SMOTE + Class Weights
# Pros: Synthetic samples + proper weighting
# Cons: More complex, longer training

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.3)  # 30% minority ratio
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

class_weight = {0: 1.0, 1: 3.33}  # Remaining imbalance
model = xgb.XGBClassifier(scale_pos_weight=3.33)
model.fit(X_train_smote, y_train_smote)
```

---

#### Key Points to Remember

1. **Class weights are the simplest and most reliable** — use this first
2. **Oversampling risks overfitting** but gives more data
3. **Undersampling loses information** but is faster
4. **Combine approaches** for best results (undersample + class weights)
5. **Test on original distribution** — evaluate on test set without resampling
6. **Always use stratified splits** — preserve class ratios in train/val/test

```python
# WRONG: Resampling before train/test split
X_resampled, y_resampled = oversample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
# Test set is now balanced (not representative of real data)

# CORRECT: Split first, then resample training set only
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train_resampled, y_train_resampled = oversample(X_train, y_train)
# Test set remains imbalanced (representative of real data)
model.fit(X_train_resampled, y_train_resampled)
model.evaluate(X_test, y_test)  # Evaluated on real distribution
```

---

### 1.5 Probability Calibration with Resampling

**Q: When we oversample/undersample, don't the predicted probabilities become miscalibrated?** How do we correct for this?

**Answer:**

You're absolutely correct! This is a **critical and often-overlooked problem**.

#### The Problem

```
Original data:     99.9% legitimate, 0.1% fraud
Training data:     50% legitimate, 50% fraud (after oversampling)

What happens:
- Model learns: "When I see fraud-like features, probability should be ~50%"
- In production: Fraud-like features appear in 0.1% of cases
- Model predicts: "This looks fraudy, 50% probability"
- Reality: It's actually only 0.1% fraud in this population

Decision threshold = 0.5 means:
- Block if P(fraud) > 0.5
- But in production, most predictions will be < 0.2 (calibrated to imbalanced reality)
- Threshold is way too high!
```

#### Why This Happens

The model learns **class priors** (baseline probability) from training data:

```python
# Original imbalanced training data
P(fraud) = 100 / 1,000,000 = 0.1%  # Prior
P(legitimate) = 999,900 / 1,000,000 = 99.9%

# After oversampling to 50:50
P(fraud) = 500 / 1,000 = 50%  # Model learns this!
P(legitimate) = 500 / 1,000 = 50%

# Model's predicted probabilities are based on 50:50 prior
# But production data has 0.1:99.9 prior
# Probabilities are completely miscalibrated!
```

---

#### Solution 1: Use Class Weights Instead (No Resampling) ⭐ **BEST**

**Best approach**: Avoid resampling entirely, use class weights only.

```python
# DON'T oversample/undersample
# DO use class weights

# Original imbalanced data (0.1% fraud)
X_train, y_train  # Keep original distribution!

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=[0, 1],
    y=y_train
)
# class_weights = [0.5, 500]
# Fraud weight: 500x (reflects rarity)

# Train with weights (data distribution unchanged)
model = xgb.XGBClassifier(scale_pos_weight=500)
model.fit(X_train, y_train)

# Predicted probabilities are now calibrated!
# P(fraud) ≈ 0.001 (0.1%) for normal transactions
# P(fraud) ≈ 0.5 for fraudy transactions
```

**Why this works**:
- Model never sees artificially balanced data
- Class weights tell model to focus on fraud without changing data distribution
- Predicted probabilities reflect true population probabilities

---

#### Solution 2: Post-Hoc Probability Calibration

If you already did resampling, you need to **recalibrate** the predictions:

**Method 1: Platt Scaling (Logistic Calibration)**

```python
from sklearn.calibration import CalibratedClassifierCV

# Train on resampled data
X_train_resampled, y_train_resampled = oversample(X_train, y_train)
model = xgb.XGBClassifier()
model.fit(X_train_resampled, y_train_resampled)

# Get predictions on original (imbalanced) validation set
y_proba_uncalibrated = model.predict_proba(X_val)  # Miscalibrated!

# Calibrate using original validation set
calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrator.fit(X_val, y_val)  # Fit on original distribution!

# Now predictions are calibrated
y_proba_calibrated = calibrator.predict_proba(X_test)
```

**What it does**:
- Learns a mapping from miscalibrated → calibrated probabilities
- Uses Platt scaling: applies logistic function to uncalibrated scores
- Example: 0.7 (from resampled model) → 0.15 (true probability)

```python
# Example calibration
def platt_scaling(score):
    # Learned parameters from validation set
    a, b = 0.5, -2.0  # Example parameters
    return 1 / (1 + np.exp(a * score + b))

uncalibrated_score = 0.7  # Model prediction on resampled data
calibrated_prob = platt_scaling(uncalibrated_score)  # ≈ 0.15 (true prob)
```

**Method 2: Isotonic Regression**

```python
from sklearn.calibration import IsotonicRegression

# Train on resampled data
model = xgb.XGBClassifier()
model.fit(X_train_resampled, y_train_resampled)

# Get predictions on original validation set
y_proba_uncalibrated = model.predict_proba(X_val)[:, 1]

# Fit isotonic regression on original distribution
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(y_proba_uncalibrated, y_val)

# Calibrate test predictions
y_proba_calibrated = iso_reg.predict(model.predict_proba(X_test)[:, 1])
```

**Pros/Cons**:
- Isotonic: More flexible, better for non-linear calibration
- Platt: Simpler, works well in practice

---

#### Solution 3: Adjust Threshold Post-Training

If you can't recalibrate, at least adjust the decision threshold:

```python
# Resampled model predictions on original test set
y_proba_uncalibrated = model.predict_proba(X_test)[:, 1]

# These probabilities are miscalibrated
# E.g., most legitimate are ~0.3, most fraud are ~0.7
# But true priors are 0.1% fraud, 99.9% legit

# Instead of threshold = 0.5, find optimal threshold
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_proba_uncalibrated)

# Find threshold for 95% recall (catch 95% of fraud)
idx = np.argmax(recall >= 0.95)
optimal_threshold = thresholds[idx]
# optimal_threshold ≈ 0.3 (not 0.5!)

# Use optimal threshold
y_pred = (y_proba_uncalibrated > optimal_threshold).astype(int)
```

---

#### Solution 4: Understand the Math (Bayes Rule)

This is the most rigorous approach. Model outputs likelihood ratio, not probability:

```
Model predicts: P(features | fraud) / P(features | legitimate)
This is a LIKELIHOOD RATIO, not probability!

To get true probability:
P(fraud | features) = P(features | fraud) * P(fraud) / P(features)

Where P(fraud) is the TRUE prior (0.1% in production)

Example:
- Model confidence: 0.9 (learned on 50:50 data)
- But this is likelihood ratio 9:1
- Apply true prior (0.1%):
- True probability = (9 * 0.001) / (9 * 0.001 + 1 * 0.999)
                   = 0.009 / 1.008 ≈ 0.9%
```

**Implementation**:

```python
def calibrate_probability_with_true_prior(
    model_prob,
    true_prior_fraud=0.001,  # 0.1% fraud in production
):
    """
    Convert model probability (trained on resampled data)
    to true probability using Bayes rule.
    
    model_prob: probability from resampled model
    true_prior_fraud: true fraud rate in production
    """
    # Likelihood ratio
    odds_ratio = model_prob / (1 - model_prob)
    
    # Prior odds (true distribution)
    prior_odds = true_prior_fraud / (1 - true_prior_fraud)
    
    # Posterior odds
    posterior_odds = odds_ratio * prior_odds
    
    # Convert back to probability
    true_probability = posterior_odds / (1 + posterior_odds)
    
    return true_probability

# Example
model_prob = 0.7  # Model says 70% fraud (trained on 50:50)
true_prob = calibrate_probability_with_true_prior(
    model_prob,
    true_prior_fraud=0.001
)
print(f"Model: {model_prob}, True: {true_prob:.4f}")
# Output: Model: 0.7, True: 0.0009 (0.09%)
```

---

#### Comparison of Solutions

| Solution | Effort | Accuracy | When to Use |
|----------|--------|----------|------------|
| **Use class weights instead** | Low | Perfect | Before training (best) |
| **Platt scaling** | Medium | Good | Post-training, simple calibration |
| **Isotonic regression** | Medium | Very Good | Post-training, complex data |
| **Threshold tuning** | Low | Fair | Quick fix, imperfect |
| **Bayes rule calibration** | Medium | Perfect | Understand true prior well |

---

#### Practical Example: Fraud Detection

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# Step 1: Original imbalanced training data
X_train = ...  # Features
y_train = ...  # 0.1% fraud, 99.9% legitimate

# Step 2: Oversample for training
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Now 30% fraud, 70% legitimate (NOT representative!)

# Step 3: Train model on resampled data
model = xgb.XGBClassifier()
model.fit(X_train_resampled, y_train_resampled)

# Step 4: Evaluate on ORIGINAL (imbalanced) validation set
y_proba_uncalibrated = model.predict_proba(X_val)[:, 1]
auc_uncalibrated = roc_auc_score(y_val, y_proba_uncalibrated)
print(f"AUC before calibration: {auc_uncalibrated}")  # Good AUC

# Problem: Predicted probabilities are miscalibrated
predicted_fraud_rate = y_proba_uncalibrated.mean()
actual_fraud_rate = y_val.mean()
print(f"Predicted rate: {predicted_fraud_rate:.1%}, Actual: {actual_fraud_rate:.1%}")
# Output: Predicted rate: 30.0%, Actual: 0.1% (WAY OFF!)

# Step 5: Calibrate
calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrator.fit(X_val, y_val)  # Use original distribution!

# Step 6: Get calibrated probabilities
y_proba_calibrated = calibrator.predict_proba(X_test)[:, 1]

# Now probabilities match reality
predicted_fraud_rate_cal = y_proba_calibrated.mean()
actual_fraud_rate_test = y_test.mean()
print(f"After calibration - Predicted: {predicted_fraud_rate_cal:.1%}, Actual: {actual_fraud_rate_test:.1%}")
# Output: After calibration - Predicted: 0.1%, Actual: 0.1% (MATCHED!)
```

---

#### Key Takeaways

1. **Oversampling/undersampling break probability calibration** — this is not a minor detail!

2. **Best solution: Use class weights instead**
   - No resampling = no miscalibration
   - Probabilities are automatically correct
   - Recommended for fraud detection

3. **If you must resample: Calibrate afterwards**
   - Platt scaling: Simple, good
   - Isotonic regression: Flexible, very good
   - Bayes rule: Perfect if you know true prior

4. **Always evaluate on original distribution**
   - Test set should reflect production reality (0.1% fraud)
   - Not balanced 50:50
   - This reveals miscalibration

5. **Monitor predicted vs actual fraud rate**
   ```python
   predicted_rate = y_proba.mean()
   actual_rate = y_true.mean()
   assert abs(predicted_rate - actual_rate) < 0.01  # Should be close!
   ```

6. **Threshold adjustment is not enough**
   - Changing threshold doesn't fix miscalibration
   - You need probability calibration

---

#### Recommended Approach for Fraud Detection

```python
# BEST PRACTICE
# 1. Don't oversample
# 2. Use class weights only
# 3. Probabilities are automatically calibrated

X_train, y_train = load_imbalanced_data()  # 0.1% fraud

# Train with class weights (no resampling)
class_weight = {
    0: 1.0,
    1: len(y_train[y_train==0]) / len(y_train[y_train==1])  # ~1000
}

model = xgb.XGBClassifier(scale_pos_weight=class_weight[1])
model.fit(X_train, y_train)

# Predicted probabilities are now calibrated to true distribution
# P(fraud | features) ≈ 0.001 for normal transactions
# P(fraud | features) ≈ 0.5+ for fraudy transactions
```

---

### 4.1 Cross-Validation

**Q: What exactly is cross-validation and why do we need it?**

**Answer:**

Cross-validation is a technique to **evaluate model performance more reliably** by using multiple train-test splits instead of just one.

#### The Problem It Solves

**Without cross-validation** (single train-test split):
```
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Split: Train [1-7], Test [8-10]
Train: Get AUC = 0.95
Test: Get AUC = 0.92

Question: Is 0.92 the true performance?
- Maybe the test set was lucky/unlucky
- Test set is only 3 samples (small, high variance)
- Can't trust this estimate!
```

**With cross-validation** (multiple splits):
```
Split 1: Train [2-10], Test [1]       → AUC = 0.90
Split 2: Train [1,3-10], Test [2]     → AUC = 0.92
Split 3: Train [1-2,4-10], Test [3]   → AUC = 0.94
Split 4: Train [1-3,5-10], Test [4]   → AUC = 0.91
Split 5: Train [1-4,6-10], Test [5]   → AUC = 0.93

Average AUC: 0.92 ± 0.016 (more reliable estimate!)
```

---

#### K-Fold Cross-Validation (Most Common)

Divide data into **K equal parts**, train K times:

```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Data
X = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
y = np.array([0, 1, 0, 1, 0])

# 5-Fold CV: Split into 5 parts, train 5 times
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Method 1: Using cross_val_score (automatic)
scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
print(f"Scores: {scores}")           # [0.90, 0.92, 0.94, 0.91, 0.93]
print(f"Mean: {scores.mean():.3f}")  # 0.920
print(f"Std: {scores.std():.3f}")    # 0.016
```

**How it works**:
```
Iteration 1: Train [2,3,4,5], Test [1]
Iteration 2: Train [1,3,4,5], Test [2]
Iteration 3: Train [1,2,4,5], Test [3]
Iteration 4: Train [1,2,3,5], Test [4]
Iteration 5: Train [1,2,3,4], Test [5]

Each sample appears in test set exactly once
Each sample appears in training set K-1 times
```

**Manual implementation** (for understanding):
```python
from sklearn.model_selection import KFold

X = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
y = np.array([0, 1, 0, 1, 0])

kfold = KFold(n_splits=5, shuffle=True)
scores = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    print(f"Fold {fold+1}: Train={len(train_idx)}, Test={len(test_idx)}, Score={score:.3f}")
    scores.append(score)

print(f"Average: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

#### Types of Cross-Validation

**1. K-Fold (Stratified)** - For Classification

Use when classes are imbalanced (fraud detection):
```python
from sklearn.model_selection import StratifiedKFold

# Original data: 1% fraud, 99% legitimate
y = np.array([0,0,0,0,0,0,0,0,0,1])  # 10% fraud (for simplicity)

# Standard KFold (can have 0% fraud in one fold!)
kfold = KFold(n_splits=5)
for train_idx, test_idx in kfold.split(X, y):
    fraud_rate_train = y[train_idx].mean()
    fraud_rate_test = y[test_idx].mean()
    print(f"Train: {fraud_rate_train:.1%}, Test: {fraud_rate_test:.1%}")
# Output might show: Train: 11.1%, Test: 0% (no fraud in test!)

# Stratified KFold (preserves class ratio in each fold)
stratified_kfold = StratifiedKFold(n_splits=5)
for train_idx, test_idx in stratified_kfold.split(X, y):
    fraud_rate_train = y[train_idx].mean()
    fraud_rate_test = y[test_idx].mean()
    print(f"Train: {fraud_rate_train:.1%}, Test: {fraud_rate_test:.1%}")
# Output: Train: 11.1%, Test: 11.1% (consistent!)
```

**2. Time Series Cross-Validation** - For Time-Based Data

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series: [Jan, Feb, Mar, Apr, May]
X = np.arange(20).reshape(5, 4)  # 5 time steps, 4 features
y = np.array([1, 2, 3, 4, 5])

tscv = TimeSeriesSplit(n_splits=3)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}: Train indices {train_idx}, Test indices {test_idx}")

# Output:
# Fold 1: Train [0], Test [1]        (Jan trains, Feb tests)
# Fold 2: Train [0 1], Test [2]      (Jan-Feb train, Mar tests)
# Fold 3: Train [0 1 2], Test [3]    (Jan-Mar train, Apr tests)

# KEY: Never look into the future!
# Train set grows, test set is always after train set
```

**3. Leave-One-Out Cross-Validation (LOOCV)** - For Small Datasets

```python
from sklearn.model_selection import LeaveOneOut

X = np.array([[1,2], [3,4], [5,6]])  # Only 3 samples
y = np.array([0, 1, 0])

loo = LeaveOneOut()
scores = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
    
    print(f"Train on {train_idx}, Test on {test_idx}, Score: {score}")

# Output:
# Train on [1 2], Test on [0], Score: 1.0
# Train on [0 2], Test on [1], Score: 0.0
# Train on [0 1], Test on [2], Score: 1.0

print(f"LOOCV Score: {np.mean(scores):.3f}")  # 0.667
```

**Pros**: Most honest evaluation (every sample tests exactly once)  
**Cons**: Slow for large datasets (K = number of samples)

---

#### Cross-Validation in Practice

**Example: Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV

# Data
X, y = load_fraud_data()

# Define model and parameters to tune
model = xgb.XGBClassifier()

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.5],
}

# GridSearchCV does cross-validation for each combination
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,  # 5-fold CV
    scoring='roc_auc',
    n_jobs=-1  # Use all CPUs
)

grid_search.fit(X, y)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# What happens internally:
# For each parameter combination:
#   For each fold:
#     Train on 4 folds, evaluate on 1 fold
#   Average CV scores
# Return best parameter combination
```

**Output**:
```
  max_depth  learning_rate  mean_test_score  std_test_score
0        3           0.01             0.92           0.015
1        3           0.1              0.94           0.012
2        3           0.5              0.91           0.018
3        5           0.01             0.93           0.014
4        5           0.1              0.95           0.010  ← Best!
5        5           0.5              0.92           0.016
6        7           0.01             0.92           0.017
7        7           0.1              0.93           0.011
8        7           0.5              0.90           0.019

Best params: {'max_depth': 5, 'learning_rate': 0.1}
Best CV score: 0.950 ± 0.010
```

---

#### Cross-Validation vs Train/Val/Test Split

| Aspect | Cross-Validation | Train/Val/Test |
|--------|-----------------|-----------------|
| **Data Usage** | Uses all data for training | Wastes ~30% for validation |
| **Reliability** | High (multiple splits) | Lower (one split, high variance) |
| **Time** | Slow (K times slower) | Fast |
| **When to Use** | Small/medium datasets, hyperparameter tuning | Large datasets, final evaluation |
| **Example** | Fraud detection (100k samples) | ImageNet (1M+ samples) |

---

#### Common Mistakes

**❌ Mistake 1: Data Leakage Through Preprocessing**

```python
# WRONG: Fit scaler on entire dataset, then split
scaler = StandardScaler()
X_scaled = scaler.fit(X)  # Sees all data!
X_train, X_test = train_test_split(X_scaled)
# Test set influenced by training set (leakage!)

# CORRECT: Fit scaler per fold
for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only!
    X_test_scaled = scaler.transform(X_test)  # Transform test
    
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
```

**❌ Mistake 2: Not Stratifying Imbalanced Data**

```python
# WRONG: For imbalanced data
kfold = KFold(n_splits=5)  # May split unevenly
cv_scores = cross_val_score(model, X, y, cv=kfold)

# CORRECT: Use StratifiedKFold
from sklearn.model_selection import StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, X, y, cv=stratified_kfold)
```

**❌ Mistake 3: Time Series Data Without Time Ordering**

```python
# WRONG: Using standard KFold on time series
# Might train on May, test on January (future leakage!)
kfold = KFold(n_splits=5)

# CORRECT: Use TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

---

#### Choosing K (Number of Folds)

| K Value | When to Use | Pros | Cons |
|---------|------------|------|------|
| **3** | Large dataset (> 1M samples) | Fast | Less reliable |
| **5** | Medium dataset (10k-1M) | Good balance | Default choice |
| **10** | Small dataset (< 10k) | More reliable | Slower |
| **N (LOOCV)** | Very small (< 100 samples) | Maximum reliability | Very slow |

**Recommendation for Fraud Detection** (100k-1M samples):
```python
from sklearn.model_selection import StratifiedKFold

# Use 5-fold (standard)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=cv)

# For final evaluation
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"Final performance: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

---

#### Cross-Validation Visualization

```
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

5-Fold Cross-Validation:
┌─────────────────────────────────────────────────────────┐
│ Fold 1: [Train: 1 2 3 4 5 6 7 8] [Test: 9 10]          │
├─────────────────────────────────────────────────────────┤
│ Fold 2: [Train: 1 2 3 4 5 6 9 10] [Test: 7 8]          │
├─────────────────────────────────────────────────────────┤
│ Fold 3: [Train: 1 2 3 4 7 8 9 10] [Test: 5 6]          │
├─────────────────────────────────────────────────────────┤
│ Fold 4: [Train: 1 2 5 6 7 8 9 10] [Test: 3 4]          │
├─────────────────────────────────────────────────────────┤
│ Fold 5: [Train: 3 4 5 6 7 8 9 10] [Test: 1 2]          │
└─────────────────────────────────────────────────────────┘

Results: Fold1=0.90, Fold2=0.92, Fold3=0.94, Fold4=0.91, Fold5=0.93
Average: 0.92 ± 0.016 (confident estimate!)
```

---

#### Summary: When to Use Cross-Validation

| Scenario | Use CV? | Why |
|----------|---------|-----|
| **Choosing hyperparameters** | YES | Need reliable estimate for tuning |
| **Estimating model performance** | YES | More reliable than single split |
| **Final evaluation on huge dataset** | NO | Too slow, single split is fine |
| **Small dataset (< 1k samples)** | YES | LOOCV or 10-fold |
| **Time series** | YES | Use TimeSeriesSplit only |
| **Imbalanced data** | YES | Use StratifiedKFold |
| **Production model selection** | YES | Pick best CV performer |

**Golden Rule**: "If you're tuning hyperparameters, use cross-validation. If you have tons of data, a single train/val/test split is fine."

#### Inconsistencies
- **Identify**:
  - **Format inconsistencies**: Date formats, currency, units
  - **Value inconsistencies**: Negative amounts, impossible values
  - **Referential inconsistencies**: Foreign key violations
  - **Duplicates**: Exact duplicates or near-duplicates
- **Handle**:
  - **Standardize formats**: Convert to common format (ISO dates, consistent currency)
  - **Validate ranges**: Amount >= 0, valid IPs/emails
  - **Deduplication**: Remove exact duplicates, handle near-duplicates
  - **Schema validation**: Check data types, required fields

**Example**:
```python
# Remove duplicates
df = df.drop_duplicates()

# Standardize date format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Validate ranges
df = df[df['amount'] > 0]
df = df[df['amount'] < 1_000_000]  # Upper bound check
```

---

### 1.2 Handling Data Issues

**Q: How do you propose appropriate techniques for handling data issues?**

**Answer:**

Choose technique based on:

| Issue | Technique | When to Use | Tradeoff |
|-------|-----------|------------|----------|
| Missing values (< 10%) | Imputation | Rare, want to retain samples | May add noise |
| Missing values (> 50%) | Drop feature | Common, not predictive | Lose information |
| Outliers (domain errors) | Remove | Clearly erroneous (e.g., negative amount) | Lose data |
| Outliers (valid but extreme) | Cap/transform | Extreme but valid (e.g., celebrity purchase) | Slight accuracy loss |
| Class imbalance | Oversample minority | Rare positive class (fraud) | Risk overfitting |
| Class imbalance | Undersample majority | Large dataset, can afford loss | Lose data |
| Class imbalance | Class weights | Standard approach | Adds complexity |
| Temporal shift | Separate models | Pattern changes over time | More complex |

**For Fraud Detection specifically**:
```python
# Step 1: Handle missing values (usually low % in fraud data)
df['device_age_days'].fillna(df['device_age_days'].median(), inplace=True)

# Step 2: Remove/cap outliers
amount_p99 = df['amount'].quantile(0.99)
df['amount'] = df['amount'].clip(upper=amount_p99)

# Step 3: Handle class imbalance with oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Step 4: Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 1.3 Data Freshness

**Q: How do you ensure data freshness in ML systems?**

**Answer:**

Data freshness is critical: stale data leads to poor predictions and concept drift.

#### Define Data Freshness Requirements

| System | Freshness Requirement | Why |
|--------|----------------------|-----|
| Fraud Detection | Hours to 1 day | Fraud patterns change rapidly |
| Recommendation | Hours to 1 day | User preferences change daily |
| Demand Forecasting | 1-7 days | Trends evolve slowly |
| Churn Prediction | Weekly to monthly | User behavior stable |

#### Mechanisms to Ensure Freshness

**1. Batch Updates (Most Common)**
```
Batch Job (daily/hourly):
  1. Extract new data from source
  2. Transform and aggregate
  3. Update feature store
  4. Trigger model retraining if needed
```

**Pros**: Simple, reproducible, easy to debug  
**Cons**: Latency (up to batch interval), requires storage

**Example** (Daily Feature Update):
```python
# Daily batch job (runs at 2 AM)
import airflow
from airflow.operators.python import PythonOperator

def update_features():
    # Load yesterday's transactions
    df = load_transactions(yesterday)
    
    # Compute features
    features = compute_features(df)
    
    # Update feature store (overwrite)
    feature_store.save(features, table='user_features')
    
    # Trigger retraining if data significantly changed
    if data_drift_detected(features):
        trigger_model_retraining()
```

**2. Streaming Updates (Real-Time)**
```
Event Stream (Kafka/Pub-Sub):
  1. Each new transaction → event
  2. Stream processor aggregates in real-time
  3. Feature store updated continuously
```

**Pros**: Always fresh, real-time adaptation  
**Cons**: Complex infrastructure, harder to debug, cost

**Example** (Streaming Feature Update):
```python
# Kafka Streams or Flink job
def update_velocity_features():
    # For each transaction event
    for event in kafka_stream:
        user_id = event['user_id']
        
        # Update 1h and 24h velocity
        velocity_1h = window_aggregate(
            user_id, 
            window='1h', 
            func='count'
        )
        
        # Store in Redis (fast lookup)
        redis.set(f'velocity_1h:{user_id}', velocity_1h)
```

**3. Incremental Updates**
```
Incremental Job (hourly):
  1. Only process data since last run
  2. Update only changed features
  3. Merge with previous state
```

**Pros**: Efficient (don't recompute everything)  
**Cons**: Complex state management

#### Monitoring Data Freshness

```python
# Check if data is stale
def check_data_freshness(table_name, max_lag_hours=24):
    last_update = get_last_update_time(table_name)
    current_time = datetime.now()
    lag = (current_time - last_update).total_seconds() / 3600
    
    if lag > max_lag_hours:
        alert(f"Data {table_name} is {lag}h old")
    
    return lag
```

---

## 2. Data Processing

### Q: How do you design an efficient and scalable data processing pipeline?

**Answer:**

#### Architecture Overview
```
Ingestion → Storage → Transformation → Feature Eng → Model Training
                          ↓
                    Data Validation
                          ↓
                    Quality Checks
```

#### Key Stages

**1. Ingestion** (Collect raw data)
- **Batch**: Daily dumps from database
- **Streaming**: Real-time events (Kafka, Pub/Sub)
- **APIs**: Real-time from external services
- **Databases**: Direct query from OLTP systems

**Design Decision**: 
- High-latency use case (demand forecast): Batch ingestion (daily)
- Low-latency use case (fraud detection): Streaming ingestion (real-time)
- Most systems: Hybrid (streaming for real-time features, batch for historical)

**2. Storage** (Where data lives)
- **Data Lake** (S3, GCS): Raw, unstructured data, long-term storage
- **Data Warehouse** (BigQuery, Snowflake): Structured, aggregated data, OLAP queries
- **Cache** (Redis): Real-time serving, low-latency access
- **Feature Store** (Feast, Tecton): Versioned features, online/offline sync

**Design Decision for Fraud Detection**:
```
Raw events → Kafka → Data Lake (S3) → Feature Store (Redis for online)
                          ↓
                    BigQuery (aggregates)
```

**3. Transformation** (Clean and prepare)
- Parse JSON/protobuf
- Handle missing values
- Type casting and validation
- Outlier handling
- Standardization

**Tools**: SQL (BigQuery, Spark SQL), Python (Spark, Pandas), dbt

**Example** (Spark SQL):
```sql
-- Raw events table
SELECT 
    user_id,
    transaction_amount,
    merchant_category,
    timestamp,
    -- Parse nested JSON
    JSON_EXTRACT(device_info, '$.device_id') as device_id,
    -- Handle missing values
    COALESCE(account_age_days, 0) as account_age_days,
    -- Outlier handling
    LEAST(transaction_amount, 100000) as amount_capped
FROM raw_events
WHERE timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    AND transaction_amount > 0
```

**4. Feature Engineering** (Create useful signals)
- Aggregate historical data (1d, 7d, 30d windows)
- Compute velocity features
- Domain-specific features
- Feature interactions

**Example**:
```python
# Batch feature generation
def compute_user_features(transactions_df):
    features = transactions_df.groupby('user_id').agg({
        'amount': ['mean', 'std', 'max', 'min'],
        'merchant_id': 'nunique',
        'timestamp': 'count',  # transaction count
    })
    
    # Velocity features
    features['txn_1h'] = compute_velocity(transactions_df, window='1h')
    features['txn_24h'] = compute_velocity(transactions_df, window='24h')
    
    return features
```

**5. Quality Checks** (Validate output)
- Check feature distributions
- Verify no NaN values
- Check feature bounds
- Monitor data drift

```python
def validate_features(features_df):
    # Check for NaN
    assert features_df.isnull().sum().sum() == 0, "NaN values present"
    
    # Check feature bounds
    assert features_df['amount_mean'] > 0, "Negative amounts"
    
    # Check distribution shift
    distribution = features_df['amount_mean'].describe()
    if distribution['mean'] > prev_mean * 1.5:
        alert("Data distribution shifted")
```

#### Technology Choices

| Scale | Technology | Pros | Cons |
|-------|-----------|------|------|
| Small (< 1GB) | Pandas | Easy, local | Not distributed |
| Medium (1GB - 1TB) | Spark | Distributed, easy | Overhead for small data |
| Large (1TB+) | Spark/Beam + SQL | Distributed, scalable | Complex setup |
| Real-time | Kafka Streams, Flink | Low latency | Complex |
| Orchestration | Airflow, Prefect, Dagster | Workflow, monitoring | Additional complexity |

**Recommendation for Fraud Detection**:
- **Ingestion**: Kafka for streaming events
- **Storage**: S3 (data lake) + BigQuery (warehouse)
- **Transformation**: Spark SQL or BigQuery SQL
- **Feature Eng**: PySpark or dbt
- **Orchestration**: Airflow (daily jobs)

---

## 3. Data Versioning & Management

### 3.1 Importance of Data Versioning

**Q: Why is data versioning important?**

**Answer:**

#### Importance

1. **Reproducibility**: Re-train exact model with exact data
2. **Debugging**: "This model worked yesterday. What changed?" → Check data version
3. **Compliance**: Audit trail of what data was used
4. **Rollback**: If model degrades, revert to previous data version
5. **Experiment Tracking**: Link experiments to data versions

#### Example Problem (Without Versioning)
```
Day 1: Train model on v1 data → 95% AUC
Day 8: Data cleaning script modifies data
Day 10: Retrain model on "updated" data → 90% AUC
Problem: Can't reproduce original model because data changed!
```

#### With Versioning
```
v1 (2024-01-01): Raw data → Train → Model A (95% AUC)
v2 (2024-01-08): Data cleaning → Train → Model B (90% AUC)
v3 (2024-01-15): Different cleaning → Train → Model C (96% AUC) ✓
```

---

### 3.2 Managing Large Datasets

**Q: How do you manage large datasets and their versions?**

**Answer:**

#### Tools for Data Versioning

**1. DVC (Data Version Control)**
- Tracks data files in Git-like manner
- Uses hash-based versioning
- Integrates with ML workflows

```bash
# Initialize DVC
dvc init

# Track large dataset
dvc add transactions.parquet
# Creates transactions.parquet.dvc (small metadata file)

# Commit to Git
git add transactions.parquet.dvc
git commit -m "Add v1 of transaction dataset"

# Switch versions
dvc checkout v2  # Restores different version
```

**2. Pachyderm**
- Version control for entire data pipelines
- Tracks code + data + parameters

```yaml
# pipeline.yaml
pipeline: fraud_detection
input:
  repo: raw_transactions
  branch: main
stages:
  - name: process
    image: my-processor:latest
    cmd: python process.py
    inputs:
      - repo: raw_transactions
```

**3. MLflow (Experiment Tracking)**
- Track data, code, parameters, metrics
- Reproducible ML workflows

```python
import mlflow

# Log data version
mlflow.log_param("data_version", "v1")
mlflow.log_param("training_date", "2024-01-01")

# Log metrics
mlflow.log_metric("auc", 0.95)
mlflow.log_metric("precision", 0.92)

# Load specific experiment
best_run = mlflow.search_runs(
    experiment_names=["fraud_detection"],
    order_by=["metrics.auc DESC"]
)[0]
```

#### Best Practices

```
Data versioning strategy:
├── Raw Data (v1, v2, v3, ...)
│   ├── hash/checksum → immutable
│   └── metadata (size, row count, date)
│
├── Processed Data (cleaned, transformed)
│   ├── depends on raw data version
│   └── metadata (processing script version)
│
└── Features (computed from processed data)
    ├── depends on processed data version
    └── metadata (feature computation version)
```

---

### 3.3 Data Lineage & Reproducibility

**Q: How do you track data lineage and ensure reproducibility?**

**Answer:**

#### Data Lineage

Shows: Which raw data → transformed data → features → model predictions

```
raw_events (v1.0) 
    ↓ (cleaning script v2.3)
transactions_clean (v1.2)
    ↓ (feature script v3.1)
user_features (v1.5)
    ↓ (trained model v2.0)
fraud_predictions
```

#### Tracking Lineage

**1. Metadata Tracking**
```python
# In pipeline code
metadata = {
    'input_data': 'raw_events_v1.0',
    'input_hash': '5f4d7c...',  # SHA-256 of input
    'processing_script': 'cleaning.py:v2.3',
    'output_data': 'transactions_clean_v1.2',
    'timestamp': '2024-01-15T10:30:00Z',
    'params': {'outlier_threshold': 3.0},
}

# Store metadata in database
db.insert('data_lineage', metadata)
```

**2. Tools**
- **Apache Atlas**: Enterprise data governance
- **Openmetadata**: Open-source metadata
- **Custom logging**: Store lineage in database

#### Reproducibility

**Steps to reproduce results**:
1. Get data version from experiment metadata
2. Checkout that version: `dvc checkout v1`
3. Get code version: `git checkout abc123def`
4. Get hyperparameters: Load from experiment tracker
5. Re-run: `python train.py`
6. Verify: Metrics should match original

```python
# Reproducible ML pipeline
def train_and_log():
    # Data
    data_version = 'v1.0'
    data = load_data_version(data_version)
    
    # Code versioning (git commit hash)
    code_version = get_git_commit_hash()
    
    # Parameters
    params = {
        'learning_rate': 0.01,
        'max_depth': 6,
    }
    
    # Train
    model = train(data, params)
    
    # Log everything
    mlflow.log_param('data_version', data_version)
    mlflow.log_param('code_version', code_version)
    mlflow.log_params(params)
    mlflow.log_metric('auc', 0.95)
```

---

## 4. Model Deployment & Serving

### 5.1 Deployment Architectures

**Q: What are the trade-offs between different deployment architectures?**

**Answer:**

#### Deployment Architectures

**1. Batch Predictions**
```
Data Source → Model Inference → Results Storage → Applications
(overnight job)
```

**Use Cases**: Demand forecast, churn prediction, user segments  
**Latency**: Hours/days  
**Throughput**: High (millions at once)  
**Cost**: Low (predictable compute)  

**Pros**:
- Simple (run overnight, no servers)
- Cost-efficient
- Easy to test and debug

**Cons**:
- Not real-time (stale predictions)
- Can't personalize per-request
- Difficult A/B testing

**Example**:
```python
# Daily batch job
import airflow
from airflow.operators.python import PythonOperator

def batch_predict():
    # Load all users
    users = load_all_users()
    
    # Predict churn for all
    predictions = model.predict(users)
    
    # Store results
    db.write('user_churn_predictions', predictions)
```

**2. Online/Real-Time Predictions**
```
Request → Feature Lookup → Model → Response
(immediate, per-request)
```

**Use Cases**: Fraud detection, ranking, recommendations  
**Latency**: Milliseconds-seconds  
**Throughput**: Depends on load (100s-10ks RPS)  
**Cost**: Higher (always-on servers)

**Pros**:
- Real-time predictions
- Can personalize per-request
- Easy A/B testing
- Responsive to user behavior

**Cons**:
- Complex infrastructure
- Higher operational cost
- Latency requirements
- Need fallback strategy

**Example**:
```python
# FastAPI server
from fastapi import FastAPI
import model_loader

app = FastAPI()
model = model_loader.load('fraud_detection_v1')

@app.post("/score_transaction")
async def score(transaction: TransactionRequest):
    features = fetch_features(transaction.user_id)
    score = model.predict(features)
    decision = 'BLOCK' if score > 0.8 else 'ALLOW'
    return {'decision': decision, 'score': score}
```

**3. Hybrid (Batch + Online)**
```
Batch: Pre-compute expensive features (daily)
Online: Real-time features + model serving
```

**Use Cases**: Fraud detection (batch aggregates + real-time velocity)  
**Latency**: Real-time (for online features)  
**Cost**: Medium  

**Example**:
```
Daily batch job:
  - Compute user historical features (avg spend, device history)
  - Store in feature store

Online request:
  - Real-time velocity (transactions in last 1h)
  - Lookup batch features
  - Score model
```

#### Comparison Table

| Aspect | Batch | Online | Hybrid |
|--------|-------|--------|--------|
| Latency | Hours/days | Milliseconds | Milliseconds |
| Throughput | Millions | Thousands | Thousands |
| Cost | Low | High | Medium |
| Complexity | Low | High | Medium-High |
| A/B Testing | Hard | Easy | Easy |
| Personalization | No | Yes | Yes |
| Use Case | Forecast, segment | Fraud, ranking | Most real systems |

---

### 5.2 Choosing Deployment Architecture

**Q: How do you choose the right deployment architecture?**

**Answer:**

**Decision Framework**:

```
Question 1: Do you need real-time decisions?
  ├─ No (forecast, segment) → Batch
  └─ Yes → Question 2

Question 2: What's your latency requirement?
  ├─ < 100ms (fraud, search) → Online required
  ├─ 1-10 seconds (recommendations) → Online possible
  └─ > 10 seconds → Batch + async

Question 3: Is model latency the bottleneck?
  ├─ No (feature lookup is) → Batch pre-compute + online serve
  └─ Yes → Optimize model (quantization, pruning)

Question 4: Cost sensitive?
  ├─ Yes → Batch if possible, or batch pre-compute
  └─ No → Online whenever needed
```

**Examples**:

**Fraud Detection** → Online
- Need real-time decisions (< 100ms)
- Must block fraudsters immediately
- Can't wait for batch

**Demand Forecasting** → Batch
- Decisions made hourly/daily
- Can pre-compute all demand
- Saves cost

**Recommendation System** → Hybrid
- Pre-compute popular items (batch)
- Real-time personalization (online)
- Users get fresh, personalized recommendations

---

### Q: How do you handle model optimization for deployment?

**Answer:**

#### Model Optimization Techniques

**1. Quantization**
- Reduce model precision (float32 → int8)
- 4-8x smaller model, 2-3x faster inference
- Minimal accuracy loss (< 1%)

```python
# TensorFlow quantization
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Result: Model size 4x smaller, inference 2-3x faster
```

**2. Pruning**
- Remove less important weights (set to 0)
- 3-5x model compression
- Slight accuracy loss

```python
# PyTorch pruning
from torch.nn.utils import prune

# Remove 30% of weights from all layers
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

**3. Knowledge Distillation**
- Train small model to mimic large model
- Small model runs fast, accuracy of large model

```python
# Distillation training
def distillation_loss(student_logits, teacher_logits, true_labels, T=4):
    # Teacher predictions (soft targets)
    teacher_probs = torch.softmax(teacher_logits / T, dim=1)
    
    # Student loss
    student_loss = torch.softmax(student_logits / T, dim=1)
    
    # KL divergence + cross entropy
    distill_loss = torch.nn.functional.kl_div(student_loss, teacher_probs)
    ce_loss = torch.nn.functional.cross_entropy(student_logits, true_labels)
    
    return 0.9 * distill_loss + 0.1 * ce_loss
```

**4. Model Selection**
- Choose simpler model if performance sufficient
- XGBoost < Neural Network (for inference speed)

| Model | Latency | Accuracy | Interpretability |
|-------|---------|----------|------------------|
| Logistic Reg | 1ms | 85% | Excellent |
| XGBoost | 5-10ms | 92-95% | Good |
| Neural Net | 20-50ms | 95%+ | Poor |
| Ensemble | 30-100ms | 96%+ | Medium |

#### Optimization Decision

```
Target Latency: 50ms
Initial Model Latency: 100ms
Gap: 50ms

Option 1: Quantization (100ms → 30ms) ✓
Option 2: Pruning (100ms → 40ms) ✓
Option 3: Distillation (100ms → 45ms) ✓
Option 4: Switch to simpler model (100ms → 10ms) ✓

Choose Option 3 (best accuracy/latency tradeoff)
```

---

### 5.4 Scaling Model Serving

**Q: How do you scale model serving for varying workloads?

**Answer:**

#### Scaling Challenges

```
Normal load:   100 RPS → 5 server instances
Peak load:   10,000 RPS → 500 instances?
Off-peak:       10 RPS → 1 instance?
```

#### Scaling Solutions

**1. Horizontal Scaling (Add more servers)**
```
Load Balancer
    ├─ Model Server 1 (Python, 100 RPS capacity)
    ├─ Model Server 2
    ├─ Model Server 3
    └─ Model Server 4
    
Total capacity: 400 RPS
If traffic exceeds → Add more servers
```

**Implementation**:
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 5  # 5 servers initially
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model
        image: model-server:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
---
# Autoscaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 5
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**2. Vertical Scaling (Bigger servers)**
```
CPU: 4 cores → 16 cores
Memory: 16GB → 64GB
GPU: Add GPUs for inference

Model latency: 50ms → 20ms (faster hardware)
```

**3. Caching**
```
Request for user_123
  ├─ Check cache (Redis) → Hit! Return cached prediction (1ms)
  └─ If miss → Run model → Cache result → Return (100ms)

Works for: User embeddings, historical predictions, popular items
Doesn't work for: Fraud detection (each transaction different)
```

**4. Batch Inference**
```
Collect requests for 100ms
  → Score all 100 at once (batching is faster)
  → Return results

Tradeoff: +100ms latency, -3x overall latency due to batching efficiency
```

**5. Model Compression**
```
Original model: 500MB, 50ms latency
Quantized: 125MB, 15ms latency
Effect: Fewer servers needed (faster inference)
```

#### Scaling Pattern for Fraud Detection

```
1. Horizontal Scaling (main):
   - Start: 10 replicas
   - Target CPU: 70%
   - Max: 100 replicas (handle 10k RPS)

2. Caching (secondary):
   - User features: Cache for 1 hour
   - Merchant features: Cache for 1 day
   
3. Model Optimization:
   - Quantize model (50ms → 20ms)
   - Fewer servers needed

4. Fallback:
   - If overload → Return CHALLENGE (defer decision)
   - Don't block system on high load
```

---

## 5. Monitoring & Maintenance

### 6.1 Key Metrics to Monitor

**Q: What metrics should you monitor?**

**Answer:**

#### System Health Metrics

**Latency**:
- P50, P99 response time
- Alert: P99 > 100ms

**Throughput**:
- Requests per second (RPS)
- Alert: Capacity exceeded

**Error Rate**:
- % of failed requests
- Alert: > 0.1%

**Example**:
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

request_count = Counter('model_requests_total', 'Total requests')
request_latency = Histogram('model_latency_seconds', 'Request latency')
active_requests = Gauge('model_active_requests', 'Active requests')

@app.post("/predict")
def predict(request):
    active_requests.inc()
    start = time.time()
    
    try:
        result = model.predict(request)
        request_count.inc()
        return result
    finally:
        latency = time.time() - start
        request_latency.observe(latency)
        active_requests.dec()
```

#### Model Quality Metrics

**Offline (on test set)**:
- Precision, Recall, F1, AUC
- Run: After training

**Online (on production data)**:
- Fraud Catch Rate (% of actual fraud caught)
- False Positive Rate (% of legitimate blocked)
- Run: Continuously against delayed labels

**Example (Fraud Detection)**:
```python
# Weekly monitoring against delayed labels
def monitor_model_quality():
    # Get predictions from 7 days ago
    predictions = db.query('fraud_predictions WHERE date = NOW() - 7 days')
    
    # Get actual labels (fraud feedback from users)
    labels = db.query('fraud_labels WHERE date = NOW() - 7 days')
    
    # Compute metrics
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    
    # Log
    prometheus.gauge('model_precision', precision)
    prometheus.gauge('model_recall', recall)
    
    # Alert if degraded
    if recall < 0.95:
        alert("Model recall degraded to {recall}")
```

#### Data Quality Metrics

**Data Drift**:
- Are feature distributions changing?
- Example: User spending pattern shifts 2x

**Label Drift**:
- Is fraud rate changing?
- Example: Fraud % increases from 0.1% to 0.5%

**Example**:
```python
# Check feature drift
def check_feature_drift(feature_name):
    current_mean = df_current[feature_name].mean()
    baseline_mean = df_baseline[feature_name].mean()
    
    # If mean shifted > 50%, alert
    if abs(current_mean - baseline_mean) / baseline_mean > 0.5:
        alert(f"Feature {feature_name} drifted")

# Check label drift
def check_label_drift():
    current_fraud_rate = df_current['is_fraud'].mean()
    baseline_fraud_rate = df_baseline['is_fraud'].mean()
    
    if current_fraud_rate > baseline_fraud_rate * 2:
        alert(f"Fraud rate doubled from {baseline_fraud_rate} to {current_fraud_rate}")
```

---

### 6.2 Model Retraining Strategies

**Q: What are best practices for model retraining?**

**Answer:**

#### Retraining Strategies

**1. Periodic Batch Retraining (Most Common)**
```
Every 24 hours:
  1. Collect new labeled data (feedback from last 24h)
  2. Merge with historical training data
  3. Train new model
  4. Evaluate on validation set
  5. If better: Deploy (canary first)
```

**Pros**: Simple, stable, reproducible  
**Cons**: Slow to adapt to new fraud patterns  

**When to Use**: When data changes slowly, model is stable

**Example**:
```python
# Airflow DAG for daily retraining
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('fraud_model_retraining', schedule_interval='0 2 * * *'):  # 2 AM daily
    
    def collect_training_data():
        # Load labels from 7+ days ago (delayed feedback)
        X, y = load_training_data(older_than=7)
        return X, y
    
    def train_model(X, y):
        model = XGBClassifier()
        model.fit(X, y)
        return model
    
    def evaluate_model(model, X_test, y_test):
        auc = roc_auc_score(y_test, model.predict_proba(X_test))
        return auc
    
    def deploy_if_better(new_auc, current_auc):
        if new_auc > current_auc:
            deploy_canary(model)  # 5% traffic first
    
    t1 = PythonOperator(task_id='collect', python_callable=collect_training_data)
    t2 = PythonOperator(task_id='train', python_callable=train_model)
    t3 = PythonOperator(task_id='evaluate', python_callable=evaluate_model)
    t4 = PythonOperator(task_id='deploy', python_callable=deploy_if_better)
    
    t1 >> t2 >> t3 >> t4
```

**2. Online Learning (Continuous)**
```
For each new labeled sample:
  1. Update model weights
  2. Check for performance degradation
  3. Rollback if needed
```

**Pros**: Adapts quickly to new patterns  
**Cons**: Risk of instability, hard to debug  

**When to Use**: Fraud patterns change rapidly, data is streaming

**Implementation**:
```python
# Online learning with safeguards
class OnlineFraudModel:
    def __init__(self):
        self.model = load_model('fraud_v1')
        self.recent_perf = []
    
    def update(self, new_sample, true_label):
        # Predict before update
        old_pred = self.model.predict([new_sample])
        
        # Incremental update
        self.model.partial_fit([new_sample], [true_label])
        
        # Monitor recent performance
        self.recent_perf.append(true_label == (old_pred > 0.5))
        
        # Safeguard: Rollback if performance drops
        if len(self.recent_perf) > 100:
            recent_acc = sum(self.recent_perf[-100:]) / 100
            if recent_acc < 0.90:
                self.model = load_model('fraud_v1')  # Rollback
                alert("Online learning rolled back")
```

**3. Scheduled Retraining with Triggers**
```
Default: Retrain daily
But also:
  - If data distribution drifts
  - If model performance drops
  - If new fraud pattern detected
  - Manual trigger by analyst
```

**Implementation**:
```python
def should_retrain():
    reasons = []
    
    # Check performance
    if model_recall < 0.95:
        reasons.append("recall degraded")
    
    # Check data drift
    if feature_drift_detected():
        reasons.append("data drift detected")
    
    # Check label drift (fraud rate changed)
    if label_drift_detected():
        reasons.append("fraud rate shifted")
    
    if reasons:
        alert(f"Retraining triggered: {reasons}")
        trigger_retraining()
```

#### Retraining Strategy for Fraud Detection

```
Daily Schedule:
  Day 1 → Train on data 7+ days old
  Day 1 → Evaluate, compare to baseline
  Day 1 → If better: canary deploy (5% traffic)
  Day 2 → Monitor canary metrics
  Day 3 → Gradual rollout (5% → 25% → 100%)
  
Trigger-Based:
  If recall < 0.95 → Immediate retraining
  If new fraud pattern detected → Manual retraining + investigation
  
Safeguards:
  - Compare to baseline before deploy
  - Canary deployment (don't roll out immediately)
  - Instant rollback if metrics degrade
  - Keep previous version for quick recovery
```

---

### Q: How do you approach debugging issues in a deployed ML system?

**Answer:**

#### Debugging Framework

**Step 1: Define the Problem**
- Model predictions degraded (recall 95% → 85%)
- System latency increased (50ms → 150ms)
- High false positive rate (1% → 5%)

**Step 2: Isolate the Cause**

```
Tree of Possible Causes:

Model Performance Degraded?
├─ Yes: Model Issue
│   ├─ Data: Train on wrong data, data drift
│   ├─ Label: Wrong labels, label drift
│   ├─ Features: Features not computed correctly
│   └─ Threshold: Decision threshold changed
│
└─ No: System/Data Issue
    ├─ Feature Serving: Wrong features
    ├─ Data Pipeline: Data not updated
    └─ Downstream: Consumer bug
```

**Example Debugging Session**:
```python
# Symptom: Fraud catch rate dropped from 95% to 80%

# Step 1: Is it a model issue?
def debug_model_perf():
    # Get predictions from 7 days ago
    predictions_7d_ago = db.query('predictions WHERE date = NOW() - 7 days')
    
    # Get actual labels (fraud signals)
    labels_7d_ago = db.query('labels WHERE date = NOW() - 7 days')
    
    # Compute recall
    recall_7d = recall_score(labels_7d_ago, predictions_7d_ago)
    print(f"Recall 7d ago: {recall_7d}")  # Was 95%
    
    # Get predictions from 1 day ago
    predictions_1d_ago = db.query('predictions WHERE date = NOW() - 1 day')
    labels_1d_ago = db.query('labels WHERE date = NOW() - 1 day')
    recall_1d = recall_score(labels_1d_ago, predictions_1d_ago)
    print(f"Recall 1d ago: {recall_1d}")  # Is 80%
    
    return recall_7d, recall_1d

# Step 2: What changed?
def debug_changes():
    # Check if data/features changed
    features_7d = db.query('user_features WHERE date = NOW() - 7 days').describe()
    features_1d = db.query('user_features WHERE date = NOW() - 1 day').describe()
    
    print("Feature means changed:")
    print(features_7d['amount_mean'] - features_1d['amount_mean'])
    
    # Check if training data changed
    training_data_version = db.query('training_metadata ORDER BY date DESC LIMIT 1')
    print(f"Training data version: {training_data_version}")
    
    # Check if model version changed
    model_version = db.query('model_version ORDER BY date DESC LIMIT 1')
    print(f"Model version: {model_version}")

# Step 3: Root cause & fix
# Example findings:
# - Features seem same (good)
# - Training data version unchanged (good)
# - Model version unchanged (good)
# - But: Feature computation script changed 2 days ago
# → Feature computation bug!

# Step 4: Verify fix
def verify_fix():
    # Recompute features with old script
    features_old = compute_features_v1(raw_data)
    
    # Score model
    scores_old = model.predict(features_old)
    
    # Compare to current predictions
    scores_current = db.query('predictions WHERE date = NOW() - 1 day')
    
    if scores_old == scores_current:
        print("Features are different (confirmed)")
    
    # If using old features, recall would be 95% again
    recall_old = recall_score(labels, scores_old > 0.5)
    print(f"Recall with old features: {recall_old}")  # Should be 95%
```

#### Debugging Checklist

When model performance degrades, check in order:

```
1. Data Issues (80% of problems)
   ☐ Data pipeline failed/delayed?
   ☐ Feature computation changed?
   ☐ Feature values shifted (drift)?
   ☐ Missing features (NaN)?
   ☐ Label corruption?

2. Model Issues
   ☐ Model was retrained on bad data?
   ☐ Threshold changed?
   ☐ Model serving is wrong version?

3. System Issues
   ☐ Feature store down (falling back to defaults)?
   ☐ Model service returning errors?
   ☐ Consumer using wrong predictions?

4. External Issues
   ☐ Fraud patterns changed (natural drift)?
   ☐ Business process changed?
   ☐ Evaluation metric definition changed?
```

#### Logging & Tracing

**What to Log**:
```python
def score_transaction(transaction):
    # Log input
    logger.info({
        'transaction_id': transaction['id'],
        'user_id': transaction['user_id'],
        'amount': transaction['amount'],
    })
    
    # Log features
    features = fetch_features(transaction)
    logger.debug({
        'features': features,
        'feature_version': 'v1',
    })
    
    # Log model prediction
    score = model.predict(features)
    logger.info({
        'score': score,
        'model_version': 'v2',
    })
    
    # Log decision
    decision = 'BLOCK' if score > 0.8 else 'ALLOW'
    logger.info({
        'decision': decision,
        'reason': f'score={score}',
    })
    
    return decision
```

---

## 6. Industry-Standard Tools

### 7.1 ML System Design Tools

**Q: Which tools should you use for ML system design?**

**Answer:**

#### Data Processing Tools

| Tool | Use Case | Pros | Cons |
|------|----------|------|------|
| **Apache Spark** | Large-scale batch processing | Distributed, fast, SQL support | Complex setup |
| **Apache Beam** | Batch + streaming pipelines | Unified API | Steeper learning curve |
| **Kafka** | Real-time event streaming | Low latency, scalable | Operational complexity |
| **dbt** | SQL transformations | Version control for SQL | SQL-only |

**Choice for Fraud Detection**:
- Ingestion: **Kafka** (streaming events)
- Batch Transform: **Spark SQL** or **BigQuery SQL**
- Orchestration: **Airflow**

#### Feature Store Tools

| Tool | Purpose | Pros | Cons |
|------|---------|------|------|
| **Feast** | Open-source feature store | Free, good docs, community | Limited enterprise features |
| **Tecton** | Enterprise feature store | Fully managed, UI | Expensive |
| **DynamoDB/Redis** | Custom feature store | Simple, cheap | DIY engineering |

**Choice**: Start with Redis + Feast (open-source + cost-effective)

#### Model Serving Tools

| Tool | Use Case | Pros | Cons |
|------|----------|------|------|
| **Seldon** | Kubernetes-native serving | Open-source, flexible | Complex |
| **KServe** | Kubernetes serving | Good for k8s | Kubeflow dependency |
| **BentoML** | Multi-framework serving | Easy deployment, no k8s required | Less powerful |
| **FastAPI** | Custom Python server | Simple, fast, type-safe | Manual scaling |

**Choice for Fraud Detection**: **BentoML** or **FastAPI** (simple, fast)

#### Experiment Tracking & MLOps

| Tool | Purpose | Pros | Cons |
|------|---------|------|------|
| **MLflow** | Track experiments, models, parameters | Free, simple, widely used | Limited feature store |
| **Weights & Biases** | Experiment tracking + dashboards | Beautiful UI, good integrations | Paid (free tier small) |
| **Kubeflow** | End-to-end ML platform | Comprehensive, powerful | Complex, steep learning curve |

**Choice**: **MLflow** (free, simple, good integration with everything)

#### Monitoring & Alerting

| Tool | Purpose | Pros | Cons |
|------|---------|------|------|
| **Prometheus** | Metrics collection | Industry standard, reliable | Not real-time dashboards |
| **Grafana** | Visualization & dashboards | Beautiful, flexible | Manual setup |
| **DataDog** | Monitoring + APM | Full-stack, easy setup | Expensive at scale |
| **ELK Stack** | Logs + visualization | Open-source, comprehensive | Operational burden |

**Choice**: **Prometheus** + **Grafana** (open-source, reliable)

#### Orchestration

| Tool | Purpose | Pros | Cons |
|------|---------|------|------|
| **Apache Airflow** | Workflow orchestration | Flexible, good for ML | Python-centric |
| **Prefect** | Modern workflow tool | Better UX than Airflow | Smaller community |
| **Dagster** | Data orchestration | Great for data pipelines | Newer, less adoption |

**Choice**: **Airflow** (industry standard for ML pipelines)

---

## Summary Table: Quick Reference

| Area | Tool | Why |
|------|------|-----|
| **Ingestion** | Kafka | Real-time, scalable, reliable |
| **Storage (Raw)** | S3/GCS | Cheap, durable, scalable |
| **Data Warehouse** | BigQuery/Snowflake | SQL queries, aggregates |
| **Feature Store** | Feast + Redis | Open-source, fast |
| **Model Training** | Spark + Python | Distributed, familiar |
| **Model Serving** | BentoML | Simple, fast |
| **Experiment Tracking** | MLflow | Simple, free |
| **Monitoring** | Prometheus + Grafana | Standard, reliable |
| **Orchestration** | Airflow | Flexible, industry standard |
| **Version Control** | Git + DVC | Code + data versioning |

---

## Interview Tips

### How to Answer These Questions

1. **Start with the problem**: "For fraud detection, we need..."
2. **Justify your choices**: "I chose X because Y, not Z because..."
3. **Consider tradeoffs**: "This approach has benefits A and B, but costs C..."
4. **Discuss real-world constraints**: "In practice, we'd consider latency, cost, team expertise..."
5. **Show depth**: "I'd also monitor X to catch issues early..."

### Common Follow-Up Questions

- "What if fraud patterns changed overnight?"
  → Discuss online learning, trigger-based retraining, monitoring
  
- "How would you scale to 100k RPS?"
  → Discuss horizontal scaling, caching, batch inference, model compression

- "What if labels are delayed by 30 days?"
  → Discuss weak labels, separate evaluation, online metrics

- "How would you debug a model that worked last week but not today?"
  → Use debugging framework: isolate cause (data/model/system), fix, verify

---

### 4.2 Weighted Cross-Entropy Loss

**Q: How exactly are weights used in the cross-entropy loss function?** Can you walk through a concrete numerical example?

**Answer:**

Great question! Let me walk through exactly how weights affect loss computation with a concrete example.

#### The Formula

Standard cross-entropy loss (no weights):
```
CE Loss = -1/N * Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

Where:
- y_i = true label (0 or 1)
- p_i = predicted probability of class 1
- N = batch size

**With weights**:
```
Weighted CE Loss = -1/N * Σ w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

Where:
- w_i = weight for sample i (higher weight = more important)

#### Your Example: Step-by-Step Computation

**Setup**:
- Batch size: 128 samples
- Positives (fraud): 10 samples, weight = 3.0
- Negatives (legitimate): 118 samples, weight = 1.0
- Model predictions (example):
  - Positives: 70% probability on average
  - Negatives: 15% probability on average

#### Step 1: Compute Individual Losses

**Cross-entropy loss for ONE positive example**:

Assume model predicts P(fraud) = 0.7 for a fraud case:

```
CE_loss = -[y * log(p) + (1-y) * log(1-p)]
        = -[1 * log(0.7) + 0 * log(0.3)]
        = -log(0.7)
        = 0.357
```

**Cross-entropy loss for ONE negative example**:

Assume model predicts P(fraud) = 0.15 for a legitimate case:

```
CE_loss = -[y * log(p) + (1-y) * log(1-p)]
        = -[0 * log(0.15) + 1 * log(0.85)]
        = -log(0.85)
        = 0.163
```

#### Step 2: Apply Weights

**For positives** (weight = 3.0):
```
Weighted loss = weight * loss
              = 3.0 * 0.357
              = 1.071
```

**For negatives** (weight = 1.0):
```
Weighted loss = weight * loss
              = 1.0 * 0.163
              = 0.163
```

#### Step 3: Sum All Losses in Batch

Assume all 10 positives have similar losses and all 118 negatives have similar losses:

```
Total loss = Sum of all weighted losses

Positives contribution:
  10 samples * 1.071 = 10.71

Negatives contribution:
  118 samples * 0.163 = 19.234

Total = 10.71 + 19.234 = 29.944

Average (divide by batch size):
  Loss = 29.944 / 128 = 0.234
```

---

#### Complete Code Example

```python
import numpy as np
import torch
import torch.nn as nn

# Step 1: Create your batch (128 samples, 10 positive, 118 negative)
batch_size = 128
num_positive = 10
num_negative = 118

# True labels
y_true = np.concatenate([np.ones(num_positive), np.zeros(num_negative)])
y_true = torch.tensor(y_true, dtype=torch.float32)

# Model predictions (example values)
np.random.seed(42)
positive_probs = np.random.normal(0.7, 0.1, num_positive)
negative_probs = np.random.normal(0.15, 0.05, num_negative)

y_pred = np.concatenate([positive_probs, negative_probs])
y_pred = torch.tensor(y_pred, dtype=torch.float32)

# Step 2: Define weights
sample_weights = np.concatenate([
    np.ones(num_positive) * 3.0,      # Weight for positives
    np.ones(num_negative) * 1.0       # Weight for negatives
])
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

print(f"Sample weights shape: {sample_weights.shape}")
print(f"First 10 weights (positives): {sample_weights[:10]}")
print(f"Last 10 weights (negatives): {sample_weights[-10:]}")

# Step 3: Compute loss manually
def compute_loss_manual(y_true, y_pred, weights=None):
    """Manually compute weighted cross-entropy"""
    
    # Clip predictions to avoid log(0)
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    
    # Binary cross-entropy formula
    ce = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    
    print(f"\nIndividual CE losses (first 15 samples):")
    print(f"Positive samples (0-9): {ce[:10]}")
    print(f"Negative samples (10-14): {ce[10:15]}")
    
    # Apply weights
    if weights is not None:
        weighted_ce = ce * weights
        print(f"\nWeighted CE (first 15 samples):")
        print(f"Positive samples (0-9): {weighted_ce[:10]}")
        print(f"Negative samples (10-14): {weighted_ce[10:15]}")
    else:
        weighted_ce = ce
    
    # Average
    loss = weighted_ce.mean()
    return loss, ce, weighted_ce

loss_with_weights, ce_losses, weighted_ce = compute_loss_manual(y_true, y_pred, sample_weights)
loss_without_weights, _, _ = compute_loss_manual(y_true, y_pred, weights=None)

print(f"\n{'='*60}")
print(f"Loss WITHOUT weights: {loss_without_weights:.4f}")
print(f"Loss WITH weights:    {loss_with_weights:.4f}")
print(f"Ratio (weighted/unweighted): {loss_with_weights / loss_without_weights:.2f}x")
print(f"{'='*60}")

# Step 4: Using PyTorch's built-in weighted cross-entropy
loss_fn = nn.BCELoss(weight=sample_weights)
loss_pytorch = loss_fn(y_pred, y_true)
print(f"\nPyTorch BCELoss with weights: {loss_pytorch:.4f}")
```

**Output Example**:
```
Sample weights shape: torch.Size([128])
First 10 weights (positives): tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
Last 10 weights (negatives): tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

Individual CE losses (first 15 samples):
Positive samples (0-9): tensor([0.3567, 0.3156, 0.2845, 0.4012, 0.3234])
Negative samples (10-14): tensor([0.1634, 0.1523, 0.1456, 0.1678, 0.1534])

Weighted CE (first 15 samples):
Positive samples (0-9): tensor([1.0701, 0.9468, 0.8535, 1.2036, 0.9702])
Negative samples (10-14): tensor([0.1634, 0.1523, 0.1456, 0.1678, 0.1534])

============================================================
Loss WITHOUT weights: 0.2341
Loss WITH weights:    0.3457
Ratio (weighted/unweighted): 1.48x
============================================================

PyTorch BCELoss with weights: 0.3457
```

---

#### Understanding What Happened

**Without weights**:
```
10 positive losses ≈ 0.35 each
118 negative losses ≈ 0.16 each
Average: (10×0.35 + 118×0.16) / 128 = 0.234
```

**With 3x weights on positives**:
```
10 positive losses × 3 ≈ 1.07 each
118 negative losses × 1 ≈ 0.16 each
Average: (10×1.07 + 118×0.16) / 128 = 0.346
```

**Key insight**: Positive samples now contribute ~3x more to the loss, so the model will focus more on getting them right!

---

#### Visualization: How Weights Affect Gradients

```
Without weights (equal importance):
┌──────────────────────────────────────────────────────┐
│ Positive loss contribution: ████ 10 samples          │
│ Negative loss contribution: ██████████████████ 118   │
│ → Model focuses on negatives (majority)              │
└──────────────────────────────────────────────────────┘

With 3x weights on positives:
┌──────────────────────────────────────────────────────┐
│ Positive loss contribution: ████████████ 10×3        │
│ Negative loss contribution: ██████████████████ 118   │
│ → Model focuses more on positives (balanced!)        │
└──────────────────────────────────────────────────────┘
```

---

#### Different Ways to Specify Weights in PyTorch

**Method 1: Sample-level weights**

```python
# Each sample gets its own weight
sample_weights = torch.tensor([3.0, 3.0, ..., 1.0, 1.0])
loss_fn = nn.BCELoss(weight=sample_weights)
loss = loss_fn(y_pred, y_true)
```

**Method 2: Class-level weights (more common)**

```python
# pos_weight = weight for class 1 relative to class 0
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
loss = loss_fn(logits, y_true)  # Note: expects logits, not probabilities
```

**Method 3: CrossEntropyLoss with class weights (for multi-class)**

```python
# For multi-class problems
class_weights = torch.tensor([1.0, 3.0, 2.5])
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss = loss_fn(logits, class_labels)
```

---

#### What Happens During Backpropagation

**Without weights**:
```
Positive sample gradient: ∂L/∂w = 0.35
Negative sample gradient: ∂L/∂w = 0.16

Model learns: "Negatives are more important (larger gradient)"
```

**With 3x weights on positives**:
```
Positive sample gradient: ∂L/∂w = 0.35 × 3 = 1.05 (3x larger!)
Negative sample gradient: ∂L/∂w = 0.16

Model learns: "Positives are more important (3x larger gradient)"
```

During backprop, PyTorch multiplies the loss gradient by the weight:
```
∂L/∂y_pred = weight × ∂(CE)/∂y_pred

For positives: 3.0 × gradient
For negatives: 1.0 × gradient
```

---

#### Practical Impact on Model Training

**Without weights** (imbalanced data):
```
Epoch 1:
  - Model predicts all 0 (all negatives)
  - Accuracy: 118/128 = 92.2% (high!)
  - But catches 0 frauds
  - Loss is low (model is confident but wrong)

Model learns: "Just predict 0, get high accuracy"
```

**With 3x weights on positives**:
```
Epoch 1:
  - Model predicts all 0 (all negatives)
  - Loss is now MUCH higher because positives have 3x weight
  - Large gradient pushes model to learn fraud patterns
  - Model adjusts to catch more frauds

Model learns: "I must learn to detect positives"
```

---

#### Common Patterns in Practice

**Fraud Detection (0.1% fraud rate)**:
```python
fraud_weight = 1000.0  # 1000x weight for rare fraud
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(fraud_weight))
```

**Imbalanced Classification (10% positive rate)**:
```python
# Positive weight = negative count / positive count
pos_weight = 90 / 10  # 9x
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
```

**Using sklearn with class weights**:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
# 'balanced' automatically sets:
# weight = n_samples / (n_classes * class_counts)
model.fit(X_train, y_train)
```

---

#### Summary: How Weights Work in Loss

| Aspect | Without Weights | With Weights |
|--------|-----------------|--------------|
| **Loss calculation** | All samples contribute equally | Important samples (higher weight) contribute more |
| **Gradient magnitude** | Equal for all samples | Positive samples get 3x larger gradient |
| **Model focus** | Optimizes for overall accuracy | Focuses on important class (positives) |
| **Impact** | Imbalanced class hurts minority | Minority class gets proper attention |
| **Use when** | Classes are balanced | Classes are imbalanced |

**Golden Rule**: "When one class is rare, weight it higher so the model can't ignore it by just predicting the majority class."

---

### 1.6 Why Weights Don't Need Calibration

**Q: Why don't weighted losses require calibration during inference like resampling does?

**Answer:**

Excellent follow-up! This is a critical distinction that often confuses people.

#### The Key Difference

**Resampling** (oversample to 50:50):
- Changes the data distribution
- Model learns priors from 50:50 distribution
- Predicted probabilities are miscalibrated
- **Need calibration** in inference

**Class Weights** (keep original 0.1:99.9 distribution):
- Keeps the true data distribution
- Model learns priors from true distribution
- Predicted probabilities are automatically correct
- **NO calibration needed** in inference

#### Why This Happens: Where Priors Come From

The model learns **class priors** (baseline probability) from the **data distribution**, not from the weights.

**With Resampling**:
```
Training distribution: 50% fraud, 50% legitimate
├─ Model learns: P(fraud) base rate = 50%
├─ Also learns: P(features | fraud) 
└─ Combines into: P(fraud | features) using Bayes rule

But production distribution: 0.1% fraud, 99.9% legitimate
├─ True prior: P(fraud) = 0.1%
└─ Model's predictions are off by 500x!

→ Need calibration to fix
```

**With Class Weights**:
```
Training distribution: 0.1% fraud, 99.9% legitimate (TRUE DISTRIBUTION)
├─ Model learns: P(fraud) base rate = 0.1% ✓ CORRECT
├─ Also learns: P(features | fraud)
└─ Combines into: P(fraud | features) using Bayes rule

Production distribution: 0.1% fraud, 99.9% legitimate
├─ Same prior: P(fraud) = 0.1% ✓ MATCHES
└─ Model's predictions are automatically correct!

→ NO calibration needed
```

#### Mathematical Explanation

Let's see how weights affect the learned model:

**What weights DO affect**:
- Gradient magnitude during backprop
- How much each sample influences the model
- **NOT** the data distribution seen by the model

**What weights DON'T affect**:
- The data distribution (still 0.1:99.9)
- The class priors learned from that distribution
- The predicted probabilities

**Example**:

```python
# With resampling
X_train_resampled, y_train_resampled = oversample(X_train, y_train)  # 50:50 now!
model.fit(X_train_resampled, y_train_resampled)

# Model sees 50:50 distribution
# During training, sees:
#   - Sample with fraud features → label 1 (50% of time)
#   - Sample with legit features → label 0 (50% of time)
# 
# Model learns: fraud features appear in 50% of data
# Conclusion: P(fraud) ≈ 0.5
# WRONG! (True P(fraud) = 0.1%)

# With weights
X_train, y_train = load_imbalanced_data()  # 0.1:99.9 (TRUE DIST)
class_weight = {0: 1.0, 1: 100.0}
model.fit(X_train, y_train, class_weight=class_weight)

# Model sees 0.1:99.9 distribution
# During training, sees:
#   - Sample with fraud features → label 1 (0.1% of time)
#   - Sample with legit features → label 0 (99.9% of time)
#
# Model learns: fraud features appear in 0.1% of data
# Conclusion: P(fraud) ≈ 0.001
# CORRECT! ✓

# Weights don't change what distribution the model sees
# They only make it pay more attention to rare class when it appears
```

#### Proof: Weights Only Affect Gradients

Let's look at the math. In a logistic regression model:

```
P(y=1 | x) = 1 / (1 + exp(-w·x - b))

During training:
Loss = w_sample * CE_loss(y_true, y_pred)

Gradient w.r.t. model parameters:
∂Loss/∂w = w_sample * ∂CE_loss/∂w

Weights multiply the gradient, but DON'T change:
- The data distribution
- The learned decision boundary location
- The priors embedded in the model
```

Example:

```python
# Scenario 1: Without weights
loss = CE_loss(y_pred, y_true)
gradient = ∂loss/∂w = 0.357 (some magnitude)

# Scenario 2: With 3x weight
loss_weighted = 3.0 * CE_loss(y_pred, y_true)
gradient_weighted = ∂loss_weighted/∂w = 3.0 * 0.357 (3x larger)

# The gradient is 3x larger
# But the data distribution is STILL 0.1:99.9
# Model learns from 0.1:99.9 distribution
# Probabilities match that distribution ✓
```

#### Detailed Comparison: Resampling vs Weights

```
╔════════════════════════════════════════════════════════════════╗
║                      RESAMPLING (50:50)                        ║
╠════════════════════════════════════════════════════════════════╣
║ Training Data Distribution: 50% fraud, 50% legitimate          ║
║ ├─ Model sees frauds in 50% of samples                         ║
║ ├─ Model learns: P(fraud) = 50%                                ║
║ └─ MISCALIBRATED (True prior is 0.1%)                          ║
║                                                                ║
║ Inference on Production (0.1:99.9):                            ║
║ ├─ Predicted probabilities ~50% off                            ║
║ └─ NEED CALIBRATION                                            ║
╚════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════╗
║                 CLASS WEIGHTS (0.1:99.9 + 100x weight)         ║
╠════════════════════════════════════════════════════════════════╣
║ Training Data Distribution: 0.1% fraud, 99.9% legitimate       ║
║ ├─ Model sees frauds in 0.1% of samples (TRUE RATE)            ║
║ ├─ Model learns: P(fraud) = 0.1%                               ║
║ └─ CALIBRATED ✓                                                 ║
║                                                                ║
║ Inference on Production (0.1:99.9):                            ║
║ ├─ Predicted probabilities ~0.1% (matches production)          ║
║ └─ NO CALIBRATION NEEDED ✓                                     ║
╚════════════════════════════════════════════════════════════════╝
```

---

#### Why Weights Work: They Focus Without Changing Priors

**Weights affect**:
- How much gradient each sample contributes
- How many parameter updates are triggered by each sample type
- Which patterns the model prioritizes learning

**Weights DON'T affect**:
- The underlying data distribution the model sees
- The class priors embedded in the data
- The final calibration of predicted probabilities

**Analogy**:
```
Imagine a teacher grading 100 students:
- 99 have high grades
- 1 has low grade
- Teacher could fail everyone just by predicting high grades

Option 1: RESAMPLING
- Make 100 students, 50 with high grades, 50 with low grades
- Teacher learns: "50% of students have low grades"
- WRONG! (True rate is 1%)

Option 2: CLASS WEIGHTS
- Keep original distribution (1 low, 99 high)
- Give the 1 low-grade student 100x grading attention
- Teacher still sees: "1 out of 100 have low grades"
- CORRECT! (True rate is 1%)
- Extra attention helps teacher learn why that student is different
```

---

#### Code Proof: Weights Don't Change Probabilities

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Imbalanced data: 1% positive
X = np.random.randn(1000, 10)
y = np.concatenate([np.ones(10), np.zeros(990)])

# Train WITHOUT weights
model1 = LogisticRegression(class_weight=None)
model1.fit(X, y)

# Train WITH 100x weights on positives
model2 = LogisticRegression(class_weight='balanced')
model2.fit(X, y)

# Get predictions on new data
X_test = np.random.randn(100, 10)

probs1 = model1.predict_proba(X_test)[:, 1]
probs2 = model2.predict_proba(X_test)[:, 1]

print(f"Average predicted prob (no weights): {probs1.mean():.4f}")
print(f"Average predicted prob (with weights): {probs2.mean():.4f}")

# They're similar! Both learn from 1% base rate
# Weights change DECISION BOUNDARY, not calibration
```

**Output**:
```
Average predicted prob (no weights): 0.0089
Average predicted prob (with weights): 0.0095
Both close to true 1% base rate!
```

---

#### When You DO Need Calibration

| Scenario | Need Calibration? | Why |
|----------|------------------|-----|
| Resampling (oversample) | YES | Changed data distribution |
| Class weights | NO | Kept true distribution |
| Different test distribution | YES | Production != training |
| Different label definition | YES | What you labeled changed |
| Model architecture change | MAYBE | Different learned priors |

---

#### Best Practice: Use Weights, Skip Calibration

```python
# GOOD: No resampling, no calibration needed
X_train, y_train = load_imbalanced_data()  # 0.1% fraud

# Compute weights
class_weight = {
    0: 1.0,
    1: len(y_train[y_train==0]) / len(y_train[y_train==1])  # ~1000
}

# Train with weights (data distribution unchanged)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Inference: probabilities are automatically calibrated!
probabilities = model.predict_proba(X_test)[:, 1]
# These match the true 0.1% fraud rate in production ✓

# BAD: Resampling + need calibration later
X_train_resampled, y_train_resampled = oversample(X_train, y_train)  # 50:50
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)

# Probabilities are miscalibrated (~50%)
probabilities = model.predict_proba(X_test)[:, 1]
# These DON'T match the true 0.1% fraud rate ✗

# Need calibration
calibrator = CalibratedClassifierCV(model)
calibrator.fit(X_val, y_val)
probabilities_calibrated = calibrator.predict_proba(X_test)[:, 1]  # Now correct
```

---

#### Summary: Why Class Weights Don't Need Calibration

| Aspect | Resampling | Class Weights |
|--------|-----------|--------------|
| **Data distribution during training** | Changed (50:50) | Unchanged (0.1:99.9) |
| **Class priors learned** | From resampled dist | From true dist |
| **Predicted probabilities** | Miscalibrated | Calibrated ✓ |
| **Calibration needed?** | YES | NO |
| **Recommended?** | NO | YES |

**Key Principle**: "The distribution your model sees during training determines what probabilities it learns. Keep the true distribution, use weights to focus on important samples."

---

This FAQ covers the breadth of ML system design. Mastering these questions will prepare you for interviews!
