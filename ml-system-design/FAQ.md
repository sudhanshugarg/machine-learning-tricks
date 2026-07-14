# ML System Design - Frequently Asked Questions

This FAQ addresses common questions across ML system design problems. These are questions you'll encounter in interviews when designing systems like fraud detection, recommendation engines, ranking systems, etc.

---

## 1. Data Pipeline

### Q: Can you identify potential data quality issues?

**Answer:**

Common data quality issues to look for:

#### Missing Values
- **Identify**: Check % of null values per feature
- **Handle**:
  - **Deletion**: Remove rows/columns if > threshold (e.g., > 50% missing)
  - **Imputation**: 
    - Mean/median for numerical features
    - Mode for categorical features
    - Forward fill / backward fill for time-series
    - Predictive imputation (KNN, regression)
  - **Create indicator feature**: Mark whether value was missing (can be predictive)

**Example (Fraud Detection)**:
```python
# Check missing values
missing_pct = df.isnull().sum() / len(df) * 100
# Drop features > 50% missing
df = df.dropna(axis=1, thresh=0.5*len(df))
# Impute remaining with mean
df.fillna(df.mean(), inplace=True)
```

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

### Q: How do you propose appropriate techniques for handling data issues?

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

### Q: How do you ensure data freshness in ML systems?

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

### Q: Why is data versioning important?

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

### Q: How do you manage large datasets and their versions?

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

### Q: How do you track data lineage and ensure reproducibility?

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

### Q: What are the trade-offs between different deployment architectures?

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

### Q: How do you choose the right deployment architecture?

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

### Q: How do you scale model serving for varying workloads?

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

### Q: What metrics should you monitor?

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

### Q: What are best practices for model retraining?

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

### Q: Which tools should you use for ML system design?

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

This FAQ covers the breadth of ML system design. Mastering these questions will prepare you for interviews!
