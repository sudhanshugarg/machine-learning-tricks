"""
Fraud Detection System - Template Implementation

This template provides a starter implementation for key components of the fraud detection system:
- Feature Store (online feature serving)
- Model Service (fraud scoring)
- Decision Service (orchestration and fraud decisions)
- Training Pipeline (offline model training)

Modify and extend these classes to implement your own fraud detection system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass


# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

@dataclass
class Transaction:
    """Represents a single transaction request."""
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    merchant_category: str
    device_id: str
    ip_address: str
    timestamp: datetime


@dataclass
class FeatureVector:
    """Represents a feature vector for fraud scoring."""
    user_id: str
    features: Dict[str, float]
    timestamp: datetime
    feature_version: str


@dataclass
class FraudDecision:
    """Represents the fraud detection decision."""
    transaction_id: str
    decision: str  # "ALLOW", "CHALLENGE", "BLOCK", "REVIEW"
    fraud_score: float
    confidence: float
    reasoning: Dict[str, str]
    timestamp: datetime


# ============================================================================
# 2. FEATURE STORE (Online Feature Serving)
# ============================================================================

class FeatureStore:
    """
    Serves real-time and precomputed features for fraud scoring.

    In a real system, this would connect to:
    - Redis/DynamoDB for online features (< 20ms latency)
    - BigQuery/Snowflake for batch features
    - Feature registry for schema and versioning
    """

    def __init__(self):
        # In production: Connect to Redis/DynamoDB, feature registry
        # For template: In-memory storage
        self.online_features = {}  # user_id -> {feature_name: value}
        self.batch_features = {}   # user_id -> {feature_name: value}

    def lookup_features(self, transaction: Transaction) -> FeatureVector:
        """
        Fetch feature vector for a transaction.

        In production, this would:
        1. Query online store for real-time features (velocity, device consistency)
        2. Query batch store for historical features (aggregates)
        3. Fall back to defaults for missing features

        Args:
            transaction: Transaction to get features for

        Returns:
            FeatureVector with all required features

        Raises:
            FeatureStoreException: If feature lookup times out or fails
        """
        features = {}
        user_id = transaction.user_id

        # Real-time features (computed at request time or cached)
        features['velocity_1h'] = self._compute_velocity(user_id, hours=1)
        features['velocity_24h'] = self._compute_velocity(user_id, hours=24)
        features['device_is_new'] = self._is_device_new(user_id, transaction.device_id)
        features['location_distance'] = self._compute_location_distance(
            user_id, transaction.ip_address
        )

        # Batch features (precomputed daily, served from cache)
        batch_feats = self.batch_features.get(user_id, {})
        features.update(batch_feats)

        # Fallback to defaults for missing features
        default_features = self._get_default_features()
        for feat_name, default_val in default_features.items():
            if feat_name not in features:
                features[feat_name] = default_val

        return FeatureVector(
            user_id=user_id,
            features=features,
            timestamp=datetime.now(),
            feature_version="v1"
        )

    def _compute_velocity(self, user_id: str, hours: int) -> float:
        """Compute transaction count in time window."""
        # In production: Query transaction log for user in time window
        return np.random.poisson(2)  # Template: random value

    def _is_device_new(self, user_id: str, device_id: str) -> float:
        """Check if device is new for user."""
        # In production: Query device history
        return np.random.choice([0, 1])  # Template: random

    def _compute_location_distance(self, user_id: str, ip_address: str) -> float:
        """Compute geographic distance from last transaction."""
        # In production: Geolocate IP, compute distance from last location
        return np.random.uniform(0, 5000)  # Template: random distance (km)

    def _get_default_features(self) -> Dict[str, float]:
        """Default feature values when features can't be computed."""
        return {
            'avg_transaction_amount': 100.0,
            'std_transaction_amount': 50.0,
            'merchant_affinity': 0.5,
            'account_age_days': 365.0,
            'num_devices': 1.0,
            'txn_per_day': 2.0,
        }


# ============================================================================
# 3. MODEL SERVICE (Fraud Scoring)
# ============================================================================

class FraudScoringModel:
    """
    Fraud detection model for scoring transactions.

    In production, this would:
    - Load XGBoost/LightGBM model from disk
    - Handle model versioning (v1, v2, v3, ...)
    - Support canary deployments (route % of traffic to new model)
    - Track inference latency and throughput
    """

    def __init__(self, model_version: str = "v1"):
        self.model_version = model_version
        self.feature_names = [
            'velocity_1h', 'velocity_24h', 'device_is_new', 'location_distance',
            'avg_transaction_amount', 'std_transaction_amount', 'merchant_affinity',
            'account_age_days', 'num_devices', 'txn_per_day'
        ]
        # In production: Load trained XGBoost model
        # For template: Simple logistic regression-like scoring
        self.weights = {feat: np.random.uniform(-1, 1) for feat in self.feature_names}
        self.bias = -2.0  # Lower baseline fraud score

    def predict(self, feature_vector: FeatureVector) -> Tuple[float, float]:
        """
        Score a transaction for fraud risk.

        Args:
            feature_vector: Feature vector for transaction

        Returns:
            (fraud_score, confidence)
            - fraud_score: [0, 1], higher = more fraudulent
            - confidence: [0, 1], model confidence in prediction
        """
        # Extract features in order
        x = np.array([
            feature_vector.features.get(feat, 0.0)
            for feat in self.feature_names
        ])

        # Normalize features (in production: use fitted scalers from training)
        x = (x - np.array([1.0, 2.0, 0.5, 1000, 100, 50, 0.5, 365, 1, 2])) / (
            np.array([2.0, 5.0, 0.5, 1000, 50, 30, 0.3, 200, 3, 2]) + 1e-8
        )

        # Score: logistic function of weighted sum
        score = self.bias + np.dot(x, list(self.weights.values()))
        fraud_prob = 1 / (1 + np.exp(-score))  # Sigmoid

        # Confidence based on model certainty
        confidence = max(fraud_prob, 1 - fraud_prob)

        return float(fraud_prob), float(confidence)


# ============================================================================
# 4. DECISION SERVICE (Orchestration)
# ============================================================================

class DecisionService:
    """
    Orchestrates the fraud detection pipeline.

    Responsibilities:
    1. Validate transaction request
    2. Fetch feature vector
    3. Score with model
    4. Apply business rules
    5. Make final decision (ALLOW, CHALLENGE, BLOCK, REVIEW)
    """

    def __init__(self, feature_store: FeatureStore, model: FraudScoringModel):
        self.feature_store = feature_store
        self.model = model

        # Decision thresholds (tune based on business requirements)
        self.allow_threshold = 0.3      # score < 0.3 → ALLOW
        self.challenge_threshold = 0.7  # 0.3-0.7 → CHALLENGE
        self.block_threshold = 0.9      # 0.7-0.9 → BLOCK
        self.review_threshold = 0.95    # score > 0.95 → REVIEW

    def make_decision(self, transaction: Transaction) -> FraudDecision:
        """
        Make fraud decision for a transaction.

        Args:
            transaction: Transaction to evaluate

        Returns:
            FraudDecision with decision and reasoning
        """
        try:
            # Step 1: Validate transaction
            self._validate_transaction(transaction)

            # Step 2: Fetch features (can fail, use fallback)
            feature_vector = self.feature_store.lookup_features(transaction)

            # Step 3: Score with model
            fraud_score, confidence = self.model.predict(feature_vector)

            # Step 4: Apply business rules
            decision_rules = self._apply_business_rules(transaction, fraud_score)

            # Step 5: Make final decision
            decision_str = self._decide(fraud_score, decision_rules)

            reasoning = {
                'fraud_score': str(fraud_score),
                'confidence': str(confidence),
                'model_version': self.model.model_version,
                'rules_applied': ', '.join(decision_rules),
            }

            return FraudDecision(
                transaction_id=transaction.transaction_id,
                decision=decision_str,
                fraud_score=fraud_score,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now()
            )

        except Exception as e:
            # Fallback: If any error, return CHALLENGE (safer than BLOCK or ALLOW)
            return FraudDecision(
                transaction_id=transaction.transaction_id,
                decision='CHALLENGE',
                fraud_score=0.5,
                confidence=0.0,
                reasoning={'error': str(e), 'using_fallback': 'true'},
                timestamp=datetime.now()
            )

    def _validate_transaction(self, txn: Transaction) -> None:
        """Validate transaction has all required fields."""
        required_fields = ['transaction_id', 'user_id', 'amount', 'timestamp']
        for field in required_fields:
            if not getattr(txn, field, None):
                raise ValueError(f"Missing required field: {field}")

    def _apply_business_rules(self, txn: Transaction, fraud_score: float) -> List[str]:
        """Apply hard-coded business rules."""
        rules = []

        # Rule 1: High-risk merchants
        high_risk_merchants = ['cryptocurrency', 'gambling']
        if txn.merchant_category.lower() in high_risk_merchants:
            rules.append('high_risk_merchant')

        # Rule 2: Suspicious amount
        if txn.amount > 10000:
            rules.append('high_amount')

        # Rule 3: Manual review list
        if txn.merchant_id in self._get_manual_review_list():
            rules.append('manual_review_list')

        return rules

    def _get_manual_review_list(self) -> List[str]:
        """Get list of merchants/users flagged for manual review."""
        return []  # Template: empty list

    def _decide(self, fraud_score: float, rules: List[str]) -> str:
        """
        Make final decision based on score and rules.

        Decision Logic:
        - ALLOW: Low risk (score < 0.3)
        - CHALLENGE: Medium risk (0.3-0.7) - require 2FA, additional verification
        - BLOCK: High risk (0.7-0.9) - block but allow manual appeal
        - REVIEW: Very high risk (> 0.9) - queue for manual investigation
        """
        # Override based on business rules
        if 'manual_review_list' in rules:
            return 'REVIEW'

        # Base decision on fraud score
        if fraud_score > self.review_threshold:
            return 'REVIEW'
        elif fraud_score > self.block_threshold:
            return 'BLOCK'
        elif fraud_score > self.challenge_threshold:
            return 'CHALLENGE'
        else:
            return 'ALLOW'


# ============================================================================
# 5. TRAINING PIPELINE (Offline Model Training)
# ============================================================================

class TrainingPipeline:
    """
    Offline model training and evaluation.

    In production, this would:
    1. Load labeled training data (transactions from 7+ days ago with fraud labels)
    2. Generate features using feature store
    3. Handle class imbalance (upsampling/downsampling)
    4. Train XGBoost/LightGBM model
    5. Evaluate on validation set
    6. Compare with baseline and promote if better
    """

    def __init__(self):
        self.best_model = None
        self.best_auc = 0.0
        self.training_history = []

    def generate_training_data(
        self,
        transactions: List[Transaction],
        labels: np.ndarray,
        feature_store: FeatureStore
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate feature matrix from transactions.

        Args:
            transactions: List of transactions
            labels: Fraud labels (1 = fraud, 0 = legitimate)
            feature_store: Feature store for feature lookup

        Returns:
            (X, y) - feature matrix and labels
        """
        X = []
        for txn in transactions:
            feature_vec = feature_store.lookup_features(txn)
            x_row = np.array([
                feature_vec.features.get(feat, 0.0)
                for feat in [
                    'velocity_1h', 'velocity_24h', 'device_is_new', 'location_distance',
                    'avg_transaction_amount', 'std_transaction_amount', 'merchant_affinity',
                    'account_age_days', 'num_devices', 'txn_per_day'
                ]
            ])
            X.append(x_row)

        return np.array(X), labels

    def handle_class_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance (fraud is rare).

        Strategy: Oversample fraud class, undersample legitimate class
        """
        fraud_idx = np.where(y == 1)[0]
        legitimate_idx = np.where(y == 0)[0]

        # Oversample fraud to match legitimate (1:1 ratio)
        oversample_size = len(legitimate_idx)
        fraud_idx_resampled = np.random.choice(fraud_idx, oversample_size, replace=True)

        # Combine
        combined_idx = np.concatenate([legitimate_idx, fraud_idx_resampled])
        X_balanced = X[combined_idx]
        y_balanced = y[combined_idx]

        return X_balanced, y_balanced

    def train(self, X: np.ndarray, y: np.ndarray, model_name: str = "v1") -> FraudScoringModel:
        """
        Train fraud detection model.

        In production:
        - Use XGBoost/LightGBM (better than logistic regression)
        - Hyperparameter tuning (grid search, Bayesian optimization)
        - Cross-validation for robust evaluation

        Args:
            X: Feature matrix
            y: Binary labels (fraud=1, legitimate=0)
            model_name: Version name for the model

        Returns:
            Trained FraudScoringModel
        """
        # In production: Train XGBoost
        # For template: Train simple logistic model
        model = FraudScoringModel(model_version=model_name)

        # Simulate training with gradient descent
        weights = {feat: 0.0 for feat in model.feature_names}
        learning_rate = 0.01

        for iteration in range(100):
            # Forward pass: compute predictions
            y_pred = 1 / (1 + np.exp(-(np.dot(X, list(weights.values())) + model.bias)))

            # Backward pass: compute gradients
            residuals = y_pred - y
            for i, feat_name in enumerate(model.feature_names):
                gradient = np.mean(residuals * X[:, i])
                weights[feat_name] -= learning_rate * gradient

            # Update bias
            model.bias -= learning_rate * np.mean(residuals)

        model.weights = weights
        return model

    def evaluate(self, model: FraudScoringModel, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with metrics (precision, recall, AUC, etc.)
        """
        # Get predictions
        y_pred = np.array([model.predict(self._features_to_vector(x))[0] for x in X])

        # Compute metrics (manual implementation for portability)
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Precision = TP / (TP + FP)
        tp = np.sum((y_pred_binary == 1) & (y == 1))
        fp = np.sum((y_pred_binary == 1) & (y == 0))
        fn = np.sum((y_pred_binary == 0) & (y == 1))
        precision = tp / (tp + fp + 1e-8)

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-8)

        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Simple AUC approximation (rank-based)
        auc = np.mean(y_pred[y == 1]) - np.mean(y_pred[y == 0])
        auc = max(0, min(1, (auc + 1) / 2))  # Normalize to [0, 1]

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
        }

        self.training_history.append({
            'model_version': model.model_version,
            'timestamp': datetime.now(),
            'metrics': metrics
        })

        return metrics

    def _features_to_vector(self, x: np.ndarray) -> FeatureVector:
        """Convert numpy array to FeatureVector for model scoring."""
        features = {feat: val for feat, val in zip(
            ['velocity_1h', 'velocity_24h', 'device_is_new', 'location_distance',
             'avg_transaction_amount', 'std_transaction_amount', 'merchant_affinity',
             'account_age_days', 'num_devices', 'txn_per_day'],
            x
        )}
        return FeatureVector(
            user_id='unknown',
            features=features,
            timestamp=datetime.now(),
            feature_version='v1'
        )


# ============================================================================
# 6. MAIN: Example Usage
# ============================================================================

def main():
    """
    Example: End-to-end fraud detection system.

    Demonstrates:
    1. Creating a transaction
    2. Looking up features
    3. Scoring with model
    4. Making decision
    """

    print("=" * 70)
    print("FRAUD DETECTION SYSTEM - TEMPLATE EXAMPLE")
    print("=" * 70)

    # Initialize components
    feature_store = FeatureStore()
    model = FraudScoringModel(model_version="v1")
    decision_service = DecisionService(feature_store, model)

    # Simulate some transactions
    transactions = [
        Transaction(
            transaction_id="txn_001",
            user_id="user_123",
            merchant_id="merchant_456",
            amount=50.0,
            merchant_category="grocery",
            device_id="device_abc",
            ip_address="192.168.1.1",
            timestamp=datetime.now()
        ),
        Transaction(
            transaction_id="txn_002",
            user_id="user_789",
            merchant_id="merchant_999",
            amount=5000.0,
            merchant_category="cryptocurrency",
            device_id="device_xyz",
            ip_address="10.0.0.1",
            timestamp=datetime.now()
        ),
    ]

    # Make fraud decisions
    print("\n--- Making Fraud Decisions ---\n")
    for txn in transactions:
        decision = decision_service.make_decision(txn)
        print(f"Transaction: {decision.transaction_id}")
        print(f"  User: {txn.user_id}")
        print(f"  Amount: ${txn.amount}")
        print(f"  Category: {txn.merchant_category}")
        print(f"  Fraud Score: {decision.fraud_score:.3f}")
        print(f"  Decision: {decision.decision}")
        print(f"  Reasoning: {decision.reasoning}\n")

    # Training pipeline example
    print("\n--- Training Pipeline Example ---\n")

    # Simulate training data
    n_samples = 1000
    X_train = np.random.randn(n_samples, 10)  # 10 features
    y_train = np.random.binomial(1, 0.01, n_samples)  # 1% fraud rate

    training_pipeline = TrainingPipeline()

    # Handle imbalance
    X_balanced, y_balanced = training_pipeline.handle_class_imbalance(X_train, y_train)
    print(f"Original dataset: {len(X_train)} samples ({np.sum(y_train)} fraud)")
    print(f"Balanced dataset: {len(X_balanced)} samples ({np.sum(y_balanced)} fraud)")

    # Train model
    print("\nTraining model...")
    new_model = training_pipeline.train(X_balanced, y_balanced, model_name="v2")

    # Evaluate on test set
    print("Evaluating model on test set...")
    metrics = training_pipeline.evaluate(new_model, X_train, y_train)
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1']:.3f}")
    print(f"  AUC: {metrics['auc']:.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
