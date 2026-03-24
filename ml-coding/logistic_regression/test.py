import numpy as np
from solution import LogisticRegression, accuracy, precision, recall, f1_score


def test_logistic_regression():
    """Test logistic regression on a simple dataset."""
    # Create synthetic dataset
    np.random.seed(42)
    n_samples, n_features = 100, 2

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate labels based on a linear boundary
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Calculate metrics
    acc = accuracy(y, y_pred)
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-Score: {f1:.3f}")

    assert acc > 0.8, "Accuracy should be > 0.8"
    assert len(y_proba) == n_samples, "Proba shape mismatch"
    assert np.all((y_proba >= 0) & (y_proba <= 1)), "Probabilities out of range"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_logistic_regression()
