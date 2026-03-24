import numpy as np
from typing import Tuple

class LogisticRegression:
    """Binary logistic regression classifier using gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: float = 0.0):
        """
        Initialize logistic regression.

        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of iterations for training
            regularization: L2 regularization parameter (lambda)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute loss (binary cross-entropy with L2 regularization)
            loss = (-1/n_samples) * np.sum(y * np.log(predictions + 1e-9) +
                                          (1 - y) * np.log(1 - predictions + 1e-9))
            loss += (self.regularization / (2 * n_samples)) * np.sum(self.weights ** 2)
            self.losses.append(loss)

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            dw += (self.regularization / n_samples) * self.weights
            db = (1/n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of positive class.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted probabilities (n_samples,)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Features (n_samples, n_features)
            threshold: Classification threshold

        Returns:
            Predicted labels (n_samples,)
        """
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision score."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall score."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0
