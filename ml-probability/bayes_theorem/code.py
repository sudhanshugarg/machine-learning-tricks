"""
Bayes' Theorem and Bayesian Inference Implementation
"""

import numpy as np
from typing import Dict, List, Tuple


def bayes_theorem(prior: float, likelihood: float, marginal: float) -> float:
    """
    Apply Bayes' theorem to compute posterior probability.

    Formula: P(H|E) = P(E|H) * P(H) / P(E)

    Args:
        prior: P(Hypothesis) - Prior probability
        likelihood: P(Evidence|Hypothesis) - Likelihood
        marginal: P(Evidence) - Marginal likelihood

    Returns:
        Posterior probability P(Hypothesis|Evidence)
    """
    return (likelihood * prior) / marginal


def compute_marginal_likelihood(
    likelihoods: List[float],
    priors: List[float]
) -> float:
    """
    Compute marginal likelihood using law of total probability.

    P(Evidence) = sum_i P(Evidence|Hypothesis_i) * P(Hypothesis_i)

    Args:
        likelihoods: List of P(Evidence|H_i)
        priors: List of P(H_i)

    Returns:
        Marginal likelihood P(Evidence)
    """
    return sum(l * p for l, p in zip(likelihoods, priors))


def bayesian_inference(
    observation_likelihoods: List[float],
    priors: List[float],
    hypotheses: List[str] = None
) -> Dict[str, float]:
    """
    Perform Bayesian inference for multiple hypotheses.

    Args:
        observation_likelihoods: P(Evidence|H_i) for each hypothesis
        priors: P(H_i) for each hypothesis
        hypotheses: Names of hypotheses

    Returns:
        Dictionary mapping hypothesis names to posterior probabilities
    """
    if hypotheses is None:
        hypotheses = [f"H_{i}" for i in range(len(priors))]

    # Compute marginal likelihood
    marginal = compute_marginal_likelihood(observation_likelihoods, priors)

    # Compute posteriors
    posteriors = {}
    for i, hyp in enumerate(hypotheses):
        post = bayes_theorem(priors[i], observation_likelihoods[i], marginal)
        posteriors[hyp] = post

    return posteriors


class BayesianClassifier:
    """
    Simple Bayesian classifier for binary classification.
    """

    def __init__(self, prior_positive: float = 0.5):
        """
        Initialize classifier.

        Args:
            prior_positive: P(Positive class) - prior probability
        """
        self.prior_positive = prior_positive
        self.prior_negative = 1 - prior_positive
        self.likelihood_pos = None
        self.likelihood_neg = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Estimate likelihoods from data.

        For simplicity, assumes X contains binary features.

        Args:
            X: Features (n_samples, n_features) with binary values
            y: Labels (n_samples,) with binary values
        """
        n_samples = len(y)

        # Separate by class
        X_pos = X[y == 1]
        X_neg = X[y == 0]

        # Estimate P(X|Positive) and P(X|Negative)
        # Using mean (proportion of 1s in each feature)
        self.likelihood_pos = X_pos.mean(axis=0) if len(X_pos) > 0 else np.zeros(X.shape[1])
        self.likelihood_neg = X_neg.mean(axis=0) if len(X_neg) > 0 else np.zeros(X.shape[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of positive class.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilities of positive class (n_samples,)
        """
        if self.likelihood_pos is None:
            raise ValueError("Model not fitted yet")

        probs = np.zeros(len(X))

        for i in range(len(X)):
            # Likelihood under positive class: product of feature likelihoods
            # P(X|Positive) = prod_j P(x_j|Positive)
            lik_pos = np.prod(
                self.likelihood_pos ** X[i] * (1 - self.likelihood_pos) ** (1 - X[i])
            )
            lik_neg = np.prod(
                self.likelihood_neg ** X[i] * (1 - self.likelihood_neg) ** (1 - X[i])
            )

            # Posterior via Bayes' theorem
            marginal = lik_pos * self.prior_positive + lik_neg * self.prior_negative
            probs[i] = (lik_pos * self.prior_positive) / (marginal + 1e-10)

        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features
            threshold: Decision threshold

        Returns:
            Predicted labels
        """
        return (self.predict_proba(X) >= threshold).astype(int)


# Example: Spam Detection Problem
def spam_detection_example():
    """Solve the spam detection problem."""
    print("=" * 70)
    print("SPAM DETECTION PROBLEM")
    print("=" * 70)

    # Given probabilities
    prior_spam = 0.05  # P(Spam)
    prior_legitimate = 1 - prior_spam  # P(Legitimate)

    likelihood_flagged_spam = 0.95  # P(Flagged|Spam)
    likelihood_flagged_legitimate = 0.02  # P(Flagged|Legitimate)

    # Compute marginal likelihood
    marginal_flagged = (
        likelihood_flagged_spam * prior_spam +
        likelihood_flagged_legitimate * prior_legitimate
    )

    # Compute posterior
    posterior_spam_given_flagged = bayes_theorem(
        prior_spam,
        likelihood_flagged_spam,
        marginal_flagged
    )

    print(f"\nPrior probabilities:")
    print(f"  P(Spam) = {prior_spam}")
    print(f"  P(Legitimate) = {prior_legitimate}")

    print(f"\nLikelihoods:")
    print(f"  P(Flagged|Spam) = {likelihood_flagged_spam}")
    print(f"  P(Flagged|Legitimate) = {likelihood_flagged_legitimate}")

    print(f"\nMarginal likelihood:")
    print(f"  P(Flagged) = {marginal_flagged:.4f}")

    print(f"\nPosterior probability:")
    print(f"  P(Spam|Flagged) = {posterior_spam_given_flagged:.4f} ({posterior_spam_given_flagged*100:.2f}%)")
    print(f"  P(Legitimate|Flagged) = {1-posterior_spam_given_flagged:.4f} ({(1-posterior_spam_given_flagged)*100:.2f}%)")

    # Intuitive explanation
    print(f"\nInterpretation (out of 10,000 emails):")
    n_spam = 10000 * prior_spam
    n_legitimate = 10000 * prior_legitimate
    n_spam_flagged = n_spam * likelihood_flagged_spam
    n_legitimate_flagged = n_legitimate * likelihood_flagged_legitimate
    n_total_flagged = n_spam_flagged + n_legitimate_flagged

    print(f"  Spam correctly flagged: {n_spam_flagged:.0f}")
    print(f"  Legitimate wrongly flagged: {n_legitimate_flagged:.0f}")
    print(f"  Total flagged: {n_total_flagged:.0f}")
    print(f"  → {n_spam_flagged/n_total_flagged*100:.1f}% of flagged are actual spam")

    return posterior_spam_given_flagged


# Example: Medical Diagnosis
def medical_diagnosis_example():
    """Solve a medical diagnosis problem."""
    print("\n" + "=" * 70)
    print("MEDICAL DIAGNOSIS PROBLEM")
    print("=" * 70)

    # Disease prevalence in population
    prior_disease = 0.001  # 0.1% have the disease

    # Test accuracy
    sensitivity = 0.99  # P(Positive Test|Disease)
    specificity = 0.98  # P(Negative Test|No Disease)
    false_positive_rate = 1 - specificity

    # Posterior after positive test
    marginal = sensitivity * prior_disease + false_positive_rate * (1 - prior_disease)
    posterior = bayes_theorem(prior_disease, sensitivity, marginal)

    print(f"\nDisease prevalence: {prior_disease*100:.2f}%")
    print(f"Test sensitivity (true positive rate): {sensitivity*100:.1f}%")
    print(f"Test specificity (true negative rate): {specificity*100:.1f}%")
    print(f"\nIf test is POSITIVE:")
    print(f"  Probability of having disease: {posterior*100:.2f}%")
    print(f"  Probability of NOT having disease (false positive): {(1-posterior)*100:.2f}%")

    return posterior


# Example: Multi-hypothesis inference
def multi_hypothesis_example():
    """Demonstrate inference with multiple hypotheses."""
    print("\n" + "=" * 70)
    print("MULTI-HYPOTHESIS INFERENCE")
    print("=" * 70)

    # Suppose we're classifying a coin based on observation: 7 heads in 10 flips
    hypotheses = [
        "Fair coin (p=0.5)",
        "Biased towards heads (p=0.7)",
        "Biased towards heads (p=0.8)"
    ]

    priors = [0.5, 0.3, 0.2]  # Prior belief distribution

    # Likelihood of observing 7 heads in 10 flips under each hypothesis
    from scipy.stats import binom
    likelihoods = [
        binom.pmf(7, 10, 0.5),
        binom.pmf(7, 10, 0.7),
        binom.pmf(7, 10, 0.8)
    ]

    posteriors = bayesian_inference(likelihoods, priors, hypotheses)

    print(f"\nHypotheses and priors:")
    for h, p in zip(hypotheses, priors):
        print(f"  {h}: {p}")

    print(f"\nObservation: 7 heads in 10 flips")
    print(f"\nLikelihoods:")
    for h, l in zip(hypotheses, likelihoods):
        print(f"  P(7 heads|{h}): {l:.4f}")

    print(f"\nPosterior probabilities:")
    for h, p in posteriors.items():
        print(f"  P({h}|observation): {p:.4f} ({p*100:.2f}%)")

    return posteriors


if __name__ == "__main__":
    spam_detection_example()
    medical_diagnosis_example()
    multi_hypothesis_example()
