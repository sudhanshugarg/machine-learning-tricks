# Bayes' Theorem and Bayesian Inference

## Problem Statement

You are working on a spam detection system. Your email filter has the following characteristics:
- **Spam rate** in incoming emails: 5% (0.05)
- **True positive rate** (filter catches spam): 95% (correctly identifies spam)
- **False positive rate** (filter flags legitimate email as spam): 2% (incorrectly flags legitimate as spam)

A user receives an email flagged as spam by the filter. What is the probability that the email is actually spam?

## Key Concepts

This is a classic application of **Bayes' theorem**, which relates conditional probabilities:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

In this problem:
- $P(\text{Spam} | \text{Flagged})$ = posterior probability (what we want to find)
- $P(\text{Flagged} | \text{Spam})$ = likelihood (test accuracy for spam)
- $P(\text{Spam})$ = prior probability (base rate of spam)
- $P(\text{Flagged})$ = marginal likelihood (total probability of being flagged)

## Challenge

1. Calculate the posterior probability using Bayes' theorem
2. Understand why the answer might be counterintuitive
3. Explore how changing priors and likelihoods affect the result
4. Generalize to multi-class problems

## Applications

- Medical diagnosis (disease testing)
- Spam/fraud detection
- A/B testing and early stopping
- Naive Bayes classification
- Document classification
