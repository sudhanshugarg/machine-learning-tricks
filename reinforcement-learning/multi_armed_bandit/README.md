# Multi-Armed Bandit Problem

## Problem Statement

You have K slot machines (arms), each with an unknown reward distribution. In each round, you can pull one arm and receive a reward. Design a strategy to maximize cumulative reward over T rounds.

This is the classic exploration-exploitation tradeoff:
- **Exploration**: Try different arms to learn their reward distributions
- **Exploitation**: Pull the best arm known so far

## Challenge

Minimize **regret**: the difference between your cumulative reward and the optimal policy (always pulling the best arm).

## Key Concepts

### Arm Properties
- Each arm i has unknown expected reward $\mu_i$
- Rewards are typically drawn from a distribution (e.g., Bernoulli, Gaussian)
- Goal: Identify and exploit the arm with highest $\mu_i$

### Regret Definition

$$\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^{T} r_t$$

Where:
- $\mu^* = \max_i \mu_i$ (best arm's expected reward)
- $r_t$ is reward at round t
- Lower regret = better strategy

## Algorithms

### 1. Epsilon-Greedy
- With probability $\epsilon$: explore (random arm)
- With probability $1-\epsilon$: exploit (best arm)
- Simple, but not optimal

### 2. Upper Confidence Bound (UCB)
- Balance exploration via confidence intervals
- Pull arm with highest upper confidence bound
- Theoretically optimal regret: $O(\log T)$

### 3. Thompson Sampling
- Bayesian approach: maintain posterior over arm rewards
- Sample from posterior, pull best sampled arm
- Often empirically outperforms UCB

### 4. Optimism in the Face of Uncertainty (OFU)
- Maintain confidence intervals for each arm
- Pull arm with highest optimistic estimate
- Related to UCB

## Problem Variants

1. **Contextual Bandits**: Arms have context-dependent rewards
2. **Restless Bandits**: Arm rewards change over time
3. **Dueling Bandits**: Compare pairs of arms instead of absolute rewards
4. **Best Arm Identification**: Find best arm with high confidence (not maximize reward)

## Applications

- Online A/B testing
- Adaptive content recommendation
- Dynamic pricing
- Hyperparameter tuning
- Clinical trials
