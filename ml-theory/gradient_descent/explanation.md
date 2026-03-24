# Gradient Descent Optimization

## Overview

Gradient descent is a fundamental optimization algorithm used to minimize cost functions in machine learning. The core idea is to iteratively move in the direction of steepest descent (negative gradient) to find the minimum of a function.

## Mathematical Foundation

### Basic Concept

Given a cost function $J(\theta)$ where $\theta$ are parameters, gradient descent updates parameters as:

$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$

Where:
- $\alpha$ is the learning rate (step size)
- $\nabla J(\theta_t)$ is the gradient at iteration $t$
- $\theta_t$ are the parameters at iteration $t$

### Gradient

The gradient is the vector of partial derivatives:

$$\nabla J(\theta) = \left[\frac{\partial J}{\partial \theta_1}, \frac{\partial J}{\partial \theta_2}, ..., \frac{\partial J}{\partial \theta_n}\right]$$

The gradient points in the direction of steepest increase; moving opposite (negative gradient) leads to steepest descent.

## Variants

### 1. Batch Gradient Descent (BGD)

**Update rule:**
$$\theta = \theta - \alpha \sum_{i=1}^{m} \nabla J(\theta; x_i, y_i)$$

**Characteristics:**
- Uses entire dataset for each update
- Guaranteed convergence for convex functions
- Slow for large datasets
- Smooth convergence curve
- Computationally expensive per iteration

**When to use:** Small datasets, when memory allows

### 2. Stochastic Gradient Descent (SGD)

**Update rule:**
$$\theta = \theta - \alpha \nabla J(\theta; x_i, y_i)$$

**Characteristics:**
- Updates using single sample
- Noisy gradient estimates
- Fast iterations, slow convergence
- Can escape local minima
- Noisy convergence curve

**When to use:** Large datasets, online learning

### 3. Mini-batch Gradient Descent

**Update rule:**
$$\theta = \theta - \alpha \sum_{i \in batch} \nabla J(\theta; x_i, y_i)$$

**Characteristics:**
- Best of both worlds: speed and stability
- Batch size typically 32-256
- Parallelizable
- Used in most deep learning frameworks

**When to use:** Standard choice for most problems

## Learning Rate Considerations

### Learning Rate Impact

- **Too small**: Slow convergence, may not reach optimum
- **Too large**: May overshoot minimum, oscillate, or diverge
- **Optimal**: Converges efficiently to minimum

### Adaptive Learning Rates

Instead of fixed $\alpha$, use algorithms that adapt:

**Momentum:**
$$v_t = \beta v_{t-1} + \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_t$$

**RMSprop:**
$$r_t = \beta r_{t-1} + (1-\beta)(\nabla J)^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{r_t + \epsilon}} \nabla J(\theta_t)$$

**Adam (Adaptive Moment Estimation):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla J$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla J)^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

## Convergence Analysis

### Convex Functions

- **Theorem:** BGD converges to global minimum for convex $J(\theta)$ with appropriate learning rate
- **Rate:** $O(1/t)$ for non-strongly convex, exponential for strongly convex

### Non-convex Functions

- May converge to local minima or saddle points
- Most deep neural networks are non-convex
- Practical convergence depends on initialization and hyperparameters

### Convergence Criteria

Stop gradient descent when:
1. Gradient norm is small: $\|\nabla J\| < \epsilon$
2. Fixed number of iterations reached
3. Loss change is small: $|J_t - J_{t-1}| < \epsilon$
4. Validation loss starts increasing (early stopping)

## Practical Tips

1. **Feature Scaling:** Normalize features to $[0,1]$ or $[-1,1]$ for faster convergence
2. **Initialize Properly:** Random initialization, avoid all zeros
3. **Monitor Loss:** Plot loss vs iterations to diagnose issues
4. **Batch Size:** Start with 32-128, adjust based on available memory
5. **Learning Rate Schedule:** Decay learning rate over time (e.g., $\alpha_t = \alpha_0 / \sqrt{t}$)
6. **Regularization:** Add L1/L2 penalty to avoid overfitting
7. **Numerical Stability:** Watch for exploding/vanishing gradients

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss increases | Learning rate too high | Reduce learning rate |
| No convergence | Learning rate too low | Increase learning rate |
| Slow convergence | Poor feature scaling | Normalize features |
| Stuck at plateau | Local minima | Restart with different init |
| NaN/Inf loss | Numerical instability | Use smaller learning rate, normalize |

