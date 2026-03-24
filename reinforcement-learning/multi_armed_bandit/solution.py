import numpy as np
from abc import ABC, abstractmethod


class BanditAlgorithm(ABC):
    """Base class for multi-armed bandit algorithms."""

    def __init__(self, n_arms: int):
        """
        Initialize bandit algorithm.

        Args:
            n_arms: Number of arms
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)  # Number of times each arm pulled
        self.values = np.zeros(n_arms)  # Estimated reward for each arm

    @abstractmethod
    def select_arm(self) -> int:
        """Select an arm to pull."""
        pass

    def update(self, arm: int, reward: float) -> None:
        """
        Update arm estimates based on reward.

        Args:
            arm: Arm that was pulled
            reward: Reward received
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]

        # Incremental mean update: new_mean = old_mean + (reward - old_mean) / n
        self.values[arm] = value + (reward - value) / n


class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-greedy strategy: explore with probability epsilon, exploit with 1-epsilon."""

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best arm
            return np.argmax(self.values)


class UCB(BanditAlgorithm):
    """Upper Confidence Bound algorithm."""

    def __init__(self, n_arms: int, c: float = 1.0):
        """
        Initialize UCB.

        Args:
            n_arms: Number of arms
            c: Exploration constant (higher = more exploration)
        """
        super().__init__(n_arms)
        self.c = c
        self.t = 0  # Total number of pulls

    def select_arm(self) -> int:
        """Select arm with highest upper confidence bound."""
        self.t += 1

        # For each arm, compute UCB = mean + c * sqrt(log(t) / count)
        ucb_values = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            if self.counts[i] == 0:
                # Unvisited arms have infinite UCB (explore first)
                ucb_values[i] = float('inf')
            else:
                exploitation = self.values[i]
                exploration = self.c * np.sqrt(np.log(self.t) / self.counts[i])
                ucb_values[i] = exploitation + exploration

        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)


class ThompsonSampling(BanditAlgorithm):
    """Thompson sampling with Bernoulli rewards."""

    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.successes = np.zeros(n_arms)  # Number of successes per arm
        self.failures = np.zeros(n_arms)   # Number of failures per arm

    def select_arm(self) -> int:
        """Sample from posterior Beta distribution for each arm."""
        # Sample from Beta(successes + 1, failures + 1) for each arm
        samples = np.array([
            np.random.beta(self.successes[i] + 1, self.failures[i] + 1)
            for i in range(self.n_arms)
        ])

        return np.argmax(samples)

    def update(self, arm: int, reward: float) -> None:
        """Update success/failure counts."""
        self.counts[arm] += 1

        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

        # Update estimated value
        n = self.counts[arm]
        self.values[arm] = self.successes[arm] / n


class BanditEnvironment:
    """Simulated multi-armed bandit environment."""

    def __init__(self, arm_rewards: np.ndarray):
        """
        Initialize environment.

        Args:
            arm_rewards: Expected reward for each arm
        """
        self.arm_rewards = arm_rewards
        self.n_arms = len(arm_rewards)
        self.best_arm = np.argmax(arm_rewards)
        self.optimal_reward = arm_rewards[self.best_arm]

    def pull(self, arm: int) -> float:
        """
        Pull an arm and get Bernoulli reward.

        Args:
            arm: Arm to pull

        Returns:
            Reward (0 or 1)
        """
        return float(np.random.random() < self.arm_rewards[arm])

    def simulate(self, algorithm: BanditAlgorithm, n_rounds: int) -> dict:
        """
        Run bandit simulation.

        Args:
            algorithm: Algorithm to test
            n_rounds: Number of rounds to simulate

        Returns:
            Results dictionary
        """
        rewards = []
        choices = []
        regrets = []
        cumulative_regret = 0

        for _ in range(n_rounds):
            # Select and pull arm
            arm = algorithm.select_arm()
            reward = self.pull(arm)
            algorithm.update(arm, reward)

            # Track results
            rewards.append(reward)
            choices.append(arm)
            regret = self.optimal_reward - self.arm_rewards[arm]
            cumulative_regret += regret
            regrets.append(cumulative_regret)

        return {
            'rewards': np.array(rewards),
            'choices': np.array(choices),
            'cumulative_regret': np.array(regrets),
            'total_reward': np.sum(rewards),
            'total_regret': cumulative_regret,
            'arm_counts': algorithm.counts.copy(),
            'arm_values': algorithm.values.copy()
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Armed Bandit Simulation")
    print("=" * 60)

    # Create environment with 3 arms
    # Arm 0: 0.5, Arm 1: 0.7, Arm 2: 0.6
    # Best arm is arm 1 with reward 0.7
    arm_rewards = np.array([0.5, 0.7, 0.6])
    env = BanditEnvironment(arm_rewards)

    print(f"\nArm rewards: {arm_rewards}")
    print(f"Best arm: {env.best_arm} (reward: {env.optimal_reward})\n")

    n_rounds = 1000

    # Test algorithms
    algorithms = {
        'Epsilon-Greedy (ε=0.1)': EpsilonGreedy(3, epsilon=0.1),
        'UCB (c=1.0)': UCB(3, c=1.0),
        'Thompson Sampling': ThompsonSampling(3)
    }

    results = {}
    for name, algo in algorithms.items():
        results[name] = env.simulate(algo, n_rounds)

    # Print results
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Total reward: {result['total_reward']:.0f}/{n_rounds}")
        print(f"  Cumulative regret: {result['total_regret']:.1f}")
        print(f"  Arm pulls: {result['arm_counts'].astype(int)}")
        print(f"  Estimated values: {result['arm_values']:.3f}")
