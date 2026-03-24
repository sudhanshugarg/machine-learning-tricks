"""
Template for Reinforcement Learning Problems

This template provides the basic structure for RL solutions.
Replace the Agent class with your implementation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any


class Agent:
    """
    Agent class for reinforcement learning.

    Implement your RL agent here.
    """

    def __init__(self, n_actions: int, **params):
        """
        Initialize the agent.

        Args:
            n_actions: Number of possible actions
            **params: Additional parameters
        """
        self.n_actions = n_actions

    def act(self, state: np.ndarray = None) -> int:
        """Select an action."""
        raise NotImplementedError

    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray = None, done: bool = False) -> None:
        """Update agent based on experience."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset agent state."""
        raise NotImplementedError


def main():
    """Main function for testing the agent."""
    # Initialize agent
    # agent = Agent(n_actions=4)

    # Simulate episodes
    # for episode in range(num_episodes):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = agent.act(state)
    #         next_state, reward, done = env.step(action)
    #         agent.learn(state, action, reward, next_state, done)
    #         state = next_state

    pass


if __name__ == "__main__":
    main()
