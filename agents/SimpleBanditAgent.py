import random
import numpy as np
from agents.BaseAgent import BaseAgent


class SimpleBanditAgent(BaseAgent):
    def __init__(self, num_of_actions: int, initial_value: int):
        self.num_of_actions = num_of_actions
        self.action_values = np.full(num_of_actions, initial_value, dtype=float)
        self.N = np.zeros(num_of_actions, dtype=int)

    def get_action(self, epsilon) -> int:
        random_value = random.uniform(0, 1)
        if random_value > epsilon:
            action_with_greater_value = int(np.argmax(self.action_values))
            return action_with_greater_value
        else:
            random_action = int(random.uniform(0, self.num_of_actions))
            return random_action

    def learn(self, action: int, reward: float, alpha: float) -> None:
        self.N[action] += 1

        if alpha == -1.0:
            self.action_values[action] += (1 / self.N[action]) * (reward - self.action_values[action])
        else:
            self.action_values[action] += (alpha) * (reward - self.action_values[action])
