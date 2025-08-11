import numpy as np
import random

class SimpleBanditEnv:

    def __init__(self, seed: int, num_of_arms: int = 10):
        self.__num_of_arms = num_of_arms
        self.true_action_values = np.random.randn(self.__num_of_arms)
        self.true_best_action = int(np.argmax(self.true_action_values))

    def step(self, action: int) -> float:
        reward_of_action = np.random.normal(self.true_action_values[action])
        return reward_of_action

    @property
    def best_action(self) -> int:
        return self.true_best_action

    @property
    def action_space(self) -> int:
        return self.__num_of_arms
