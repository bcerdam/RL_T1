import random
import numpy as np
from agents.BaseAgent import BaseAgent
from scipy.special import softmax


class SimpleBanditAgent(BaseAgent):
    def __init__(self, num_of_actions: int, initial_value: int):
        self.num_of_actions = num_of_actions
        self.action_values = np.full(num_of_actions, initial_value, dtype=float)
        self.N = np.zeros(num_of_actions, dtype=int)

        self.action_preferences = np.zeros(num_of_actions, dtype=float)
        self.policy = softmax(self.action_preferences)
        self.incremental_average_reward = 0.0

    def get_action(self, epsilon, sga) -> int:

        if sga == True:
            chosen_action = np.random.choice(np.arange(self.num_of_actions), p=self.policy)
            return chosen_action

        else:
            random_value = random.uniform(0, 1)
            if random_value > epsilon:
                action_with_greater_value = int(np.argmax(self.action_values))
                return action_with_greater_value
            else:
                random_action = int(random.uniform(0, self.num_of_actions))
                return random_action

    def learn(self, action: int, reward: float, alpha: float, sga: bool, baseline: bool) -> None:
        self.N[action] += 1

        if sga == True:
            if baseline == True:
                self.incremental_average_reward += (1/np.sum(self.N)) * (reward - self.incremental_average_reward)
            self.action_preferences[action] += alpha * (reward - self.incremental_average_reward) * (1 - self.policy[action])

            for a in range(self.num_of_actions):
                if a != action:
                    self.action_preferences[a] -= alpha * (reward - self.incremental_average_reward) * self.policy[a]
            self.policy = softmax(self.action_preferences)

        else:
            if alpha == -1.0: # Step size no constante
                self.action_values[action] += (1 / self.N[action]) * (reward - self.action_values[action])
            else: # Step size constante
                self.action_values[action] += (alpha) * (reward - self.action_values[action])
