from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def get_action(self, epsilon) -> int:
        pass

    @abstractmethod
    def learn(self, action: int, reward: float):
        pass
