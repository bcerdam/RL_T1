from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def get_action(self, epsilon, sga) -> int:
        pass

    @abstractmethod
    def learn(self, action: int, reward: float, alpha: float, sga: bool, baseline: bool) -> None:
        pass
