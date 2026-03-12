from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def learn(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_env(self):
        raise NotImplementedError
