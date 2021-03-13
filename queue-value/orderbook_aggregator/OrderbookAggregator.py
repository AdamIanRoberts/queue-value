import abc
import numpy as np


class OrderbookAggregator(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def time_steps(self) -> list:
        pass

    @property
    @abc.abstractmethod
    def orderbook_states(self) -> list:
        pass
