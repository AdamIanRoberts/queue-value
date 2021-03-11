import abc
import itertools
import numpy as np


class OrderbookAggregator(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def time_steps(self) -> np.array:
        pass

    @property
    @abc.abstractmethod
    def orderbook_states(self) -> itertools.product:
        pass
