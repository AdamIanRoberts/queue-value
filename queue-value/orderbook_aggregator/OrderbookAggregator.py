import abc


class OrderbookAggregator(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def orderbook_states(self) -> list:
        pass
