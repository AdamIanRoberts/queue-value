"""
Aggregates processed order book data into a discrete time step with discrete imbalance and bid-ask spread states.
"""
import os

from datetime import datetime, time, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix

from data_processing.lobster.process_lobster_data import PROCESSED_DATA_DIR
from orderbook_aggregator.OrderbookAggregator import OrderbookAggregator


class StoikovAggregator(OrderbookAggregator):
    def __init__(
        self,
        date: datetime,
        ticker: str,
        levels: int,
        time_step_size: float = 100,  # Given in milliseconds
        imbalance_step_size: Optional[float] = 0.1,
        imbalance_end_points: bool = True,
        spread_step_size: Optional[float] = 100,
        max_spread: Optional[float] = None,
    ) -> None:
        self.date = date
        self.ticker = ticker
        self.levels = levels
        self.imbalance_step_size = imbalance_step_size
        self.imbalance_end_points = imbalance_end_points
        self.spread_step_size = spread_step_size
        self.time_step_size = time_step_size
        self.raw_orderbook = self._build_raw_orderbook()
        self.date = min(self.raw_orderbook["datetime"]).date()
        self.max_spread = max_spread or self.raw_orderbook["spread"].max()
        self.start_time = datetime.combine(self.date, time(9, 35))
        self.end_time = datetime.combine(self.date, time(16, 55))  # trim noisy first and last 5 minutes from data
        self.orderbook = self._discretise_orderbook()

    def _build_raw_orderbook(self) -> pd.DataFrame:
        file_name = f"{self.ticker}_{str(self.date.date())}_orderbook_{self.levels}.csv"
        raw_orderbook = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, self.ticker, file_name))
        return self._process_raw_orderbook(raw_orderbook)

    @property
    def time_steps(self) -> np.array:
        step_size = timedelta(milliseconds=self.time_step_size)
        start_time = max(self.start_time, min(self.raw_orderbook["datetime"]))
        end_time = min(self.end_time, max(self.raw_orderbook["datetime"]))
        return pd.date_range(start_time, end_time, freq=step_size)

    @property
    def imbalance_steps(self) -> np.array:
        imbalance_steps = np.arange(0, 1 + self.imbalance_step_size, self.imbalance_step_size)
        if not self.imbalance_end_points:
            imbalance_steps = imbalance_steps[1:-1]
        return imbalance_steps

    @property
    def spread_steps(self) -> np.array:
        return np.arange(
            self.spread_step_size,
            self.max_spread + self.spread_step_size,
            self.spread_step_size,
        )

    @property
    def mid_price_steps(self) -> np.array:
        max_value = 2 * self.max_spread
        return np.setdiff1d(0.5 * np.arange(-max_value, max_value + self.spread_step_size, self.spread_step_size), [0])

    @property
    def orderbook_states(self) -> np.array:
        imbalance_decimals = int(round(abs(np.log(self.imbalance_step_size) / np.log(10))))
        return np.array([(round(x, imbalance_decimals), y) for x in self.imbalance_steps for y in self.spread_steps])

    def absorbing_states(self) -> Tuple[csc_matrix, pd.Series]:
        orderbook_states_dict = dict(zip(self.orderbook_states, range(len(self.orderbook_states))))
        mid_price_steps_dict = dict(zip(self.mid_price_steps, range(len(self.mid_price_steps))))
        absorbing_probs = (
            self.orderbook[self.orderbook["mid_price_move"] != 0]
            .groupby(["state", "mid_price_move"])
            .count()
            .rename(columns={"size": "prob"})
        )["prob"]
        absorbing_count = self.orderbook.groupby(["state"]).count().rename(columns={"size": "prob"})["prob"]
        absorbing_probs = (absorbing_probs / absorbing_count).reset_index()
        row = [orderbook_states_dict[state] for state in absorbing_probs["state"]]
        col = [mid_price_steps_dict[price] for price in absorbing_probs["mid_price_move"]]
        data = absorbing_probs["prob"].values
        absorbing_states = csc_matrix(
            (data, (row, col)),
            shape=(len(self.orderbook_states), len(mid_price_steps_dict)),
        )
        return (
            absorbing_states,
            absorbing_count.reindex(self.orderbook_states).fillna(0),
        )

    def transient_states(self) -> Tuple[csc_matrix, pd.Series]:
        num_of_states = len(self.orderbook_states)
        orderbook_states_dict = dict(zip(self.orderbook_states, range(num_of_states)))
        transient_probs = (
            self.orderbook[self.orderbook["mid_price_move"] == 0]
            .groupby(["state", "next_state"])
            .count()
            .rename(columns={"size": "prob"})
        )["prob"]
        transient_count = self.orderbook.groupby(["state"]).count().rename(columns={"size": "prob"})["prob"]
        transient_probs = (transient_probs / transient_count).reset_index()
        row = [orderbook_states_dict[state] for state in transient_probs["state"]]
        col = [orderbook_states_dict[state] for state in transient_probs["next_state"]]
        data = transient_probs["prob"].values
        transient_states = csc_matrix((data, (row, col)), shape=(num_of_states, num_of_states))
        return (
            transient_states,
            transient_count.reindex(self.orderbook_states).fillna(0),
        )

    def absorbing_states_2(self) -> Tuple[csc_matrix, int]:
        num_of_states = len(self.orderbook_states)
        orderbook_states_dict = dict(zip(self.orderbook_states, range(num_of_states)))
        absorbing_2_probs = (
            self.orderbook[self.orderbook["mid_price_move"] != 0]
            .groupby(["state", "next_state"])
            .count()
            .rename(columns={"size": "prob"})
        )["prob"]
        absorbing_2_count = self.orderbook.groupby(["state"]).count().rename(columns={"size": "prob"})["prob"]
        absorbing_2_probs = (absorbing_2_probs / absorbing_2_count).reset_index()
        row = [orderbook_states_dict[state] for state in absorbing_2_probs["state"]]
        col = [orderbook_states_dict[state] for state in absorbing_2_probs["next_state"]]
        data = absorbing_2_probs["prob"].values
        absorbing_2_states = csc_matrix((data, (row, col)), shape=(num_of_states, num_of_states))
        return (
            absorbing_2_states,
            absorbing_2_count.reindex(self.orderbook_states).fillna(0),
        )

    def _process_raw_orderbook(self, raw_orderbook: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [col for col in raw_orderbook.columns if col[-1] in [str(x) for x in range(1, self.levels)]]
        for col in ["index", "order_id"]:
            drop_cols.append(col)
        raw_orderbook = raw_orderbook.drop(columns=drop_cols)
        raw_orderbook["datetime"] = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f") for d in raw_orderbook["datetime"]]
        raw_orderbook["imbalance"] = self._calculate_raw_imbalance(raw_orderbook)
        raw_orderbook["spread"] = self._calculate_raw_spread(raw_orderbook)
        raw_orderbook["mid_price"] = self._calculate_raw_mid_price(raw_orderbook)
        raw_orderbook["mid_price_move"] = self._calculate_raw_mid_price_move(raw_orderbook)
        return raw_orderbook

    def _discretise_orderbook(self) -> pd.DataFrame:
        orderbook = self.raw_orderbook.drop_duplicates(["datetime"], keep="last").set_index("datetime")
        orderbook = orderbook.reindex(self.time_steps, method="ffill")
        orderbook["imbalance"] = self._calculate_discrete_imbalance(orderbook["imbalance"])
        orderbook["spread"] = self._calculate_discrete_spread(orderbook["spread"])
        orderbook["spread"] = self._clip_max_spread(orderbook["spread"])
        orderbook["mid_price"] = self._calculate_discrete_mid_price(orderbook["mid_price"])
        orderbook["mid_price_move"] = self._calculate_raw_mid_price_move(orderbook)
        orderbook["mid_price_move"] = self._clip_max_mid_price_move(orderbook["mid_price_move"])
        orderbook["state"] = list(zip(orderbook["imbalance"], orderbook["spread"]))
        orderbook["next_state"] = orderbook["state"].shift(-1).values
        return orderbook.dropna()

    def _calculate_discrete_imbalance(self, imbalance: pd.Series) -> pd.Series:
        imbalance_decimals = int(round(abs(np.log(self.imbalance_step_size) / np.log(10))))
        return imbalance.round(decimals=imbalance_decimals)

    def _calculate_discrete_spread(self, spreads: pd.Series) -> pd.Series:
        return round(spreads / self.spread_step_size) * self.spread_step_size

    def _calculate_discrete_mid_price(self, mid_prices: pd.Series) -> pd.Series:
        return round(mid_prices / (0.5 * self.spread_step_size)) * 0.5 * self.spread_step_size

    def _clip_max_spread(self, spreads: pd.Series) -> pd.Series:
        return np.minimum(spreads, self.max_spread)

    def _clip_max_mid_price_move(self, mid_price_moves: pd.Series) -> pd.Series:
        # We don't *2 here, the max mid price move is one spread
        return np.maximum(np.minimum(mid_price_moves, self.max_spread), -self.max_spread)

    @staticmethod
    def _calculate_raw_imbalance(orderbook) -> pd.Series:
        return orderbook["bid_size_0"] / (orderbook["bid_size_0"] + orderbook["ask_size_0"])

    @staticmethod
    def _calculate_raw_spread(orderbook) -> pd.Series:
        return orderbook["ask_price_0"] - orderbook["bid_price_0"]

    @staticmethod
    def _calculate_raw_mid_price(orderbook) -> pd.Series:
        return 0.5 * (orderbook["ask_price_0"] + orderbook["bid_price_0"])

    @staticmethod
    def _calculate_raw_mid_price_move(orderbook) -> pd.Series:
        return orderbook["mid_price"].shift(-1) - orderbook["mid_price"]
