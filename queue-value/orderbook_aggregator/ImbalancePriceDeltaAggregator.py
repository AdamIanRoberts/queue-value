from datetime import datetime, time, timedelta
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from data_processing.lobster.process_lobster_data import PROCESSED_DATA_DIR
from orderbook_aggregator.OrderbookAggregator import OrderbookAggregator


class ImbalancePriceDeltaAggregator(OrderbookAggregator):
    def __init__(
        self,
        date: datetime,
        ticker: str,
        levels: int,
        spread_size: float = 100,
        imbalance_step_size: Optional[float] = 0.25,
        max_mid_price_move: float = 100,
        imbalance_end_points: bool = True,
        price_delta_window: int = 1000,
        max_price_delta: int = 200,
    ):
        self.date = date
        self.ticker = ticker
        self.levels = levels
        self.imbalance_step_size = imbalance_step_size
        self.imbalance_end_points = imbalance_end_points
        self.price_delta_window = price_delta_window
        self.max_price_delta = max_price_delta
        self.spread_size = spread_size
        self.max_mid_price_move = max_mid_price_move
        self.time_step_size = 100  # Given in milliseconds
        self.raw_orderbook = self._build_raw_orderbook()
        self.start_time = datetime.combine(self.date, time(9, 45))
        self.end_time = datetime.combine(self.date, time(16, 45))  # trim noisy first and last 5 minutes from data
        self.orderbook = self._discretise_orderbook()

    def _build_raw_orderbook(self) -> pd.DataFrame:
        file_name = f"{self.ticker}_{str(self.date.date())}_orderbook_{self.levels}.csv"
        raw_orderbook = pd.read_csv(PROCESSED_DATA_DIR + f"/{self.ticker}/" + file_name)
        raw_orderbook = self._process_raw_orderbook(raw_orderbook)
        self.date = min(raw_orderbook["datetime"]).date()
        return raw_orderbook

    @property
    def time_steps(self) -> list:
        step_size = timedelta(milliseconds=self.time_step_size)
        start_time = max(self.start_time, min(self.raw_orderbook["datetime"]))
        end_time = min(self.end_time, max(self.raw_orderbook["datetime"]))
        return pd.date_range(start_time, end_time, freq=step_size)

    @property
    def imbalance_steps(self) -> list:
        imbalance_steps = np.arange(0, 1 + self.imbalance_step_size, self.imbalance_step_size)
        if not self.imbalance_end_points:
            imbalance_steps = imbalance_steps[1:-1]
        return imbalance_steps

    @property
    def price_delta_steps(self) -> list:
        return np.arange(-self.max_price_delta, self.max_price_delta + self.spread_size, self.spread_size)

    @property
    def mid_price_steps(self) -> list:
        return np.arange(-self.max_mid_price_move, self.max_mid_price_move + self.spread_size, self.spread_size)

    @property
    def orderbook_states(self) -> List:
        return [(x, y) for x in self.imbalance_steps for y in self.price_delta_steps]

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

    def _discretise_orderbook(self):
        orderbook = self.raw_orderbook.drop_duplicates(["datetime"], keep="last").set_index("datetime")
        orderbook = orderbook.reindex(self.time_steps, method="ffill")
        orderbook["imbalance"] = self._calculate_discrete_imbalance(orderbook["imbalance"])
        orderbook["mid_price_move"] = self._calculate_raw_mid_price_move(orderbook)
        orderbook["mid_price_move"] = self._clip_max_mid_price_move(orderbook["mid_price_move"])
        orderbook["mid_price_delta"] = self._clip_max_mid_price_delta(
            self._calculate_discrete_mid_price_delta(orderbook)
        )
        orderbook["next_mid_price_delta"] = orderbook["mid_price_delta"].shift(-1)
        orderbook["state"] = [state for state in list(zip(orderbook["imbalance"], orderbook["mid_price_delta"]))]
        orderbook["next_state"] = orderbook["state"].shift(-1).values

        orderbook["next_spread"] = orderbook["spread"].shift(-1).values
        orderbook = orderbook[orderbook["spread"] == self.spread_size]
        orderbook = orderbook[orderbook["next_spread"] == self.spread_size]
        # Only consider cases where spread is 1 tick because we do not place quotes in these cases, so not a fair
        # representation of the relevant micro-price

        return orderbook.dropna()

    def _calculate_discrete_imbalance(self, imbalance: pd.Series) -> pd.Series:
        return round(imbalance / self.imbalance_step_size) * self.imbalance_step_size

    def _calculate_discrete_mid_price(self, mid_prices: pd.Series) -> pd.Series:
        return round(mid_prices / (0.5 * self.spread_size)) * 0.5 * self.spread_size

    def _calculate_discrete_spread(self, spreads: pd.Series) -> pd.Series:
        return round(spreads / self.spread_size) * self.spread_size

    def _clip_max_mid_price_move(self, mid_price_moves: pd.Series) -> pd.Series:
        # * 2 to round half tick jumps to full tick jumps
        return np.maximum(np.minimum(mid_price_moves, self.max_mid_price_move), -self.max_mid_price_move)

    def _clip_max_mid_price_delta(self, mid_price_deltas: pd.Series) -> pd.Series:
        return np.maximum(np.minimum(mid_price_deltas, self.max_price_delta), -self.max_price_delta)

    def _calculate_raw_mid_price_delta(self, orderbook) -> pd.Series:
        shift_quantity = int(self.price_delta_window / self.time_step_size)
        return orderbook["mid_price"] - orderbook["mid_price"].shift(shift_quantity)

    def _calculate_discrete_mid_price_delta(self, orderbook) -> pd.Series:
        raw_mid_price_delta = self._calculate_raw_mid_price_delta(orderbook)
        return round(raw_mid_price_delta / self.spread_size) * self.spread_size

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
