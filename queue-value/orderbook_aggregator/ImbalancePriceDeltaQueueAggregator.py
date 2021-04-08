from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data_processing.lobster.process_lobster_data import PROCESSED_DATA_DIR
from orderbook_aggregator.OrderbookAggregator import OrderbookAggregator


class ImbalancePriceDeltaQueueAggregator(OrderbookAggregator):
    def __init__(
        self,
        date: datetime,
        ticker: str,
        levels: int,
        spread_size: float,
        imbalance_step_size: Optional[float] = 0.25,
        imbalance_end_points: bool = True,
        price_delta_window: int = 1000,
        max_price_delta: int = 200,
        queue_position_step_size: Optional[float] = 300,
        max_queue_position: float = 3_000,
    ):
        self.date = date
        self.ticker = ticker
        self.levels = levels
        self.spread_size = spread_size
        self.imbalance_step_size = imbalance_step_size
        self.imbalance_end_points = imbalance_end_points
        self.price_delta_window = price_delta_window
        self.max_price_delta = max_price_delta
        self.queue_position_step_size = queue_position_step_size
        self.max_queue_position = max_queue_position
        self.time_step_size = 100  # Given in milliseconds
        self.raw_orderbook = self._build_raw_orderbook()
        self.start_time = datetime.combine(self.date, time(9, 45))
        self.end_time = datetime.combine(self.date, time(15, 45))  # trim noisy first and last 15 minutes from data
        self.orderbook = self._discretise_orderbook()

    def _build_raw_orderbook(self) -> pd.DataFrame:
        file_name = f"{self.ticker}_{str(self.date.date())}_orderbook_{self.levels}.csv"
        raw_orderbook = pd.read_csv(PROCESSED_DATA_DIR + f"/{self.ticker}/" + file_name)
        raw_orderbook = self._process_raw_orderbook(raw_orderbook)
        self.date = min(raw_orderbook["datetime"]).date()
        return raw_orderbook

    def time_steps(self, start_buffer: timedelta = timedelta(seconds=0)) -> list:
        step_size = timedelta(milliseconds=self.time_step_size)
        start_time = max(self.start_time, min(self.raw_orderbook["datetime"])) - start_buffer
        end_time = min(self.end_time, max(self.raw_orderbook["datetime"]))
        return list(pd.date_range(start_time, end_time, freq=step_size))

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
    def queue_position_steps(self) -> list:
        return np.arange(0, self.max_queue_position + self.queue_position_step_size, self.queue_position_step_size)

    @property
    def mid_price_steps(self) -> list:
        return [-self.spread_size, 0, self.spread_size]

    @property
    def orderbook_states(self) -> list:
        return [
            (w, x, y, z)
            for w in self.mid_price_steps
            for x in self.imbalance_steps
            for y in self.price_delta_steps
            for z in self.queue_position_steps
        ]

    @property
    def orderbook_starting_states(self) -> list:
        return [
            (0.0, x, y, z)
            for x in self.imbalance_steps
            for y in self.price_delta_steps
            for z in self.queue_position_steps
            if z > 0
        ]

    @property
    def orderbook_absorbing_states(self) -> list:
        return [
            (w, x, y, z)
            for w in self.mid_price_steps  # Separate case where mid-price move == -spread_size (?)
            for x in self.imbalance_steps
            for y in self.price_delta_steps
            for z in self.queue_position_steps
            if z == 0
        ] + [
            (-self.spread_size, -1, -1, -1)
        ]  # Represents tick down that was not filled

    @property
    def orderbook_transient_states(self) -> list:
        return [
            (0.0, x, y, z)
            for x in self.imbalance_steps
            for y in self.price_delta_steps
            for z in self.queue_position_steps
            if (z > 0)
        ]

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
        orderbook = orderbook.reindex(
            self.time_steps(start_buffer=timedelta(milliseconds=self.price_delta_window)), method="ffill"
        )
        orderbook["imbalance"] = self._calculate_discrete_imbalance(orderbook["imbalance"])
        orderbook["next_imbalance"] = orderbook["imbalance"].shift(-1)
        orderbook["mid_price"] = self._calculate_discrete_mid_price(orderbook["mid_price"])
        orderbook["mid_price_move"] = self._clip_max_mid_price_move(self._calculate_raw_mid_price_move(orderbook))
        orderbook["prev_mid_move"] = self._clip_max_mid_price_move(self._calculate_raw_prev_mid_move(orderbook))
        orderbook["mid_price_delta"] = self._clip_max_mid_price_delta(
            self._calculate_discrete_mid_price_delta(orderbook)
        )
        orderbook["next_mid_price_delta"] = orderbook["mid_price_delta"].shift(-1)
        orderbook = orderbook.dropna()
        orderbook = orderbook.drop(columns=["event_type", "size", "price", "direction"])

        event_quantities = self.raw_orderbook.copy()
        event_quantities["time_bucket"] = event_quantities["datetime"].dt.floor(f"{self.time_step_size}ms")
        event_quantities = event_quantities[
            [self.start_time < t < self.end_time for t in event_quantities["time_bucket"]]
        ]
        event_quantities["ask_touch"] = orderbook.loc[event_quantities["time_bucket"].values]["ask_price_0"].values
        event_quantities["bid_touch"] = orderbook.loc[event_quantities["time_bucket"].values]["bid_price_0"].values
        event_quantities = event_quantities[
            (event_quantities["price"] == event_quantities["ask_touch"])
            | (event_quantities["price"] == event_quantities["bid_touch"])
        ]
        event_quantities = (
            event_quantities.groupby(["time_bucket", "event_type", "direction"])
            .sum()["size"]
            .unstack(level=-1)
            .unstack(level=-1)
            .fillna(0)
        )  # Sum if price == ask_price_0
        event_quantities.columns = [f"event_{c}" for c in event_quantities.columns]
        event_quantities = pd.DataFrame(
            index=event_quantities.index,
            columns=["limit_buy", "limit_sell", "cancel_buy", "cancel_sell", "market_buy", "market_sell"],
            data=np.array(
                [
                    event_quantities["event_(1, 1)"].values,
                    event_quantities["event_(-1, 2)"].values,
                    (event_quantities["event_(1, 2)"] + event_quantities["event_(1, 3)"]).values,
                    (event_quantities["event_(-1, 2)"] + event_quantities["event_(-1, 3)"]).values,
                    (event_quantities["event_(-1, 4)"] + event_quantities["event_(-1, 5)"]).values,
                    (event_quantities["event_(1, 4)"] + event_quantities["event_(1, 5)"]).values,
                ]
            ).T,
        )

        orderbook["next_spread"] = orderbook["spread"].shift(-1).values
        orderbook = orderbook[orderbook["spread"] == self.spread_size]  # Ignore states with initial spread > 1 tick
        orderbook = orderbook[orderbook["next_spread"] == self.spread_size]

        return orderbook.join(event_quantities).fillna(0)

    def _calculate_discrete_imbalance(self, imbalance: pd.Series) -> pd.Series:
        return round(imbalance / self.imbalance_step_size) * self.imbalance_step_size

    def _calculate_discrete_mid_price(self, mid_prices: pd.Series) -> pd.Series:
        return round(mid_prices / (0.5 * self.spread_size)) * 0.5 * self.spread_size

    def _clip_max_mid_price_move(self, mid_price_moves: pd.Series) -> pd.Series:
        # * 2 to round half tick jumps to full tick jumps
        return np.maximum(np.minimum(2 * mid_price_moves, self.spread_size), -self.spread_size)

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

    @staticmethod
    def _calculate_raw_prev_mid_move(orderbook) -> pd.Series:
        return orderbook["mid_price"] - orderbook["mid_price"].shift(1)

    @staticmethod
    def round_up_to_step_size(value: float, step_size: float) -> float:
        return np.ceil(value / step_size) * step_size
