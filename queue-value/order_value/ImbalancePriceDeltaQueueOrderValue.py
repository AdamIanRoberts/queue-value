from tqdm import tqdm
from typing import List, Dict

import numpy as np
import pandas as pd

from scipy.sparse import identity

from orderbook_aggregator.ImbalancePriceDeltaQueueAggregator import ImbalancePriceDeltaQueueAggregator


class ImbalancePriceDeltaQueueOrderValue:
    def __init__(self, orderbooks: List[ImbalancePriceDeltaQueueAggregator], micro_prices: List[float]) -> None:
        self.orderbooks = orderbooks
        assert (
            len(self.orderbooks) > 0
        ), "ImbalanceQueueOrderValue must be instantiated with at least one ImbalanceQueueAggregator"
        self.spread_size = self.orderbooks[0].spread_size
        self.orderbook_absorbing_states = self.orderbooks[0].orderbook_absorbing_states
        self.orderbook_transient_states = self.orderbooks[0].orderbook_transient_states
        self.orderbook_starting_states = self.orderbooks[0].orderbook_starting_states
        self.absorbing_count = None
        self.transient_count = None
        self.absorbing_prob_df = None
        self.transient_prob_df = None
        self.final_state_values = self._calculate_final_state_values(micro_prices)
        self.calculate_absorbing_and_transient_probs()

    def calculate_absorbing_and_transient_probs(self) -> None:
        self.absorbing_count: Dict[tuple, Dict[tuple, float]] = {
            s_0: {s_1: 0 for s_1 in self.orderbook_absorbing_states} for s_0 in self.orderbook_starting_states
        }
        self.transient_count: Dict[tuple, Dict[tuple, float]] = {
            s_0: {s_1: 0 for s_1 in self.orderbook_transient_states} for s_0 in self.orderbook_starting_states
        }
        for orderbook in tqdm(self.orderbooks):
            for idx, row in orderbook.orderbook.iterrows():
                state_changes = self._next_states(row, orderbook)
                for k, v in state_changes.items():
                    if v[0] > 0:
                        # tick up so we get filled
                        self.absorbing_count[k][(v[0], v[1], v[2], 0.0)] += 1
                    elif v[0] == 0:
                        # no mid-price move
                        if v[3] == 0:
                            self.absorbing_count[k][v] += 1
                        else:
                            self.transient_count[k][v] += 1
                    else:
                        if v[3] == 0:
                            self.absorbing_count[k][v] += 1
                        else:  # tick down and we did not get filled
                            self.absorbing_count[k][(-orderbook.spread_size, -1, -1, -1)] += 1
        absorbing_df = pd.DataFrame.from_dict(self.absorbing_count).T
        transient_df = pd.DataFrame.from_dict(self.transient_count).T
        all_df = transient_df.join(absorbing_df)
        self.absorbing_prob_df = absorbing_df.div(all_df.sum(axis=1), axis=0).fillna(0)
        self.transient_prob_df = transient_df.div(all_df.sum(axis=1), axis=0).fillna(0)

    def calculate_order_values(self) -> pd.DataFrame:
        transient_sum = identity(len(self.transient_prob_df.columns)).todense() - self.transient_prob_df.values
        inv_trans_sum = pd.DataFrame(
            index=self.transient_prob_df.index,
            columns=self.transient_prob_df.columns,
            data=np.linalg.inv(transient_sum),
        ).round(3)
        abs_delta_product = np.dot(self.absorbing_prob_df.values, self.final_state_values)
        queue_value = np.dot(inv_trans_sum, abs_delta_product)
        queue_value_df = pd.DataFrame(index=self.transient_prob_df.index, columns=["queue_value"], data=queue_value.T)
        return queue_value_df.unstack().T.droplevel(0).droplevel(0, axis=1)

    def _calculate_final_state_values(self, micro_prices: List[float]) -> List[float]:
        tick_down_deltas = [self.spread_size * 1.5 - micro_price for micro_price in micro_prices]
        no_tick_deltas = [0.5 * self.spread_size - micro_price for micro_price in micro_prices]
        tick_up_deltas = [micro_price - 0.5 * self.spread_size for micro_price in micro_prices[::-1]]
        no_fill_value = [0.0]
        return tick_down_deltas + no_tick_deltas + tick_up_deltas + no_fill_value

    @classmethod
    def _next_states(cls, row: pd.Series, orderbook: ImbalancePriceDeltaQueueAggregator) -> Dict[tuple, tuple]:
        return {
            (0.0, row["imbalance"], row["mid_price_delta"], q): (
                row["mid_price_move"],
                row["next_imbalance"],
                row["next_mid_price_delta"],
                cls._calculate_next_queue_position(
                    q, row["cancel_sell"], row["ask_size_0"], row["market_buy"], orderbook
                ),
            )
            for q in orderbook.queue_position_steps
            if (row["ask_size_0"] + orderbook.queue_position_step_size) > q > 0
        }

    @staticmethod
    def _calculate_next_queue_position(
        queue_position: float,
        cancel_qty: float,
        ask_size: float,
        buy_qty: float,
        orderbook: ImbalancePriceDeltaQueueAggregator,
    ) -> float:
        queue_pos_after_cancel = max(queue_position * (1 - cancel_qty / ask_size), 1)
        # Can't be better than front of queue
        return orderbook.round_up_to_step_size(
            min(max(queue_pos_after_cancel - buy_qty, 0), orderbook.max_queue_position),
            orderbook.queue_position_step_size,
        )
