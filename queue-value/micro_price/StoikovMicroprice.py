from typing import List

import numpy as np
import pandas as pd

from scipy.linalg import eig
from scipy.sparse import csc_matrix, identity, linalg

from orderbook_aggregator.StoikovAggregator import StoikovAggregator


class StoikovMicroPrice:
    def __init__(self, stoikov_aggregations: List[StoikovAggregator]) -> None:
        self.imbalance_steps: np.array = None
        self.spread_steps: np.array = None
        self.mid_price_steps: np.array = None
        self.orderbook_states: np.array = None
        self.absorbing_states = None
        self.absorbing_count = 0
        self.transient_states = None
        self.transient_count = 0
        self.absorbing_states_2 = None
        self.absorbing_2_count = 0
        self.g1 = None
        self.B = None
        self.micro_price = None
        self.build_from_orderbooks(stoikov_aggregations)

    def build_from_orderbooks(self, stoikov_aggregations: List[StoikovAggregator]) -> None:
        for orderbook in stoikov_aggregations:
            self._update_imbalance_steps(orderbook)
            self._update_spread_steps(orderbook)
            self._update_mid_price_steps(orderbook)
            self._update_orderbook_states(orderbook)
            self._update_absorbing_states(orderbook)
            self._update_transient_states(orderbook)
            self._update_absorbing_states_2(orderbook)

    def _update_imbalance_steps(self, stoikov_aggregation: StoikovAggregator) -> None:
        if self.imbalance_steps is None:
            self.imbalance_steps = stoikov_aggregation.imbalance_steps
        else:
            assert all(
                self.imbalance_steps == stoikov_aggregation.imbalance_steps
            ), f"Differing imbalance steps for date {stoikov_aggregation.date}"

    def _update_spread_steps(self, stoikov_aggregation: StoikovAggregator) -> None:
        if self.spread_steps is None:
            self.spread_steps = stoikov_aggregation.spread_steps
        else:
            assert all(
                self.spread_steps == stoikov_aggregation.spread_steps
            ), f"Differing imbalance steps for date {stoikov_aggregation.date}"

    def _update_mid_price_steps(self, stoikov_aggregation: StoikovAggregator) -> None:
        if self.mid_price_steps is None:
            self.mid_price_steps = stoikov_aggregation.mid_price_steps
        else:
            assert all(
                self.mid_price_steps == stoikov_aggregation.mid_price_steps
            ), f"Differing imbalance steps for date {stoikov_aggregation.date}"

    def _update_orderbook_states(self, stoikov_aggregation: StoikovAggregator) -> None:
        if self.orderbook_states is None:
            self.orderbook_states = stoikov_aggregation.orderbook_states
        else:
            pass

    def _update_absorbing_states(self, stoikov_aggregation: StoikovAggregator) -> None:
        if self.absorbing_states is None:
            (
                self.absorbing_states,
                self.absorbing_count,
            ) = stoikov_aggregation.absorbing_states()
            self.absorbing_states = pd.DataFrame(
                self.absorbing_states.todense(),
                index=self.orderbook_states,
                columns=self.mid_price_steps,
            )
        else:
            (
                next_absorbing_states,
                next_absorbing_count,
            ) = stoikov_aggregation.absorbing_states()
            next_absorbing_states = pd.DataFrame(
                next_absorbing_states.todense(),
                index=self.orderbook_states,
                columns=self.mid_price_steps,
            )
            self.absorbing_states = (
                (
                    self.absorbing_states.mul(self.absorbing_count, axis=0)
                    + next_absorbing_states.mul(next_absorbing_count, axis=0)
                )
                .div((self.absorbing_count + next_absorbing_count), axis=0)
                .fillna(0)
            )
            self.absorbing_count += next_absorbing_count

    def _update_transient_states(self, stoikov_aggregation: StoikovAggregator) -> None:
        if self.transient_states is None:
            (
                self.transient_states,
                self.transient_count,
            ) = stoikov_aggregation.transient_states()
            self.transient_states = pd.DataFrame(
                self.transient_states.todense(),
                index=self.orderbook_states,
                columns=self.orderbook_states,
            )
        else:
            (
                next_transient_states,
                next_transient_count,
            ) = stoikov_aggregation.transient_states()
            next_transient_states = pd.DataFrame(
                next_transient_states.todense(),
                index=self.orderbook_states,
                columns=self.orderbook_states,
            )
            self.transient_states = (
                (
                    self.transient_states.mul(self.transient_count, axis=0)
                    + next_transient_states.mul(next_transient_count, axis=0)
                )
                .div((self.transient_count + next_transient_count), axis=0)
                .fillna(0)
            )
            self.transient_count += next_transient_count

    def _update_absorbing_states_2(self, stoikov_aggregation: StoikovAggregator) -> None:
        if self.absorbing_states_2 is None:
            (
                self.absorbing_states_2,
                self.absorbing_2_count,
            ) = stoikov_aggregation.absorbing_states_2()
            self.absorbing_states_2 = pd.DataFrame(
                self.absorbing_states_2.todense(),
                index=self.orderbook_states,
                columns=self.orderbook_states,
            )
        else:
            (
                next_absorbing_states_2,
                next_absorbing_2_count,
            ) = stoikov_aggregation.absorbing_states_2()
            next_absorbing_states_2 = pd.DataFrame(
                next_absorbing_states_2.todense(),
                index=self.orderbook_states,
                columns=self.orderbook_states,
            )
            self.absorbing_states_2 = (
                (
                    self.absorbing_states_2.mul(self.absorbing_2_count, axis=0)
                    + next_absorbing_states_2.mul(next_absorbing_2_count, axis=0)
                )
                .div((self.absorbing_2_count + next_absorbing_2_count), axis=0)
                .fillna(0)
            )
            self.absorbing_2_count += next_absorbing_2_count

    def calculate_g1(self) -> csc_matrix:
        mid_price_steps = csc_matrix(self.mid_price_steps.reshape((-1, 1)))
        transient_sum = identity(len(self.transient_states), format="csc") - csc_matrix(self.transient_states.values)
        abs_mid = csc_matrix(self.absorbing_states.values) * mid_price_steps
        self.g1 = csc_matrix(linalg.spsolve(transient_sum, abs_mid).reshape((-1, 1)))
        return self.g1

    def calculate_B(self) -> csc_matrix:
        transient_sum = identity(len(self.transient_states), format="csc") - csc_matrix(self.transient_states.values)
        self.B = csc_matrix(linalg.spsolve(transient_sum, csc_matrix(self.absorbing_states_2.values)))
        return self.B

    def calculate_micro_price(self) -> pd.DataFrame:
        G1 = self.calculate_g1()
        B = self.calculate_B()
        right_eig_vals, right_eig_vecs = eig(B.todense(), left=False, right=True)
        left_eig_vals, left_eig_vecs = eig(B.todense(), left=True, right=False)
        assert all(right_eig_vals == left_eig_vals), "Left and right eigenvalues do not match."
        assert (
            np.imag(right_eig_vals).max() < 0.1
        ), f"Large imaginary value in eigenvalues: {np.imag(right_eig_vals).max()}"
        eigenvalues = np.real(right_eig_vals)[1:]  # Don't include first eigenvalue, eigenvector pair in sum
        right_eig_vecs = right_eig_vecs[1:]
        left_eig_vecs = left_eig_vecs[1:]
        micro_price = G1
        for i in range(len(self.orderbook_states) - 1):  # Don't include first eigenvalue, eigenvector pair in sum
            eig_multiplier = eigenvalues[i] / (1 - eigenvalues[i])
            micro_price += (
                csc_matrix(eig_multiplier * left_eig_vecs[i].reshape((-1, 1)))
                .dot(csc_matrix(right_eig_vecs[i]))
                .dot(G1)
            )
        micro_price = micro_price.todense().reshape((len(self.imbalance_steps), len(self.spread_steps)))
        return pd.DataFrame(
            data=np.real(micro_price),
            index=[round(x*10) / 10 for x in self.imbalance_steps],
            columns=self.spread_steps
        )
