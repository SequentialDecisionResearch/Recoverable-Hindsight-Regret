from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .bounded_switch import (
    BinaryTransitionTable,
    OracleResult,
    build_binary_transition_table,
    solve_bounded_switch_from_transitions,
)


@dataclass(frozen=True)
class LocalContinuationResult:
    decision_date: pd.Timestamp
    horizon: int
    initial_position: int
    max_switches: int
    best: OracleResult
    q_action_0: OracleResult
    q_action_1: OracleResult

    @property
    def best_log_utility(self) -> float:
        return float(self.best.log_utility)

    def action_regret(self, action: int) -> float:
        if int(action) == 0:
            q = self.q_action_0.log_utility
        elif int(action) == 1:
            q = self.q_action_1.log_utility
        else:
            raise ValueError("action must be 0 or 1")
        regret = self.best.log_utility - q
        return float(max(regret, 0.0))


def slice_transitions(
    transitions: BinaryTransitionTable,
    *,
    decision_index: int,
    horizon: int,
) -> BinaryTransitionTable:
    if decision_index < 0:
        raise ValueError("decision_index must be non-negative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    end = decision_index + horizon
    if end > transitions.n_intervals:
        raise ValueError("Local horizon extends beyond available market path")
    return BinaryTransitionTable(
        decision_dates=transitions.decision_dates[decision_index:end],
        execution_dates=transitions.execution_dates[decision_index:end],
        log_gross=transitions.log_gross[decision_index:end].copy(),
    )


def solve_local_continuation_from_transitions(
    transitions: BinaryTransitionTable,
    *,
    decision_index: int,
    horizon: int,
    initial_position: int,
    max_switches: int,
) -> LocalContinuationResult:
    local = slice_transitions(transitions, decision_index=decision_index, horizon=horizon)
    best = solve_bounded_switch_from_transitions(
        local, max_switches=max_switches, initial_position=initial_position
    )
    q0 = solve_bounded_switch_from_transitions(
        local,
        max_switches=max_switches,
        initial_position=initial_position,
        forced_first_action=0,
    )
    q1 = solve_bounded_switch_from_transitions(
        local,
        max_switches=max_switches,
        initial_position=initial_position,
        forced_first_action=1,
    )
    if abs(best.log_utility - max(q0.log_utility, q1.log_utility)) > 1e-12:
        raise AssertionError("Unconstrained local oracle must equal max over first actions")
    return LocalContinuationResult(
        decision_date=local.decision_dates[0],
        horizon=int(horizon),
        initial_position=int(initial_position),
        max_switches=min(int(max_switches), int(horizon)),
        best=best,
        q_action_0=q0,
        q_action_1=q1,
    )


def solve_local_continuation_oracle(
    asset_market: pd.DataFrame,
    *,
    decision_index: int,
    horizon: int,
    initial_position: int,
    max_switches: int,
    cost_bps_per_side: float,
) -> LocalContinuationResult:
    transitions = build_binary_transition_table(asset_market, cost_bps_per_side=cost_bps_per_side)
    return solve_local_continuation_from_transitions(
        transitions,
        decision_index=decision_index,
        horizon=horizon,
        initial_position=initial_position,
        max_switches=max_switches,
    )
