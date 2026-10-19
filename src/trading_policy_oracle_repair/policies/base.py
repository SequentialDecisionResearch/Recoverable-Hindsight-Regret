from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..dates import normalize_daily_dates
from ..state import AccountState


@dataclass(frozen=True)
class Action:
    """A policy decision produced at a daily close.

    Policies may only propose a target exposure. They never mutate account cash,
    shares, or the execution engine. `ready=False` means an indicator is still in
    warmup; the canonical warmup behavior is to hold the current exposure.
    """

    policy_id: str
    decision_date: pd.Timestamp
    target_exposure: float
    ready: bool
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.target_exposure) <= 1.0:
            raise ValueError("target_exposure must lie in [0, 1]")


class BasePolicy(ABC):
    policy_id: str

    @abstractmethod
    def act(self, history: pd.DataFrame, state: AccountState) -> Action:
        raise NotImplementedError

    @abstractmethod
    def definition(self) -> dict[str, Any]:
        raise NotImplementedError

    def _validate_history(self, history: pd.DataFrame, state: AccountState) -> pd.DataFrame:
        if history is None or history.empty:
            raise ValueError("history may not be empty")
        required = {"date", "asset_id", "close", "signal_price"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"history missing required columns: {sorted(missing)}")

        # Phase 4 replay validates the entire market path once, then passes
        # past-only slices marked as prevalidated. This preserves the exact
        # policy API while avoiding an O(T^2) sequence of full-frame copies.
        if bool(history.attrs.get("_policy_history_prevalidated", False)):
            h = history
        else:
            h = history.copy()
            h["date"] = normalize_daily_dates(h["date"])
            h = h.sort_values("date").reset_index(drop=True)
            if h["date"].duplicated().any():
                raise ValueError("history contains duplicate dates")
            if h["asset_id"].astype(str).nunique() != 1:
                raise ValueError("policy history must contain exactly one asset")
            for col in ["close", "signal_price"]:
                values = pd.to_numeric(h[col], errors="raise").astype(float)
                if (~np.isfinite(values)).any() or (values <= 0).any():
                    raise ValueError(f"history contains invalid {col}")
                h[col] = values

        last_date = pd.Timestamp(h.iloc[-1]["date"]).normalize()
        state_date = pd.Timestamp(state.date).normalize()
        if state_date != last_date:
            raise ValueError(
                f"state date {state_date.date()} must equal history close date {last_date.date()}"
            )
        if str(state.asset_id) != str(h.iloc[-1]["asset_id"]):
            raise ValueError("state asset_id does not match history")
        return h

    @staticmethod
    def _current_exposure(history: pd.DataFrame, state: AccountState) -> float:
        close = float(history.iloc[-1]["close"])
        exposure = float(state.exposure(close))
        return float(np.clip(exposure, 0.0, 1.0))

    def _warmup_action(
        self,
        history: pd.DataFrame,
        state: AccountState,
        *,
        needed_rows: int,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> Action:
        current = self._current_exposure(history, state)
        return Action(
            policy_id=self.policy_id,
            decision_date=pd.Timestamp(history.iloc[-1]["date"]).normalize(),
            target_exposure=current,
            ready=False,
            reason=f"warmup: need {needed_rows} rows, have {len(history)}",
            diagnostics=dict(diagnostics or {}),
        )
