from __future__ import annotations

import pandas as pd

from ..state import AccountState
from .base import Action, BasePolicy


class SMACrossPolicy(BasePolicy):
    def __init__(self, short_window: int, long_window: int, policy_id: str = "sma_cross") -> None:
        if short_window < 1 or long_window < 2 or short_window >= long_window:
            raise ValueError("Require 1 <= short_window < long_window")
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.policy_id = policy_id

    def act(self, history: pd.DataFrame, state: AccountState) -> Action:
        h = self._validate_history(history, state)
        if len(h) < self.long_window:
            return self._warmup_action(h, state, needed_rows=self.long_window)
        p = h["signal_price"]
        short_sma = float(p.iloc[-self.short_window :].mean())
        long_sma = float(p.iloc[-self.long_window :].mean())
        target = 1.0 if short_sma > long_sma else 0.0
        return Action(
            policy_id=self.policy_id,
            decision_date=pd.Timestamp(h.iloc[-1]["date"]).normalize(),
            target_exposure=target,
            ready=True,
            reason="short_sma_above_long_sma" if target == 1.0 else "short_sma_not_above_long_sma",
            diagnostics={"short_sma": short_sma, "long_sma": long_sma, "margin": short_sma - long_sma},
        )

    def definition(self) -> dict:
        return {
            "id": self.policy_id,
            "type": "sma_cross",
            "short_window": self.short_window,
            "long_window": self.long_window,
        }
