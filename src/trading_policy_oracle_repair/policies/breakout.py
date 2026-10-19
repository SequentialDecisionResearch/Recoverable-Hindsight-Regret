from __future__ import annotations

import pandas as pd

from ..state import AccountState
from .base import Action, BasePolicy


class DonchianBreakoutPolicy(BasePolicy):
    """Restartable long/cash Donchian rule.

    Entry uses today's close versus the maximum *prior* high over entry_window.
    Exit uses today's close versus the minimum *prior* low over exit_window.
    The current day is excluded from both channel thresholds.
    """

    def __init__(self, entry_window: int, exit_window: int, policy_id: str = "donchian") -> None:
        if entry_window < 1 or exit_window < 1:
            raise ValueError("Donchian windows must be positive")
        self.entry_window = int(entry_window)
        self.exit_window = int(exit_window)
        self.policy_id = policy_id

    def act(self, history: pd.DataFrame, state: AccountState) -> Action:
        h = self._validate_history(history, state)
        for col in ["high", "low"]:
            if col not in h.columns:
                raise ValueError(f"Donchian policy requires {col}")
        needed = max(self.entry_window, self.exit_window) + 1
        if len(h) < needed:
            return self._warmup_action(h, state, needed_rows=needed)

        current_exposure = self._current_exposure(h, state)
        close = float(h.iloc[-1]["close"])
        prior_entry_high = float(h["high"].iloc[-(self.entry_window + 1) : -1].max())
        prior_exit_low = float(h["low"].iloc[-(self.exit_window + 1) : -1].min())

        if close > prior_entry_high:
            target = 1.0
            reason = "entry_breakout"
        elif close < prior_exit_low:
            target = 0.0
            reason = "exit_breakdown"
        else:
            target = current_exposure
            reason = "inside_channel_hold"

        return Action(
            policy_id=self.policy_id,
            decision_date=pd.Timestamp(h.iloc[-1]["date"]).normalize(),
            target_exposure=target,
            ready=True,
            reason=reason,
            diagnostics={
                "close": close,
                "prior_entry_high": prior_entry_high,
                "prior_exit_low": prior_exit_low,
            },
        )

    def definition(self) -> dict:
        return {
            "id": self.policy_id,
            "type": "breakout",
            "entry_window": self.entry_window,
            "exit_window": self.exit_window,
        }
