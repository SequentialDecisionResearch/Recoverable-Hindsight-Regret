from __future__ import annotations

import numpy as np
import pandas as pd

from ..state import AccountState
from .base import Action, BasePolicy


def wilder_rsi(prices: pd.Series, window: int) -> float:
    values = pd.to_numeric(prices, errors="raise").astype(float).to_numpy()
    if len(values) < window + 1:
        raise ValueError("Not enough prices for RSI")
    deltas = np.diff(values)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)

    avg_gain = float(gains[:window].mean())
    avg_loss = float(losses[:window].mean())
    for i in range(window, len(deltas)):
        avg_gain = ((window - 1) * avg_gain + float(gains[i])) / window
        avg_loss = ((window - 1) * avg_loss + float(losses[i])) / window

    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


class RSIMeanReversionPolicy(BasePolicy):
    def __init__(
        self,
        window: int,
        entry: float,
        exit: float,
        policy_id: str = "rsi_mean_reversion",
    ) -> None:
        if window < 1:
            raise ValueError("RSI window must be positive")
        if not 0.0 <= entry < exit <= 100.0:
            raise ValueError("Require 0 <= entry < exit <= 100")
        self.window = int(window)
        self.entry = float(entry)
        self.exit = float(exit)
        self.policy_id = policy_id

    def act(self, history: pd.DataFrame, state: AccountState) -> Action:
        h = self._validate_history(history, state)
        needed = self.window + 1
        if len(h) < needed:
            return self._warmup_action(h, state, needed_rows=needed)

        rsi = wilder_rsi(h["signal_price"], self.window)
        current = self._current_exposure(h, state)
        if rsi < self.entry:
            target = 1.0
            reason = "rsi_entry"
        elif rsi > self.exit:
            target = 0.0
            reason = "rsi_exit"
        else:
            target = current
            reason = "rsi_band_hold"

        return Action(
            policy_id=self.policy_id,
            decision_date=pd.Timestamp(h.iloc[-1]["date"]).normalize(),
            target_exposure=target,
            ready=True,
            reason=reason,
            diagnostics={"rsi": rsi, "entry": self.entry, "exit": self.exit},
        )

    def definition(self) -> dict:
        return {
            "id": self.policy_id,
            "type": "rsi_mean_reversion",
            "window": self.window,
            "entry": self.entry,
            "exit": self.exit,
        }
