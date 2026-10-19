from __future__ import annotations

import pandas as pd

from ..state import AccountState
from .base import Action, BasePolicy


class MomentumPolicy(BasePolicy):
    def __init__(self, lookback: int, policy_id: str = "momentum") -> None:
        if lookback < 1:
            raise ValueError("lookback must be positive")
        self.lookback = int(lookback)
        self.policy_id = policy_id

    def act(self, history: pd.DataFrame, state: AccountState) -> Action:
        h = self._validate_history(history, state)
        needed = self.lookback + 1
        if len(h) < needed:
            return self._warmup_action(h, state, needed_rows=needed)
        p = h["signal_price"]
        past = float(p.iloc[-(self.lookback + 1)])
        current = float(p.iloc[-1])
        momentum_return = current / past - 1.0
        target = 1.0 if momentum_return > 0.0 else 0.0
        return Action(
            policy_id=self.policy_id,
            decision_date=pd.Timestamp(h.iloc[-1]["date"]).normalize(),
            target_exposure=target,
            ready=True,
            reason="positive_momentum" if target == 1.0 else "nonpositive_momentum",
            diagnostics={"lookback_return": momentum_return},
        )

    def definition(self) -> dict:
        return {"id": self.policy_id, "type": "momentum", "lookback": self.lookback}
